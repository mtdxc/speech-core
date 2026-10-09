#include "speech_core/models/onnx_zipformer_kws.h"
#include "speech_core/models/onnx_engine.h"
#include "speech_core/audio/kws_fbank.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>

namespace speech_core {
namespace {
struct ValueDeleter { const OrtApi* api; void operator()(OrtValue* value) const { if (value) api->ReleaseValue(value); } };
struct SessionDeleter { const OrtApi* api; void operator()(OrtSession* session) const { if (session) api->ReleaseSession(session); } };
using Value = std::unique_ptr<OrtValue, ValueDeleter>;
using Session = std::unique_ptr<OrtSession, SessionDeleter>;
struct TensorSpec { std::string name; std::vector<int64_t> shape; ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };
void require(bool value, const char* message) { if (!value) throw std::invalid_argument(message); }
size_t element_count(const std::vector<int64_t>& shape) {
    require(!shape.empty() && shape.size() <= 5, "Invalid KWS tensor rank");
    size_t count = 1;
    for (auto dimension : shape) {
        require(dimension > 0 && dimension <= 1000000 && count <= 1000000 / static_cast<size_t>(dimension), "Oversized KWS tensor");
        count *= static_cast<size_t>(dimension);
    }
    return count;
}
}

class OnnxZipformerKws::Impl {
public:
    const OrtApi* api = OnnxEngine::get().api();
    OrtAllocator* allocator = nullptr;
    Session encoder{nullptr, {api}}, decoder{nullptr, {api}}, joiner{nullptr, {api}};
    std::vector<TensorSpec> encoder_inputs, encoder_outputs;
    std::vector<Value> states;
    std::unique_ptr<KwsDecoder> search;
    audio::KwsFbank fbank;
    std::vector<float> mel;
    int chunk = 0, frames = 0, dim = 0, vocabulary = 0, context = 0;
    int64_t samples = 0;
    bool failed = false, finished = false;

    Impl(const std::string& directory, std::vector<KwsKeyword> keywords, int beam) {
        std::ifstream input(directory + "/config.json", std::ios::binary);
        require(input.good(), "KWS config.json is required");
        input.seekg(0, std::ios::end);
        require(input.tellg() > 0 && input.tellg() <= 65536, "Invalid KWS configuration size");
        input.seekg(0);
        auto config = nlohmann::json::parse(input);
        const auto& feature = config.at("feature");
        require(config.at("runtime") == "soniqo-kws-onnx-v1" && feature.at("type") == "kaldi-fbank" &&
            feature.at("sampleRate") == 16000 && feature.at("numMelBins") == 80 &&
            feature.at("frameLengthMs") == 25 && feature.at("frameShiftMs") == 10 &&
            feature.at("dither") == 0 && feature.at("snipEdges") == false &&
            feature.at("normalizeSamples") == true && feature.at("highFreq") == -400,
            "Unsupported KWS feature contract");
        const auto& enc = config.at("encoder");
        chunk = enc.at("chunkSize").get<int>(); frames = enc.at("outputFrames").get<int>();
        dim = enc.at("joinerDim").get<int>();
        require(chunk > 0 && chunk <= 64 && chunk % 2 == 0 && frames == chunk / 2 &&
            enc.at("totalInputFrames") == chunk * 2 + 13 && dim > 0 && dim <= 2048,
            "Invalid KWS encoder configuration");
        const auto& dec = config.at("decoder");
        vocabulary = dec.at("vocabSize").get<int>(); context = dec.at("contextSize").get<int>();
        require(vocabulary > 1 && vocabulary <= 65536 && context > 0 && context <= 8 &&
            dec.at("decoderDim") == dim, "Invalid KWS decoder configuration");
        encoder_inputs.push_back({"x", {1, chunk * 2 + 13, 80}});
        encoder_outputs.push_back({"encoder_out", {1, frames, dim}});
        auto names = enc.at("layerStateNames").get<std::vector<std::string>>();
        auto shapes = enc.at("layerStateShapes").get<std::vector<std::vector<int64_t>>>();
        require(!names.empty() && names.size() <= 128 && names.size() == shapes.size(), "Invalid KWS encoder states");
        std::set<std::string> seen{"x", "cached_embed_left_pad", "processed_lens"};
        size_t state_size = 0;
        for (size_t i = 0; i < names.size(); ++i) {
            require(!names[i].empty() && names[i].size() <= 128 && seen.insert(names[i]).second,
                    "Invalid or duplicate KWS state name");
            state_size += element_count(shapes[i]);
            require(state_size <= 1000000, "KWS state budget exceeded");
            encoder_inputs.push_back({names[i], shapes[i]});
            encoder_outputs.push_back({"new_" + names[i], shapes[i]});
        }
        auto pad = enc.at("cachedEmbedLeftPadShape").get<std::vector<int64_t>>();
        element_count(pad);
        encoder_inputs.push_back({"cached_embed_left_pad", pad});
        encoder_outputs.push_back({"new_cached_embed_left_pad", pad});
        encoder_inputs.push_back({"processed_lens", {1}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64});
        encoder_outputs.push_back({"new_processed_lens", {1}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64});
        ort_check(api, api->GetAllocatorWithDefaultOptions(&allocator));
        encoder.reset(OnnxEngine::get().load(directory + "/encoder.onnx", false, false, 1));
        decoder.reset(OnnxEngine::get().load(directory + "/decoder.onnx", false, false, 1));
        joiner.reset(OnnxEngine::get().load(directory + "/joiner.onnx", false, false, 1));
        validate_session(encoder.get(), encoder_inputs, encoder_outputs);
        validate_session(decoder.get(), {{"y", {1, context}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64}}, {{"decoder_out", {1, dim}}});
        validate_session(joiner.get(), {{"encoder_out", {1, dim}}, {"decoder_out", {1, dim}}}, {{"logits", {1, vocabulary}}});
        search = std::make_unique<KwsDecoder>(std::move(keywords),
            [this](const std::vector<int>& tokens) {
                auto value = zeros({"y", {1, context}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64});
                void* data = nullptr; ort_check(api, api->GetTensorMutableData(value.get(), &data));
                auto* integers = static_cast<int64_t*>(data);
                for (size_t i = 0; i < tokens.size(); ++i) integers[i] = tokens[i];
                auto result = run(decoder.get(), {"y"}, {value.get()}, {{"decoder_out", {1, dim}}});
                return floats(result[0].get(), static_cast<size_t>(dim));
            },
            [this](const std::vector<float>& frame, const std::vector<float>& decoded) {
                auto a = float_value({1, dim}, frame.data(), frame.size());
                auto b = float_value({1, dim}, decoded.data(), decoded.size());
                auto result = run(joiner.get(), {"encoder_out", "decoder_out"}, {a.get(), b.get()}, {{"logits", {1, vocabulary}}});
                return floats(result[0].get(), static_cast<size_t>(vocabulary));
            }, vocabulary, dec.at("blankId").get<int>(), context, beam,
            config.at("kws").at("defaultNumTrailingBlanks").get<int>(),
            config.at("kws").at("autoResetSeconds").get<double>());
        reset();
    }

    void validate_shape(const OrtTensorTypeAndShapeInfo* info, const TensorSpec& spec, bool dynamic) {
        ONNXTensorElementDataType type; size_t rank = 0;
        ort_check(api, api->GetTensorElementType(info, &type));
        ort_check(api, api->GetDimensionsCount(info, &rank));
        require(type == spec.type && rank == spec.shape.size(), "KWS tensor type/rank does not match configuration");
        std::vector<int64_t> shape(rank);
        ort_check(api, api->GetDimensions(info, shape.data(), rank));
        for (size_t i = 0; i < rank; ++i) require(shape[i] == spec.shape[i] || (dynamic && shape[i] == -1), "KWS tensor shape does not match configuration");
    }
    void validate_session(OrtSession* session, const std::vector<TensorSpec>& inputs, const std::vector<TensorSpec>& outputs) {
        for (bool output : {false, true}) {
            const auto& specs = output ? outputs : inputs;
            size_t count = 0;
            ort_check(api, output ? api->SessionGetOutputCount(session, &count) : api->SessionGetInputCount(session, &count));
            require(count == specs.size(), "KWS model has unexpected tensor count");
            for (size_t i = 0; i < count; ++i) {
                char* name = nullptr;
                ort_check(api, output ? api->SessionGetOutputName(session, i, allocator, &name) : api->SessionGetInputName(session, i, allocator, &name));
                std::string observed(name); allocator->Free(allocator, name);
                require(observed == specs[i].name, "KWS model tensor names do not match configuration");
                OrtTypeInfo* raw = nullptr;
                ort_check(api, output ? api->SessionGetOutputTypeInfo(session, i, &raw) : api->SessionGetInputTypeInfo(session, i, &raw));
                std::unique_ptr<OrtTypeInfo, void(*)(OrtTypeInfo*)> info(raw, api->ReleaseTypeInfo);
                const OrtTensorTypeAndShapeInfo* tensor = nullptr;
                ort_check(api, api->CastTypeInfoToTensorInfo(raw, &tensor));
                require(tensor != nullptr, "KWS model contains a non-tensor input/output");
                validate_shape(tensor, specs[i], output);
            }
        }
    }
    Value zeros(const TensorSpec& spec) {
        size_t count = element_count(spec.shape);
        OrtValue* raw = nullptr;
        ort_check(api, api->CreateTensorAsOrtValue(allocator, spec.shape.data(), spec.shape.size(), spec.type, &raw));
        Value value(raw, {api}); void* data = nullptr;
        ort_check(api, api->GetTensorMutableData(raw, &data));
        std::memset(data, 0, count * (spec.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ? sizeof(int64_t) : sizeof(float)));
        return value;
    }
    Value float_value(const std::vector<int64_t>& shape, const float* data, size_t count) {
        require(count == element_count(shape), "KWS input size does not match configuration");
        auto value = zeros({"", shape}); void* destination = nullptr;
        ort_check(api, api->GetTensorMutableData(value.get(), &destination));
        std::memcpy(destination, data, count * sizeof(float)); return value;
    }
    std::vector<float> floats(OrtValue* value, size_t count) {
        void* data = nullptr; ort_check(api, api->GetTensorMutableData(value, &data));
        auto* first = static_cast<float*>(data);
        require(std::all_of(first, first + count, [](float x) { return std::isfinite(x); }), "KWS model returned non-finite output");
        return std::vector<float>(first, first + count);
    }
    std::vector<Value> run(OrtSession* session, const std::vector<std::string>& inputs,
        const std::vector<const OrtValue*>& values, const std::vector<TensorSpec>& outputs) {
        std::vector<const char*> in_names, out_names;
        for (const auto& name : inputs) in_names.push_back(name.c_str());
        for (const auto& spec : outputs) out_names.push_back(spec.name.c_str());
        std::vector<OrtValue*> raw(outputs.size(), nullptr);
        OrtStatus* status = api->Run(session, nullptr, in_names.data(), values.data(), values.size(), out_names.data(), outputs.size(), raw.data());
        std::vector<Value> result;
        for (auto* value : raw) result.emplace_back(value, ValueDeleter{api});
        ort_check(api, status);
        for (size_t i = 0; i < result.size(); ++i) {
            OrtTensorTypeAndShapeInfo* raw_info = nullptr;
            ort_check(api, api->GetTensorTypeAndShape(result[i].get(), &raw_info));
            std::unique_ptr<OrtTensorTypeAndShapeInfo, void(*)(OrtTensorTypeAndShapeInfo*)> info(raw_info, api->ReleaseTensorTypeAndShapeInfo);
            validate_shape(raw_info, outputs[i], false);
        }
        return result;
    }
    void reset() {
        states.clear();
        for (size_t i = 1; i < encoder_inputs.size(); ++i) states.push_back(zeros(encoder_inputs[i]));
        fbank.reset(); mel.clear(); search->reset(); samples = 0; failed = finished = false;
    }
    std::vector<KwsDetection> drain() {
        const size_t needed = static_cast<size_t>(chunk * 2 + 13) * 80;
        std::vector<KwsDetection> detections;
        while (mel.size() >= needed) {
            auto x = float_value({1, chunk * 2 + 13, 80}, mel.data(), needed);
            std::vector<const OrtValue*> values{x.get()}; std::vector<std::string> names;
            for (const auto& spec : encoder_inputs) names.push_back(spec.name);
            for (const auto& state : states) values.push_back(state.get());
            auto outputs = run(encoder.get(), names, values, encoder_outputs);
            const size_t frame_size = static_cast<size_t>(dim);
            auto encoded = floats(outputs[0].get(), static_cast<size_t>(frames) * frame_size);
            for (size_t i = 1; i + 1 < outputs.size(); ++i)
                floats(outputs[i].get(), element_count(encoder_outputs[i].shape));
            states.clear();
            for (size_t i = 1; i < outputs.size(); ++i) states.push_back(std::move(outputs[i]));
            for (size_t frame = 0; frame < static_cast<size_t>(frames); ++frame) {
                const auto start = encoded.begin() + frame * frame_size;
                auto hits = search->step(std::vector<float>(start, start + frame_size));
                for (auto& hit : hits) { hit.audio_end_seconds = samples / 16000.0; detections.push_back(std::move(hit)); }
            }
            mel.erase(mel.begin(), mel.begin() + static_cast<size_t>(chunk) * 2 * 80);
        }
        return detections;
    }
    std::vector<KwsDetection> accept(const float* pcm, size_t count, bool final) {
        require(!failed && !finished, "KWS session must be reset after finalization or failure");
        try {
            auto features = final ? fbank.finish() : fbank.push(pcm, count);
            samples += static_cast<int64_t>(count); mel.insert(mel.end(), features.begin(), features.end());
            auto detections = drain();
            if (final) {
                if (!mel.empty()) {
                    mel.resize(static_cast<size_t>(chunk * 2 + 13) * 80, -15);
                    auto tail = drain(); detections.insert(detections.end(), tail.begin(), tail.end());
                }
                finished = true;
            }
            return detections;
        } catch (...) { failed = true; throw; }
    }
};
OnnxZipformerKws::OnnxZipformerKws(const std::string& directory, std::vector<KwsKeyword> keywords, int beam)
    : impl_(std::make_unique<Impl>(directory, std::move(keywords), beam)) {}
OnnxZipformerKws::~OnnxZipformerKws() = default;
std::vector<KwsDetection> OnnxZipformerKws::push(const float* samples, size_t count) { return impl_->accept(samples, count, false); }
std::vector<KwsDetection> OnnxZipformerKws::finish() { return impl_->accept(nullptr, 0, true); }
void OnnxZipformerKws::reset() { impl_->reset(); }
}
