// CosyVoice3 ONNX synthesis example.
//
// Renders text with a prepared conditioning blob (see
// OnnxCosyVoice3Tts::encode_conditioning_blob) whose prompt_text_ids may be
// empty — they are filled here from --transcript via the bundle tokenizer.
//
// Usage:
//   speech_cosyvoice3_synth_onnx <bundle_dir> <conditioning.blob> \
//       <reference transcript> <text> <out.wav> [seed]

#include <speech_core/models/onnx_cosyvoice3_tts.h>
#include <speech_core/audio/wav_io.h>
#include "../common/utf8_args.h"
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream f(std::filesystem::u8path(path), std::ios::binary | std::ios::ate);
    if (!f.good()) throw std::runtime_error("cannot read " + path);
    const std::streamsize n = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> data(static_cast<size_t>(n));
    if (!f.read(reinterpret_cast<char*>(data.data()), n)) {
        throw std::runtime_error("short read " + path);
    }
    return data;
}

}  // namespace

int main(int argc, char** argv) {
    const auto args = speech_examples::utf8_args(argc, argv);
    argc = static_cast<int>(args.size());
    if (argc < 6) {
        std::fprintf(stderr,
            "usage: %s <bundle_dir> <conditioning.blob> <transcript> <text> <out.wav> [seed]\n",
            args.empty() ? "speech_cosyvoice3_synth_onnx" : args[0].c_str());
        return 2;
    }
    const std::string bundle_dir = args[1];
    const std::string blob_path = args[2];
    const std::string transcript = args[3];
    const std::string text = args[4];
    const std::string out_path = args[5];
    const uint32_t seed = argc > 6 ? static_cast<uint32_t>(std::stoul(args[6])) : 7u;

    speech_core::OnnxCosyVoice3Tts tts(bundle_dir, /*hw_accel=*/false);

    const auto blob = read_file(blob_path);
    auto cond = speech_core::OnnxCosyVoice3Tts::decode_conditioning_blob(
        blob.data(), blob.size());
    if (cond.prompt_text_ids.empty()) {
        cond.prompt_text_ids = tts.encode_prompt_text(
            speech_core::OnnxCosyVoice3Tts::prompt_text_from_transcript(transcript));
    }
    tts.set_conditioning(std::move(cond));
    tts.set_seed(seed);

    std::vector<float> pcm;
    tts.synthesize(text, "english", [&](const float* data, size_t n, bool) {
        pcm.insert(pcm.end(), data, data + n);
    });

    speech_core::write_wav_mono_pcm16(out_path, pcm, tts.output_sample_rate());
    std::printf("tokens=%d stop=%d prefill=%lldms ar=%lldms decode=%lldms samples=%zu -> %s\n",
                tts.tokens_generated(), tts.stopped_on_stop_token() ? 1 : 0,
                static_cast<long long>(tts.prefill_ms()),
                static_cast<long long>(tts.ar_ms()),
                static_cast<long long>(tts.audio_decode_ms()),
                pcm.size(), out_path.c_str());
    return 0;
}
