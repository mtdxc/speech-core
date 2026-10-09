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
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
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
    if (argc < 6) {
        std::fprintf(stderr,
            "usage: %s <bundle_dir> <conditioning.blob> <transcript> <text> <out.wav> [seed]\n",
            argv[0]);
        return 2;
    }
    const std::string bundle_dir = argv[1];
    const std::string blob_path = argv[2];
    const std::string transcript = argv[3];
    const std::string text = argv[4];
    const std::string out_path = argv[5];
    const uint32_t seed = argc > 6 ? static_cast<uint32_t>(std::stoul(argv[6])) : 7u;

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

    write_wav_mono_pcm16(out_path, pcm, tts.output_sample_rate());
    std::printf("tokens=%d stop=%d prefill=%lldms ar=%lldms decode=%lldms samples=%zu -> %s\n",
                tts.tokens_generated(), tts.stopped_on_stop_token() ? 1 : 0,
                static_cast<long long>(tts.prefill_ms()),
                static_cast<long long>(tts.ar_ms()),
                static_cast<long long>(tts.audio_decode_ms()),
                pcm.size(), out_path.c_str());
    return 0;
}
