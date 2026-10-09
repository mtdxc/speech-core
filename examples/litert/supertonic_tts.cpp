// SupertonicTTS-3 LiteRT smoke test / example.
//
//   supertonic_tts <bundle_dir> [text] [lang] [voice] [out.wav]
//
// bundle_dir = the soniqo/Supertonic-3-LiteRT layout: the four .tflite graphs +
// unicode_indexer.json + tts.json + voice_styles/<id>.json.

#include "speech_core/supertonic_c.h"
#include "speech_core/audio/wav_io.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

std::vector<float> g_pcm;

void on_chunk(const float* samples, size_t length, bool is_final, void* /*user*/) {
    g_pcm.insert(g_pcm.end(), samples, samples + length);
    std::printf("  chunk: %zu samples%s\n", length, is_final ? " (final)" : "");
}


}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <bundle_dir> [text] [lang] [voice] [out.wav]\n", argv[0]);
        return 2;
    }
    const char* bundle = argv[1];
    const char* text   = argc > 2 ? argv[2] : "Hello from soniqo dot audio.";
    const char* lang   = argc > 3 ? argv[3] : "en";
    const char* voice  = argc > 4 ? argv[4] : "F1";
    const std::string out = argc > 5 ? argv[5] : "supertonic_out.wav";

    sc_supertonic_t s = sc_supertonic_create(bundle);
    if (!s) {
        std::fprintf(stderr, "create failed: %s\n", sc_supertonic_last_error(nullptr));
        return 1;
    }
    sc_supertonic_set_voice(s, voice);
    sc_supertonic_set_total_step(s, 8);
    sc_supertonic_set_seed(s, 1234);  // reproducible

    std::printf("synthesizing [%s] \"%s\" voice=%s\n", lang, text, voice);
    const int rc = sc_supertonic_synthesize(s, text, lang, on_chunk, nullptr);
    if (rc != 0) {
        std::fprintf(stderr, "synthesize failed (%d): %s\n", rc, sc_supertonic_last_error(s));
        sc_supertonic_destroy(s);
        return 1;
    }

    const int sr = sc_supertonic_output_sample_rate(s);
    std::printf("OK: %zu samples @ %d Hz = %.2fs\n", g_pcm.size(), sr,
                static_cast<double>(g_pcm.size()) / sr);
    write_wav_mono_pcm16(out, g_pcm, sr);
    std::printf("wrote %s\n", out.c_str());
    sc_supertonic_destroy(s);
    return 0;
}
