// Tiny CLI that runs Kokoro TTS on a piece of text and writes the audio to a WAV.
//
// Usage: speech_synthesize [model_dir] <output.wav> "<text>" [language]
//        (model_dir defaults to $SPEECH_MODEL_DIR, else ~/.cache/speech-core/models)
//
// Pairs with speech_transcribe — round-trip a known prompt through synthesis
// and back through STT to surface phonemizer / tokenizer / decoder bugs
// without bouncing through Android.
//
// Calls KokoroTts directly (skipping the speech-core pipeline) so we can
// inspect the raw audio buffer the model emits.

#include <speech_core/models/kokoro_tts.h>
#include <speech_core/audio/wav_io.h>

#include "../../common/default_model_dir.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

constexpr int kSampleRate = 24000;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: %s [model_dir] <output.wav> \"<text>\" [language]\n"
            "  model_dir : directory holding kokoro-e2e.onnx + voices/*.bin\n"
            "              (default: $SPEECH_MODEL_DIR, else %s)\n"
            "  language  : BCP-47 tag (default: en). Auto-switches voice.\n",
            argv[0], speech_example_model_dir().c_str());
        return 2;
    }
    // model_dir is optional. Old form: <model_dir> <out.wav> <text> [lang];
    // new form drops model_dir. With 4 args, both parses are plausible —
    // disambiguate by whether argv[1] is an existing directory.
    const bool has_dir = (argc >= 5)
        || (argc == 4 && std::filesystem::is_directory(argv[1]));
    const int base = has_dir ? 2 : 1;
    if (argc < base + 2) {
        std::fprintf(stderr, "usage: %s [model_dir] <output.wav> \"<text>\" [language]\n", argv[0]);
        return 2;
    }
    const std::string model_dir = has_dir ? argv[1] : speech_example_model_dir();
    const std::string out_wav   = argv[base];
    const std::string text      = argv[base + 1];
    const std::string language  = (argc >= base + 3) ? argv[base + 2] : "en";

    speech_core::KokoroTts tts(model_dir + "/kokoro-e2e.onnx",
                               model_dir + "/voices",
                               model_dir,
                               /*hw_accel=*/false);

    std::vector<float> samples;
    tts.synthesize(text, language,
        [&](const float* chunk, size_t length, bool /*is_final*/) {
            samples.insert(samples.end(), chunk, chunk + length);
        });

    if (samples.empty()) {
        std::fprintf(stderr, "synthesis produced no audio\n");
        return 1;
    }

    if (!speech_core::write_wav_mono_pcm16(out_wav, samples, kSampleRate)) {
        std::fprintf(stderr, "could not write %s\n", out_wav.c_str());
        return 1;
    }
    std::fprintf(stderr, "wrote %zu samples (%.2fs @ %d Hz) to %s\n",
                 samples.size(),
                 double(samples.size()) / double(kSampleRate),
                 kSampleRate, out_wav.c_str());
    return 0;
}
