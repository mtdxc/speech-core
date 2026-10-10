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
#include "../../common/utf8_args.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

constexpr int kSampleRate = 24000;

int main(int argc, char** argv) {
    const auto args = speech_examples::utf8_args(argc, argv);
    argc = static_cast<int>(args.size());
    const char* argv0 = args.empty() ? "speech_synthesize" : args[0].c_str();
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: %s [model_dir] <output.wav> \"<text>\" [language]\n"
            "  model_dir : directory holding kokoro-e2e.onnx + voices/*.bin\n"
            "              (default: $SPEECH_MODEL_DIR, else %s)\n"
            "  language  : BCP-47 tag (default: en). Auto-switches voice.\n",
            argv0, speech_example_model_dir().c_str());
        return 2;
    }
    // model_dir is optional. Old form: <model_dir> <out.wav> <text> [lang];
    // new form drops model_dir. With 4 args, both parses are plausible —
    // disambiguate by whether args[1] is an existing directory.
    const bool has_dir = (argc >= 5)
        || (argc == 4 && std::filesystem::is_directory(std::filesystem::u8path(args[1])));
    const int base = has_dir ? 2 : 1;
    if (argc < base + 2) {
        std::fprintf(stderr, "usage: %s [model_dir] <output.wav> \"<text>\" [language]\n", argv0);
        return 2;
    }
    const std::string model_dir = has_dir ? args[1] : speech_example_model_dir();
    const std::string out_wav   = args[base];
    const std::string text      = args[base + 1];
    const std::string language  = (argc >= base + 3) ? args[base + 2] : "en";

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

    if (!speech_core::WavData::write_mono(out_wav, samples, kSampleRate)) {
        std::fprintf(stderr, "could not write %s\n", out_wav.c_str());
        return 1;
    }
    std::fprintf(stderr, "wrote %zu samples (%.2fs @ %d Hz) to %s\n",
                 samples.size(),
                 double(samples.size()) / double(kSampleRate),
                 kSampleRate, out_wav.c_str());
    return 0;
}
