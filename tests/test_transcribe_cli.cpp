// Exercise the actual transcription entry point with a recording C ABI stub.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include "speech.h"
#include "wav_test_fixture.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

int speech_transcribe_cli_main(int argc, char** argv);

struct speech_pipeline_s {
    speech_event_fn callback;
    void* context;
};

namespace {

std::vector<float> captured;
int create_count = 0;
int destroy_count = 0;

int run_cli(const std::string& path) {
    captured.clear();
    create_count = 0;
    destroy_count = 0;
    std::string program = "speech_transcribe";
    std::string model_dir = "unused-test-models";
    std::string input_path = path;
    char* args[] = {program.data(), model_dir.data(), input_path.data(), nullptr};
    return speech_transcribe_cli_main(3, args);
}

void check_input(const std::filesystem::path& path, uint16_t format, uint16_t bits,
                 uint16_t channels, uint32_t rate, uint32_t sample) {
    // One second of constant audio makes rate and downmix errors unambiguous.
    std::vector<uint32_t> samples(static_cast<size_t>(rate) * channels, sample);
    if (channels == 2) {
        for (size_t i = 1; i < samples.size(); i += 2) samples[i] = 0;
    }
    wav_test::write_fixture(path, format, bits, channels, rate, samples);
    assert(run_cli(path.u8string()) == 0);
    assert(create_count == 1 && destroy_count == 1);
    // The C ABI accepts 16 kHz audio, regardless of the WAV's native rate.
    constexpr size_t audio_samples = 16000;
    assert(captured.size() >= audio_samples + 24000);  // at least 1.5 s of silence
    const float amplitude = channels == 1 ? 0.25f : 0.125f;
    for (size_t i = 0; i < audio_samples; ++i) {
        assert(std::fabs(captured[i] - amplitude) < 1e-5f);
    }
    assert(std::all_of(captured.begin() + audio_samples, captured.end(),
                       [](float value) { return value == 0.0f; }));
}

}  // namespace

extern "C" speech_config_t speech_config_default(void) { return {}; }

extern "C" speech_pipeline_t speech_create(speech_config_t config,
                                            speech_event_fn callback, void* context) {
    assert(config.transcribe_only);
    ++create_count;
    return new speech_pipeline_s{callback, context};
}

extern "C" void speech_start(speech_pipeline_t pipeline) {
    // Complete immediately so this input-contract test needs no model or wait.
    speech_event_t event{};
    event.type = SPEECH_EVENT_TRANSCRIPTION;
    event.text = "test transcription";
    pipeline->callback(&event, pipeline->context);
}

extern "C" void speech_push_audio(speech_pipeline_t, const float* samples, size_t count) {
    captured.insert(captured.end(), samples, samples + count);
}

extern "C" void speech_destroy(speech_pipeline_t pipeline) {
    ++destroy_count;
    delete pipeline;
}

int main() {
    wav_test::TemporaryDirectory directory("speech_core_test_transcribe_cli");
    const auto path = directory.path / "fixture.wav";
    check_input(path, 1, 16, 1, 16000, 0x2000);
    check_input(path, 1, 16, 1, 48000, 0x2000);
    check_input(path, 1, 16, 2, 44100, 0x2000);
    check_input(path, 1, 24, 1, 24000, 0x200000);
    check_input(path, 1, 32, 1, 48000, 0x20000000);
    check_input(path, 3, 32, 2, 48000, 0x3e800000);
    assert(run_cli((directory.path / "missing.wav").u8string()) == 1);
    assert(create_count == 0 && destroy_count == 0 && captured.empty());
    std::puts("All transcription CLI input tests passed.");
    return 0;
}
