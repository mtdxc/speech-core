// Exercise the production transcription entry point through the OS command
// line, with a recording C ABI stub instead of downloaded models.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include "speech.h"
#include "wav_test_fixture.h"
#include "../examples/common/utf8_args.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

#if !defined(_WIN32)
#include <cerrno>
#include <sys/wait.h>
#include <unistd.h>
#endif

int speech_transcribe_cli_main(int argc, char** argv);

struct speech_pipeline_s {
    speech_event_fn callback;
    void* context;
};

namespace {

std::vector<float> captured;
std::string captured_model_dir;
int create_count = 0;
int destroy_count = 0;
const std::string model_prefix = u8"unused-\u092e\u0947\u0930\u093e \u00e4udio-models/";

#if defined(_WIN32)
std::wstring quote_arg(const std::wstring& argument) {
    std::wstring quoted = L"\"";
    size_t backslashes = 0;
    for (wchar_t c : argument) {
        if (c == L'\\') {
            ++backslashes;
        } else {
            quoted.append(c == L'"' ? backslashes * 2 + 1 : backslashes, L'\\');
            quoted += c;
            backslashes = 0;
        }
    }
    quoted.append(backslashes * 2, L'\\');
    return quoted + L'"';
}

int run_cli(std::vector<std::string> args) {
    std::wstring command;
    for (const auto& arg : args) {
        if (!command.empty()) command += L' ';
        command += quote_arg(std::filesystem::u8path(arg).native());
    }
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    const auto executable = std::filesystem::u8path(args[0]).native();
    if (!CreateProcessW(executable.c_str(), command.data(), nullptr, nullptr,
                        FALSE, 0, nullptr, nullptr, &startup, &process)) return -1;
    const DWORD wait_result = WaitForSingleObject(process.hProcess, 10000);
    DWORD exit_code = 1;
    if (wait_result == WAIT_OBJECT_0) {
        GetExitCodeProcess(process.hProcess, &exit_code);
    } else {
        TerminateProcess(process.hProcess, 1);
    }
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    return static_cast<int>(exit_code);
}
#else
int run_cli(std::vector<std::string> args) {
    const pid_t child = fork();
    if (child < 0) return -1;
    if (child == 0) {
        std::vector<char*> argv;
        for (auto& arg : args) argv.push_back(arg.data());
        argv.push_back(nullptr);
        execv(args[0].c_str(), argv.data());
        _exit(127);
    }
    int status = 0;
    pid_t result;
    do { result = waitpid(child, &status, 0); } while (result < 0 && errno == EINTR);
    return result == child && WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}
#endif

void check_input(const std::string& executable, const std::filesystem::path& path,
                 uint16_t format, uint16_t bits,
                 uint16_t channels, uint32_t rate, uint32_t sample) {
    // One second of constant audio makes rate and downmix errors unambiguous.
    std::vector<uint32_t> samples(static_cast<size_t>(rate) * channels, sample);
    if (channels == 2) {
        for (size_t i = 1; i < samples.size(); i += 2) samples[i] = 0;
    }
    wav_test::write_fixture(path, format, bits, channels, rate, samples);
    assert(run_cli({executable, model_prefix + (channels == 1 ? "mono" : "stereo"),
                    path.u8string()}) == 0);
}

void check_child(int argc, char** argv, const std::vector<std::string>& args) {
    const int result = speech_transcribe_cli_main(argc, argv);
    if (args[1] == "missing") {
        assert(result == 1);
        assert(create_count == 0 && destroy_count == 0 && captured.empty());
        return;
    }
    assert(result == 0);
    assert(create_count == 1 && destroy_count == 1);
    assert(captured_model_dir == args[1]);
    assert(captured_model_dir == model_prefix + "mono" ||
           captured_model_dir == model_prefix + "stereo");
    // The C ABI accepts 16 kHz audio, regardless of the WAV's native rate.
    constexpr size_t audio_samples = 16000;
    assert(captured.size() >= audio_samples + 24000);  // at least 1.5 s of silence
    const float amplitude = captured_model_dir == model_prefix + "mono" ? 0.25f : 0.125f;
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
    captured_model_dir = config.model_dir;
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

int main(int argc, char** argv) {
    const auto args = speech_examples::utf8_args(argc, argv);
    if (args.size() == 3) {
        check_child(argc, argv, args);
        return 0;
    }
    assert(args.size() == 1);
    const auto executable = std::filesystem::absolute(std::filesystem::u8path(args[0])).u8string();
    wav_test::TemporaryDirectory directory("speech_core_test_transcribe_cli");
    const auto path = directory.path / std::filesystem::u8path(u8"\u092e\u0947\u0930\u093e \u00e4udio.wav");
    check_input(executable, path, 1, 16, 1, 16000, 0x2000);
    check_input(executable, path, 1, 16, 1, 48000, 0x2000);
    check_input(executable, path, 1, 16, 2, 44100, 0x2000);
    check_input(executable, path, 1, 24, 1, 24000, 0x200000);
    check_input(executable, path, 1, 32, 1, 48000, 0x20000000);
    check_input(executable, path, 3, 32, 2, 48000, 0x3e800000);
    assert(run_cli({executable, "missing", (directory.path / "missing.wav").u8string()}) == 0);
    std::puts("All transcription CLI input tests passed.");
    return 0;
}
