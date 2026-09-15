// StreamingMelSpectrogram against the whole-buffer mel_spectrogram it
// replaces in the Nemotron multilingual streaming front-end: pre-emphasis
// 0.97, reflect-centred legacy framing, Slaney bank, log(x + 2^-24).
//
// A streamed frame must equal the whole-buffer frame of the same index for
// every frame whose samples have all arrived, however the audio was split
// into pushes, and the stream must keep a bounded amount of audio so a
// frame's cost does not grow with the stream. Random audio, no fixtures.

#include "speech_core/audio/mel.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr int kSampleRate = 16000;
constexpr int kNfft = 512;
constexpr int kHop = 160;
constexpr int kWin = 400;
constexpr int kMels = 128;
constexpr float kPreEmphasis = 0.97f;
const float kLogFloor = 1.0f / static_cast<float>(1 << 24);
constexpr float kTolerance = 1e-5f;

int failures = 0;

void expect(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

std::vector<float> random_audio(size_t length, std::uint32_t seed) {
    std::vector<float> audio(length);
    std::uint32_t state = seed;
    for (size_t i = 0; i < length; ++i) {
        state = state * 1664525u + 1013904223u;
        audio[i] = static_cast<float>(state >> 8) / static_cast<float>(1u << 24) - 0.5f;
    }
    // A silent stretch, so frames near the log floor are compared as well.
    const size_t quiet_from = length / 3;
    const size_t quiet_to = std::min(length, quiet_from + 2400);
    std::fill(audio.begin() + static_cast<std::ptrdiff_t>(quiet_from),
              audio.begin() + static_cast<std::ptrdiff_t>(quiet_to), 0.0f);
    return audio;
}

std::vector<float> whole_buffer_mel(const std::vector<float>& audio) {
    // Exactly what the Nemotron wrappers computed before streaming features.
    std::vector<float> emphasized(audio.size());
    emphasized[0] = audio[0];
    for (size_t i = 1; i < audio.size(); ++i) {
        emphasized[i] = audio[i] - kPreEmphasis * audio[i - 1];
    }
    return speech_core::audio::mel_spectrogram(
        emphasized.data(), emphasized.size(), kSampleRate, kNfft, kHop, kWin, kMels,
        /*slaney_norm=*/true, kLogFloor, /*center=*/true);
}

speech_core::audio::StreamingMelSpectrogram::Config config() {
    speech_core::audio::StreamingMelSpectrogram::Config c;
    c.sample_rate = kSampleRate;
    c.n_fft = kNfft;
    c.hop_length = kHop;
    c.win_length = kWin;
    c.num_mel_bins = kMels;
    c.slaney_norm = true;
    c.log_floor = kLogFloor;
    c.pre_emphasis = kPreEmphasis;
    return c;
}

/// Frames whose every sample lies inside `length` samples: the last sample a
/// frame reads is t*hop + win - n_fft/2 - 1, and the reflected head of the
/// first frames reads up to index n_fft/2.
size_t complete_frames(size_t length) {
    if (length <= static_cast<size_t>(kNfft / 2)) return 0;
    const size_t reach = static_cast<size_t>(kWin - kNfft / 2);
    if (length < reach) return 0;
    return (length - reach) / kHop + 1;
}

float streamed_max_difference(
    const std::vector<float>& audio, size_t chunk, size_t* frames_out,
    size_t* max_buffered_out)
{
    speech_core::audio::StreamingMelSpectrogram stream(config());
    std::vector<float> frames;
    size_t max_buffered = 0;
    for (size_t offset = 0; offset < audio.size(); offset += chunk) {
        const size_t count = std::min(chunk, audio.size() - offset);
        stream.push(audio.data() + offset, count, frames);
        max_buffered = std::max(max_buffered, stream.buffered_samples());
    }
    const auto whole = whole_buffer_mel(audio);
    const size_t whole_frames = whole.size() / kMels;
    const size_t streamed = frames.size() / kMels;
    *frames_out = streamed;
    *max_buffered_out = max_buffered;
    if (streamed > whole_frames) return 1e9f;

    float max_difference = 0.0f;
    for (size_t t = 0; t < streamed; ++t) {
        for (size_t m = 0; m < static_cast<size_t>(kMels); ++m) {
            const float a = frames[t * kMels + m];
            const float b = whole[m * whole_frames + t];
            max_difference = std::max(max_difference, std::fabs(a - b));
        }
    }
    return max_difference;
}

}  // namespace

int main() {
    float worst = 0.0f;

    // Every frame whose samples have arrived equals the whole-buffer frame,
    // for lengths around the first frames and a full minute's worth of
    // windows, pushed in sizes that land everywhere relative to the hop.
    const size_t lengths[] = {257, 1000, 5376, 16000, 48017};
    const size_t chunks[] = {1, 7, 97, 160, 401, 1601, 4999, 1u << 20};
    std::uint32_t seed = 7;
    for (size_t length : lengths) {
        const auto audio = random_audio(length, seed++);
        for (size_t chunk : chunks) {
            size_t frames = 0;
            size_t max_buffered = 0;
            const float difference = streamed_max_difference(audio, chunk, &frames, &max_buffered);
            worst = std::max(worst, difference);
            if (difference > kTolerance) {
                std::printf("  length=%zu chunk=%zu max|diff|=%g\n", length, chunk, difference);
            }
            expect(difference <= kTolerance, "streamed frames equal whole-buffer frames");
            if (frames != complete_frames(length)) {
                std::printf("  length=%zu chunk=%zu frames=%zu expected=%zu\n",
                            length, chunk, frames, complete_frames(length));
            }
            expect(frames == complete_frames(length),
                   "a frame appears exactly when its last sample arrives");
            if (chunk < static_cast<size_t>(kNfft)) {
                expect(max_buffered <= static_cast<size_t>(kNfft + kWin + kHop) + chunk,
                       "the stream keeps a bounded amount of audio");
            }
        }
    }

    // Too few samples for the reflected head: no frame yet.
    {
        speech_core::audio::StreamingMelSpectrogram stream(config());
        std::vector<float> frames;
        const auto audio = random_audio(kNfft / 2, 99);
        stream.push(audio.data(), audio.size(), frames);
        expect(frames.empty(), "no frame before n_fft/2 + 1 samples");
    }

    // The Nemotron end of stream: zeros padded to a whole window continue the
    // stream, and the frames equal the whole-buffer mel of the padded audio.
    {
        auto audio = random_audio(5200, 1234);
        speech_core::audio::StreamingMelSpectrogram stream(config());
        std::vector<float> frames;
        stream.push(audio.data(), audio.size(), frames);
        std::vector<float> zeros(10240 - audio.size(), 0.0f);
        stream.push(zeros.data(), zeros.size(), frames);
        audio.insert(audio.end(), zeros.begin(), zeros.end());
        const auto whole = whole_buffer_mel(audio);
        const size_t whole_frames = whole.size() / kMels;
        const size_t streamed = frames.size() / kMels;
        expect(streamed >= 64, "a padded tail completes both 32-frame windows");
        float difference = 0.0f;
        for (size_t t = 0; t < std::min<size_t>(streamed, 64); ++t) {
            for (size_t m = 0; m < static_cast<size_t>(kMels); ++m) {
                difference = std::max(difference,
                    std::fabs(frames[t * kMels + m] - whole[m * whole_frames + t]));
            }
        }
        worst = std::max(worst, difference);
        expect(difference <= kTolerance, "padded tail equals whole-buffer mel of the padded audio");
    }

    // A long stream holds only what its next frame needs.
    {
        speech_core::audio::StreamingMelSpectrogram stream(config());
        std::vector<float> frames;
        const auto audio = random_audio(60 * kSampleRate, 42);
        for (size_t offset = 0; offset < audio.size(); offset += 1600) {
            stream.push(audio.data() + offset, std::min<size_t>(1600, audio.size() - offset), frames);
            frames.clear();
        }
        expect(stream.samples_pushed() == audio.size(), "every sample is counted");
        expect(stream.frames_produced() == complete_frames(audio.size()),
               "a minute of audio produces every complete frame");
        expect(stream.buffered_samples() <= static_cast<size_t>(kNfft + kWin + kHop),
               "a minute of audio leaves under one n_fft + window + hop buffered");
    }

    // Reset starts sample 0 again: the same audio streams to the same frames.
    {
        const auto audio = random_audio(8000, 5);
        speech_core::audio::StreamingMelSpectrogram stream(config());
        std::vector<float> first;
        stream.push(audio.data(), audio.size(), first);
        stream.reset();
        std::vector<float> second;
        stream.push(audio.data(), audio.size(), second);
        expect(first == second, "reset restarts the stream");
        expect(stream.samples_pushed() == audio.size(), "reset clears the sample count");
    }

    std::printf("streaming mel: worst max|diff| vs whole buffer = %g (tolerance %g)\n",
                worst, kTolerance);
    if (failures > 0) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
