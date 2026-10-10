#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace speech_core {

/// RIFF/WAVE audio decoded to mono float32 at the file's native sample rate.
struct WavData {
    std::vector<float> samples;
    int sample_rate = 0;
    double duration() const {
        return sample_rate > 0
            ? static_cast<double>(samples.size()) / sample_rate : 0.0;
    }
    bool load(const std::string& path);
    bool save(const std::string& path) const;
};

/// Load PCM16/24/32 or IEEE float32 WAV audio, averaging channels to mono.
/// The native sample rate is preserved; callers resample when needed.
/// The historical pcm16 name also covers the other supported encodings.
/// Paths are UTF-8. Returns false on I/O or format errors and clears *out.
bool load_wav_mono_pcm16(const std::string& path, WavData* out);

/// Write mono PCM16 RIFF/WAVE audio, clamping float samples to [-1, 1].
/// Paths are UTF-8. Returns false on I/O errors or invalid arguments.
bool write_wav_mono_pcm16(const std::string& path, const WavData& data);
bool write_wav_mono_pcm16(const std::string& path,
                          const std::vector<float>& data, int sample_rate);
bool write_wav_mono_pcm16(const std::string& path,
                          const float* samples, size_t count,
                          int sample_rate);

}  // namespace speech_core
