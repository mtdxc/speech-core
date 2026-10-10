#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
namespace speech_core {

/// RIFF/WAVE audio decoded to mono float32 at the file's native sample rate.
struct WavData {
    std::vector<float> samples;
    int sample_rate = 0;
    double duration() const {
        return sample_rate > 0
            ? static_cast<double>(samples.size()) / sample_rate : 0.0;
    }
    void clear() {
        samples.clear();
        sample_rate = 0;
    }

    /// Load PCM16/24/32 or IEEE float32 WAV audio, averaging channels to mono.
    /// The native sample rate is preserved; callers resample when needed.
    /// The historical pcm16 name also covers the other supported encodings.
    /// Paths are UTF-8. Returns false on I/O or format errors and clears the object.
    bool load_mono(const std::string& path) {
        return load_mono(std::filesystem::u8path(path));
    }
    bool load_mono(const std::filesystem::path& path) {
        std::ifstream file(path, std::ios::binary);
        return load_mono(file);
    }
    bool load_mono(std::istream& file);
    /// Save PCM16/24/32 or IEEE float32 WAV audio, averaging channels to mono.
    /// The native sample rate is preserved; callers resample when needed.
    /// The historical pcm16 name also covers the other supported encodings.
    /// Paths are UTF-8. Returns false on I/O errors or invalid arguments.
    bool save(const std::string& path) const {
        std::ofstream file(std::filesystem::u8path(path), std::ios::binary);
        return write_mono(file, samples.data(), samples.size(), sample_rate);
    }
    bool save(const std::filesystem::path& path) const {
        std::ofstream file(path, std::ios::binary);
        return write_mono(file, samples.data(), samples.size(), sample_rate);
    }
    bool save(std::ostream& file) const {
        return write_mono(file, samples.data(), samples.size(), sample_rate);
    }
    static bool write_mono(const std::string& path,
                            const std::vector<float>& data, int sample_rate) {
        std::ofstream file(std::filesystem::u8path(path), std::ios::binary);
        return write_mono(file, data.data(), data.size(), sample_rate);
    }
    /// Write mono PCM16 RIFF/WAVE audio, clamping float samples to [-1, 1].
    /// Paths are UTF-8. Returns false on I/O errors or invalid arguments.
    static bool write_mono(std::ostream& os,
                            const float* samples, size_t count,
                            int sample_rate);
};


}  // namespace speech_core
