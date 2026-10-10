#pragma once

#include <cassert>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace wav_test {

struct TemporaryDirectory {
    std::filesystem::path path;

    explicit TemporaryDirectory(const char* name) {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        for (int attempt = 0; attempt < 10; ++attempt) {
            path = std::filesystem::temp_directory_path() /
                (std::string(name) + "_" + std::to_string(stamp) + "_" + std::to_string(attempt));
            if (std::filesystem::create_directory(path)) return;
        }
        throw std::runtime_error("Could not create WAV test directory");
    }

    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

// Samples are on-disk bit patterns, independent of the production encoder.
inline void write_fixture(const std::filesystem::path& path,
                          uint16_t format, uint16_t bits, uint16_t channels,
                          uint32_t rate, const std::vector<uint32_t>& samples,
                          bool odd_junk = false) {
    std::ofstream out(path, std::ios::binary);
    assert(out);
    auto integer = [&](uint32_t value, int bytes) {
        for (int i = 0; i < bytes; ++i) out.put(static_cast<char>((value >> (8 * i)) & 0xff));
    };
    const uint32_t width = bits / 8;
    const uint32_t data_bytes = static_cast<uint32_t>(samples.size()) * width;
    out.write("RIFF", 4);
    integer(36 + data_bytes + (data_bytes & 1u) + (odd_junk ? 10 : 0), 4);
    out.write("WAVE", 4);
    if (odd_junk) {
        out.write("JUNK", 4);
        integer(1, 4);
        out.put('x');
        out.put('\0');
    }
    out.write("fmt ", 4);
    integer(16, 4);
    integer(format, 2);
    integer(channels, 2);
    integer(rate, 4);
    integer(rate * channels * width, 4);
    integer(channels * width, 2);
    integer(bits, 2);
    out.write("data", 4);
    integer(data_bytes, 4);
    for (uint32_t sample : samples) integer(sample, static_cast<int>(width));
    if (data_bytes & 1u) out.put('\0');
    out.close();
    assert(out);
}

}  // namespace wav_test
