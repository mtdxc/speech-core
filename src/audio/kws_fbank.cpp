#include "speech_core/audio/kws_fbank.h"
#include "speech_core/audio/fft.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace speech_core::audio {
KwsFbank::KwsFbank() : window_(400), filters_(80 * 257) {
    const double pi = std::acos(-1.0);
    for (int n = 0; n < 400; ++n) window_[n] = static_cast<float>(std::pow(.5 - .5 * std::cos(2 * pi * n / 399), .85));
    auto mel = [](double hz) { return 1127 * std::log1p(hz / 700); };
    double low = mel(20), delta = (mel(7600) - low) / 81;
    for (int m = 0; m < 80; ++m) for (int k = 0; k < 257; ++k) {
        double distance = (mel(k * 16000.0 / 512) - low - (m + 1) * delta) / delta;
        filters_[m * 257 + k] = static_cast<float>(std::max(0.0, 1 - std::abs(distance)));
    }
}
void KwsFbank::reset() { pcm_.clear(); offset_ = emitted_ = 0; }
std::vector<float> KwsFbank::push(const float* samples, size_t count) {
    if ((count && !samples) || count > 32000) throw std::invalid_argument("Invalid KWS audio block");
    for (size_t i = 0; i < count; ++i) if (!std::isfinite(samples[i]) || std::abs(samples[i]) > 1)
        throw std::invalid_argument("KWS expects finite normalized PCM");
    if (count) pcm_.insert(pcm_.end(), samples, samples + count);
    return drain(false);
}
std::vector<float> KwsFbank::finish() { return drain(true); }
std::vector<float> KwsFbank::drain(bool final) {
    int64_t observed = offset_ * 160 + static_cast<int64_t>(pcm_.size());
    int64_t ready = final ? (observed + 80) / 160 : observed < 280 ? 0 : (observed - 280) / 160 + 1;
    std::vector<float> out;
    out.reserve(static_cast<size_t>(std::max<int64_t>(0, ready - emitted_)) * 80);
    for (; emitted_ < ready; ++emitted_) {
        float frame[512] = {}, real[257], imaginary[257], power[257];
        int64_t start = (emitted_ - offset_) * 160 - 120, size = static_cast<int64_t>(pcm_.size());
        float mean = 0;
        for (int n = 0; n < 400; ++n) {
            int64_t index = start + n;
            while (index < 0 || index >= size) {
                if (index < 0) index = -index - 1;
                if (index >= size) index = 2 * size - index - 1;
            }
            frame[n] = pcm_[static_cast<size_t>(index)]; mean += frame[n] / 400;
        }
        for (int n = 0; n < 400; ++n) frame[n] -= mean;
        for (int n = 399; n > 0; --n) frame[n] -= .97f * frame[n - 1];
        frame[0] *= .03f;
        for (int n = 0; n < 400; ++n) frame[n] *= window_[n];
        fft_real(frame, 512, real, imaginary);
        for (int k = 0; k < 257; ++k) power[k] = real[k] * real[k] + imaginary[k] * imaginary[k];
        for (int m = 0; m < 80; ++m) {
            float energy = 0;
            for (int k = 0; k < 257; ++k) energy += filters_[m * 257 + k] * power[k];
            out.push_back(std::log(std::max(energy, std::numeric_limits<float>::epsilon())));
        }
    }
    int64_t target = std::max<int64_t>(0, emitted_ - 1);
    if (target > offset_) {
        size_t count = std::min(static_cast<size_t>((target - offset_) * 160), pcm_.size());
        pcm_.erase(pcm_.begin(), pcm_.begin() + count); offset_ += static_cast<int64_t>(count / 160);
    }
    return out;
}
}
