#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace speech_core::audio {
/// Kaldi online 80-bin fbank: normalized 16 kHz PCM, 25/10 ms Povey
/// windows, DC removal, 0.97 preemphasis, snip_edges=false, high_freq=-400.
/// This is distinct from speaker embedding's CMVN/frontend contract.
class KwsFbank {
public:
    KwsFbank();
    std::vector<float> push(const float* samples, size_t count);
    std::vector<float> finish();
    void reset();
private:
    std::vector<float> window_, filters_, pcm_;
    int64_t offset_ = 0, emitted_ = 0;
    std::vector<float> drain(bool final);
};
}
