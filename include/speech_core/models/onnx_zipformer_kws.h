#pragma once
#include "speech_core/models/kws_decoder.h"
#include <memory>

namespace speech_core {
/// Streaming ONNX Zipformer phrase spotter. Loads only the supplied local
/// directory; has no microphone, downloader, command routing or fallback.
/// Feed normalized mono 16 kHz PCM on one serial worker. Each hit includes
/// a monotonic stream frame and the accepted audio's session end time.
class OnnxZipformerKws {
public:
    OnnxZipformerKws(const std::string& directory, std::vector<KwsKeyword> keywords, int beam = 4);
    ~OnnxZipformerKws();
    OnnxZipformerKws(const OnnxZipformerKws&) = delete;
    OnnxZipformerKws& operator=(const OnnxZipformerKws&) = delete;
    std::vector<KwsDetection> push(const float* samples, size_t count);
    std::vector<KwsDetection> finish();
    void reset();
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
}
