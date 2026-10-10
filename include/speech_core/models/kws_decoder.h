#pragma once
#include <functional>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace speech_core {

/// The host registers exact acoustic phrases. This engine never executes actions.
struct KwsKeyword {
    std::string phrase;
    std::vector<int> tokens;
    double threshold = 0.25;
    double boost = 0.1;
};
struct KwsDetection {
    std::string phrase;
    std::vector<int> tokens;
    std::vector<int64_t> token_frames;
    int64_t stream_frame = 0;
    double audio_end_seconds = 0;
};

/// Stateless-transducer modified beam search; reset only after idle frames,
/// rather than on a timer from the last keyword. Backend failures propagate.
class KwsDecoder {
public:
    using Decoder = std::function<std::vector<float>(const std::vector<int>&)>;
    using Joiner = std::function<std::vector<float>(const std::vector<float>&, const std::vector<float>&)>;
    KwsDecoder(std::vector<KwsKeyword> keywords, Decoder decoder, Joiner joiner,
               int vocabulary_size, int blank = 0, int context_size = 2,
               int beam = 4, int trailing_blanks = 1, double idle_seconds = 1.5);
    std::vector<KwsDetection> step(const std::vector<float>& frame);
    void reset();
private:
    struct Node {
        std::map<int, int> next;
        int fail = 0, output = -1, level = 0, keyword = -1;
        double token_score = 0, node_score = 0, output_score = 0;
    };
    struct Hypothesis {
        std::vector<int> tokens;
        std::vector<double> probabilities;
        std::vector<int64_t> times;
        int node = 0, blanks = 0;
        double score = 0;
    };
    std::vector<KwsKeyword> keywords_;
    std::vector<Node> graph_;
    Decoder decoder_;
    Joiner joiner_;
    std::map<std::vector<int>, std::vector<float>> decoder_cache_;
    std::vector<Hypothesis> hypotheses_;
    int vocabulary_, blank_, context_, beam_, trailing_, idle_frames_, idle_ = 0;
    int64_t stream_frame_ = 0;
    std::vector<int> initial_context() const;
    void reset_search();
    std::pair<double, int> advance(int node, int token) const;
};
}  // namespace speech_core
