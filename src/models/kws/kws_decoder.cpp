#include "speech_core/models/kws_decoder.h"
#include <algorithm>
#include <cmath>
#include <queue>
#include <set>
#include <stdexcept>

namespace speech_core {
KwsDecoder::KwsDecoder(std::vector<KwsKeyword> keywords, Decoder decoder, Joiner joiner,
                       int vocabulary, int blank, int context, int beam, int trailing, double idle)
    : keywords_(std::move(keywords)), decoder_(std::move(decoder)), joiner_(std::move(joiner)),
      vocabulary_(vocabulary), blank_(blank), context_(context), beam_(beam), trailing_(trailing) {
    if (keywords_.empty() || keywords_.size() > 128 || vocabulary < 2 || vocabulary > 65536 ||
        blank < 0 || blank >= vocabulary || context < 1 || context > 8 || beam < 1 || beam > 32 ||
        trailing < 0 || trailing > 100 || !std::isfinite(idle) || idle <= 0 || idle > 3600 ||
        !decoder_ || !joiner_) throw std::invalid_argument("Invalid KWS decoding configuration");
    idle_frames_ = std::max(1, static_cast<int>(std::round(idle / .04)));
    graph_.push_back(Node{});
    std::set<std::vector<int>> seen;
    for (size_t k = 0; k < keywords_.size(); ++k) {
        const auto& keyword = keywords_[k];
        if (keyword.phrase.empty() || keyword.tokens.empty() || keyword.tokens.size() > 128 ||
            !std::isfinite(keyword.threshold) || keyword.threshold <= 0 || keyword.threshold > 1 ||
            !std::isfinite(keyword.boost) || keyword.boost < 0 || keyword.boost > 10 ||
            !seen.insert(keyword.tokens).second) throw std::invalid_argument("Invalid or duplicate KWS phrase");
        int node = 0;
        for (int token : keyword.tokens) {
            if (token < 0 || token >= vocabulary || token == blank) throw std::invalid_argument("Invalid KWS token");
            auto found = graph_[node].next.find(token);
            if (found == graph_[node].next.end()) {
                Node child;
                child.level = graph_[node].level + 1;
                child.token_score = keyword.boost;
                child.node_score = graph_[node].node_score + keyword.boost;
                int next = static_cast<int>(graph_.size());
                graph_[node].next[token] = next;
                graph_.push_back(child);
                node = next;
            } else {
                node = found->second;
                if (graph_[node].token_score != keyword.boost)
                    throw std::invalid_argument("Shared KWS prefix must use one boost");
            }
        }
        graph_[node].keyword = static_cast<int>(k);
        graph_[node].output_score = graph_[node].node_score;
    }
    std::queue<int> pending;
    for (const auto& edge : graph_[0].next) pending.push(edge.second);
    while (!pending.empty()) {
        int current = pending.front(); pending.pop();
        for (const auto& edge : graph_[current].next) {
            int token = edge.first, child = edge.second, fail = graph_[current].fail;
            while (fail != 0 && !graph_[fail].next.count(token)) fail = graph_[fail].fail;
            auto target = graph_[fail].next.find(token);
            graph_[child].fail = target == graph_[fail].next.end() ? 0 : target->second;
            int output = graph_[child].fail;
            while (output != 0 && graph_[output].keyword < 0) output = graph_[output].fail;
            graph_[child].output = output == 0 ? -1 : output;
            if (output != 0) graph_[child].output_score += graph_[output].output_score;
            pending.push(child);
        }
    }
    reset();
}

std::vector<int> KwsDecoder::initial_context() const {
    std::vector<int> tokens(context_, -1); tokens.back() = blank_; return tokens;
}
void KwsDecoder::reset_search() {
    decoder_cache_.clear(); idle_ = 0;
    Hypothesis initial; initial.tokens = initial_context(); hypotheses_ = {initial};
}
void KwsDecoder::reset() { stream_frame_ = 0; reset_search(); }
std::pair<double, int> KwsDecoder::advance(int state, int token) const {
    auto direct = graph_[state].next.find(token);
    if (direct != graph_[state].next.end()) {
        int next = direct->second;
        return {graph_[next].token_score + graph_[next].output_score, next};
    }
    int fail = graph_[state].fail;
    while (fail != 0 && !graph_[fail].next.count(token)) fail = graph_[fail].fail;
    auto edge = graph_[fail].next.find(token);
    int next = edge == graph_[fail].next.end() ? fail : edge->second;
    return {graph_[next].node_score - graph_[state].node_score + graph_[next].output_score, next};
}

std::vector<KwsDetection> KwsDecoder::step(const std::vector<float>& frame) {
    struct Candidate { double score, probability; size_t hypothesis; int token; int order; };
    std::vector<Candidate> candidates;
    int order = 0;
    auto before = [](const Candidate& a, const Candidate& b) {
        return a.score != b.score ? a.score > b.score : a.order < b.order;
    };
    for (size_t h = 0; h < hypotheses_.size(); ++h) {
        const auto& hypothesis = hypotheses_[h];
        std::vector<int> context(hypothesis.tokens.end() - context_, hypothesis.tokens.end());
        auto decoded = decoder_cache_.find(context);
        if (decoded == decoder_cache_.end()) {
            auto values = decoder_(context);
            if (values.empty() || !std::all_of(values.begin(), values.end(), [](float x) { return std::isfinite(x); }))
                throw std::runtime_error("Invalid KWS decoder output");
            decoded = decoder_cache_.emplace(context, std::move(values)).first;
        }
        auto logits = joiner_(frame, decoded->second);
        if (logits.size() != static_cast<size_t>(vocabulary_) ||
            !std::all_of(logits.begin(), logits.end(), [](float x) { return std::isfinite(x); }))
            throw std::runtime_error("Invalid KWS joiner output");
        float maximum = *std::max_element(logits.begin(), logits.end()), sum = 0;
        for (float& value : logits) { value = std::exp(value - maximum); sum += value; }
        for (int token = 0; token < vocabulary_; ++token) {
            float probability = logits[token] / sum;
            Candidate candidate{hypothesis.score + (probability > 0 ? std::log(probability) : -INFINITY),
                                probability, h, token, order++};
            auto place = std::lower_bound(candidates.begin(), candidates.end(), candidate, before);
            if (place != candidates.end() || candidates.size() < static_cast<size_t>(beam_)) {
                candidates.insert(place, candidate);
                if (candidates.size() > static_cast<size_t>(beam_)) candidates.pop_back();
            }
        }
    }
    std::map<std::vector<int>, Hypothesis> next_beam;
    for (const auto& candidate : candidates) {
        Hypothesis hypothesis = hypotheses_[candidate.hypothesis];
        hypothesis.blanks++;
        double boost = 0;
        if (candidate.token != blank_) {
            hypothesis.tokens.push_back(candidate.token);
            hypothesis.times.push_back(stream_frame_);
            hypothesis.probabilities.push_back(candidate.probability);
            auto advanced = advance(hypothesis.node, candidate.token);
            boost = advanced.first; hypothesis.node = advanced.second; hypothesis.blanks = 0;
            if (hypothesis.node == 0) {
                hypothesis.tokens.resize(hypothesis.tokens.size() - std::min(static_cast<size_t>(context_), hypothesis.tokens.size()));
                auto initial = initial_context();
                hypothesis.tokens.insert(hypothesis.tokens.end(), initial.begin(), initial.end());
            }
        }
        hypothesis.score = candidate.score + boost;
        auto found = next_beam.find(hypothesis.tokens);
        if (found == next_beam.end()) {
            auto key = hypothesis.tokens;
            next_beam.emplace(std::move(key), std::move(hypothesis));
        }
        else {
            double a = found->second.score, b = hypothesis.score;
            found->second.score = std::isinf(a) && a < 0 ? b : std::isinf(b) && b < 0 ? a :
                std::max(a, b) + std::log1p(std::exp(-std::abs(a - b)));
        }
    }
    hypotheses_.clear();
    for (auto& entry : next_beam) hypotheses_.push_back(std::move(entry.second));
    const auto& top = *std::max_element(hypotheses_.begin(), hypotheses_.end(), [](const Hypothesis& a, const Hypothesis& b) {
        double left = a.score / a.tokens.size(), right = b.score / b.tokens.size();
        return left != right ? left < right : a.tokens > b.tokens;
    });
    int match = graph_[top.node].keyword >= 0 ? top.node : graph_[top.node].output;
    if (match >= 0) {
        size_t level = static_cast<size_t>(graph_[match].level);
        if (level > 0 && top.probabilities.size() >= level && top.blanks > trailing_) {
            double probability = 0;
            for (auto i = top.probabilities.end() - level; i != top.probabilities.end(); ++i) probability += *i / level;
            const auto& keyword = keywords_[graph_[match].keyword];
            if (probability >= keyword.threshold) {
                KwsDetection detection{keyword.phrase, std::vector<int>(top.tokens.end() - level, top.tokens.end()),
                    std::vector<int64_t>(top.times.end() - level, top.times.end()), stream_frame_, 0};
                ++stream_frame_; reset_search(); return {detection};
            }
        }
    }
    ++stream_frame_;
    if (top.blanks == 0) idle_ = 0;
    else if (++idle_ >= idle_frames_) reset_search();
    // Length-normalized beam ranking depends on the complete token history.
    // Never silently truncate it: fail closed at a bounded memory budget.
    for (auto& hypothesis : hypotheses_) {
        if (hypothesis.tokens.size() > 16384)
            throw std::runtime_error("KWS decoding history budget exceeded; reset required");
    }
    if (decoder_cache_.size() > 4096) decoder_cache_.clear();
    return {};
}
}  // namespace speech_core
