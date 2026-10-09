#include "speech_core/models/kws_decoder.h"
#include <limits>
#include <stdexcept>

static void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
static speech_core::KwsDecoder make(float threshold = .25f, float boost = .1f, int beam = 1) {
    return speech_core::KwsDecoder({{"alpha beta", {1, 2}, threshold, boost}},
        [](const std::vector<int>&) { return std::vector<float>{0}; },
        [](const std::vector<float>& frame, const std::vector<float>&) { return frame; }, 4, 0, 2, beam, 1, 1.5);
}
static std::vector<float> logits(int token) { std::vector<float> x(4, -15); x[token] = 15; return x; }
static speech_core::KwsDetection complete(speech_core::KwsDecoder& search) {
    check(search.step(logits(1)).empty(), "Prefix emitted");
    check(search.step(logits(2)).empty(), "Phrase emitted without trailing blanks");
    check(search.step(logits(0)).empty(), "One trailing blank emitted");
    auto hits = search.step(logits(0)); check(hits.size() == 1, "Complete phrase missing"); return hits[0];
}
int main() {
    auto search = make(); auto first = complete(search); auto second = complete(search);
    check(first.phrase == "alpha beta" && first.token_frames == std::vector<int64_t>({0, 1}), "Phrase timing changed");
    check(second.stream_frame > first.stream_frame && second.token_frames[0] > first.token_frames.back(), "Hit reset rewound stream clock");
    search.reset(); auto fresh = complete(search); check(fresh.stream_frame == first.stream_frame, "Explicit reset did not rewind clock");
    auto gap = make(); gap.step(logits(1));
    for (int i = 0; i < 37; ++i) gap.step(logits(0));
    gap.step(logits(2)); gap.step(logits(0)); check(gap.step(logits(0)).size() == 1, "Idle reset used session age instead of last token");
    auto expired = make(); expired.step(logits(1));
    for (int i = 0; i < 38; ++i) expired.step(logits(0));
    expired.step(logits(2)); expired.step(logits(0)); check(expired.step(logits(0)).empty(), "Expired prefix promoted");
    auto uncertain = make(.9f, 10);
    uncertain.step({-10, .01f, 0, -10}); uncertain.step({-10, 0, .01f, -10});
    uncertain.step(logits(0)); check(uncertain.step(logits(0)).empty(), "Context boost promoted acoustically weak phrase");
    bool rejected = false;
    try { search.step({0, std::numeric_limits<float>::quiet_NaN(), 0, 0}); } catch (const std::runtime_error&) { rejected = true; }
    check(rejected, "Non-finite model output accepted");
    rejected = false;
    try { speech_core::KwsDecoder duplicate({{"a", {1}, .25f, .1f}, {"b", {1}, .25f, .1f}},
        [](const std::vector<int>&) { return std::vector<float>{0}; },
        [](const std::vector<float>& x, const std::vector<float>&) { return x; }, 4); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "Ambiguous duplicate token phrase accepted");
}
