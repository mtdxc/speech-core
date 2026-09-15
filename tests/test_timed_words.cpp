// Words assembled from SentencePiece tokens and the encoder frames they were
// emitted on, as the Nemotron streaming wrappers report them.

#include "speech_core/transcription/timed_words.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

bool near(float a, float b) { return std::fabs(a - b) < 1e-6f; }

}  // namespace

int main() {
    using speech_core::TimedWord;
    using speech_core::transcription::append_sentencepiece_token;
    using speech_core::transcription::clamp_word_times;

    constexpr float kFrame = 0.08f;  // hop 160 x subsampling 8 / 16 kHz
    struct Token { const char* text; long frame; };
    // " The meeting starts, 9" — a word split over two pieces, punctuation
    // attached to its word, and a bare word-start piece before a digit.
    const Token tokens[] = {
        {" The", 0}, {" meet", 2}, {"ing", 3}, {" starts", 5}, {",", 5},
        {" ", 7}, {"9", 8}, {"", 9},
    };

    std::vector<TimedWord> words;
    std::string all;
    for (const auto& token : tokens) {
        append_sentencepiece_token(words, token.text, token.frame, kFrame);
        all += token.text;
    }

    expect(words.size() == 4, "four words");
    std::string joined;
    for (const auto& word : words) joined += word.text;
    expect(joined == all, "words concatenated equal the tokens concatenated");

    expect(words[0].text == " The" && near(words[0].start_time, 0.0f)
           && near(words[0].end_time, 0.08f), "a one-token word spans its frame");
    expect(words[1].text == " meeting" && near(words[1].start_time, 0.16f)
           && near(words[1].end_time, 0.32f), "a word runs from its first token to its last");
    expect(words[2].text == " starts," && near(words[2].end_time, 0.48f),
           "punctuation stays on its word");
    expect(words[3].text == " 9" && near(words[3].start_time, 0.56f)
           && near(words[3].end_time, 0.72f), "a bare word start joins the piece after it");

    for (size_t i = 0; i < words.size(); ++i) {
        expect(words[i].start_time < words[i].end_time, "a word ends after it starts");
        if (i > 0) {
            expect(words[i].start_time >= words[i - 1].start_time, "starts never go back");
            expect(words[i].end_time >= words[i - 1].end_time, "ends never go back");
        }
    }

    // The first token opens a word even without its space.
    {
        std::vector<TimedWord> first;
        append_sentencepiece_token(first, "Hello", 4, kFrame);
        append_sentencepiece_token(first, " world", 6, kFrame);
        expect(first.size() == 2 && first[0].text == "Hello", "a leading piece opens a word");
    }

    // Times never pass the audio the stream was given.
    {
        const auto clamped = clamp_word_times(words, 0.6f);
        expect(near(clamped[3].start_time, 0.56f) && near(clamped[3].end_time, 0.6f),
               "an end past the audio is clamped to it");
        expect(near(clamped[0].end_time, 0.08f), "times inside the audio are unchanged");
    }

    if (failures > 0) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
