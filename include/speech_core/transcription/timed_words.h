#pragma once

#include "speech_core/interfaces.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace speech_core::transcription {

/// Appends one decoded token to the words of a stream.
///
/// `text` is the token as a SentencePiece decoder renders it: a piece that
/// starts a word (U+2581 in the vocabulary) arrives with a leading space.
/// Such a token opens a new word; any other token continues the word before
/// it, which is how punctuation and word pieces stay attached ("morning,").
/// A stream's first token opens a word with or without the space.
///
/// `frame` is the encoder output frame the token was emitted on, and
/// `frame_seconds` the audio one frame covers. A word starts at its first
/// token's frame and ends where its last token's frame ends. That is emission
/// time, so it trails the speech by whatever lookahead the model decodes with.
/// Word text keeps its leading space, so the words concatenated are exactly
/// the tokens concatenated.
inline void append_sentencepiece_token(
    std::vector<TimedWord>& words, const std::string& text,
    std::int64_t frame, float frame_seconds)
{
    if (text.empty()) return;
    const float start = static_cast<float>(frame) * frame_seconds;
    const float end = static_cast<float>(frame + 1) * frame_seconds;
    if (words.empty() || text.front() == ' ') {
        words.push_back(TimedWord{text, start, end});
    } else {
        words.back().text += text;
        words.back().end_time = end;
    }
}

/// `words` with every time limited to `max_seconds`: a stream padded to a
/// whole decoder window can emit on frames past the audio it was given, and
/// no word ends after the audio does.
inline std::vector<TimedWord> clamp_word_times(
    std::vector<TimedWord> words, float max_seconds)
{
    for (auto& word : words) {
        word.start_time = std::min(word.start_time, max_seconds);
        word.end_time = std::min(word.end_time, max_seconds);
    }
    return words;
}

}  // namespace speech_core::transcription
