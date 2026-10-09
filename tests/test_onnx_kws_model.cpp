#include "speech_core/models/onnx_zipformer_kws.h"
#include <nlohmann/json.hpp>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        const char* directory = argc > 1 ? argv[1] : std::getenv("SPEECH_KWS_MODEL_DIR");
        if (!directory) { std::cout << "Set SPEECH_KWS_MODEL_DIR for the real KWS model test\n"; return 77; }
        std::vector<speech_core::KwsKeyword> phrases{{"alpha beta", {1, 2}, .25, .1}};
        if (argc == 4) {
            std::ifstream file(argv[2]); auto json = nlohmann::json::parse(file); phrases.clear();
            for (const auto& item : json) phrases.push_back({item.at("phrase"), item.at("tokens").get<std::vector<int>>(), .25, .1});
        }
        speech_core::OnnxZipformerKws spotter(directory, phrases);
        if (argc == 4) {
            std::ifstream audio(argv[3], std::ios::binary | std::ios::ate);
            auto bytes = audio.tellg();
            if (bytes <= 0 || bytes > 256 * 1024 * 1024 || bytes % 4) throw std::runtime_error("Invalid normalized Float32 replay");
            audio.seekg(0); std::vector<float> samples(static_cast<size_t>(bytes) / 4);
            audio.read(reinterpret_cast<char*>(samples.data()), bytes);
            nlohmann::json results = nlohmann::json::array();
            auto append = [&](const std::vector<speech_core::KwsDetection>& hits) {
                for (const auto& hit : hits) results.push_back({{"phrase", hit.phrase}, {"frame", hit.stream_frame},
                    {"tokenFrames", hit.token_frames}, {"audioEndSeconds", hit.audio_end_seconds}});
            };
            for (size_t i = 0; i < samples.size(); i += 320) append(spotter.push(samples.data() + i, std::min<size_t>(320, samples.size() - i)));
            append(spotter.finish()); std::cout << results.dump() << '\n'; return 0;
        }
        std::vector<float> silence(320);
        for (int i = 0; i < 100; ++i) if (!spotter.push(silence.data(), silence.size()).empty()) throw std::runtime_error("Silence emitted a phrase");
        if (!spotter.finish().empty()) throw std::runtime_error("Silence tail emitted a phrase");
        bool rejected = false;
        try { spotter.push(silence.data(), silence.size()); } catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) throw std::runtime_error("Finalized stream accepted more audio");
        spotter.reset(); rejected = false; float invalid = std::numeric_limits<float>::quiet_NaN();
        try { spotter.push(&invalid, 1); } catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) throw std::runtime_error("Non-finite audio accepted");
        rejected = false;
        try { spotter.push(silence.data(), silence.size()); } catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) throw std::runtime_error("Failed stream resumed silently");
        spotter.reset(); spotter.push(silence.data(), silence.size());
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
