#include "speech_core/audio/kws_fbank.h"
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>

static void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
int main() {
    const std::string base = std::string(__FILE__).substr(0, std::string(__FILE__).find_last_of("/\\")) + "/data/kws/";
    std::ifstream pcm_file(base + "input.f32", std::ios::binary), reference_file(base + "reference.bin", std::ios::binary);
    check(pcm_file.good() && reference_file.good(), "KWS fixtures missing");
    std::vector<float> pcm(16000); pcm_file.read(reinterpret_cast<char*>(pcm.data()), pcm.size() * sizeof(float));
    int32_t rows = 0, bins = 0;
    reference_file.read(reinterpret_cast<char*>(&rows), 4); reference_file.read(reinterpret_cast<char*>(&bins), 4);
    check(rows == 100 && bins == 80, "KWS reference dimensions changed");
    std::vector<float> reference(rows * bins); reference_file.read(reinterpret_cast<char*>(reference.data()), reference.size() * sizeof(float));
    speech_core::audio::KwsFbank whole, streaming;
    auto actual = whole.push(pcm.data(), pcm.size());
    auto tail = whole.finish(); actual.insert(actual.end(), tail.begin(), tail.end());
    std::vector<float> chunked;
    for (size_t i = 0; i < pcm.size(); i += 511) {
        auto values = streaming.push(pcm.data() + i, std::min<size_t>(511, pcm.size() - i));
        chunked.insert(chunked.end(), values.begin(), values.end());
    }
    tail = streaming.finish(); chunked.insert(chunked.end(), tail.begin(), tail.end());
    check(actual.size() == reference.size() && actual == chunked, "KWS streaming boundary changed features");
    double mean = 0, maximum = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        double error = std::abs(actual[i] - reference[i]); mean += error / actual.size(); maximum = std::max(maximum, error);
    }
    check(maximum <= .003 && mean <= .00005, "KWS frontend differs from frozen Kaldi reference");
    streaming.reset(); check(streaming.finish().empty(), "Empty KWS stream produced features");
    std::vector<float> silence(400, 0); auto quiet = streaming.push(silence.data(), silence.size());
    for (float x : quiet) check(std::abs(x - std::log(std::numeric_limits<float>::epsilon())) < .00001, "Silence floor changed");
    bool rejected = false; float invalid = std::numeric_limits<float>::quiet_NaN();
    try { streaming.push(&invalid, 1); } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "Non-finite PCM accepted");
}
