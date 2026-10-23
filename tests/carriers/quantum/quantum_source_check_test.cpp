// M5 D3 source check: no strings, hashing or parsing in the quantum numerical
// sources, and independent test references. The quantum sources (every file
// of src/carriers/quantum/ and src/qubo/, src/backends/cpu/pqc_edges.cpp,
// lanes.hpp, fock_edges.cpp and src/backends/cuda/quantum.cu, where they
// exist; the PQC ones must) must not match the plan's regular expression,
// extended by `using namespace std` and namespace aliases of std (which would
// let an unqualified `string` through), every std string type (wide, pmr,
// basic_string), the remaining sto*/strto*/ato* conversions, from_chars/
// to_chars, std::format and their headers (independent review S1); the dense test references include only
// standard headers (nothing from src/ or kan/).
#include "support/test.hpp"
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace {
namespace fs = std::filesystem;
const fs::path root = KAN_SOURCE_DIR;

const std::regex& forbidden() {
    static const std::regex pattern(
        R"(\bstd\s*::\s*(pmr\s*::\s*)?(\w*string\w*|sto(i|l|ll|ul|ull|f|d|ld)\b|hash\b|unordered_\w+|regex|from_chars|to_chars|format\w*)|\b(strto\w+|sscanf|ato(f|i|l|ll))\b|#include\s*<(string|string_view|sstream|regex|unordered_map|unordered_set|charconv|format)>|\busing\s+namespace\s+(::)?std\s*;|\bnamespace\s+\w+\s*=\s*(::)?std\s*;)");
    return pattern;
}

std::string read(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    std::ostringstream text;
    text << in.rdbuf();
    return text.str();
}

std::vector<fs::path> quantum_sources() {
    std::vector<fs::path> files;
    for (const char* directory : {"src/carriers/quantum", "src/qubo"})
        if (fs::is_directory(root / directory))
            for (const auto& entry : fs::recursive_directory_iterator(root / directory))
                if (entry.is_regular_file()) files.push_back(entry.path());
    for (const char* file : {"src/backends/cpu/pqc_edges.cpp", "src/backends/cpu/lanes.hpp",
                             "src/backends/cpu/fock_edges.cpp", "src/backends/cuda/quantum.cu"})
        if (fs::exists(root / file)) files.push_back(root / file);
    return files;
}
} // namespace

TEST(the_pattern_catches_what_it_must) {
    for (const char* bad :
         {"std::string s;", "std::string_view v", "std::to_string(1)", "std::stod(t)", "std::hash<int>{}",
          "std::unordered_map<int, int>", "std::regex r", "std::ostringstream o", "x = atoi(p);", "strtod(p, nullptr)",
          "#include <string>", "#include <sstream>", "#include <unordered_set>", "using std::string;",
          "using std::to_string;", "using namespace std;", "using namespace ::std;", "namespace s = std;",
          "namespace s = ::std ;", "std::wstring w;", "std::u8string u;", "std::basic_string<char> b;",
          "std::pmr::string p;", "std :: string s;", "std::stoul(t)", "std::stoull(t)", "std::from_chars(a, b, v)",
          "std::format(\"{}\", 1)", "strtol(p, nullptr, 10)", "atol(p)", "#include <string_view>",
          "#include <charconv>", "#include <format>"})
        REQUIRE(std::regex_search(std::string(bad), forbidden()));
    for (const char* good : {"std::size_t n;", "std::invalid_argument(\"message\")", "#include <stdexcept>",
                             "#include <span>", "stringent", "std::span<const double>", "using namespace pqc;",
                             "using namespace detail::pqc;", "using namespace stdx;", "namespace pqc = detail::pqc;",
                             "std::stop_token t;", "// no strings, hashing or parsing", "std::numeric_limits<double>"})
        REQUIRE(!std::regex_search(std::string(good), forbidden()));
}

TEST(quantum_numerical_sources_have_no_strings_hashing_or_parsing) {
    REQUIRE(fs::exists(root / "src/backends/cpu/pqc_edges.cpp"));
    REQUIRE(fs::exists(root / "src/backends/cpu/lanes.hpp"));
    REQUIRE(fs::is_directory(root / "src/carriers/quantum"));
    const auto files = quantum_sources();
    REQUIRE(files.size() >= 3);
    std::string failures;
    for (const auto& file : files) {
        const auto text = read(file);
        std::smatch match;
        if (std::regex_search(text, match, forbidden()))
            failures += file.generic_string() + ": " + match.str() + "; ";
        else
            std::printf("clean %s\n", fs::relative(file, root).generic_string().c_str());
    }
    if (!failures.empty()) throw std::runtime_error(failures);
}

TEST(test_references_are_independent_of_the_library) {
    REQUIRE(fs::exists(root / "tests/support/dense_qubit_reference.hpp"));
    std::string failures;
    for (const char* file : {"tests/support/dense_qubit_reference.hpp", "tests/support/dense_fock_reference.hpp"}) {
        if (!fs::exists(root / file)) continue;
        std::istringstream lines(read(root / file));
        std::string line;
        const std::regex include(R"(^\s*#\s*include\s*(\S+))"), standard(R"(<[a-z_]+>)");
        while (std::getline(lines, line)) {
            std::smatch m;
            if (std::regex_search(line, m, include) && !std::regex_match(m[1].str(), standard))
                failures += std::string(file) + ": " + line + "; ";
        }
    }
    if (!failures.empty()) throw std::runtime_error(failures);
}

int main() { return test::run(); }
