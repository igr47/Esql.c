#include "plotter_includes/plotter.h"
#include <algorithm>
#include <map>

namespace Visualization {

    std::string Plotter::parseLineStyle(const std::string& styleStr) {
        static const std::map<std::string, std::string> styleMap = {
            {"-", "-"},
            {"--", "--"},
            {":", ":"},
            {"-.", "-."},
            {"solid", "-"},
            {"dashed", "--"},
            {"dotted", ":"},
            {"dashdot", "-."},
            {"none", ""},
            {" ", ""}
        };

        auto it = styleMap.find(styleStr);
        if (it != styleMap.end()) {
            return it->second;
        }

        // Try lowercase
        std::string lowerStr = styleStr;
        std::transform(lowerStr.begin(), lowerStr.end(), lowerStr.begin(), ::tolower);
        it = styleMap.find(lowerStr);
        if (it != styleMap.end()) {
            return it->second;
        }

        return "-";
    }

    std::string Plotter::parseMarker(const std::string& markerStr) {
        static const std::map<std::string, std::string> markerMap = {
            {".", "."},
            {",", ","},
            {"o", "o"},
            {"v", "v"},
            {"^", "^"},
            {"<", "<"},
            {">", ">"},
            {"1", "1"},
            {"2", "2"},
            {"3", "3"},
            {"4", "4"},
            {"8", "8"},
            {"s", "s"},
            {"p", "p"},
            {"P", "P"},
            {"*", "*"},
            {"h", "h"},
            {"H", "H"},
            {"+", "+"},
            {"x", "x"},
            {"X", "X"},
            {"D", "D"},
            {"d", "d"},
            {"|", "|"},
            {"_", "_"},
            {"point", "."},
            {"pixel", ","},
            {"circle", "o"},
            {"triangle_down", "v"},
            {"triangle_up", "^"},
            {"triangle_left", "<"},
            {"triangle_right", ">"},
            {"tri_down", "1"},
            {"tri_up", "2"},
            {"tri_left", "3"},
            {"tri_right", "4"},
            {"octagon", "8"},
            {"square", "s"},
            {"pentagon", "p"},
            {"plus", "+"},
            {"star", "*"},
            {"hexagon1", "h"},
            {"hexagon2", "H"},
            {"diamond", "D"},
            {"thin_diamond", "d"},
            {"vline", "|"},
            {"hline", "_"},
            {"none", ""},
            {" ", ""}
        };

        auto it = markerMap.find(markerStr);
        if (it != markerMap.end()) {
            return it->second;
        }

        // Try lowercase
        std::string lowerStr = markerStr;
        std::transform(lowerStr.begin(), lowerStr.end(), lowerStr.begin(), ::tolower);
        it = markerMap.find(lowerStr);
        if (it != markerMap.end()) {
            return it->second;
        }

        return "o";
    }
}
