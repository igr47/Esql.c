#include "plotter_includes/plotter.h"
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <map>
#include <vector>

namespace Visualization {

    std::array<float, 4> Plotter::parseColor(const std::string& colorStr) {
        static const std::map<std::string, std::array<float, 4>> colorMap = {
            {"blue", {0.0f, 0.0f, 1.0f, 1.0f}},
            {"red", {1.0f, 0.0f, 0.0f, 1.0f}},
            {"green", {0.0f, 1.0f, 0.0f, 1.0f}},
            {"black", {0.0f, 0.0f, 0.0f, 1.0f}},
            {"white", {1.0f, 1.0f, 1.0f, 1.0f}},
            {"yellow", {1.0f, 1.0f, 0.0f, 1.0f}},
            {"cyan", {0.0f, 1.0f, 1.0f, 1.0f}},
            {"magenta", {1.0f, 0.0f, 1.0f, 1.0f}},
            {"gray", {0.5f, 0.5f, 0.5f, 1.0f}},
            {"grey", {0.5f, 0.5f, 0.5f, 1.0f}},
            {"lightgray", {0.8f, 0.8f, 0.8f, 1.0f}},
            {"lightgrey", {0.8f, 0.8f, 0.8f, 1.0f}},
            {"darkgray", {0.3f, 0.3f, 0.3f, 1.0f}},
            {"darkgrey", {0.3f, 0.3f, 0.3f, 1.0f}},
            {"lightblue", {0.68f, 0.85f, 0.9f, 1.0f}},
            {"skyblue", {0.53f, 0.81f, 0.92f, 1.0f}},
            {"steelblue", {0.27f, 0.51f, 0.71f, 1.0f}},
            {"lightgreen", {0.56f, 0.93f, 0.56f, 1.0f}},
            {"mediumseagreen", {0.24f, 0.7f, 0.44f, 1.0f}},
            {"darkgreen", {0.0f, 0.39f, 0.0f, 1.0f}},
            {"orange", {1.0f, 0.65f, 0.0f, 1.0f}},
            {"gold", {1.0f, 0.84f, 0.0f, 1.0f}},
            {"goldenrod", {0.85f, 0.65f, 0.13f, 1.0f}},
            {"saddlebrown", {0.55f, 0.27f, 0.07f, 1.0f}},
            {"lightcoral", {0.94f, 0.5f, 0.5f, 1.0f}},
            {"coral", {1.0f, 0.5f, 0.31f, 1.0f}},
            {"tomato", {1.0f, 0.39f, 0.28f, 1.0f}},
            {"pink", {1.0f, 0.75f, 0.8f, 1.0f}},
            {"purple", {0.5f, 0.0f, 0.5f, 1.0f}},
            {"violet", {0.93f, 0.51f, 0.93f, 1.0f}},
            {"indigo", {0.29f, 0.0f, 0.51f, 1.0f}},
            {"teal", {0.0f, 0.5f, 0.5f, 1.0f}},
            {"navy", {0.0f, 0.0f, 0.5f, 1.0f}},
            {"maroon", {0.5f, 0.0f, 0.0f, 1.0f}},
            {"olive", {0.5f, 0.5f, 0.0f, 1.0f}},
            {"orangered", {1.0f, 0.27f, 0.0f, 1.0f}},
            {"darkorange", {1.0f, 0.55f, 0.0f, 1.0f}},
            {"lime", {0.0f, 1.0f, 0.0f, 1.0f}},
            {"springgreen", {0.0f, 1.0f, 0.5f, 1.0f}},
            {"turquoise", {0.25f, 0.88f, 0.82f, 1.0f}},
            {"royalblue", {0.25f, 0.41f, 0.88f, 1.0f}},
            {"mediumblue", {0.0f, 0.0f, 0.8f, 1.0f}},
            {"darkblue", {0.0f, 0.0f, 0.55f, 1.0f}},
            {"silver", {0.75f, 0.75f, 0.75f, 1.0f}},
            {"brown", {0.65f, 0.16f, 0.16f, 1.0f}},
            {"beige", {0.96f, 0.96f, 0.86f, 1.0f}},
            {"lavender", {0.9f, 0.9f, 0.98f, 1.0f}},
            {"khaki", {0.94f, 0.9f, 0.55f, 1.0f}},
            {"plum", {0.87f, 0.63f, 0.87f, 1.0f}},
            {"salmon", {0.98f, 0.5f, 0.45f, 1.0f}},
            {"tan", {0.82f, 0.71f, 0.55f, 1.0f}},
            {"aqua", {0.0f, 1.0f, 1.0f, 1.0f}},
            {"fuchsia", {1.0f, 0.0f, 1.0f, 1.0f}},
            {"limegreen", {0.2f, 0.8f, 0.2f, 1.0f}},
            {"forestgreen", {0.13f, 0.55f, 0.13f, 1.0f}},
            {"darkslategray", {0.18f, 0.31f, 0.31f, 1.0f}},
            {"dimgray", {0.41f, 0.41f, 0.41f, 1.0f}},
            {"slategray", {0.44f, 0.5f, 0.56f, 1.0f}},
            {"lightsteelblue", {0.69f, 0.77f, 0.87f, 1.0f}},
            {"powderblue", {0.69f, 0.88f, 0.9f, 1.0f}},
            {"palegreen", {0.6f, 0.98f, 0.6f, 1.0f}},
            {"lightpink", {1.0f, 0.71f, 0.76f, 1.0f}},
            {"thistle", {0.85f, 0.75f, 0.85f, 1.0f}},
            {"gainsboro", {0.86f, 0.86f, 0.86f, 1.0f}},
            {"whitesmoke", {0.96f, 0.96f, 0.96f, 1.0f}},
            {"snow", {1.0f, 0.98f, 0.98f, 1.0f}},
            {"mintcream", {0.96f, 1.0f, 0.98f, 1.0f}},
            {"azure", {0.94f, 1.0f, 1.0f, 1.0f}},
            {"aliceblue", {0.94f, 0.97f, 1.0f, 1.0f}},
            {"ghostwhite", {0.97f, 0.97f, 1.0f, 1.0f}},
            {"seashell", {1.0f, 0.96f, 0.93f, 1.0f}},
            {"oldlace", {0.99f, 0.96f, 0.9f, 1.0f}},
            {"linen", {0.98f, 0.94f, 0.9f, 1.0f}},
            {"antiquewhite", {0.98f, 0.92f, 0.84f, 1.0f}},
            {"papayawhip", {1.0f, 0.94f, 0.84f, 1.0f}},
            {"blanchedalmond", {1.0f, 0.92f, 0.8f, 1.0f}},
            {"bisque", {1.0f, 0.89f, 0.77f, 1.0f}},
            {"peachpuff", {1.0f, 0.85f, 0.73f, 1.0f}},
            {"navajowhite", {1.0f, 0.87f, 0.68f, 1.0f}},
            {"moccasin", {1.0f, 0.89f, 0.71f, 1.0f}},
            {"cornsilk", {1.0f, 0.97f, 0.86f, 1.0f}},
            {"ivory", {1.0f, 1.0f, 0.94f, 1.0f}},
            {"lemonchiffon", {1.0f, 0.98f, 0.8f, 1.0f}},
            {"honeydew", {0.94f, 1.0f, 0.94f, 1.0f}},
            {"chartreuse", {0.5f, 1.0f, 0.0f, 1.0f}},
            {"lawngreen", {0.49f, 0.99f, 0.0f, 1.0f}},
            {"greenyellow", {0.68f, 1.0f, 0.18f, 1.0f}},
            {"palegoldenrod", {0.93f, 0.91f, 0.67f, 1.0f}},
            {"lightgoldenrodyellow", {0.98f, 0.98f, 0.82f, 1.0f}},
            {"lightyellow", {1.0f, 1.0f, 0.88f, 1.0f}},
            {"yellowgreen", {0.6f, 0.8f, 0.2f, 1.0f}},
            {"darkolivegreen", {0.33f, 0.42f, 0.18f, 1.0f}},
            {"olivedrab", {0.42f, 0.56f, 0.14f, 1.0f}},
            {"darkkhaki", {0.74f, 0.72f, 0.42f, 1.0f}},
            {"peru", {0.8f, 0.52f, 0.25f, 1.0f}},
            {"rosybrown", {0.74f, 0.56f, 0.56f, 1.0f}},
            {"sienna", {0.63f, 0.32f, 0.18f, 1.0f}},
            {"sandybrown", {0.96f, 0.64f, 0.38f, 1.0f}},
            {"burlywood", {0.87f, 0.72f, 0.53f, 1.0f}},
            {"wheat", {0.96f, 0.87f, 0.7f, 1.0f}},
            {"chocolate", {0.82f, 0.41f, 0.12f, 1.0f}}
        };

        // Check if it's a hex color
        if (colorStr.size() > 1 && colorStr[0] == '#') {
            try {
                std::string hex = colorStr.substr(1);
                if (hex.size() == 6) {
                    int r, g, b;
                    std::stringstream ss;
                    ss << std::hex << hex.substr(0, 2);
                    ss >> r;
                    ss.clear();
                    ss << std::hex << hex.substr(2, 2);
                    ss >> g;
                    ss.clear();
                    ss << std::hex << hex.substr(4, 2);
                    ss >> b;
                    return {r/255.0f, g/255.0f, b/255.0f, 1.0f};
                } else if (hex.size() == 8) {
                    int r, g, b, a;
                    std::stringstream ss;
                    ss << std::hex << hex.substr(0, 2);
                    ss >> r;
                    ss.clear();
                    ss << std::hex << hex.substr(2, 2);
                    ss >> g;
                    ss.clear();
                    ss << std::hex << hex.substr(4, 2);
                    ss >> b;
                    ss.clear();
                    ss << std::hex << hex.substr(6, 2);
                    ss >> a;
                    return {r/255.0f, g/255.0f, b/255.0f, a/255.0f};
                }
            } catch (...) {
                // Fall through to default
            }
        }

        // Check named colors
        auto it = colorMap.find(colorStr);
        if (it != colorMap.end()) {
            return it->second;
        }

        // Try lowercase version
        std::string lowerStr = colorStr;
        std::transform(lowerStr.begin(), lowerStr.end(), lowerStr.begin(), ::tolower);
        it = colorMap.find(lowerStr);
        if (it != colorMap.end()) {
            return it->second;
        }

        // Default to blue
        return {0.0f, 0.0f, 1.0f, 1.0f};
    }

    std::vector<std::string> Plotter::getColorPalette(int n) {
        std::vector<std::string> colors;

        if (currentPalette == "tab10" || currentPalette.empty()) {
            // Tableau 10 color palette
            static const std::vector<std::string> tab10 = {
                "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
            };
            for (int i = 0; i < n; ++i) {
                colors.push_back(tab10[i % tab10.size()]);
            }
        } else if (currentPalette == "tab20") {
            // Tableau 20 color palette
            static const std::vector<std::string> tab20 = {
                "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
                "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
                "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
                "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"
            };
            for (int i = 0; i < n; ++i) {
                colors.push_back(tab20[i % tab20.size()]);
            }
        } else if (currentPalette == "set1") {
            // Set1 palette
            static const std::vector<std::string> set1 = {
                "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
                "#ffff33", "#a65628", "#f781bf", "#999999"
            };
            for (int i = 0; i < n; ++i) {
                colors.push_back(set1[i % set1.size()]);
            }
        } else if (currentPalette == "set2") {
            // Set2 palette
            static const std::vector<std::string> set2 = {
                "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854",
                "#ffd92f", "#e5c494", "#b3b3b3"
            };
            for (int i = 0; i < n; ++i) {
                colors.push_back(set2[i % set2.size()]);
            }
        } else if (currentPalette == "set3") {
            // Set3 palette
            static const std::vector<std::string> set3 = {
                "#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3",
                "#fdb462", "#b3de69", "#fccde5", "#d9d9d9", "#bc80bd",
                "#ccebc5", "#ffed6f"
            };
            for (int i = 0; i < n; ++i) {
                colors.push_back(set3[i % set3.size()]);
            }
        } else if (currentPalette == "viridis") {
            // Viridis palette (approximation)
            static const std::vector<std::string> viridis = {
                "#440154", "#482777", "#3e4989", "#31688e", "#26828e",
                "#1f9e89", "#35b779", "#6ece58", "#b5de2b", "#fde725"
            };
            for (int i = 0; i < n; ++i) {
                colors.push_back(viridis[i % viridis.size()]);
            }
        } else if (currentPalette == "plasma") {
            // Plasma palette (approximation)
            static const std::vector<std::string> plasma = {
                "#0d0887", "#46039f", "#7201a8", "#9c179e", "#bd3786",
                "#d8576b", "#ed7953", "#fb9f3a", "#fdca26", "#f0f921"
            };
            for (int i = 0; i < n; ++i) {
                colors.push_back(plasma[i % plasma.size()]);
            }
        } else if (currentPalette == "inferno") {
            // Inferno palette (approximation)
            static const std::vector<std::string> inferno = {
                "#000004", "#160b39", "#420a68", "#6a176e", "#932667",
                "#bc3754", "#dd513a", "#f37819", "#fca50a", "#f6d746"
            };
            for (int i = 0; i < n; ++i) {
                colors.push_back(inferno[i % inferno.size()]);
            }
        } else if (currentPalette == "magma") {
            // Magma palette (approximation)
            static const std::vector<std::string> magma = {
                "#000004", "#120d32", "#331068", "#5a167e", "#7f2485",
                "#a3307e", "#c73e6c", "#e95554", "#f97c3c", "#feab2f"
            };
            for (int i = 0; i < n; ++i) {
                colors.push_back(magma[i % magma.size()]);
            }
        } else {
            // Default palette
            static const std::vector<std::string> defaultPalette = {
                "blue", "green", "red", "cyan", "magenta", "yellow",
                "black", "orange", "purple", "brown", "pink", "gray",
                "olive", "teal", "navy", "maroon", "lime", "gold",
                "skyblue", "lightgreen", "lightcoral", "violet"
            };
            for (int i = 0; i < n; ++i) {
                colors.push_back(defaultPalette[i % defaultPalette.size()]);
            }
        }

        return colors;
    }

    std::vector<std::string> Plotter::getSequentialPalette(int n) {
        // Generate sequential colors (light to dark)
        std::vector<std::string> colors;
        for (int i = 0; i < n; ++i) {
            float intensity = static_cast<float>(i) / (n - 1);
            // Create a blue sequential palette
            int r = static_cast<int>(255 * (1 - intensity * 0.5));
            int g = static_cast<int>(255 * (1 - intensity * 0.3));
            int b = static_cast<int>(255);
            std::stringstream ss;
            ss << "#" << std::hex << std::setw(2) << std::setfill('0') << r
               << std::setw(2) << std::setfill('0') << g
               << std::setw(2) << std::setfill('0') << b;
            colors.push_back(ss.str());
        }
        return colors;
    }

    std::vector<std::string> Plotter::getDivergingPalette(int n) {
        // Generate diverging colors (two contrasting colors meeting in middle)
        std::vector<std::string> colors;
        for (int i = 0; i < n; ++i) {
            float t = static_cast<float>(i) / (n - 1);
            int r, g, b;

            if (t < 0.5) {
                // Blue to white
                r = static_cast<int>(255 * (t * 2));
                g = static_cast<int>(255 * (t * 2));
                b = 255;
            } else {
                // White to red
                r = 255;
                g = static_cast<int>(255 * (1 - (t - 0.5) * 2));
                b = static_cast<int>(255 * (1 - (t - 0.5) * 2));
            }

            std::stringstream ss;
            ss << "#" << std::hex << std::setw(2) << std::setfill('0') << r
               << std::setw(2) << std::setfill('0') << g
               << std::setw(2) << std::setfill('0') << b;
            colors.push_back(ss.str());
        }
        return colors;
    }

    std::vector<std::string> Plotter::getQualitativePalette(int n) {
        // Generate qualitative colors (distinct colors for categories)
        return getColorPalette(n);
    }

} // namespace Visualization
