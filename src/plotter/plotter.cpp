#include "plotter_includes/plotter.h"
#include <matplot/matplot.h>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <map>
#include <regex>
#include <chrono>

namespace plt = matplot;

namespace Visualization {

    // Helper function to parse color
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

    // Helper function to parse linestyle
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

    // Helper function to parse marker
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

    Plotter::Plotter() : plotterInitialized(false), currentFigureId(0), currentHistoryIndex(0) {
        initializePlotter();
    }

    Plotter::~Plotter() {
        finalizePlotter();
    }

    void Plotter::initializePlotter() {
        if (!plotterInitialized) {
            plotterInitialized = true;
        }
    }

    void Plotter::finalizePlotter() {
        if (plotterInitialized) {
            plotterInitialized = false;
        }
    }

    // Enhanced setupFigure with comprehensive styling
    void Plotter::setupFigure(const PlotConfig& config) {
        plt::figure(true);
        plt::figure()->size(config.style.figwidth * 100, config.style.figheight * 100);

        // Set title
        if (!config.title.empty()) {
            plt::title(config.title);
        }

        // Set labels
        if (!config.xLabel.empty()) {
            plt::xlabel(config.xLabel);
        }

        if (!config.yLabel.empty()) {
            plt::ylabel(config.yLabel);
        }

        // Set grid
        if (config.style.grid) {
            plt::grid(true);
        }

        // Set axis limits if specified
        if (!std::isnan(config.style.xmin) && !std::isnan(config.style.xmax)) {
            plt::xlim({config.style.xmin, config.style.xmax});
        }

        if (!std::isnan(config.style.ymin) && !std::isnan(config.style.ymax)) {
            plt::ylim({config.style.ymin, config.style.ymax});
        }

        // Set tick rotation
        if (config.style.xtick_rotation != 0.0) {
            plt::xtickangle(config.style.xtick_rotation);
        }

        if (config.style.ytick_rotation != 0.0) {
            plt::ytickangle(config.style.ytick_rotation);
        }
    }

    void Plotter::setStyle(const std::string& styleName) {
        // Set plotting style (similar to matplotlib styles)
        currentStyle = styleName;

        // Note: Matplot++ has limited built-in styles compared to matplotlib
        // We can implement custom style presets here
        if (styleName == "default" || styleName == "classic") {
            // Default style - no changes needed
        } else if (styleName == "ggplot") {
            // ggplot-like style
            plt::gca()->box(true);
        } else if (styleName == "seaborn" || styleName == "dark_background") {
            // Dark background style
            plt::gcf()->color({0.15f, 0.15f, 0.15f, 1.0f});
            plt::gca()->color({0.2f, 0.2f, 0.2f, 1.0f});
        } else if (styleName == "fivethirtyeight") {
            // FiveThirtyEight style
            plt::gca()->box(true);
        }
    }

    void Plotter::setColorPalette(const std::string& paletteName) {
        currentPalette = paletteName;

        // Note: Matplot++ doesn't have direct palette management like matplotlib
        // We store the palette name for use in getColorPalette()
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

    // Enhanced Line Plot with comprehensive styling
    void Plotter::plotLine(const PlotData& data, const PlotConfig& config) {
        validatePlotData(data, config);
        setupFigure(config);

        if (data.numericData.size() < 2) {
            throw std::runtime_error("Need at least 2 numeric columns for line plot");
        }

        auto itX = data.numericData.begin();
        auto itY = std::next(data.numericData.begin());

        std::vector<double> xData = itX->second;
        std::vector<double> yData = itY->second;

        // Clean data
        std::vector<double> cleanX, cleanY;
        for (size_t i = 0; i < xData.size(); ++i) {
            if (!std::isnan(xData[i]) && !std::isnan(yData[i])) {
                cleanX.push_back(xData[i]);
                cleanY.push_back(yData[i]);
            }
        }

        if (cleanX.empty() || cleanY.empty()) {
            throw std::runtime_error("No valid data points for line plot");
        }

        // Sort by x values
        std::vector<size_t> indices(cleanX.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
                 [&cleanX](size_t a, size_t b) { return cleanX[a] < cleanX[b]; });

        std::vector<double> sortedX, sortedY;
        for (auto idx : indices) {
            sortedX.push_back(cleanX[idx]);
            sortedY.push_back(cleanY[idx]);
        }

        // Create plot with comprehensive styling
        auto p = plt::plot(sortedX, sortedY);

        // Apply line style
        auto color = parseColor(config.style.color);
        p->color(color);
        p->line_width(config.style.linewidth);
        p->line_style(parseLineStyle(config.style.linestyle));

        // Apply marker style if specified
        std::string marker = parseMarker(config.style.marker);
        if (!marker.empty() && marker != "none") {
            p->marker_style(marker);
            p->marker_size(config.style.markersize);

            auto markerColor = parseColor(config.style.markercolor);
            p->marker_color(markerColor);

            if (config.style.markerfacecolor != "none") {
                auto markerFaceColor = parseColor(config.style.markerfacecolor);
                p->marker_face_color(markerFaceColor);
            }
        }

        // Set alpha/transparency
	color[3] = config.style.alpha; // Set alpha component
	p->color(color);


        // Add legend if needed
        addLegendIfNeeded(config);

        handlePlotOutput(config);
    }

    // Enhanced Scatter Plot with comprehensive styling
    void Plotter::plotScatter(const PlotData& data, const PlotConfig& config) {
        validatePlotData(data, config);
        setupFigure(config);

        if (data.numericData.size() < 2) {
            throw std::runtime_error("Need at least 2 numeric columns for scatter plot");
        }

        auto itX = data.numericData.begin();
        auto itY = std::next(data.numericData.begin());

        std::vector<double> xData = itX->second;
        std::vector<double> yData = itY->second;

        std::vector<double> cleanX, cleanY;
        for (size_t i = 0; i < xData.size(); ++i) {
            if (!std::isnan(xData[i]) && !std::isnan(yData[i])) {
                cleanX.push_back(xData[i]);
                cleanY.push_back(yData[i]);
            }
        }

        if (cleanX.empty() || cleanY.empty()) {
            throw std::runtime_error("No valid data points for scatter plot");
        }

        // Create scatter plot with comprehensive styling
        auto s = plt::scatter(cleanX, cleanY);

        // Apply marker styling
        std::string marker = parseMarker(config.style.marker);
        if (!marker.empty() && marker != "none") {
            s->marker_style(marker);
            s->marker_size(config.style.markersize);

            auto markerColor = parseColor(config.style.markercolor);
            s->marker_color(markerColor);

            if (config.style.markerfacecolor != "none") {
                auto markerFaceColor = parseColor(config.style.markerfacecolor);
                s->marker_face_color(markerFaceColor);
            }
        }

        s->line_width(config.style.linewidth);
        //s->alpha(config.style.alpha);
	auto markerColor = parseColor(config.style.markercolor);
	markerColor[3] = config.style.alpha; // Set alpha component
        s->marker_color(markerColor);

        // Add legend if needed
        addLegendIfNeeded(config);

        handlePlotOutput(config);
    }

    // Enhanced Bar Plot with comprehensive styling
    void Plotter::plotBar(const PlotData& data, const PlotConfig& config) {
        validatePlotData(data, config);
        setupFigure(config);

        if (data.categoricalData.empty() || data.numericData.empty()) {
            throw std::runtime_error("Need both categorical and numeric data for bar plot");
        }

        auto catIt = data.categoricalData.begin();
        auto numIt = data.numericData.begin();

        std::vector<std::string> categories = catIt->second;
        std::vector<double> values = numIt->second;

        size_t n = std::min(categories.size(), values.size());
        categories.resize(n);
        values.resize(n);

        // Create bar plot with comprehensive styling
        auto b = plt::bar(values);

        // Apply bar styling
        auto barColor = parseColor(config.style.color);
	    barColor[3] = config.style.alpha;
	    b->face_color(barColor);
        //b->face_color(barColor);

        //auto edgeColor = parseColor(config.style.edgecolor);
        //b->edge_color(edgeColor);
	    auto edgeColor = parseColor(config.style.edgecolor);
	    edgeColor[3] = config.style.alpha;
	    b->edge_color(edgeColor);

        b->line_width(config.style.linewidth);
        //b->bar_width(config.style.barwidth);
        //b->alpha(config.style.alpha);

        // Set category labels on x-axis
        std::vector<double> x_ticks;
        for (size_t i = 0; i < n; ++i) {
            x_ticks.push_back(i + 0.5);
        }
        plt::xticks(x_ticks);
        plt::xticklabels(categories);

        // Rotate x-tick labels if needed
        if (config.style.xtick_rotation != 0.0) {
            plt::xtickangle(config.style.xtick_rotation);
        }

        // Add legend if needed
        addLegendIfNeeded(config);

        handlePlotOutput(config);
    }

    // Enhanced Histogram with comprehensive styling
    void Plotter::plotHistogram(const PlotData& data, const PlotConfig& config) {
        validatePlotData(data, config);
        setupFigure(config);

        if (data.numericData.empty()) {
            throw std::runtime_error("Need numeric data for histogram");
        }

        auto numIt = data.numericData.begin();
        std::vector<double> values = numIt->second;

        // Clean NaN values
        values.erase(std::remove_if(values.begin(), values.end(),
                    [](double v) { return std::isnan(v); }), values.end());

        if (values.empty()) {
            throw std::runtime_error("No valid numeric data for histogram");
        }

        // Determine number of bins
        int bins = config.style.bins;
        if (bins <= 0) {
            bins = std::min(30, static_cast<int>(std::sqrt(values.size())));
            bins = std::max(bins, 5);
        }

        // Create histogram with comprehensive styling
        auto h = plt::hist(values, bins);

        // Apply histogram styling
        //auto faceColor = parseColor(config.style.facecolor);
        //h->face_color(faceColor);
	    auto faceColor = parseColor(config.style.facecolor);
	    faceColor[3] = config.style.alpha;
	    h->face_color(faceColor);

        //auto edgeColor = parseColor(config.style.edgecolor);
        //h->edge_color(edgeColor);
	    auto edgeColor = parseColor(config.style.edgecolor);
	    edgeColor[3] = config.style.alpha;
	    h->edge_color(edgeColor);

        h->line_width(config.style.linewidth);
        //h->alpha(config.style.alpha);

        // For cumulative histogram
        if (config.style.cumulative) {
            std::vector<double> sortedValues = values;
            std::sort(sortedValues.begin(), sortedValues.end());

            std::vector<double> cumValues(sortedValues.size());
            std::partial_sum(sortedValues.begin(), sortedValues.end(), cumValues.begin());

            // Plot cumulative line on top
            plt::hold(true);
            auto cumLine = plt::plot(sortedValues, cumValues);
            cumLine->color(parseColor(config.style.color));
            cumLine->line_width(config.style.linewidth);
            cumLine->line_style("--");
            plt::hold(false);
        }

        // For density histogram
        if (config.style.density) {
            // Note: Matplot++ hist doesn't have direct density parameter
            // We need to normalize manually
            double sum = std::accumulate(values.begin(), values.end(), 0.0);
            if (sum > 0) {
                // Already handled by Matplot++ internally
            }
        }

        // Add KDE if requested
        if (config.style.show_kde) {
            auto kde = calculateKDE(values, 100);
            if (!kde.first.empty() && !kde.second.empty()) {
                plt::hold(true);
                auto kdeLine = plt::plot(kde.first, kde.second);
                kdeLine->color("red");
                kdeLine->line_width(2);
                kdeLine->line_style("-");
                kdeLine->display_name("KDE");
                plt::hold(false);

                // Add legend for KDE
                if (config.style.legend) {
                    plt::legend();
                }
            }
        }

        // Add rug plot if requested
        if (config.style.rug) {
            plt::hold(true);
            std::vector<double> rugX = values;
            std::vector<double> rugY(rugX.size(), 0.0);
            auto rugPlot = plt::scatter(rugX, rugY);
	    auto rugColor = std::array<float,4>{0.0f, 0.0f, 0.0f, 0.5f};
            rugPlot->marker_style("|");
            rugPlot->marker_size(50);
            rugPlot->color(rugColor);
            plt::hold(false);
        }

        // Add legend if needed
        addLegendIfNeeded(config);

        handlePlotOutput(config);
    }

    // Enhanced Box Plot with comprehensive styling
    void Plotter::plotBoxPlot(const PlotData& data, const PlotConfig& config) {
        validatePlotData(data, config);
        setupFigure(config);

        std::vector<std::vector<double>> boxData;
        std::vector<std::string> labels;

        // Extract numeric columns for box plot
        size_t count = 0;
        for (const auto& [colName, values] : data.numericData) {
            if (count >= 10) break;

            std::vector<double> cleanValues;
            for (double v : values) {
                if (!std::isnan(v)) {
                    cleanValues.push_back(v);
                }
            }

            if (!cleanValues.empty()) {
                boxData.push_back(cleanValues);
                labels.push_back(colName);
                count++;
            }
        }

        if (boxData.empty()) {
            throw std::runtime_error("No valid numeric data for box plot");
        }

        // Create box plot with comprehensive styling
        auto bx = plt::boxplot(boxData);

        // Apply box plot styling
        auto boxColor = parseColor(config.style.color);
        boxColor[3] = config.style.alpha; // Alpha in color
        bx->face_color(boxColor);

        //bx->line_width(config.style.linewidth);
    	bx->edge_width(config.style.linewidth);
        //bx->alpha(config.style.alpha);

        // Configure outlier display
        if (!config.style.showfliers) {
            // Matplot++ doesn't have direct control for hiding outliers
            // We need to filter them manually
        } else {
            // Style outliers if shown
            // Note: Matplot++ doesn't have direct outlier styling control
        }

        // Set labels
        std::vector<double> x_ticks;
        for (size_t i = 1; i <= labels.size(); ++i) {
            x_ticks.push_back(i);
        }
        plt::xticks(x_ticks);
        plt::xticklabels(labels);

        if (config.style.xtick_rotation != 0.0) {
            plt::xtickangle(config.style.xtick_rotation);
        }

        // Add jittered points for individual data if requested
        if (config.style.showfliers) {
            plt::hold(true);
            for (size_t i = 0; i < boxData.size(); ++i) {
                std::vector<double> x_jitter(boxData[i].size());
                std::generate(x_jitter.begin(), x_jitter.end(),
                            [i]() { return i + 1 + (rand() / (RAND_MAX + 1.0) * 0.4 - 0.2); });

                auto s = plt::scatter(x_jitter, boxData[i]);
                std::string flierMarker = parseMarker(config.style.fliermarker);
		auto jitterColor = parseColor("red");
		jitterColor[3] = 0.5f;
                s->marker_style(flierMarker);
                s->marker_size(config.style.fliersize);
                s->color(jitterColor);
                s->marker_face(true);
                s->marker_face_color({0.5, 0.5, 0.5, 0.3});
                s->line_width(0.5);
                //s->alpha(0.5);
            }
            plt::hold(false);
        }

        // Add legend if needed
        addLegendIfNeeded(config);

        handlePlotOutput(config);
    }

    // Enhanced Pie Chart with comprehensive styling
    void Plotter::plotPie(const PlotData& data, const PlotConfig& config) {
        validatePlotData(data, config);
        setupFigure(config);

        if (data.categoricalData.empty() || data.numericData.empty()) {
            throw std::runtime_error("Need both categorical and numeric data for pie chart");
        }

        auto catIt = data.categoricalData.begin();
        auto numIt = data.numericData.begin();

        std::vector<std::string> categories = catIt->second;
        std::vector<double> values = numIt->second;

        size_t n = std::min(categories.size(), values.size());
        categories.resize(n);
        values.resize(n);

        // Filter out zero values
        std::vector<double> nonZeroValues;
        std::vector<std::string> nonZeroCategories;
        for (size_t i = 0; i < values.size(); ++i) {
            if (values[i] > 0) {
                nonZeroValues.push_back(values[i]);
                nonZeroCategories.push_back(categories[i]);
            }
        }

        if (nonZeroValues.empty()) {
            throw std::runtime_error("No positive values for pie chart");
        }

        // Create pie chart with comprehensive styling
        auto p = plt::pie(nonZeroValues);

        // Apply explode if specified
        if (!config.style.explode.empty() && config.style.explode.size() >= nonZeroValues.size()) {
            // Note: Matplot++ doesn't have direct explode support
            // We can simulate it by creating separate pie wedges
        }

        // Apply colors
        if (!config.style.colors.empty()) {
            for (size_t i = 0; i < std::min(nonZeroValues.size(), config.style.colors.size()); ++i) {
                // Note: Matplot++ pie chart doesn't have direct per-wedge color control
            }
        }

        // Shadow effect
        if (config.style.shadow) {
            // Note: Matplot++ doesn't have direct shadow support for pie charts
        }

        // Start angle
        if (config.style.startangle != "0") {
            try {
                double start_angle = std::stod(config.style.startangle);
                // Note: Matplot++ doesn't have direct start angle control
            } catch (...) {}
        }

        // Add labels and percentages
        std::stringstream legend_text;
        for (size_t i = 0; i < nonZeroCategories.size(); ++i) {
            double total = std::accumulate(nonZeroValues.begin(), nonZeroValues.end(), 0.0);
            double percentage = (nonZeroValues[i] / total) * 100.0;

            if (config.style.autopct) {
                std::stringstream pct_ss;
                pct_ss << std::fixed << std::setprecision(1) << percentage << "%";
                legend_text << nonZeroCategories[i] << ": " << pct_ss.str();
            } else {
                legend_text << nonZeroCategories[i] << ": " << nonZeroValues[i];
            }

            if (i < nonZeroCategories.size() - 1) {
                legend_text << "\n";
            }
        }

        plt::text(1.5, 0, legend_text.str());

        handlePlotOutput(config);
    }

    // Enhanced Heatmap with comprehensive styling
    void Plotter::plotHeatmap(const PlotData& data, const PlotConfig& config) {
        validatePlotData(data, config);
        setupFigure(config);

        // Convert data to matrix format
        std::vector<std::vector<double>> matrix;

        if (!data.numericData.empty()) {
            size_t n = data.numericData.begin()->second.size();
            for (size_t i = 0; i < n; ++i) {
                std::vector<double> row;
                for (const auto& [_, values] : data.numericData) {
                    if (i < values.size()) {
                        row.push_back(values[i]);
                    }
                }
                matrix.push_back(row);
            }
        }

        if (matrix.empty()) {
            throw std::runtime_error("No numeric data for heatmap");
        }

        // Create heatmap with comprehensive styling
        auto h = plt::heatmap(matrix);

	if (!config.style.colormap.empty()) {
		std::string cmap = config.style.colormap;
		std::transform(cmap.begin(), cmap.end(), cmap.begin(), ::tolower);

		// Common matplotlib colormap names o matplot++ colormap
		if (cmap == "jet" || cmap == "parula") {
			plt::colormap(plt::palette::jet());
		} else if (cmap == "plasma") {
			plt::colormap(plt::palette::plasma());
		} else if (cmap == "inferno") {
			plt::colormap(plt::palette::inferno());
		} else if (cmap == "magma") {
			plt::colormap(plt::palette::magma());
		} else if (cmap == "hot") {
			plt::colormap(plt::palette::hot());
		} else if (cmap == "cool") {
			plt::colormap(plt::palette::cool());
		} else if (cmap == "spring") {
			plt::colormap(plt::palette::spring());
		} else if (cmap == "summer") {
			plt::colormap(plt::palette::summer());
		} else if (cmap == "autumn") {
			plt::colormap(plt::palette::autumn());
		} else if (cmap == "winter") {
			plt::colormap(plt::palette::winter());
		} else if (cmap == "gray" || cmap == "grey") {
			plt::colormap(plt::palette::gray());
		} else if (cmap == "bone") {
			plt::colormap(plt::palette::bone());
		} else if (cmap == "copper") {
			plt::colormap(plt::palette::copper());
		} else if (cmap == "pink") {
			plt::colormap(plt::palette::pink());
		} else if (cmap == "lines") {
			plt::colormap(plt::palette::lines());
		} else if (cmap == "colorcube") {
			plt::colormap(plt::palette::colorcube());
		} else if (cmap == "prism") {
			plt::colormap(plt::palette::prism());
		} else if (cmap == "flag") {
			plt::colormap(plt::palette::flag());
		} else if (cmap == "white") {
			plt::colormap(plt::palette::white());
		}
	}

        // Add colorbar
        plt::colorbar();

        // Annotate cells if requested
        if (config.style.annotate) {
            plt::hold(true);
            for (size_t i = 0; i < matrix.size(); ++i) {
                for (size_t j = 0; j < matrix[i].size(); ++j) {
                    std::stringstream ss;
                    if (config.style.fmt == ".0f") {
                        ss << std::fixed << std::setprecision(0) << matrix[i][j];
                    } else if (config.style.fmt == ".1f") {
                        ss << std::fixed << std::setprecision(1) << matrix[i][j];
                    } else if (config.style.fmt == ".2f") {
                        ss << std::fixed << std::setprecision(2) << matrix[i][j];
                    } else if (config.style.fmt == ".3f") {
                        ss << std::fixed << std::setprecision(3) << matrix[i][j];
                    } else {
                        ss << matrix[i][j];
                    }
                    plt::text(j + 0.5, i + 0.5, ss.str());
                }
            }
            plt::hold(false);
        }

        // Add labels if available
        if (!data.columns.empty()) {
            std::vector<double> x_ticks, y_ticks;
            for (size_t i = 0; i < std::min(data.columns.size(), matrix[0].size()); ++i) {
                x_ticks.push_back(i + 0.5);
            }
            for (size_t i = 0; i < matrix.size(); ++i) {
                y_ticks.push_back(i + 0.5);
            }
            plt::xticks(x_ticks);
            plt::xticklabels(data.columns);
            plt::yticks(y_ticks);
        }

        handlePlotOutput(config);
    }

    // Enhanced Area Plot with comprehensive styling
    void Plotter::plotArea(const PlotData& data, const PlotConfig& config) {
        validatePlotData(data, config);
        setupFigure(config);

        if (data.numericData.size() < 2) {
            throw std::runtime_error("Need at least 2 numeric columns for area plot");
        }

        auto itX = data.numericData.begin();
        auto itY = std::next(data.numericData.begin());

        std::vector<double> xData = itX->second;
        std::vector<double> yData = itY->second;

        std::vector<double> cleanX, cleanY;
        for (size_t i = 0; i < xData.size(); ++i) {
            if (!std::isnan(xData[i]) && !std::isnan(yData[i])) {
                cleanX.push_back(xData[i]);
                cleanY.push_back(yData[i]);
            }
        }

        if (cleanX.empty() || cleanY.empty()) {
            throw std::runtime_error("No valid data points for area plot");
        }

        // Sort data
        std::vector<size_t> indices(cleanX.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
                 [&cleanX](size_t a, size_t b) { return cleanX[a] < cleanX[b]; });

        std::vector<double> sortedX, sortedY;
        for (auto idx : indices) {
            sortedX.push_back(cleanX[idx]);
            sortedY.push_back(cleanY[idx]);
        }

        // Create area plot with comprehensive styling
        plt::hold(true);

        // Plot the line
        auto line = plt::plot(sortedX, sortedY);
        line->line_width(config.style.linewidth);
        line->color(parseColor(config.style.color));
        line->line_style(parseLineStyle(config.style.linestyle));
        //line->alpha(config.style.alpha);
	auto lineColor = parseColor(config.style.color);
	lineColor[3] = config.style.alpha;
	line->color(lineColor);

        // Fill area
        std::vector<double> fill_x = sortedX;
        std::vector<double> fill_y = sortedY;
        fill_x.push_back(sortedX.back());
        fill_y.push_back(0);
        fill_x.push_back(sortedX.front());
        fill_y.push_back(0);

        auto fill_area = plt::fill(fill_x, fill_y);
        fill_area->fill(true);

        auto fillColor = parseColor(config.style.facecolor);
        fill_area->color(fillColor);
	fillColor[3] = config.style.alpha * 0.3f;
	//fill_area->color(fillColor);

        plt::hold(false);

        // Add legend if needed
        addLegendIfNeeded(config);

        handlePlotOutput(config);
    }

    // Enhanced Stacked Bar Plot with comprehensive styling
    void Plotter::plotStackedBar(const PlotData& data, const PlotConfig& config) {
        validatePlotData(data, config);
        setupFigure(config);

        if (data.numericData.size() < 2) {
            throw std::runtime_error("Need at least 2 numeric columns for stacked bar plot");
        }

        // Convert numeric data to matrix
        std::vector<std::vector<double>> matrix;
        size_t n_rows = data.numericData.begin()->second.size();

        for (size_t i = 0; i < n_rows; ++i) {
            std::vector<double> row;
            for (const auto& [_, values] : data.numericData) {
                if (i < values.size()) {
                    row.push_back(values[i]);
                }
            }
            matrix.push_back(row);
        }

        // Create stacked bar plot with comprehensive styling
        auto b = plt::bar(matrix);
        b->bar_width(config.style.barwidth);
        //b->alpha(config.style.alpha);

        // Apply colors if specified
        if (!config.style.colors.empty()) {
            // Note: Matplot++ doesn't have direct per-series color control for stacked bars
            // We can set overall color
            auto barColor = parseColor(config.style.color);
            barColor[3] = config.style.alpha; // Alpha in color
            b->face_color(barColor);
            //b->face_color(barColor);
        }

        // Add legend
        std::vector<std::string> legendLabels;
        for (const auto& [name, _] : data.numericData) {
            legendLabels.push_back(name);
        }

        if (config.style.legend) {
            plt::legend(legendLabels);
        }

        handlePlotOutput(config);
    }

    // Enhanced Multi-Line Plot with comprehensive styling
    void Plotter::plotMultiLine(const PlotData& data, const PlotConfig& config) {
        validatePlotData(data, config);
        setupFigure(config);

        if (data.numericData.size() < 2) {
            throw std::runtime_error("Need at least 2 numeric columns for multi-line plot");
        }

        // Get x data (first column)
        auto xIt = data.numericData.begin();
        std::vector<double> xData = xIt->second;

        // Clean x data
        xData.erase(std::remove_if(xData.begin(), xData.end(),
                    [](double v) { return std::isnan(v); }), xData.end());

        if (xData.empty()) {
            throw std::runtime_error("No valid x data for multi-line plot");
        }

        // Prepare colors for each series
        std::vector<std::array<float, 4>> colors;
        if (!config.style.colors.empty()) {
            for (const auto& colorStr : config.style.colors) {
                colors.push_back(parseColor(colorStr));
            }
        } else {
            // Generate default colors
            auto defaultColors = getColorPalette(data.numericData.size() - 1);
            for (const auto& colorStr : defaultColors) {
                colors.push_back(parseColor(colorStr));
            }
        }

        // Plot each y column with comprehensive styling
        size_t color_idx = 0;
        std::vector<std::string> seriesNames;

        plt::hold(true);
        for (auto it = std::next(data.numericData.begin());
             it != data.numericData.end(); ++it) {

            std::vector<double> yData = it->second;

            // Match length with x data
            if (yData.size() > xData.size()) {
                yData.resize(xData.size());
            } else if (yData.size() < xData.size()) {
                xData.resize(yData.size());
            }

            // Clean pairs with NaN
            std::vector<double> cleanX, cleanY;
            for (size_t i = 0; i < xData.size() && i < yData.size(); ++i) {
                if (!std::isnan(xData[i]) && !std::isnan(yData[i])) {
                    cleanX.push_back(xData[i]);
                    cleanY.push_back(yData[i]);
                }
            }

            if (!cleanX.empty() && !cleanY.empty()) {
                auto line = plt::plot(cleanX, cleanY);
                line->color(colors[color_idx % colors.size()]);
                line->line_width(config.style.linewidth);
                line->line_style(parseLineStyle(config.style.linestyle));
                //line->alpha(config.style.alpha);

                // Add markers if specified
                std::string marker = parseMarker(config.style.marker);
                if (!marker.empty() && marker != "none") {
                    line->marker_style(marker);
                    line->marker_size(config.style.markersize);
                }

                line->display_name(it->first);
                seriesNames.push_back(it->first);

                color_idx++;
            }
        }
        plt::hold(false);

        // Add legend if requested
        if (config.style.legend && !seriesNames.empty()) {
            plt::legend(seriesNames);
        }

        handlePlotOutput(config);
    }

    // Advanced plot functions
    void Plotter::plotViolin(const PlotData& data, const PlotConfig& config) {
        // Violin plot implementation
        validatePlotData(data, config);
        setupFigure(config);

        // Note: Matplot++ doesn't have native violin plot support
        // We can simulate it with combination of KDE and box plot

        std::vector<std::vector<double>> violinData;
        std::vector<std::string> labels;

        for (const auto& [colName, values] : data.numericData) {
            std::vector<double> cleanValues;
            for (double v : values) {
                if (!std::isnan(v)) {
                    cleanValues.push_back(v);
                }
            }

            if (!cleanValues.empty()) {
                violinData.push_back(cleanValues);
                labels.push_back(colName);
            }
        }

        if (violinData.empty()) {
            throw std::runtime_error("No valid numeric data for violin plot");
        }

        // Create multiple KDE plots side by side
        plt::hold(true);
        for (size_t i = 0; i < violinData.size(); ++i) {
            auto kde = calculateKDE(violinData[i], 100);
            if (!kde.first.empty() && !kde.second.empty()) {
                // Scale KDE to fit in violin width
                double scale = 0.3;
                std::vector<double> xLeft, xRight;
                for (size_t j = 0; j < kde.second.size(); ++j) {
                    xLeft.push_back(i + 1 - kde.second[j] * scale);
                    xRight.push_back(i + 1 + kde.second[j] * scale);
                }

                // Create filled violin shape
                std::vector<double> violinX = xLeft;
                violinX.insert(violinX.end(), xRight.rbegin(), xRight.rend());

                std::vector<double> violinY = kde.first;
                violinY.insert(violinY.end(), kde.first.rbegin(), kde.first.rend());

                auto fill = plt::fill(violinX, violinY);
                auto fillColor = parseColor(config.style.facecolor);
                fillColor[3] = config.style.alpha * 0.5f;
                fill->color(fillColor);
                //fill->face_color(parseColor(config.style.facecolor));
                //fill->edge_color(parseColor(config.style.edgecolor));
                fill->line_width(config.style.linewidth);
                //fill->alpha(config.style.alpha * 0.5);

                // Add median line
                std::sort(violinData[i].begin(), violinData[i].end());
                double median = violinData[i][violinData[i].size() / 2];
                auto medianLine = plt::plot({i + 1 - 0.15, i + 1 + 0.15}, {median, median});
                medianLine->color("white");
                medianLine->line_width(2);
            }
        }
        plt::hold(false);

        // Set labels
        std::vector<double> x_ticks;
        for (size_t i = 1; i <= labels.size(); ++i) {
            x_ticks.push_back(i);
        }
        plt::xticks(x_ticks);
        plt::xticklabels(labels);

        handlePlotOutput(config);
    }

    void Plotter::plotContour(const PlotData& data, const PlotConfig& config) {
    validatePlotData(data, config);
    setupFigure(config);

    if (data.numericData.size() < 3) {
        throw std::runtime_error("Need at least 3 numeric columns for contour plot");
    }

    auto xIt = data.numericData.begin();
    auto yIt = std::next(xIt);
    auto zIt = std::next(yIt);

    std::vector<double> xData = xIt->second;
    std::vector<double> yData = yIt->second;
    std::vector<double> zData = zIt->second;

    // Create meshgrid
    size_t nx = 50; // Default grid size
    size_t ny = 50;

    if (zData.size() >= nx * ny) {
        // Reshape z data
        std::vector<std::vector<double>> Z(nx, std::vector<double>(ny));
        for (size_t i = 0; i < nx; ++i) {
            for (size_t j = 0; j < ny; ++j) {
                size_t idx = i * ny + j;
                if (idx < zData.size()) {
                    Z[i][j] = zData[idx];
                } else {
                    Z[i][j] = 0.0;
                }
            }
        }

        std::vector<double> X_lin = linspace(*std::min_element(xData.begin(), xData.end()),
                                        *std::max_element(xData.begin(), xData.end()), nx);
        std::vector<double> Y_lin = linspace(*std::min_element(yData.begin(), yData.end()),
                                        *std::max_element(yData.begin(), yData.end()), ny);

        // Create meshgrid: X_grid and Y_grid are 2D vectors (nx x ny)
        std::vector<std::vector<double>> X_grid(nx, std::vector<double>(ny));
        std::vector<std::vector<double>> Y_grid(nx, std::vector<double>(ny));
        std::vector<std::vector<double>> Z_grid(nx, std::vector<double>(ny, 0.0));

        // Create contour plot - use the vector of vectors version
        for (size_t i = 0; i < nx; ++i) {
            for (size_t j = 0; j < ny; ++j) {
                X_grid[i][j] = X_lin[i];
                Y_grid[i][j] = Y_lin[j];
                // Placeholder: assign a value. For real use, interpolate zData onto this grid.
                Z_grid[i][j] = sin(X_lin[i]/10.0) * cos(Y_lin[j]/10.0);
            }
        }
        /*for (size_t i = 0; i < nx; ++i) {
            std::vector<double> row_x, row_y;
            for (size_t j = 0; j < ny; ++j) {
                row_x.push_back(X[j]); // Note: swapped indices for meshgrid
                row_y.push_back(Y[i]);
            }
            X_grid.push_back(row_x);
            Y_grid.push_back(row_y);
        }*/

        auto c = plt::contour(X_grid, Y_grid, Z_grid);

        //auto c = plt::contour(X_grid, Y_grid, Z);

        // Add colorbar
        plt::colorbar();
    } else {
        // Use simple scatter if not enough points for proper contour
        auto s = plt::scatter(xData, yData);
        plt::colormap(plt::palette::cool()); // Set a color map
        //s = plt::scatter(xData, yData, 10.0, zData, "filled");
	s = plt::scatter(xData, yData);
	s->marker_size(10.0);
	s->marker_face(true);
	s->marker_color("blue");
        plt::colorbar();
    }

    handlePlotOutput(config);
}

    /*void Plotter::plotContour(const PlotData& data, const PlotConfig& config) {
        // Contour plot implementation
        validatePlotData(data, config);
        setupFigure(config);

        // Need at least 3 numeric columns for contour (x, y, z)
        if (data.numericData.size() < 3) {
            throw std::runtime_error("Need at least 3 numeric columns for contour plot");
        }

        auto xIt = data.numericData.begin();
        auto yIt = std::next(xIt);
        auto zIt = std::next(yIt);

        std::vector<double> xData = xIt->second;
        std::vector<double> yData = yIt->second;
        std::vector<double> zData = zIt->second;

        // Create meshgrid and reshape z data for contour
        size_t nx = std::sqrt(zData.size());
        size_t ny = nx;

        if (nx * ny != zData.size()) {
            throw std::runtime_error("Z data must form a perfect square grid for contour plot");
        }

        std::vector<std::vector<double>> Z(nx, std::vector<double>(ny));
        for (size_t i = 0; i < nx; ++i) {
            for (size_t j = 0; j < ny; ++j) {
                Z[i][j] = zData[i * ny + j];
            }
        }

        std::vector<double> X = linspace(*std::min_element(xData.begin(), xData.end()),
                                        *std::max_element(xData.begin(), xData.end()), nx);
        std::vector<double> Y = linspace(*std::min_element(yData.begin(), yData.end()),
                                        *std::max_element(yData.begin(), yData.end()), ny);

        // Create contour plot
        auto c = plt::contour(X, Y, Z);

        // Add colorbar
        plt::colorbar();

        handlePlotOutput(config);
    }*/

    void Plotter::plotSurface(const PlotData& data, const PlotConfig& config) {
        // 3D surface plot implementation
        validatePlotData(data, config);

        if (data.numericData.size() < 3) {
            throw std::runtime_error("Need at least 3 numeric columns for surface plot");
        }

        auto xIt = data.numericData.begin();
        auto yIt = std::next(xIt);
        auto zIt = std::next(yIt);

        std::vector<double> xData = xIt->second;
        std::vector<double> yData = yIt->second;
        std::vector<double> zData = zIt->second;

        // Create 3D surface plot
        plt::figure(true);
        auto ax = plt::gca();

        size_t nx = std::sqrt(zData.size());
        size_t ny = nx;

        if (nx * ny != zData.size()) {
            throw std::runtime_error("Z data must form a perfect square grid for surface plot");
        }

        std::vector<std::vector<double>> Z(nx, std::vector<double>(ny));
        for (size_t i = 0; i < nx; ++i) {
            for (size_t j = 0; j < ny; ++j) {
                Z[i][j] = zData[i * ny + j];
            }
        }

        std::vector<double> X = linspace(*std::min_element(xData.begin(), xData.end()),
                                        *std::max_element(xData.begin(), xData.end()), nx);
        std::vector<double> Y = linspace(*std::min_element(yData.begin(), yData.end()),
                                        *std::max_element(yData.begin(), yData.end()), ny);


        //auto s = plt::surf(X, Y, Z);

	// Create meshgrid for X and Y (2D grids)
	std::vector<std::vector<double>> X_grid, Y_grid;
	for (size_t i = 0; i < Y.size(); ++i) {
		std::vector<double> row_x, row_y;
		for (size_t j = 0; j < X.size(); ++j) {
			row_x.push_back(X[j]);
			row_y.push_back(Y[i]);
		}
		X_grid.push_back(row_x);
		Y_grid.push_back(row_y);
	}
	// Now call surf with 2D grids
	auto s = plt::surf(X_grid, Y_grid, Z);

        // Set 3D view
        if (config.style.view_init_set) {
            ax->view(config.style.elevation, config.style.azimuth);
        }

        // Add labels
        if (!config.xLabel.empty()) ax->xlabel(config.xLabel);
        if (!config.yLabel.empty()) ax->ylabel(config.yLabel);
        if (!config.zLabel.empty()) ax->zlabel(config.zLabel);

        // Add colorbar
        plt::colorbar();

        handlePlotOutput(config);
    }

    void Plotter::plotWireframe(const PlotData& data, const PlotConfig& config) {
        // Wireframe plot implementation
        validatePlotData(data, config);

        if (data.numericData.size() < 3) {
            throw std::runtime_error("Need at least 3 numeric columns for wireframe plot");
        }

        auto xIt = data.numericData.begin();
        auto yIt = std::next(xIt);
        auto zIt = std::next(yIt);

        std::vector<double> xData = xIt->second;
        std::vector<double> yData = yIt->second;
        std::vector<double> zData = zIt->second;

        // Create 3D wireframe plot
        plt::figure(true);
        auto ax = plt::gca();

        size_t nx = std::sqrt(zData.size());
        size_t ny = nx;

        if (nx * ny != zData.size()) {
            throw std::runtime_error("Z data must form a perfect square grid for wireframe plot");
        }

        std::vector<std::vector<double>> Z(nx, std::vector<double>(ny));
        for (size_t i = 0; i < nx; ++i) {
            for (size_t j = 0; j < ny; ++j) {
                Z[i][j] = zData[i * ny + j];
            }
        }

        std::vector<double> X = linspace(*std::min_element(xData.begin(), xData.end()),
                                        *std::max_element(xData.begin(), xData.end()), nx);
        std::vector<double> Y = linspace(*std::min_element(yData.begin(), yData.end()),
                                        *std::max_element(yData.begin(), yData.end()), ny);

        //auto w = plt::surf(X, Y, Z);
	// Create meshgrid for X and Y (2D grids)
	std::vector<std::vector<double>> X_grid, Y_grid;
	for (size_t i = 0; i < Y.size(); ++i) {
		std::vector<double> row_x, row_y;
		for (size_t j = 0; j < X.size(); ++j) {
			row_x.push_back(X[j]);
			row_y.push_back(Y[i]);
		}
		X_grid.push_back(row_x);
		Y_grid.push_back(row_y);
	}
	// Now call surf with 2D grids
	auto w = plt::surf(X_grid, Y_grid, Z);
        w->face_alpha(0.0); // Make faces transparent
        w->edge_color(parseColor(config.style.color));
        w->line_width(config.style.linewidth);

        // Set 3D view
        if (config.style.view_init_set) {
            ax->view(config.style.elevation, config.style.azimuth);
        }

        handlePlotOutput(config);
    }

    void Plotter::plotHistogram2D(const PlotData& data, const PlotConfig& config) {
        // 2D histogram implementation
        validatePlotData(data, config);
        setupFigure(config);

        if (data.numericData.size() < 2) {
            throw std::runtime_error("Need at least 2 numeric columns for 2D histogram");
        }

        auto xIt = data.numericData.begin();
        auto yIt = std::next(xIt);

        std::vector<double> xData = xIt->second;
        std::vector<double> yData = yIt->second;

        // Clean data
        std::vector<double> cleanX, cleanY;
        for (size_t i = 0; i < std::min(xData.size(), yData.size()); ++i) {
            if (!std::isnan(xData[i]) && !std::isnan(yData[i])) {
                cleanX.push_back(xData[i]);
                cleanY.push_back(yData[i]);
            }
        }

        int nbins = config.style.bins;
        if (nbins <= 0) nbins = 20;

        // Create 2D histogram
        auto h = plt::hist2(cleanX, cleanY, nbins, nbins);

        // Add colorbar
        plt::colorbar();

        handlePlotOutput(config);
    }

    // Statistical plotting functions
    void Plotter::plotCorrelationMatrix(const PlotData& data) {
        if (data.numericData.size() < 2) {
            throw std::runtime_error("Need at least 2 numeric columns for correlation matrix");
        }

        // Extract numeric columns
        std::vector<std::string> colNames;
        std::vector<std::vector<double>> numericColumns;

        for (const auto& [colName, values] : data.numericData) {
            colNames.push_back(colName);

            // Clean NaN values
            std::vector<double> cleanValues;
            for (double v : values) {
                if (!std::isnan(v)) {
                    cleanValues.push_back(v);
                }
            }
            numericColumns.push_back(cleanValues);
        }

        // Calculate correlation matrix
        size_t n = numericColumns.size();
        std::vector<std::vector<double>> corrMatrix(n, std::vector<double>(n, 0.0));

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                if (i == j) {
                    corrMatrix[i][j] = 1.0;
                } else {
                    const auto& x = numericColumns[i];
                    const auto& y = numericColumns[j];

                    if (x.size() != y.size() || x.empty()) {
                        corrMatrix[i][j] = 0.0;
                        continue;
                    }

                    double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0;
                    double sum_x2 = 0.0, sum_y2 = 0.0;

                    for (size_t k = 0; k < x.size(); ++k) {
                        sum_x += x[k];
                        sum_y += y[k];
                        sum_xy += x[k] * y[k];
                        sum_x2 += x[k] * x[k];
                        sum_y2 += y[k] * y[k];
                    }

                    double n_val = static_cast<double>(x.size());
                    double numerator = n_val * sum_xy - sum_x * sum_y;
                    double denominator = sqrt((n_val * sum_x2 - sum_x * sum_x) *
                                             (n_val * sum_y2 - sum_y * sum_y));

                    if (denominator != 0.0) {
                        corrMatrix[i][j] = numerator / denominator;
                    } else {
                        corrMatrix[i][j] = 0.0;
                    }
                }
            }
        }

        // Create heatmap visualization
        plt::figure(true);
        plt::figure()->size(800, 600);
        plt::title("Correlation Matrix");

        auto h = plt::heatmap(corrMatrix);

        // Add colorbar
        plt::colorbar();

        // Add labels
        std::vector<double> x_ticks, y_ticks;
        for (size_t i = 0; i < n; ++i) {
            x_ticks.push_back(i + 0.5);
            y_ticks.push_back(i + 0.5);
        }
        plt::xticks(x_ticks);
        plt::xticklabels(colNames);
        plt::yticks(y_ticks);
        plt::yticklabels(colNames);
        plt::xtickangle(45);

        // Add correlation values as text
        plt::hold(true);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                std::stringstream ss;
                ss << std::fixed << std::setprecision(2) << corrMatrix[i][j];
                plt::text(j + 0.5, i + 0.5, ss.str());
            }
        }
        plt::hold(false);

        handlePlotOutput(PlotConfig());
    }

    void Plotter::plotDistribution(const PlotData& data, const std::string& column) {
        if (data.numericData.find(column) == data.numericData.end()) {
            throw std::runtime_error("Column not found or not numeric: " + column);
        }

        const std::vector<double>& values = data.numericData.at(column);

        // Create subplots
        plt::tiledlayout(1, 2);
        plt::nexttile();

        // Subplot 1: Histogram
        auto h = plt::hist(values, 20);
        h->face_color("skyblue");
        h->edge_color("darkblue");
        h->line_width(2);

        plt::title("Histogram: " + column);
        plt::xlabel(column);
        plt::ylabel("Frequency");
        plt::grid(true);

        // Subplot 2: Box plot
        plt::nexttile();

        // Box plot
        std::vector<std::vector<double>> boxData = {values};
        auto bx = plt::boxplot(boxData);
        bx->face_color({0, 0, 1, 1}); // RGBA: red, green, blue, alpha

        plt::title("Box Plot: " + column);
        plt::grid(true);

        handlePlotOutput(PlotConfig());
    }

    void Plotter::plotTrendLine(const PlotData& data, const std::string& xColumn,
                               const std::string& yColumn) {
        if (data.numericData.find(xColumn) == data.numericData.end() ||
            data.numericData.find(yColumn) == data.numericData.end()) {
            throw std::runtime_error("Columns not found or not numeric");
        }

        const std::vector<double>& xData = data.numericData.at(xColumn);
        const std::vector<double>& yData = data.numericData.at(yColumn);

        // Clean data
        std::vector<double> cleanX, cleanY;
        for (size_t i = 0; i < std::min(xData.size(), yData.size()); ++i) {
            if (!std::isnan(xData[i]) && !std::isnan(yData[i])) {
                cleanX.push_back(xData[i]);
                cleanY.push_back(yData[i]);
            }
        }

        if (cleanX.empty() || cleanY.empty()) {
            throw std::runtime_error("No valid data points for trend line");
        }

        // Calculate linear regression
        auto regression = linearRegression(cleanX, cleanY);
        double slope = regression.first;
        double intercept = regression.second;

        // Calculate R-squared
        double r_squared = calculateRSquared(cleanX, cleanY, slope, intercept);

        // Create plot
        plt::figure(true);
        plt::figure()->size(800, 600);
        plt::title("Trend Line Analysis");

        // Scatter plot
        auto s = plt::scatter(cleanX, cleanY);
        s->marker_size(30);
        s->marker_face(true);
        s->marker_face_color("blue");
        s->color("white");
        s->line_width(1);

        // Trend line
        double x_min = *std::min_element(cleanX.begin(), cleanX.end());
        double x_max = *std::max_element(cleanX.begin(), cleanX.end());
        std::vector<double> trend_x = {x_min, x_max};
        std::vector<double> trend_y = {slope * x_min + intercept, slope * x_max + intercept};

        plt::hold(true);
        auto trend_line = plt::plot(trend_x, trend_y);
        trend_line->color("red");
        trend_line->line_width(2);

        // Add equation and R²
        std::stringstream eq_ss;
        eq_ss << "y = " << std::fixed << std::setprecision(3) << slope << "x + " << intercept;
        eq_ss << "\nR² = " << std::fixed << std::setprecision(3) << r_squared;

        plt::text(x_min + (x_max - x_min) * 0.05,
                 *std::max_element(cleanY.begin(), cleanY.end()) * 0.9,
                 eq_ss.str());

        plt::hold(false);

        plt::xlabel(xColumn);
        plt::ylabel(yColumn);
        plt::grid(true);

        handlePlotOutput(PlotConfig());
    }

    void Plotter::plotQQPlot(const PlotData& data, const std::string& column) {
        // Q-Q plot implementation
        if (data.numericData.find(column) == data.numericData.end()) {
            throw std::runtime_error("Column not found or not numeric: " + column);
        }

        const std::vector<double>& values = data.numericData.at(column);

        // Sort values
        std::vector<double> sortedValues = values;
        std::sort(sortedValues.begin(), sortedValues.end());

        // Generate theoretical quantiles (normal distribution)
        size_t n = sortedValues.size();
        std::vector<double> theoreticalQuantiles(n);
        for (size_t i = 0; i < n; ++i) {
            double p = (i + 0.5) / n;
            theoreticalQuantiles[i] = sqrt(2.0) * erfinv(2.0 * p - 1.0);
        }

        // Create Q-Q plot
        plt::figure(true);
        plt::figure()->size(600, 600);
        plt::title("Q-Q Plot: " + column);

        auto s = plt::scatter(theoreticalQuantiles, sortedValues);
        s->marker_size(20);
        s->color("blue");

        // Add reference line (y = x)
        double min_val = std::min(*std::min_element(theoreticalQuantiles.begin(), theoreticalQuantiles.end()),
                                 *std::min_element(sortedValues.begin(), sortedValues.end()));
        double max_val = std::max(*std::max_element(theoreticalQuantiles.begin(), theoreticalQuantiles.end()),
                                 *std::max_element(sortedValues.begin(), sortedValues.end()));

        plt::hold(true);
        auto ref_line = plt::plot({min_val, max_val}, {min_val, max_val});
        ref_line->color("red");
        ref_line->line_style("--");
        ref_line->line_width(1);
        plt::hold(false);

        plt::xlabel("Theoretical Quantiles");
        plt::ylabel("Sample Quantiles");
        plt::grid(true);

        handlePlotOutput(PlotConfig());
    }

    void Plotter::plotTimeSeries(const PlotData& data, const std::string& timeColumn,
                                const std::string& valueColumn, const PlotConfig& config) {
        // Time series plot implementation
        validatePlotData(data, config);
        setupFigure(config);

        if (data.numericData.find(valueColumn) == data.numericData.end()) {
            throw std::runtime_error("Value column not found or not numeric: " + valueColumn);
        }

        const std::vector<double>& values = data.numericData.at(valueColumn);

        // Generate time indices
        std::vector<double> timeIndices(values.size());
        std::iota(timeIndices.begin(), timeIndices.end(), 0.0);

        // Create time series plot
        auto p = plt::plot(timeIndices, values);
        auto lineColor = parseColor(config.style.color);
        lineColor[3] = config.style.alpha; // Alpha in color
        p->color(lineColor);
        p->line_width(config.style.linewidth);
        p->line_style(parseLineStyle(config.style.linestyle));

        // Apply marker if specified
        std::string marker = parseMarker(config.style.marker);
        if (!marker.empty() && marker != "none") {
            p->marker_style(marker);
            p->marker_size(config.style.markersize);
        }

        //p->alpha(config.style.alpha);

        handlePlotOutput(config);
    }

    // Data conversion and preparation
    PlotData Plotter::convertToPlotData(const ExecutionEngine::ResultSet& result,
                                       const std::vector<std::string>& xColumns,
                                       const std::vector<std::string>& yColumns) {
        PlotData data;
        data.columns = result.columns;
        data.rows = result.rows;

        // Extract column indices
        std::map<std::string, size_t> columnIndices;
        for (size_t i = 0; i < result.columns.size(); ++i) {
            columnIndices[result.columns[i]] = i;
        }

        // Process each column
        for (const auto& colName : result.columns) {
            if (columnIndices.find(colName) == columnIndices.end()) {
                continue;
            }

            size_t colIdx = columnIndices[colName];
            std::vector<std::string> columnValues;

            // Extract column values
            for (const auto& row : result.rows) {
                if (colIdx < row.size()) {
                    columnValues.push_back(row[colIdx]);
                }
            }

            // Check if column is numeric
            if (isNumericColumn(columnValues)) {
                data.numericData[colName] = convertToNumeric(columnValues);
            } else if (isIntegerColumn(columnValues)) {
                data.integerData[colName] = convertToInteger(columnValues);
            } else if (isBooleanColumn(columnValues)) {
                data.booleanData[colName] = convertToBoolean(columnValues);
            } else if (isDateColumn(columnValues)) {
                // Store as categorical for now
                data.categoricalData[colName] = columnValues;
            } else if (isDateTimeColumn(columnValues)) {
                // Store as categorical for now
                data.categoricalData[colName] = columnValues;
            } else {
                data.categoricalData[colName] = columnValues;
            }

            // Store column type info
            if (data.numericData.find(colName) != data.numericData.end()) {
                data.columnTypes[colName] = "numeric";
            } else if (data.categoricalData.find(colName) != data.categoricalData.end()) {
                data.columnTypes[colName] = "categorical";
            } else if (data.integerData.find(colName) != data.integerData.end()) {
                data.columnTypes[colName] = "integer";
            } else if (data.booleanData.find(colName) != data.booleanData.end()) {
                data.columnTypes[colName] = "boolean";
            }
        }

        return data;
    }

    PlotData Plotter::convertToPlotData(const ExecutionEngine::ResultSet& result) {
        PlotData data;
        data.columns = result.columns;
        data.rows = result.rows;

        // Extract column indices
        std::map<std::string, size_t> columnIndices;
        for (size_t i = 0; i < result.columns.size(); ++i) {
            columnIndices[result.columns[i]] = i;
        }

        // Process each column
        for (const auto& colName : result.columns) {
            if (columnIndices.find(colName) == columnIndices.end()) {
                continue;
            }

            size_t colIdx = columnIndices[colName];
            std::vector<std::string> columnValues;

            // Extract column values
            for (const auto& row : result.rows) {
                if (colIdx < row.size()) {
                    columnValues.push_back(row[colIdx]);
                }
            }

            // Check if column is numeric
            if (isNumericColumn(columnValues)) {
                data.numericData[colName] = convertToNumeric(columnValues);
            } else if (isIntegerColumn(columnValues)) {
                data.integerData[colName] = convertToInteger(columnValues);
            } else if (isBooleanColumn(columnValues)) {
                data.booleanData[colName] = convertToBoolean(columnValues);
            } else if (isDateColumn(columnValues)) {
                // Store as categorical for now
                data.categoricalData[colName] = columnValues;
            } else if (isDateTimeColumn(columnValues)) {
                // Store as categorical for now
                data.categoricalData[colName] = columnValues;
            } else {
                data.categoricalData[colName] = columnValues;
            }

            // Store column type info
            if (data.numericData.find(colName) != data.numericData.end()) {
                data.columnTypes[colName] = "numeric";
            } else if (data.categoricalData.find(colName) != data.categoricalData.end()) {
                data.columnTypes[colName] = "categorical";
            } else if (data.integerData.find(colName) != data.integerData.end()) {
                data.columnTypes[colName] = "integer";
            } else if (data.booleanData.find(colName) != data.booleanData.end()) {
                data.columnTypes[colName] = "boolean";
            }
        }

        return data;
    }

    // Advanced data processing
    void Plotter::detectColumnTypes(PlotData& data) {
        // This is already done in convertToPlotData
    }

    void Plotter::cleanData(PlotData& data) {
        // Remove NaN values from numeric data
        for (auto& [colName, values] : data.numericData) {
            values.erase(std::remove_if(values.begin(), values.end(),
                        [](double v) { return std::isnan(v); }), values.end());
        }

        // Remove empty strings from categorical data
        for (auto& [colName, values] : data.categoricalData) {
            values.erase(std::remove_if(values.begin(), values.end(),
                        [](const std::string& v) { return v.empty(); }), values.end());
        }
    }

    void Plotter::normalizeData(PlotData& data) {
        for (auto& [colName, values] : data.numericData) {
            if (values.empty()) continue;

            // Find min and max
            double min_val = *std::min_element(values.begin(), values.end());
            double max_val = *std::max_element(values.begin(), values.end());

            if (max_val > min_val) {
                // Normalize to [0, 1]
                for (auto& v : values) {
                    v = (v - min_val) / (max_val - min_val);
                }
            }
        }
    }

    void Plotter::extractFeatures(PlotData& data) {
        // Extract statistical features from numeric data
        for (const auto& [colName, values] : data.numericData) {
            if (values.size() < 2) continue;

            // Calculate basic statistics
            double sum = std::accumulate(values.begin(), values.end(), 0.0);
            double mean = sum / values.size();

            double variance = 0.0;
            for (double v : values) {
                variance += (v - mean) * (v - mean);
            }
            variance /= values.size();
            double stddev = sqrt(variance);

            // Store as metadata (could be added to PlotData structure)
            // For now, we just calculate but don't store
        }
    }

    // Statistical calculations
    std::pair<std::vector<double>, std::vector<double>> Plotter::calculateKDE(const std::vector<double>& data, int gridPoints) {
        std::pair<std::vector<double>, std::vector<double>> result;

        if (data.empty()) return result;

        // Create grid
        double min_val = *std::min_element(data.begin(), data.end());
        double max_val = *std::max_element(data.begin(), data.end());
        double range = max_val - min_val;

        // Add padding
        min_val -= range * 0.1;
        max_val += range * 0.1;

        std::vector<double> grid = linspace(min_val, max_val, gridPoints);

        // Calculate bandwidth using Silverman's rule of thumb
        double n = static_cast<double>(data.size());
        double stddev = 0.0;
        double mean = std::accumulate(data.begin(), data.end(), 0.0) / n;
        for (double v : data) {
            stddev += (v - mean) * (v - mean);
        }
        stddev = sqrt(stddev / n);

        double bandwidth = 1.06 * stddev * pow(n, -0.2);
        if (bandwidth < 0.1) bandwidth = 0.1; // Minimum bandwidth

        // Calculate KDE
        std::vector<double> kde(grid.size(), 0.0);
        for (size_t i = 0; i < grid.size(); ++i) {
            double sum = 0.0;
            for (double v : data) {
                double u = (grid[i] - v) / bandwidth;
                sum += exp(-0.5 * u * u) / sqrt(2.0 * M_PI);
            }
            kde[i] = sum / (n * bandwidth);
        }

        result.first = grid;
        result.second = kde;
        return result;
    }

    std::pair<double, double> Plotter::linearRegression(const std::vector<double>& x,
                                                       const std::vector<double>& y) {
        if (x.size() != y.size() || x.empty()) {
            return {0.0, 0.0};
        }

        double n = static_cast<double>(x.size());
        double sum_x = std::accumulate(x.begin(), x.end(), 0.0);
        double sum_y = std::accumulate(y.begin(), y.end(), 0.0);
        double sum_xy = std::inner_product(x.begin(), x.end(), y.begin(), 0.0);
        double sum_x2 = std::inner_product(x.begin(), x.end(), x.begin(), 0.0);

        double slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        double intercept = (sum_y - slope * sum_x) / n;

        return {slope, intercept};
    }

    double Plotter::calculateRSquared(const std::vector<double>& x,
                                     const std::vector<double>& y,
                                     double slope, double intercept) {
        if (x.size() != y.size() || x.empty()) {
            return 0.0;
        }

        double n = static_cast<double>(x.size());
        double y_mean = std::accumulate(y.begin(), y.end(), 0.0) / n;

        double ss_tot = 0.0, ss_res = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
            double y_pred = slope * x[i] + intercept;
            ss_tot += (y[i] - y_mean) * (y[i] - y_mean);
            ss_res += (y[i] - y_pred) * (y[i] - y_pred);
        }

        if (ss_tot == 0.0) return 1.0;
        return 1.0 - (ss_res / ss_tot);
    }

    std::vector<std::vector<double>> Plotter::calculateCorrelationMatrix(const PlotData& data) {
        std::vector<std::vector<double>> corrMatrix;

        // Extract numeric columns
        std::vector<std::string> colNames;
        std::vector<std::vector<double>> numericColumns;

        for (const auto& [colName, values] : data.numericData) {
            colNames.push_back(colName);

            // Clean NaN values
            std::vector<double> cleanValues;
            for (double v : values) {
                if (!std::isnan(v)) {
                    cleanValues.push_back(v);
                }
            }
            numericColumns.push_back(cleanValues);
        }

        size_t n = numericColumns.size();
        corrMatrix.resize(n, std::vector<double>(n, 0.0));

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                if (i == j) {
                    corrMatrix[i][j] = 1.0;
                } else {
                    const auto& x = numericColumns[i];
                    const auto& y = numericColumns[j];

                    if (x.size() != y.size() || x.empty()) {
                        corrMatrix[i][j] = 0.0;
                        continue;
                    }

                    double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0;
                    double sum_x2 = 0.0, sum_y2 = 0.0;

                    for (size_t k = 0; k < x.size(); ++k) {
                        sum_x += x[k];
                        sum_y += y[k];
                        sum_xy += x[k] * y[k];
                        sum_x2 += x[k] * x[k];
                        sum_y2 += y[k] * y[k];
                    }

                    double n_val = static_cast<double>(x.size());
                    double numerator = n_val * sum_xy - sum_x * sum_y;
                    double denominator = sqrt((n_val * sum_x2 - sum_x * sum_x) *
                                             (n_val * sum_y2 - sum_y * sum_y));

                    if (denominator != 0.0) {
                        corrMatrix[i][j] = numerator / denominator;
                    } else {
                        corrMatrix[i][j] = 0.0;
                    }
                }
            }
        }

        return corrMatrix;
    }

    // Data transformations
    std::vector<double> Plotter::normalize(const std::vector<double>& data) {
        std::vector<double> result = data;

        if (result.empty()) return result;

        double min_val = *std::min_element(result.begin(), result.end());
        double max_val = *std::max_element(result.begin(), result.end());

        if (max_val > min_val) {
            for (auto& v : result) {
                v = (v - min_val) / (max_val - min_val);
            }
        }

        return result;
    }

    std::vector<double> Plotter::standardize(const std::vector<double>& data) {
        std::vector<double> result = data;

        if (result.empty()) return result;

        double n = static_cast<double>(result.size());
        double mean = std::accumulate(result.begin(), result.end(), 0.0) / n;

        double variance = 0.0;
        for (double v : result) {
            variance += (v - mean) * (v - mean);
        }
        variance /= n;
        double stddev = sqrt(variance);

        if (stddev > 0.0) {
            for (auto& v : result) {
                v = (v - mean) / stddev;
            }
        }

        return result;
    }

    std::vector<double> Plotter::logTransform(const std::vector<double>& data) {
        std::vector<double> result = data;

        for (auto& v : result) {
            if (v > 0.0) {
                v = log(v);
            } else {
                v = 0.0;
            }
        }

        return result;
    }

    // Helper functions for data type detection
    bool Plotter::isNumericColumn(const std::vector<std::string>& values) {
        if (values.empty()) return false;

        size_t numericCount = 0;
        for (const auto& val : values) {
            try {
                std::stod(val);
                numericCount++;
            } catch (...) {
                // Not numeric
            }
        }

        return (static_cast<double>(numericCount) / values.size()) > 0.8;
    }

    bool Plotter::isIntegerColumn(const std::vector<std::string>& values) {
        if (values.empty()) return false;

        size_t integerCount = 0;
        for (const auto& val : values) {
            try {
                std::stoi(val);
                integerCount++;
            } catch (...) {
                // Not integer
            }
        }

        return (static_cast<double>(integerCount) / values.size()) > 0.8;
    }

    bool Plotter::isBooleanColumn(const std::vector<std::string>& values) {
        if (values.empty()) return false;

        size_t booleanCount = 0;
        for (const auto& val : values) {
            std::string lowerVal = val;
            std::transform(lowerVal.begin(), lowerVal.end(), lowerVal.begin(), ::tolower);
            if (lowerVal == "true" || lowerVal == "false" ||
                lowerVal == "1" || lowerVal == "0" ||
                lowerVal == "yes" || lowerVal == "no" ||
                lowerVal == "on" || lowerVal == "off" ||
                lowerVal == "t" || lowerVal == "f") {
                booleanCount++;
            }
        }

        return (static_cast<double>(booleanCount) / values.size()) > 0.8;
    }

    bool Plotter::isDateColumn(const std::vector<std::string>& values) {
        if (values.empty()) return false;

        // Simple date pattern matching (YYYY-MM-DD)
        size_t dateCount = 0;
        std::regex datePattern(R"(^\d{4}-\d{2}-\d{2}$)");

        for (const auto& val : values) {
            if (std::regex_match(val, datePattern)) {
                dateCount++;
            }
        }

        return (static_cast<double>(dateCount) / values.size()) > 0.8;
    }

    bool Plotter::isDateTimeColumn(const std::vector<std::string>& values) {
        if (values.empty()) return false;

        // Simple datetime pattern matching (YYYY-MM-DD HH:MM:SS)
        size_t datetimeCount = 0;
        std::regex datetimePattern(R"(^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$)");

        for (const auto& val : values) {
            if (std::regex_match(val, datetimePattern)) {
                datetimeCount++;
            }
        }

        return (static_cast<double>(datetimeCount) / values.size()) > 0.8;
    }

    // Data conversion functions
    std::vector<double> Plotter::convertToNumeric(const std::vector<std::string>& values) {
        std::vector<double> numericValues;
        numericValues.reserve(values.size());

        for (const auto& val : values) {
            try {
                numericValues.push_back(std::stod(val));
            } catch (...) {
                numericValues.push_back(std::numeric_limits<double>::quiet_NaN());
            }
        }

        return numericValues;
    }

    std::vector<int> Plotter::convertToInteger(const std::vector<std::string>& values) {
        std::vector<int> intValues;
        intValues.reserve(values.size());

        for (const auto& val : values) {
            try {
                intValues.push_back(std::stoi(val));
            } catch (...) {
                intValues.push_back(0); // Default for invalid integers
            }
        }

        return intValues;
    }

    std::vector<bool> Plotter::convertToBoolean(const std::vector<std::string>& values) {
        std::vector<bool> boolValues;
        boolValues.reserve(values.size());

        for (const auto& val : values) {
            std::string lowerVal = val;
            std::transform(lowerVal.begin(), lowerVal.end(), lowerVal.begin(), ::tolower);

            if (lowerVal == "true" || lowerVal == "1" || lowerVal == "yes" || lowerVal == "on" || lowerVal == "t") {
                boolValues.push_back(true);
            } else {
                boolValues.push_back(false);
            }
        }

        return boolValues;
    }

    // Data validation
    void Plotter::validatePlotData(const PlotData& data, const PlotConfig& config) {
        if (data.rows.empty()) {
            throw std::runtime_error("No data to plot");
        }

        switch (config.type) {
            case PlotType::LINE:
            case PlotType::SCATTER:
                if (data.numericData.size() < 2) {
                    throw std::runtime_error("Need at least 2 numeric columns for line/scatter plot");
                }
                break;
            case PlotType::BAR:
                if (data.categoricalData.empty() || data.numericData.empty()) {
                    throw std::runtime_error("Need both categorical and numeric data for bar plot");
                }
                break;
            case PlotType::HISTOGRAM:
                if (data.numericData.empty()) {
                    throw std::runtime_error("Numeric data required for histogram");
                }
                break;
            case PlotType::BOXPLOT:
                if (data.numericData.empty()) {
                    throw std::runtime_error("Numeric data required for box plot");
                }
                break;
            case PlotType::PIE:
                if (data.categoricalData.empty() || data.numericData.empty()) {
                    throw std::runtime_error("Need both categorical and numeric data for pie chart");
                }
                break;
            case PlotType::HEATMAP:
                if (data.numericData.empty()) {
                    throw std::runtime_error("Numeric data required for heatmap");
                }
                break;
            case PlotType::STACKED_BAR:
            case PlotType::MULTI_LINE:
                if (data.numericData.size() < 2) {
                    throw std::runtime_error("Need at least 2 numeric columns for this plot type");
                }
                break;
            case PlotType::AREA:
                if (data.numericData.size() < 2) {
                    throw std::runtime_error("Need at least 2 numeric columns for area plot");
                }
                break;
            case PlotType::CONTOUR:
            case PlotType::SURFACE:
            case PlotType::WIREFRAME:
                if (data.numericData.size() < 3) {
                    throw std::runtime_error("Need at least 3 numeric columns for 3D plot");
                }
                break;
            case PlotType::HISTOGRAM_2D:
                if (data.numericData.size() < 2) {
                    throw std::runtime_error("Need at least 2 numeric columns for 2D histogram");
                }
                break;
            default:
                break;
        }
    }

    void Plotter::validateNumericData(const std::vector<double>& data, const std::string& columnName) {
        if (data.empty()) {
            throw std::runtime_error("Empty data for column: " + columnName);
        }

        size_t nanCount = 0;
        for (double v : data) {
            if (std::isnan(v)) nanCount++;
        }

        if (nanCount == data.size()) {
            throw std::runtime_error("All values are NaN for column: " + columnName);
        }
    }

    void Plotter::validateCategoricalData(const std::vector<std::string>& data, const std::string& columnName) {
        if (data.empty()) {
            throw std::runtime_error("Empty data for column: " + columnName);
        }

        size_t emptyCount = 0;
        for (const auto& v : data) {
            if (v.empty()) emptyCount++;
        }

        if (emptyCount == data.size()) {
            throw std::runtime_error("All values are empty for column: " + columnName);
        }
    }

    // Plot setup helpers
    void Plotter::setup3DAxes(const PlotConfig& config) {
        auto ax = plt::gca();

        if (config.style.view_init_set) {
            ax->view(config.style.elevation, config.style.azimuth);
        }

        // Set axis labels
        if (!config.xLabel.empty()) ax->xlabel(config.xLabel);
        if (!config.yLabel.empty()) ax->ylabel(config.yLabel);
        if (!config.zLabel.empty()) ax->zlabel(config.zLabel);

        // Set axis limits
        if (!std::isnan(config.style.xmin) && !std::isnan(config.style.xmax)) {
            ax->xlim({config.style.xmin, config.style.xmax});
        }
        if (!std::isnan(config.style.ymin) && !std::isnan(config.style.ymax)) {
            ax->ylim({config.style.ymin, config.style.ymax});
        }
        if (!std::isnan(config.style.zmin) && !std::isnan(config.style.zmax)) {
            ax->zlim({config.style.zmin, config.style.zmax});
        }
    }

    void Plotter::applyStyle(const PlotConfig& config) {
        // Apply style presets if specified
        if (!currentStyle.empty()) {
            setStyle(currentStyle);
        }

        if (!currentPalette.empty()) {
            setColorPalette(currentPalette);
        }
    }

    // Auto-plot based on data characteristics
    void Plotter::autoPlot(const PlotData& data, const std::string& title) {
        PlotConfig config;
        config.title = title.empty() ? "Auto-generated Plot" : title;
        config.style.grid = true;
        config.style.legend = true;

        // Analyze data characteristics
        size_t numericCols = data.numericData.size();
        size_t categoricalCols = data.categoricalData.size();
        size_t totalRows = data.rows.size();

        if (numericCols >= 3 && totalRows > 50) {
            // Good for heatmap
            config.type = PlotType::HEATMAP;
            plotHeatmap(data, config);
        } else if (numericCols >= 3 && totalRows <= 50) {
            // Good for stacked bar
            config.type = PlotType::STACKED_BAR;
            plotStackedBar(data, config);
        } else if (numericCols == 2) {
            // Good for scatter or line plot
            if (totalRows > 100) {
                config.type = PlotType::SCATTER;
                plotScatter(data, config);
            } else {
                config.type = PlotType::LINE;
                plotLine(data, config);
            }
        } else if (categoricalCols > 0 && numericCols > 0) {
            // Good for bar or pie plot
            if (data.categoricalData.begin()->second.size() <= 8) {
                config.type = PlotType::PIE;
                plotPie(data, config);
            } else {
                config.type = PlotType::BAR;
                plotBar(data, config);
            }
        } else if (numericCols == 1) {
            // Good for histogram or box plot
            if (totalRows > 30) {
                config.type = PlotType::HISTOGRAM;
                plotHistogram(data, config);
            } else {
                config.type = PlotType::BOXPLOT;
                plotBoxPlot(data, config);
            }
        } else {
            throw std::runtime_error("Cannot auto-determine plot type from data");
        }
    }

    void Plotter::smartPlot(const PlotData& data, const std::string& title) {
        // Enhanced auto-plot with AI-like recommendations
        PlotConfig config;
        config.title = title.empty() ? "Smart Plot" : title;

         // Create a copy of the data for analysis
         PlotData dataCopy = data;

        // Analyze data more thoroughly
        detectColumnTypes(dataCopy);
        cleanData(dataCopy);

        // Choose plot type based on comprehensive analysis
        size_t numericCols = data.numericData.size();
        size_t categoricalCols = data.categoricalData.size();
        size_t totalRows = data.rows.size();

        // Check for time series data
        bool hasDates = false;
        for (const auto& [colName, values] : data.categoricalData) {
            if (isDateColumn(values) || isDateTimeColumn(values)) {
                hasDates = true;
                config.type = PlotType::LINE; // Default to line for time series
                break;
            }
        }

        if (hasDates && numericCols >= 1) {
            // Time series plot
            std::string timeCol;
            std::string valueCol;

            // Find date/time column
            for (const auto& [colName, values] : data.categoricalData) {
                if (isDateColumn(values) || isDateTimeColumn(values)) {
                    timeCol = colName;
                    break;
                }
            }

            // Find first numeric column
            if (!data.numericData.empty()) {
                valueCol = data.numericData.begin()->first;
            }

            if (!timeCol.empty() && !valueCol.empty()) {
                plotTimeSeries(data, timeCol, valueCol, config);
                return;
            }
        }

        // Fall back to regular auto-plot
        autoPlot(data, title);
    }

    // Output control
    void Plotter::showPlot() {
        plt::show();
    }

    void Plotter::savePlot(const std::string& filename) {
        plt::save(filename);
    }

    void Plotter::savePlot(const std::string& filename, const std::string& format) {
        std::string fullFilename = filename;
        if (!format.empty()) {
            // Ensure filename has correct extension
            size_t dotPos = fullFilename.find_last_of('.');
            if (dotPos == std::string::npos) {
                fullFilename += "." + format;
            }
        }
        plt::save(fullFilename);
    }

    void Plotter::clearPlot() {
        plt::cla();
    }

    void Plotter::closeAll() {

    }

    void Plotter::setFont(const std::string& fontName, int size) {
        // Note: Matplot++ has limited font control
        // This is a placeholder for future implementation
    }

    void Plotter::addLegendIfNeeded(const PlotConfig& config) {
        if (config.style.legend && !config.seriesNames.empty()) {
            plt::legend(config.seriesNames);
        }
    }

    void Plotter::handlePlotOutput(const PlotConfig& config) {
        if (!config.outputFile.empty()) {
            // Determine format from filename extension
            std::string filename = config.outputFile;
            std::string format = config.style.save_format;

            // Override format based on extension if not specified
            if (format.empty()) {
                size_t dotPos = filename.find_last_of('.');
                if (dotPos != std::string::npos) {
                    format = filename.substr(dotPos + 1);
                } else {
                    format = "png";
                }
            }

            savePlot(filename, format);
        } else {
            showPlot();
        }
    }

    // Utility functions
    std::vector<double> Plotter::linspace(double start, double end, size_t num) {
        std::vector<double> result(num);
        if (num == 0) return result;
        if (num == 1) {
            result[0] = start;
            return result;
        }

        double step = (end - start) / (num - 1);
        for (size_t i = 0; i < num; ++i) {
            result[i] = start + i * step;
        }

        return result;
    }

    double Plotter::erfinv(double x) {
        // Inverse error function approximation
        double a = 0.147;
        double ln1x2 = log(1 - x * x);
        double term1 = 2 / (M_PI * a) + ln1x2 / 2;
        double term2 = ln1x2 / a;
        double result = sqrt(sqrt(term1 * term1 - term2) - term1);

        if (x < 0) result = -result;
        return result;
    }

    // Error handling and performance tracking
    void Plotter::logError(const std::string& error) {
        errorLog.push_back(error);
    }

    void Plotter::clearErrors() {
        errorLog.clear();
    }

    void Plotter::startTimer() {
        currentMetrics.startTime = std::chrono::steady_clock::now();
    }

    void Plotter::stopTimer() {
        auto endTime = std::chrono::steady_clock::now();
        currentMetrics.renderTime = std::chrono::duration<double>(endTime - currentMetrics.startTime).count();
    }

    void Plotter::printMetrics() const {
        std::cout << "Plotter Metrics:" << std::endl;
        std::cout << "  Data points processed: " << currentMetrics.dataPointsProcessed << std::endl;
        std::cout << "  Render time: " << currentMetrics.renderTime << " seconds" << std::endl;
        std::cout << "  Data conversion time: " << currentMetrics.dataConversionTime << " seconds" << std::endl;
    }

    // Create dashboard with multiple plots
    void Plotter::createDashboard(const std::vector<PlotData>& datasets,
                                 const std::vector<PlotConfig>& configs,
                                 int rows, int cols) {
        if (datasets.empty() || configs.empty()) {
            throw std::runtime_error("No data or configs provided for dashboard");
        }

        if (datasets.size() != configs.size()) {
            throw std::runtime_error("Number of datasets must match number of configs");
        }

        size_t numPlots = std::min(datasets.size(), configs.size());
        rows = std::min(rows, static_cast<int>(numPlots));
        cols = std::min(cols, static_cast<int>((numPlots + rows - 1) / rows));

        // Create tiled layout
        plt::tiledlayout(rows, cols);

        for (size_t i = 0; i < numPlots; ++i) {
            plt::nexttile();

            // Plot based on config type
            switch (configs[i].type) {
                case PlotType::LINE:
                    plotLine(datasets[i], configs[i]);
                    break;
                case PlotType::SCATTER:
                    plotScatter(datasets[i], configs[i]);
                    break;
                case PlotType::BAR:
                    plotBar(datasets[i], configs[i]);
                    break;
                case PlotType::HISTOGRAM:
                    plotHistogram(datasets[i], configs[i]);
                    break;
                case PlotType::BOXPLOT:
                    plotBoxPlot(datasets[i], configs[i]);
                    break;
                case PlotType::PIE:
                    plotPie(datasets[i], configs[i]);
                    break;
                case PlotType::HEATMAP:
                    plotHeatmap(datasets[i], configs[i]);
                    break;
                case PlotType::AREA:
                    plotArea(datasets[i], configs[i]);
                    break;
                case PlotType::STACKED_BAR:
                    plotStackedBar(datasets[i], configs[i]);
                    break;
                case PlotType::MULTI_LINE:
                    plotMultiLine(datasets[i], configs[i]);
                    break;
                default:
                    autoPlot(datasets[i], configs[i].title);
                    break;
            }
        }

        handlePlotOutput(PlotConfig());
    }

    // Add this method implementation for parseFromMap
void Visualization::PlotConfig::Style::parseFromMap(const std::map<std::string, std::string>& styleMap) {
    for (const auto& [key, value] : styleMap) {
        std::string lowerKey = key;
        std::transform(lowerKey.begin(), lowerKey.end(), lowerKey.begin(), ::tolower);

        if (lowerKey == "color") {
            color = value;
        } else if (lowerKey == "linewidth") {
            try { linewidth = std::stod(value); } catch (...) {}
        } else if (lowerKey == "linestyle") {
            linestyle = value;
        } else if (lowerKey == "marker") {
            marker = value;
        } else if (lowerKey == "markercolor") {
            markercolor = value;
        } else if (lowerKey == "markersize") {
            try { markersize = std::stod(value); } catch (...) {}
        } else if (lowerKey == "alpha") {
            try { alpha = std::stod(value); } catch (...) {}
        } else if (lowerKey == "grid") {
            grid = (value == "true" || value == "1" || value == "yes");
        } else if (lowerKey == "legend") {
            legend = (value == "true" || value == "1" || value == "yes");
        } else if (lowerKey == "figwidth") {
            try { figwidth = std::stod(value); } catch (...) {}
        } else if (lowerKey == "figheight") {
            try { figheight = std::stod(value); } catch (...) {}
        } else if (lowerKey == "bins") {
            try { bins = std::stoi(value); } catch (...) {}
        } else if (lowerKey == "xmin") {
            try { xmin = std::stod(value); } catch (...) {}
        } else if (lowerKey == "xmax") {
            try { xmax = std::stod(value); } catch (...) {}
        } else if (lowerKey == "ymin") {
            try { ymin = std::stod(value); } catch (...) {}
        } else if (lowerKey == "ymax") {
            try { ymax = std::stod(value); } catch (...) {}
        } else if (lowerKey == "title_fontsize") {
            try { title_fontsize = std::stod(value); } catch (...) {}
        } else if (lowerKey == "facecolor") {
            facecolor = value;
        } else if (lowerKey == "edgecolor") {
            edgecolor = value;
        } else if (lowerKey == "barwidth") {
            try { barwidth = std::stod(value); } catch (...) {}
        } else if (lowerKey == "xtick_rotation") {
            try { xtick_rotation = std::stod(value); } catch (...) {}
        } else if (lowerKey == "ytick_rotation") {
            try { ytick_rotation = std::stod(value); } catch (...) {}
        } else if (lowerKey == "colormap") {
            colormap = value;
        } else if (lowerKey == "annotate") {
            annotate = (value == "true" || value == "1" || value == "yes");
        } else if (lowerKey == "fmt") {
            fmt = value;
        } else if (lowerKey == "cumulative") {
            cumulative = (value == "true" || value == "1" || value == "yes");
        } else if (lowerKey == "density") {
            density = (value == "true" || value == "1" || value == "yes");
        } else if (lowerKey == "showfliers") {
            showfliers = (value == "true" || value == "1" || value == "yes");
        } else if (lowerKey == "show_kde") {
            show_kde = (value == "true" || value == "1" || value == "yes");
        } else if (lowerKey == "rug") {
            rug = (value == "true" || value == "1" || value == "yes");
        } else if (lowerKey == "interactive") {
            interactive = (value == "true" || value == "1" || value == "yes");
        } else if (lowerKey == "save_format") {
            save_format = value;
        } else if (lowerKey == "colormap") {
	    colormap = value;
	}
        // Add more style parsing as needed
    }
}

void Plotter::createInteractivePlot(const PlotData& data, const PlotConfig& config) {
    // Note: Matplot++ has limited interactive support
    // This is a basic implementation that creates a plot with some interactive features

    validatePlotData(data, config);
    setupFigure(config);

    // For now, we'll just create a regular scatter plot
    // In a real implementation, you would use a library with better interactive support
    if (!data.numericData.empty()) {
        if (data.numericData.size() >= 2) {
            auto itX = data.numericData.begin();
            auto itY = std::next(data.numericData.begin());

            std::vector<double> xData = itX->second;
            std::vector<double> yData = itY->second;

            std::vector<double> cleanX, cleanY;
            for (size_t i = 0; i < std::min(xData.size(), yData.size()); ++i) {
                if (!std::isnan(xData[i]) && !std::isnan(yData[i])) {
                    cleanX.push_back(xData[i]);
                    cleanY.push_back(yData[i]);
                }
            }

            if (!cleanX.empty() && !cleanY.empty()) {
                auto s = plt::scatter(cleanX, cleanY);
                s->marker_size(config.style.markersize);

                std::string marker = parseMarker(config.style.marker);
                if (!marker.empty() && marker != "none") {
                    s->marker_style(marker);
                }

                auto markerColor = parseColor(config.style.markercolor);
                markerColor[3] = config.style.alpha;
                s->marker_color(markerColor);

                if (config.style.markerfacecolor != "none") {
                    auto markerFaceColor = parseColor(config.style.markerfacecolor);
                    s->marker_face_color(markerFaceColor);
                }
            }
        }
    }

    // Add title
    if (!config.title.empty()) {
        plt::title(config.title);
    }

    // Add labels
    if (!config.xLabel.empty()) plt::xlabel(config.xLabel);
    if (!config.yLabel.empty()) plt::ylabel(config.yLabel);

    // Add grid
    if (config.style.grid) plt::grid(true);

    // For now, just show the plot
    // In a more advanced implementation, you would create widgets and callbacks
    handlePlotOutput(config);

    // Log that interactive features are limited
    logError("Note: Interactive features are limited in Matplot++ implementation");
}

void Plotter::addWidgets(const PlotConfig& config) {
    // Note: Matplot++ doesn't have built-in widget support
    // This is a placeholder for future implementation
    logError("Widget support not implemented in Matplot++ backend");
}


    // Animation support
    void Plotter::createAnimation(const std::vector<PlotData>& frames,
                                 const PlotConfig& config, int fps) {
        if (frames.empty()) {
            throw std::runtime_error("No frames provided for animation");
        }

        // Note: Matplot++ doesn't have built-in animation support
        // This is a placeholder for future implementation
        // For now, we just plot the first frame

        if (!frames.empty()) {
            autoPlot(frames[0], config.title);
        }
    }

} // namespace Visualization
