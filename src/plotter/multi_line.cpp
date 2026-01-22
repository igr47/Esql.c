#include "plotter_includes/plotter.h"
#include <matplot/matplot.h>

namespace plt = matplot;

namespace Visualization {

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

} // namespace Visualization
