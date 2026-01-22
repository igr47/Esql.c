#include "plotter_includes/plotter.h"
#include <matplot/matplot.h>

namespace plt = matplot;

namespace Visualization {

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

} // namespace Visualization
