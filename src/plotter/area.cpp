#include "plotter_includes/plotter.h"
#include <matplot/matplot.h>

namespace plt = matplot;

namespace Visualization {

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
        line->line_style(parseLineStyle(config.style.linestyle));
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
        fillColor[3] = config.style.alpha * 0.3f;
        fill_area->color(fillColor);

        plt::hold(false);

        // Add legend if needed
        addLegendIfNeeded(config);

        handlePlotOutput(config);
    }

} // namespace Visualization
