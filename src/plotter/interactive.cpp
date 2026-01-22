#include "plotter_includes/plotter.h"
#include <matplot/matplot.h>

namespace plt = matplot;

namespace Visualization {

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

} // namespace Visualization
