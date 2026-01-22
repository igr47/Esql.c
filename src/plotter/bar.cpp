#include "plotter_includes/plotter.h"
#include <matplot/matplot.h>

namespace plt = matplot;

namespace Visualization {

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

        auto edgeColor = parseColor(config.style.edgecolor);
        edgeColor[3] = config.style.alpha;
        b->edge_color(edgeColor);

        b->line_width(config.style.linewidth);

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

} // namespace Visualization
