#include "plotter_includes/plotter.h"
#include <matplot/matplot.h>

namespace plt = matplot;

namespace Visualization {

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

        // Apply colors if specified
        if (!config.style.colors.empty()) {
            // Note: Matplot++ doesn't have direct per-series color control for stacked bars
            // We can set overall color
            auto barColor = parseColor(config.style.color);
            barColor[3] = config.style.alpha; // Alpha in color
            b->face_color(barColor);
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

} // namespace Visualization
