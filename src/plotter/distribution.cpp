#include "plotter_includes/plotter.h"
#include <matplot/matplot.h>

namespace plt = matplot;

namespace Visualization {

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

} // namespace Visualization
