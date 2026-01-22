#include "plotter_includes/plotter.h"
#include <matplot/matplot.h>
#include <algorithm>
#include <cmath>

namespace plt = matplot;

namespace Visualization {

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

} // namespace Visualization
