#include "plotter_includes/plotter.h"
#include <matplot/matplot.h>
#include <sstream>
#include <iomanip>
#include <algorithm>

namespace plt = matplot;

namespace Visualization {

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

} // namespace Visualization
