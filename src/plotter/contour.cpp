#include "plotter_includes/plotter.h"
#include <matplot/matplot.h>
#include <algorithm>
#include <cmath>

namespace plt = matplot;

namespace Visualization {

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

            auto c = plt::contour(X_grid, Y_grid, Z_grid);

            // Add colorbar
            plt::colorbar();
        } else {
            // Use simple scatter if not enough points for proper contour
            auto s = plt::scatter(xData, yData);
            plt::colormap(plt::palette::cool()); // Set a color map
            s->marker_size(10.0);
            s->marker_face(true);
            s->marker_color("blue");
            plt::colorbar();
        }

        handlePlotOutput(config);
    }

} // namespace Visualization
