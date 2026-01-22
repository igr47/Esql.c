#include "plotter_includes/plotter.h"
#include <matplot/matplot.h>
#include <algorithm>
#include <cmath>

namespace plt = matplot;

namespace Visualization {

    void Plotter::plotSurface(const PlotData& data, const PlotConfig& config) {
        // 3D surface plot implementation
        validatePlotData(data, config);

        if (data.numericData.size() < 3) {
            throw std::runtime_error("Need at least 3 numeric columns for surface plot");
        }

        auto xIt = data.numericData.begin();
        auto yIt = std::next(xIt);
        auto zIt = std::next(yIt);

        std::vector<double> xData = xIt->second;
        std::vector<double> yData = yIt->second;
        std::vector<double> zData = zIt->second;

        // Create 3D surface plot
        plt::figure(true);
        auto ax = plt::gca();

        size_t nx = std::sqrt(zData.size());
        size_t ny = nx;

        if (nx * ny != zData.size()) {
            throw std::runtime_error("Z data must form a perfect square grid for surface plot");
        }

        std::vector<std::vector<double>> Z(nx, std::vector<double>(ny));
        for (size_t i = 0; i < nx; ++i) {
            for (size_t j = 0; j < ny; ++j) {
                Z[i][j] = zData[i * ny + j];
            }
        }

        std::vector<double> X = linspace(*std::min_element(xData.begin(), xData.end()),
                                        *std::max_element(xData.begin(), xData.end()), nx);
        std::vector<double> Y = linspace(*std::min_element(yData.begin(), yData.end()),
                                        *std::max_element(yData.begin(), yData.end()), ny);


        // Create meshgrid for X and Y (2D grids)
        std::vector<std::vector<double>> X_grid, Y_grid;
        for (size_t i = 0; i < Y.size(); ++i) {
            std::vector<double> row_x, row_y;
            for (size_t j = 0; j < X.size(); ++j) {
                row_x.push_back(X[j]);
                row_y.push_back(Y[i]);
            }
            X_grid.push_back(row_x);
            Y_grid.push_back(row_y);
        }
        // Now call surf with 2D grids
        auto s = plt::surf(X_grid, Y_grid, Z);

        // Set 3D view
        if (config.style.view_init_set) {
            ax->view(config.style.elevation, config.style.azimuth);
        }

        // Add labels
        if (!config.xLabel.empty()) ax->xlabel(config.xLabel);
        if (!config.yLabel.empty()) ax->ylabel(config.yLabel);
        if (!config.zLabel.empty()) ax->zlabel(config.zLabel);

        // Add colorbar
        plt::colorbar();

        handlePlotOutput(config);
    }

} // namespace Visualization
