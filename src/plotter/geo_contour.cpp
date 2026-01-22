#include "plotter_includes/plotter.h"
#include <matplot/matplot.h>

namespace plt = matplot;
using json = nlohmann::json;

namespace Visualization {
    void Plotter::plotGeoContour(const PlotData& data, const PlotConfig& config) {
        validatePlotData(data, config);
        validateGeoData(data, config);
        setupGeoAxes(config);

        // Add basemap
        addBasemap(config);

        // Get latitude, longitude, and value data
        auto latIt = data.latitudeData.begin();
        auto lonIt = data.longitudeData.begin();

        if (latIt == data.latitudeData.end() || lonIt == data.longitudeData.end()) {
            throw std::runtime_error("No latitude/longitude data found for geographic contour plot");
        }

        std::vector<double> lats = latIt->second;
        std::vector<double> lons = lonIt->second;

        // Get value data
        std::vector<double> values;
        if (!data.numericData.empty()) {
            auto valIt = data.numericData.begin();
            values = valIt->second;
        } else {
            throw std::runtime_error("Contour plot requires value data");
        }

        // Clean data
        std::vector<double> clean_lats, clean_lons, clean_values;
        for (size_t i = 0; i < lats.size(); ++i) {
            if (!std::isnan(lats[i]) && !std::isnan(lons[i]) &&
                i < values.size() && !std::isnan(values[i])) {
                clean_lats.push_back(lats[i]);
                clean_lons.push_back(lons[i]);
                clean_values.push_back(values[i]);
            }
        }

        if (clean_lats.empty() || clean_lons.empty()) {
            throw std::runtime_error("No valid data points for contour plot");
        }

        // Create grid for contour plot
        int grid_size = 30;

        // Find bounds
        double min_lat = *std::min_element(clean_lats.begin(), clean_lats.end());
        double max_lat = *std::max_element(clean_lats.begin(), clean_lats.end());
        double min_lon = *std::min_element(clean_lons.begin(), clean_lons.end());
        double max_lon = *std::max_element(clean_lons.begin(), clean_lons.end());

        // Create grid points
        std::vector<double> lat_grid = linspace(min_lat, max_lat, grid_size);
        std::vector<double> lon_grid = linspace(min_lon, max_lon, grid_size);

        // Create 2D grid coordinates
        std::vector<std::vector<double>> X(grid_size, std::vector<double>(grid_size));
        std::vector<std::vector<double>> Y(grid_size, std::vector<double>(grid_size));
        std::vector<std::vector<double>> Z(grid_size, std::vector<double>(grid_size, 0.0));

        // Initialize X and Y grids
        for (int i = 0; i < grid_size; ++i) {
            for (int j = 0; j < grid_size; ++j) {
                X[i][j] = lon_grid[j];  // Note: j for longitude (X axis)
                Y[i][j] = lat_grid[i];  // Note: i for latitude (Y axis)
            }
        }

        // Interpolate values onto grid (simplified nearest neighbor)
        for (int i = 0; i < grid_size; ++i) {
            for (int j = 0; j < grid_size; ++j) {
                // Find nearest data point
                double min_dist = std::numeric_limits<double>::max();
                double nearest_value = 0.0;

                for (size_t k = 0; k < clean_lats.size(); ++k) {
                    double dist = std::pow(clean_lats[k] - lat_grid[i], 2) +
                                 std::pow(clean_lons[k] - lon_grid[j], 2);

                    if (dist < min_dist) {
                        min_dist = dist;
                        nearest_value = clean_values[k];
                    }
                }

                Z[i][j] = nearest_value;
            }
        }

        // Create contour plot using the correct signature
        auto c = plt::contour(X, Y, Z);

        // Apply styling if needed
        if (!config.style.color.empty()) {
            c->font_color(config.style.color);
        }
        
        if (config.style.linewidth > 0) {
            c->line_width(config.style.linewidth);
        }

        // Add colorbar
        plt::colorbar();
        
        // Set map bounds
        auto ax = plt::gca();
        ax->xlim({min_lon, max_lon});
        ax->ylim({min_lat, max_lat});
        
        // Add scale bar and north arrow
        addScaleBar(config);
        addNorthArrow(config);
        
        handlePlotOutput(config);
    }
}
