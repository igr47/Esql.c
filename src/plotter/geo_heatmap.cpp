#include "plotter_includes/plotter.h"
#include <matplot/matplot.h>

namespace plt = matplot;
using json = nlohmann::json;

namespace Visualization {
	void Plotter::plotGeoHeatmap(const PlotData& data, const PlotConfig& config) {
		validatePlotData(data, config);
		validateGeoData(data, config);
                setupGeoAxes(config);

                // Add basemap
                addBasemap(config);

		// Get latitude, longitude, and value data
                auto latIt = data.latitudeData.begin();
                auto lonIt = data.longitudeData.begin();

                if (latIt == data.latitudeData.end() || lonIt == data.longitudeData.end()) {
			throw std::runtime_error("No latitude/longitude data found for geographic heatmap");
                }

		std::vector<double> lats = latIt->second;
		std::vector<double> lons = lonIt->second;

		// Get value data if available
		std::vector<double> values;
                if (!data.numericData.empty()) {
			auto valIt = data.numericData.begin();
			values = valIt->second;
		} else {
			// If no value data, use density
			values = std::vector<double>(lats.size(), 1.0);
		}

		// Clean data
		std::vector<double> clean_lats, clean_lons, clean_values;
		for (size_t i = 0; i < lats.size(); ++i) {
			if (!std::isnan(lats[i]) && !std::isnan(lons[i])) {
				clean_lats.push_back(lats[i]);
				clean_lons.push_back(lons[i]);
				if (i < values.size() && !std::isnan(values[i])) {
					clean_values.push_back(values[i]);
				} else {
					clean_values.push_back(1.0);
				}
			}
		}

		if (clean_lats.empty() || clean_lons.empty()) {
			throw std::runtime_error("No valid geographic data points for heatmap");
                }

                // Create grid for heatmap
                int grid_size = 50;
                std::vector<std::vector<double>> heat_grid(grid_size, std::vector<double>(grid_size, 0.0));
                std::vector<double> counts(grid_size * grid_size, 0.0);

                // Find bounds
                double min_lat = *std::min_element(clean_lats.begin(), clean_lats.end());
                double max_lat = *std::max_element(clean_lats.begin(), clean_lats.end());
                double min_lon = *std::min_element(clean_lons.begin(), clean_lons.end());
                double max_lon = *std::max_element(clean_lons.begin(), clean_lons.end());

                // Bin data into grid
                for (size_t i = 0; i < clean_lats.size(); ++i) {
			int lat_idx = static_cast<int>((clean_lats[i] - min_lat) / (max_lat - min_lat) * (grid_size - 1));
	                int lon_idx = static_cast<int>((clean_lons[i] - min_lon) / (max_lon - min_lon) * (grid_size - 1));

                        lat_idx = std::max(0, std::min(grid_size - 1, lat_idx));
                        lon_idx = std::max(0, std::min(grid_size - 1, lon_idx));

                        heat_grid[lat_idx][lon_idx] += clean_values[i];
                        counts[lat_idx * grid_size + lon_idx] += 1.0;
		}

                // Average the values
                for (int i = 0; i < grid_size; ++i) {
			for (int j = 0; j < grid_size; ++j) {
				if (counts[i * grid_size + j] > 0) {
					heat_grid[i][j] /= counts[i * grid_size + j];
			        }
			}
		}

		// Create heatmap
		auto h = plt::heatmap(heat_grid);
		
		// Set colormap
		if (!config.style.colormap.empty()) {
			std::string cmap = config.style.colormap;
                        std::transform(cmap.begin(), cmap.end(), cmap.begin(), ::tolower);
            
			if (cmap == "terrain") {
				// Custom terrain colormap
				auto colors = getTerrainPalette(256);
                                // Note: Matplot++ doesn't have direct custom colormap support
                                // We'll use a built-in one as fallback
                                plt::colormap(plt::palette::jet());
			} else if (cmap == "topographic") {
                                plt::colormap(plt::palette::hot());
                        } else if (cmap == "ocean") {
                                plt::colormap(plt::palette::cool());
                        } else {
				// Use standard colormap parsing
                               // ... [existing colormap code
			}
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
