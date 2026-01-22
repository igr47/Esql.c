#include "plotter_includes/plotter.h"
#include <matplot/matplot.h>

namespace plt = matplot;
using json = nlohmann::json;

namespace Visualization {
	    void Plotter::plotGeoChoropleth(const PlotData& data, const PlotConfig& config) {
        validatePlotData(data, config);
        validateGeoData(data, config);
        setupGeoAxes(config);

        // Add basemap
        addBasemap(config);

        // Check for region data
        if (data.regionData.empty()) {
            throw std::runtime_error("Choropleth plot requires region data");
        }

        auto regionIt = data.regionData.begin();
        std::vector<std::string> regions = regionIt->second;

        // Get value data
        std::vector<double> values;
        if (!data.numericData.empty()) {
            auto valIt = data.numericData.begin();
            values = valIt->second;
        } else {
            throw std::runtime_error("Choropleth plot requires value data");
        }

	        // Match regions with values
        std::map<std::string, double> region_values;
        for (size_t i = 0; i < std::min(regions.size(), values.size()); ++i) {
            if (!regions[i].empty() && !std::isnan(values[i])) {
                region_values[regions[i]] = values[i];
            }
        }

        if (region_values.empty()) {
            throw std::runtime_error("No valid region-value pairs for choropleth");
        }

        // Get color palette
        auto colors = getColorPalette(static_cast<int>(region_values.size()));

        // For each region, draw its boundary (simplified)
        // Note: In a real implementation, you would load shapefiles
        size_t color_idx = 0;
        for (const auto& [region, value] : region_values) {
		            // Simplified region boundaries (in practice, load from shapefile)
            std::vector<double> region_lons, region_lats;

            // This is a simplified example - real implementation would load actual boundaries
            if (region == "USA" || region == "United States") {
                region_lons = {-125.0, -66.0, -66.0, -125.0, -125.0};
                region_lats = {25.0, 25.0, 49.0, 49.0, 25.0};
            } else if (region == "Canada") {
                region_lons = {-141.0, -52.0, -52.0, -141.0, -141.0};
                region_lats = {42.0, 42.0, 83.0, 83.0, 42.0};
            } else if (region == "Mexico") {
                region_lons = {-118.0, -86.0, -86.0, -118.0, -118.0};
                region_lats = {15.0, 15.0, 32.0, 32.0, 15.0};
            } else {
                // Skip regions we don't have boundaries for
                continue;
            }

            // Fill the region
            auto fill = plt::fill(region_lons, region_lats);
            auto fillColor = parseColor(colors[color_idx % colors.size()]);
            fillColor[3] = config.style.alpha * 0.7;
	               fill->color(fillColor);

            // Add border
            auto border = plt::plot(region_lons, region_lats);
            border->color(parseColor(config.style.edgecolor));
            border->line_width(config.style.linewidth);

            // Add region label at center
            double center_lon = std::accumulate(region_lons.begin(), region_lons.end(), 0.0) / region_lons.size();
            double center_lat = std::accumulate(region_lats.begin(), region_lats.end(), 0.0) / region_lats.size();

            std::stringstream label;
            label << region << "\n" << PlotUtils::formatNumber(value, 2);
            plt::text(center_lon, center_lat, label.str());

            color_idx++;
        }

        // Add colorbar
        plt::colorbar();
               // Add scale bar and north arrow
        addScaleBar(config);
        addNorthArrow(config);

        handlePlotOutput(config);
    }
}

