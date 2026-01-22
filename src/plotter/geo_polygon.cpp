#include "plotter_includes/plotter.h"
#include <matplot/matplot.h>

namespace plt = matplot;
using json = nlohmann::json;

namespace Visualization {
    void Plotter::plotGeoPolygon(const PlotData& data, const PlotConfig& config) {
        validatePlotData(data, config);
        validateGeoData(data, config);
        setupGeoAxes(config);

        // Add basemap
        addBasemap(config);

        // For polygon plot, we need sequences of points that form closed shapes
        // This implementation assumes data is organized with polygon IDs

        // Get latitude and longitude data
        auto latIt = data.latitudeData.begin();
        auto lonIt = data.longitudeData.begin();

        if (latIt == data.latitudeData.end() || lonIt == data.longitudeData.end()) {
            throw std::runtime_error("No latitude/longitude data found for polygon plot");
        }

        std::vector<double> lats = latIt->second;
        std::vector<double> lons = lonIt->second;

        // Get polygon IDs if available
        std::vector<int> polygon_ids;
        
        // Use find() instead of operator[] for const maps
        auto poly_id_it = data.integerData.find("polygon_id");
        if (poly_id_it != data.integerData.end()) {
            polygon_ids = poly_id_it->second;
        } else {
            // Also check for alternative column names
            for (const auto& [col_name, col_data] : data.integerData) {
                if (col_name.find("polygon") != std::string::npos || 
                    col_name.find("id") != std::string::npos ||
                    col_name.find("group") != std::string::npos) {
                    polygon_ids = col_data;
                    break;
                }
            }
            
            // If still not found, assume all points belong to one polygon
            if (polygon_ids.empty()) {
                polygon_ids = std::vector<int>(lats.size(), 1);
            }
        }

        // Group points by polygon ID
        std::map<int, std::vector<std::pair<double, double>>> polygons;
        for (size_t i = 0; i < lats.size(); ++i) {
            if (!std::isnan(lats[i]) && !std::isnan(lons[i]) && i < polygon_ids.size()) {
                int poly_id = polygon_ids[i];
                polygons[poly_id].emplace_back(lons[i], lats[i]);
            }
        }

        if (polygons.empty()) {
            throw std::runtime_error("No valid polygon data");
        }

        // Draw each polygon
        auto colors = getColorPalette(static_cast<int>(polygons.size()));
        int color_idx = 0;

        for (const auto& [poly_id, points] : polygons) {
            if (points.size() < 3) continue; // Need at least 3 points for a polygon

            // Extract coordinates
            std::vector<double> poly_lons, poly_lats;
            for (const auto& [lon, lat] : points) {
                poly_lons.push_back(lon);
                poly_lats.push_back(lat);
            }

            // Close the polygon if not already closed
            if (poly_lons.front() != poly_lons.back() || poly_lats.front() != poly_lats.back()) {
                poly_lons.push_back(poly_lons.front());
                poly_lats.push_back(poly_lats.front());
            }

            // Fill the polygon
            auto fill = plt::fill(poly_lons, poly_lats);
            auto fillColor = parseColor(colors[color_idx % colors.size()]);
            fillColor[3] = config.style.alpha * 0.5;
            fill->color(fillColor);

            // Draw polygon border
            auto border = plt::plot(poly_lons, poly_lats);
            border->color(parseColor(config.style.edgecolor));
            border->line_width(config.style.linewidth);

            color_idx++;
        }

        // Set map bounds based on all points
        auto bounds = calculateMapBounds(data);
        auto ax = plt::gca();
        ax->xlim({bounds.first, bounds.second});

        // Calculate latitude bounds
        double min_lat = *std::min_element(lats.begin(), lats.end());
        double max_lat = *std::max_element(lats.begin(), lats.end());
        double lat_padding = (max_lat - min_lat) * 0.1;
        ax->ylim({min_lat - lat_padding, max_lat + lat_padding});

        // Add scale bar and north arrow
        addScaleBar(config);
        addNorthArrow(config);

        handlePlotOutput(config);
    }
}
