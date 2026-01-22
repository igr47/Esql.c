#include "plotter_includes/plotter.h"
#include <matplot/matplot.h>

namespace plt = matplot;
using json = nlohmann::json;

namespace Visualization {
    void Plotter::plotGeoFlow(const PlotData& data, const PlotConfig& config) {
        validatePlotData(data, config);
        validateGeoData(data, config);
        setupGeoAxes(config);

        // Add basemap
        addBasemap(config);

        // For flow plot, we need starting points and direction vectors
        auto latIt = data.latitudeData.begin();
        auto lonIt = data.longitudeData.begin();

        if (latIt == data.latitudeData.end() || lonIt == data.longitudeData.end()) {
            throw std::runtime_error("No latitude/longitude data found for flow plot");
        }

        std::vector<double> lats = latIt->second;
        std::vector<double> lons = lonIt->second;

        // Get direction data (u = east-west, v = north-south)
        std::vector<double> u_component, v_component;
        
        // Use find() instead of operator[] for const maps
        auto u_it = data.numericData.find("u");
        auto v_it = data.numericData.find("v");
        
        if (u_it != data.numericData.end()) {
            u_component = u_it->second;
        }
        if (v_it != data.numericData.end()) {
            v_component = v_it->second;
        }

        // Also check for alternative column names
        if (u_component.empty()) {
            // Try to find any column that might contain u-component data
            for (const auto& [col_name, col_data] : data.numericData) {
                if (col_name.find("u") != std::string::npos || 
                    col_name.find("east") != std::string::npos ||
                    col_name.find("x") != std::string::npos) {
                    u_component = col_data;
                    break;
                }
            }
        }
        
        if (v_component.empty()) {
            // Try to find any column that might contain v-component data
            for (const auto& [col_name, col_data] : data.numericData) {
                if (col_name.find("v") != std::string::npos || 
                    col_name.find("north") != std::string::npos ||
                    col_name.find("y") != std::string::npos) {
                    v_component = col_data;
                    break;
                }
            }
        }

        if (u_component.empty() || v_component.empty()) {
            throw std::runtime_error("Flow plot requires u and v component data. "
                                   "Columns should contain 'u'/'east'/'x' and 'v'/'north'/'y' in their names.");
        }

        // Clean data
        std::vector<double> clean_lats, clean_lons, clean_u, clean_v;
        for (size_t i = 0; i < lats.size(); ++i) {
            if (!std::isnan(lats[i]) && !std::isnan(lons[i]) &&
                i < u_component.size() && !std::isnan(u_component[i]) &&
                i < v_component.size() && !std::isnan(v_component[i])) {
                clean_lats.push_back(lats[i]);
                clean_lons.push_back(lons[i]);
                clean_u.push_back(u_component[i]);
                clean_v.push_back(v_component[i]);
            }
        }

        if (clean_lats.empty()) {
            throw std::runtime_error("No valid flow data");
        }

        // Create quiver plot (flow arrows)
        auto q = plt::quiver(clean_lons, clean_lats, clean_u, clean_v);

        // Apply styling
        q->color(parseColor(config.style.color));
        q->line_width(config.style.linewidth);

        // Set map bounds
        auto bounds = calculateMapBounds(data);
        auto ax = plt::gca();
        ax->xlim({bounds.first, bounds.second});

        // Calculate latitude bounds
        double min_lat = *std::min_element(clean_lats.begin(), clean_lats.end());
        double max_lat = *std::max_element(clean_lats.begin(), clean_lats.end());
        double lat_padding = (max_lat - min_lat) * 0.1;
        ax->ylim({min_lat - lat_padding, max_lat + lat_padding});

        // Add scale bar and north arrow
        addScaleBar(config);
        addNorthArrow(config);

        handlePlotOutput(config);
    }
}
