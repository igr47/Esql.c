#include "plotter_includes/plotter.h"
#include <matplot/matplot.h>

namespace plt = matplot;
using json = nlohmann::json;


namespace Visualization {

	    void Plotter::plotGeoLine(const PlotData& data, const PlotConfig& config) {
        validatePlotData(data, config);
        validateGeoData(data, config);
        setupGeoAxes(config);

        // Add basemap
        addBasemap(config);

        // Get latitude and longitude data
        auto latIt = data.latitudeData.begin();
        auto lonIt = data.longitudeData.begin();

        if (latIt == data.latitudeData.end() || lonIt == data.longitudeData.end()) {
            throw std::runtime_error("No latitude/longitude data found for geographic line plot");
        }

        std::vector<double> lats = latIt->second;
        std::vector<double> lons = lonIt->second;
                // Clean data
        std::vector<double> clean_lats, clean_lons;
        for (size_t i = 0; i < lats.size(); ++i) {
            if (!std::isnan(lats[i]) && !std::isnan(lons[i])) {
                clean_lats.push_back(lats[i]);
                clean_lons.push_back(lons[i]);
            }
        }

        if (clean_lats.empty() || clean_lons.empty()) {
            throw std::runtime_error("No valid geographic data points for line plot");
        }

        // Create line plot
        auto p = plt::plot(clean_lons, clean_lats);

        // Apply styling
        auto lineColor = parseColor(config.style.color);
        lineColor[3] = config.style.alpha;
        p->color(lineColor);
        p->line_width(config.style.linewidth);
        p->line_style(parseLineStyle(config.style.linestyle));

	        // Add markers if specified
        std::string marker = parseMarker(config.style.marker);
        if (!marker.empty() && marker != "none") {
            p->marker_style(marker);
            p->marker_size(config.style.markersize);
        }

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

        // Add legend if needed
        addLegendIfNeeded(config);

	       handlePlotOutput(config);
    }
}

