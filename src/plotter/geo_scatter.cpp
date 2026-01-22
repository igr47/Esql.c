#include "plotter_includes/plotter.h"
#include <matplot/matplot.h>

namespace plt = matplot;
using json = nlohmann::json;

namespace Visualization {
	void Plotter::plotGeoScatter(const PlotData& data, const PlotConfig& config) {
		validatePlotData(data, config);
		validateGeoData(data, config);
		setupGeoAxes(config);

		// Add basemap
		addBasemap(config);

		// Get latitude and longitude data
		auto latIt = data.latitudeData.begin();
		auto lonIt = data.longitudeData.begin();

		if (latIt == data.latitudeData.end() || lonIt == data.longitudeData.end()) {
			throw std::runtime_error("No latitude/longitude data found for geographic scatter plot");
		}

		std::vector<double> lats = latIt->second;
		std::vector<double> lons = lonIt->second;

		// Clean data (remove NaN values)
		std::vector<double> clean_lats, clean_lons;
                for (size_t i = 0; i < lats.size(); ++i) {
			if (!std::isnan(lats[i]) && !std::isnan(lons[i])) {
				clean_lats.push_back(lats[i]);
				clean_lons.push_back(lons[i]);
			}
		}

		if (clean_lats.empty() || clean_lons.empty()) {
			throw std::runtime_error("No valid geographic data points for scatter plot");
		}

		// Create scatter plot
		auto s = plt::scatter(clean_lons, clean_lats);

		// Apply styling
		std::string marker = parseMarker(config.style.marker);
		if (!marker.empty() && marker != "none") {
			s->marker_style(marker);
                        s->marker_size(config.style.markersize);

			auto markerColor = parseColor(config.style.markercolor);
                        markerColor[3] = config.style.alpha;
			s->marker_color(markerColor);

			if (config.style.markerfacecolor != "none") {
				auto markerFaceColor = parseColor(config.style.markerfacecolor);
                                s->marker_face_color(markerFaceColor);
			}
		}

		s->line_width(config.style.linewidth);

                // Set map bounds based on data
                auto bounds = calculateMapBounds(data);
                auto ax = plt::gca();
                ax->xlim({bounds.first, bounds.second});

		// Calculate latitude bounds with padding
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

