 #include "plotter_includes/plotter.h"
#include <matplot/matplot.h>
namespace plt = matplot;
using json = nlohmann::json;

namespace Visualization {

	void Plotter::plotGeoBubble(const PlotData& data, const PlotConfig& config) {
		validatePlotData(data, config);
                validateGeoData(data, config);
                setupGeoAxes(config);

                // Add basemap
                addBasemap(config);

                // Get latitude and longitude data
                auto latIt = data.latitudeData.begin();
                auto lonIt = data.longitudeData.begin();

                if (latIt == data.latitudeData.end() || lonIt == data.longitudeData.end()) {
			throw std::runtime_error("No latitude/longitude data found for geographic bubble plot");
                }

		std::vector<double> lats = latIt->second;
                std::vector<double> lons = lonIt->second;

                // Get value data for bubble sizes
                std::vector<double> values;
		if (!data.numericData.empty()) {
			auto valIt = data.numericData.begin();
                        values = valIt->second;
                } else {
			throw std::runtime_error("Bubble plot requires value data for bubble sizes");
		}

		// Clean data
                std::vector<double> clean_lats, clean_lons, clean_values;
                for (size_t i = 0; i < lats.size(); ++i) {
			if (!std::isnan(lats[i]) && !std::isnan(lons[i]) && i < values.size() && !std::isnan(values[i]) && values[i] > 0) {
				clean_lats.push_back(lats[i]);
                                clean_lons.push_back(lons[i]);
                                clean_values.push_back(values[i]);
			}
		}

                if (clean_lats.empty() || clean_lons.empty()) {
			throw std::runtime_error("No valid data points for bubble plot");
		}

		// Normalize bubble sizes
		double min_val = *std::min_element(clean_values.begin(), clean_values.end());
                double max_val = *std::max_element(clean_values.begin(), clean_values.end());

		std::vector<double> bubble_sizes;
		for (double val : clean_values) {
			double normalized = (val - min_val) / (max_val - min_val);
			double size = config.style.bubble_min_size + normalized * (config.style.bubble_max_size - config.style.bubble_min_size);
			if (config.style.bubble_scale == "radius") {
				size = size * size; // Area proportional to radius squared
                        }
			bubble_sizes.push_back(size);
		}

		// Create bubble plot
                auto s = plt::scatter(clean_lons, clean_lats, bubble_sizes);

                // Apply styling
	        auto markerColor = parseColor(config.style.markercolor);
                markerColor[3] = config.style.alpha * 0.7; // Slightly transparent
                s->marker_color(markerColor);
        
                s->marker_face(true);
                auto faceColor = parseColor(config.style.markerfacecolor);
                faceColor[3] = config.style.alpha * 0.5;
                s->marker_face_color(faceColor);
        
                s->line_width(config.style.linewidth);
        
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

		// Add legend for bubble sizes
                if (config.style.legend) {
			// Create dummy scatter points for legend
                        std::vector<double> legend_lons = {bounds.first + 5.0};
                        std::vector<double> legend_lats = {min_lat - lat_padding + 5.0};
                        std::vector<double> legend_sizes = {config.style.bubble_min_size};

                        auto legend1 = plt::scatter(legend_lons, legend_lats, legend_sizes);
                        legend1->display_name(PlotUtils::formatNumber(min_val, 2));

			legend_sizes[0] = (config.style.bubble_min_size + config.style.bubble_max_size) / 2;
                        //auto legend2 = plt::scatter(legend_lons, {legend_lats[0] + 2.0}, legend_sizes);
			std::vector<double> legend_lats2 = {legend_lats[0] + 2.0};
			auto legend2 = plt::scatter(legend_lons, legend_lats2, legend_sizes);
                        legend2->display_name(PlotUtils::formatNumber((min_val + max_val) / 2, 2));

	                legend_sizes[0] = config.style.bubble_max_size;
                        //auto legend3 = plt::scatter(legend_lons, {legend_lats[0] + 4.0}, legend_sizes);
			std::vector<double> legend_lats3 = {legend_lats[0] + 4.0};
			auto legend3 = plt::scatter(legend_lons, legend_lats3, legend_sizes);
                        legend3->display_name(PlotUtils::formatNumber(max_val, 2));

                        plt::legend();
		}

		handlePlotOutput(config);
	}
}

