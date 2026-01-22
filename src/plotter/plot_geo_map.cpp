#include "plotter_includes/plotter.h"
#include <matplot/matplot.h>

namespace plt = matplot;
using json = nlohmann::json;

namespace Visualization {
   void Plotter::plotGeoMap(const PlotData& data, const PlotConfig& config) {
        validatePlotData(data, config);
        setupGeoAxes(config);

        // Add basemap
        addBasemap(config);

        // Calculate bounds if we have data
        if (!data.latitudeData.empty() && !data.longitudeData.empty()) {
            auto bounds = calculateMapBounds(data);
            auto ax = plt::gca();
            ax->xlim({bounds.first, bounds.second});

            // Estimate latitude bounds
            double lat_span = 60.0; // Default span
            ax->ylim({-lat_span/2, lat_span/2});
        }

        // Add scale bar and north arrow
        addScaleBar(config);
        addNorthArrow(config);

        // Add legend if needed
        addLegendIfNeeded(config);

        handlePlotOutput(config);
    }
}

