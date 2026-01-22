#include "plotter_includes/plotter.h"
#include <matplot/matplot.h>

namespace Visualization {
	void Plotter::plotGeoGrid(const PlotData& data, const PlotConfig& config) {
		// Similar to heatmap but with grid cells
		plotGeoHeatmap(data, config); // Use heatmap implementation 
	}
}
