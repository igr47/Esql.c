#include "plotter_includes/plotter.h"
#include <matplot/matplot.h>

namespace plt = matplot;

namespace Visualization {

    void Plotter::plotTimeSeries(const PlotData& data, const std::string& timeColumn,
                                const std::string& valueColumn, const PlotConfig& config) {
        // Time series plot implementation
        validatePlotData(data, config);
        setupFigure(config);

        if (data.numericData.find(valueColumn) == data.numericData.end()) {
            throw std::runtime_error("Value column not found or not numeric: " + valueColumn);
        }

        const std::vector<double>& values = data.numericData.at(valueColumn);

        // Generate time indices
        std::vector<double> timeIndices(values.size());
        std::iota(timeIndices.begin(), timeIndices.end(), 0.0);

        // Create time series plot
        auto p = plt::plot(timeIndices, values);
        auto lineColor = parseColor(config.style.color);
        lineColor[3] = config.style.alpha; // Alpha in color
        p->color(lineColor);
        p->line_width(config.style.linewidth);
        p->line_style(parseLineStyle(config.style.linestyle));

        // Apply marker if specified
        std::string marker = parseMarker(config.style.marker);
        if (!marker.empty() && marker != "none") {
            p->marker_style(marker);
            p->marker_size(config.style.markersize);
        }

        handlePlotOutput(config);
    }

} // namespace Visualization
