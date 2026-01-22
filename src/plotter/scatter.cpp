#include "plotter_includes/plotter.h"
#include <matplot/matplot.h>

namespace plt = matplot;

namespace Visualization {

    // Enhanced Scatter Plot with comprehensive styling
    void Plotter::plotScatter(const PlotData& data, const PlotConfig& config) {
        validatePlotData(data, config);
        setupFigure(config);

        if (data.numericData.size() < 2) {
            throw std::runtime_error("Need at least 2 numeric columns for scatter plot");
        }

        auto itX = data.numericData.begin();
        auto itY = std::next(data.numericData.begin());

        std::vector<double> xData = itX->second;
        std::vector<double> yData = itY->second;

        std::vector<double> cleanX, cleanY;
        for (size_t i = 0; i < xData.size(); ++i) {
            if (!std::isnan(xData[i]) && !std::isnan(yData[i])) {
                cleanX.push_back(xData[i]);
                cleanY.push_back(yData[i]);
            }
        }

        if (cleanX.empty() || cleanY.empty()) {
            throw std::runtime_error("No valid data points for scatter plot");
        }

        // Create scatter plot with comprehensive styling
        auto s = plt::scatter(cleanX, cleanY);

        if (!config.style.colormap.empty()) {
            std::string cmap = config.style.colormap;
            std::transform(cmap.begin(), cmap.end(), cmap.begin(), ::tolower);

            // Map common matplotlib colormap names to Matplot++ colormaps
            if (cmap == "jet" || cmap == "parula") {
                plt::colormap(plt::palette::jet());
            } else if (cmap == "hot") {
                plt::colormap(plt::palette::hot());
            } else if (cmap == "cool") {
                plt::colormap(plt::palette::cool());
            } else if (cmap == "spring") {
                plt::colormap(plt::palette::spring());
            } else if (cmap == "summer") {
                plt::colormap(plt::palette::summer());
            } else if (cmap == "autumn") {
                plt::colormap(plt::palette::autumn());
            } else if (cmap == "winter") {
                plt::colormap(plt::palette::winter());
            } else if (cmap == "gray" || cmap == "grey") {
                plt::colormap(plt::palette::gray());
            } else if (cmap == "bone") {
                plt::colormap(plt::palette::bone());
            } else if (cmap == "copper") {
                plt::colormap(plt::palette::copper());
            } else if (cmap == "pink") {
                plt::colormap(plt::palette::pink());
            } else if (cmap == "lines") {
                plt::colormap(plt::palette::lines());
            } else if (cmap == "colorcube") {
                plt::colormap(plt::palette::colorcube());
            } else if (cmap == "prism") {
                plt::colormap(plt::palette::prism());
            } else if (cmap == "flag") {
                plt::colormap(plt::palette::flag());
            } else if (cmap == "white") {
                plt::colormap(plt::palette::white());
            }
            // Add colorbar
            plt::colorbar();
        }

        // Apply marker styling
        std::string marker = parseMarker(config.style.marker);
        if (!marker.empty() && marker != "none") {
            s->marker_style(marker);
            s->marker_size(config.style.markersize);

            auto markerColor = parseColor(config.style.markercolor);
            s->marker_color(markerColor);

            if (config.style.markerfacecolor != "none") {
                auto markerFaceColor = parseColor(config.style.markerfacecolor);
                s->marker_face_color(markerFaceColor);
            }
        }

        s->line_width(config.style.linewidth);
        auto markerColor = parseColor(config.style.markercolor);
        markerColor[3] = config.style.alpha; // Set alpha component
        s->marker_color(markerColor);

        // Add legend if needed
        addLegendIfNeeded(config);

        handlePlotOutput(config);
    }

} // namespace Visualization
