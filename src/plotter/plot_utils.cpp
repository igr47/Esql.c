#include "plotter_includes/plotter.h"
#include <matplot/matplot.h>

namespace plt = matplot;

namespace Visualization {

    void Plotter::setup3DAxes(const PlotConfig& config) {
        auto ax = plt::gca();

        if (config.style.view_init_set) {
            ax->view(config.style.elevation, config.style.azimuth);
        }

        // Set axis labels
        if (!config.xLabel.empty()) ax->xlabel(config.xLabel);
        if (!config.yLabel.empty()) ax->ylabel(config.yLabel);
        if (!config.zLabel.empty()) ax->zlabel(config.zLabel);

        // Set axis limits
        if (!std::isnan(config.style.xmin) && !std::isnan(config.style.xmax)) {
            ax->xlim({config.style.xmin, config.style.xmax});
        }
        if (!std::isnan(config.style.ymin) && !std::isnan(config.style.ymax)) {
            ax->ylim({config.style.ymin, config.style.ymax});
        }
        if (!std::isnan(config.style.zmin) && !std::isnan(config.style.zmax)) {
            ax->zlim({config.style.zmin, config.style.zmax});
        }
    }

    void Plotter::applyStyle(const PlotConfig& config) {
        // Apply style presets if specified
        if (!currentStyle.empty()) {
            setStyle(currentStyle);
        }

        if (!currentPalette.empty()) {
            setColorPalette(currentPalette);
        }
    }

} // namespace Visualization
