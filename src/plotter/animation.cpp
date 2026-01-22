#include "plotter_includes/plotter.h"
#include <matplot/matplot.h>

namespace plt = matplot;

namespace Visualization {

    // Animation support
    void Plotter::createAnimation(const std::vector<PlotData>& frames,
                                 const PlotConfig& config, int fps) {
        if (frames.empty()) {
            throw std::runtime_error("No frames provided for animation");
        }

        // Note: Matplot++ doesn't have built-in animation support
        // This is a placeholder for future implementation
        // For now, we just plot the first frame

        if (!frames.empty()) {
            autoPlot(frames[0], config.title);
        }
    }

} // namespace Visualization
