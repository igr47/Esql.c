#include "plotter_includes/plotter.h"
#include <matplot/matplot.h>

namespace plt = matplot;

namespace Visualization {

    void Plotter::plotHistogram2D(const PlotData& data, const PlotConfig& config) {
        // 2D histogram implementation
        validatePlotData(data, config);
        setupFigure(config);

        if (data.numericData.size() < 2) {
            throw std::runtime_error("Need at least 2 numeric columns for 2D histogram");
        }

        auto xIt = data.numericData.begin();
        auto yIt = std::next(xIt);

        std::vector<double> xData = xIt->second;
        std::vector<double> yData = yIt->second;

        // Clean data
        std::vector<double> cleanX, cleanY;
        for (size_t i = 0; i < std::min(xData.size(), yData.size()); ++i) {
            if (!std::isnan(xData[i]) && !std::isnan(yData[i])) {
                cleanX.push_back(xData[i]);
                cleanY.push_back(yData[i]);
            }
        }

        int nbins = config.style.bins;
        if (nbins <= 0) nbins = 20;

        // Create 2D histogram
        auto h = plt::hist2(cleanX, cleanY, nbins, nbins);

        // Add colorbar
        plt::colorbar();

        handlePlotOutput(config);
    }

} // namespace Visualization
