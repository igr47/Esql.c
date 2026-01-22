#include "plotter_includes/plotter.h"
#include <matplot/matplot.h>

namespace plt = matplot;

namespace Visualization {

    // Create dashboard with multiple plots
    void Plotter::createDashboard(const std::vector<PlotData>& datasets,
                                 const std::vector<PlotConfig>& configs,
                                 int rows, int cols) {
        if (datasets.empty() || configs.empty()) {
            throw std::runtime_error("No data or configs provided for dashboard");
        }

        if (datasets.size() != configs.size()) {
            throw std::runtime_error("Number of datasets must match number of configs");
        }

        size_t numPlots = std::min(datasets.size(), configs.size());
        rows = std::min(rows, static_cast<int>(numPlots));
        cols = std::min(cols, static_cast<int>((numPlots + rows - 1) / rows));

        // Create tiled layout
        plt::tiledlayout(rows, cols);

        for (size_t i = 0; i < numPlots; ++i) {
            plt::nexttile();

            // Plot based on config type
            switch (configs[i].type) {
                case PlotType::LINE:
                    plotLine(datasets[i], configs[i]);
                    break;
                case PlotType::SCATTER:
                    plotScatter(datasets[i], configs[i]);
                    break;
                case PlotType::BAR:
                    plotBar(datasets[i], configs[i]);
                    break;
                case PlotType::HISTOGRAM:
                    plotHistogram(datasets[i], configs[i]);
                    break;
                case PlotType::BOXPLOT:
                    plotBoxPlot(datasets[i], configs[i]);
                    break;
                case PlotType::PIE:
                    plotPie(datasets[i], configs[i]);
                    break;
                case PlotType::HEATMAP:
                    plotHeatmap(datasets[i], configs[i]);
                    break;
                case PlotType::AREA:
                    plotArea(datasets[i], configs[i]);
                    break;
                case PlotType::STACKED_BAR:
                    plotStackedBar(datasets[i], configs[i]);
                    break;
                case PlotType::MULTI_LINE:
                    plotMultiLine(datasets[i], configs[i]);
                    break;
                default:
                    autoPlot(datasets[i], configs[i].title);
                    break;
            }
        }

        handlePlotOutput(PlotConfig());
    }

} // namespace Visualization
