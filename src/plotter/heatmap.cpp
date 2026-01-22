#include "plotter_includes/plotter.h"
#include <matplot/matplot.h>
#include <algorithm>

namespace plt = matplot;

namespace Visualization {

    // Enhanced Heatmap with comprehensive styling
    void Plotter::plotHeatmap(const PlotData& data, const PlotConfig& config) {
        validatePlotData(data, config);
        setupFigure(config);

        // Convert data to matrix format
        std::vector<std::vector<double>> matrix;

        if (!data.numericData.empty()) {
            size_t n = data.numericData.begin()->second.size();
            for (size_t i = 0; i < n; ++i) {
                std::vector<double> row;
                for (const auto& [_, values] : data.numericData) {
                    if (i < values.size()) {
                        row.push_back(values[i]);
                    }
                }
                matrix.push_back(row);
            }
        }

        if (matrix.empty()) {
            throw std::runtime_error("No numeric data for heatmap");
        }

        // Create heatmap with comprehensive styling
        auto h = plt::heatmap(matrix);

        if (!config.style.colormap.empty()) {
            std::string cmap = config.style.colormap;
            std::transform(cmap.begin(), cmap.end(), cmap.begin(), ::tolower);

            // Common matplotlib colormap names to matplot++ colormap
            if (cmap == "jet" || cmap == "parula") {
                plt::colormap(plt::palette::jet());
            } else if (cmap == "plasma") {
                plt::colormap(plt::palette::plasma());
            } else if (cmap == "inferno") {
                plt::colormap(plt::palette::inferno());
            } else if (cmap == "magma") {
                plt::colormap(plt::palette::magma());
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
        }

        // Add colorbar
        plt::colorbar();

        // Annotate cells if requested
        if (config.style.annotate) {
            plt::hold(true);
            for (size_t i = 0; i < matrix.size(); ++i) {
                for (size_t j = 0; j < matrix[i].size(); ++j) {
                    std::stringstream ss;
                    if (config.style.fmt == ".0f") {
                        ss << std::fixed << std::setprecision(0) << matrix[i][j];
                    } else if (config.style.fmt == ".1f") {
                        ss << std::fixed << std::setprecision(1) << matrix[i][j];
                    } else if (config.style.fmt == ".2f") {
                        ss << std::fixed << std::setprecision(2) << matrix[i][j];
                    } else if (config.style.fmt == ".3f") {
                        ss << std::fixed << std::setprecision(3) << matrix[i][j];
                    } else {
                        ss << matrix[i][j];
                    }
                    plt::text(j + 0.5, i + 0.5, ss.str());
                }
            }
            plt::hold(false);
        }

        // Add labels if available
        if (!data.columns.empty()) {
            std::vector<double> x_ticks, y_ticks;
            for (size_t i = 0; i < std::min(data.columns.size(), matrix[0].size()); ++i) {
                x_ticks.push_back(i + 0.5);
            }
            for (size_t i = 0; i < matrix.size(); ++i) {
                y_ticks.push_back(i + 0.5);
            }
            plt::xticks(x_ticks);
            plt::xticklabels(data.columns);
            plt::yticks(y_ticks);
        }

        handlePlotOutput(config);
    }

} // namespace Visualization
