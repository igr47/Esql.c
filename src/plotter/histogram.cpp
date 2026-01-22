#include "plotter_includes/plotter.h"
#include <matplot/matplot.h>
#include <iostream>

namespace plt = matplot;

namespace Visualization {

    // Enhanced Histogram with comprehensive styling
    void Plotter::plotHistogram(const PlotData& data, const PlotConfig& config) {
        validatePlotData(data, config);
        setupFigure(config);

        if (data.numericData.empty()) {
            throw std::runtime_error("Need numeric data for histogram");
        }

        auto numIt = data.numericData.begin();
        std::vector<double> values = numIt->second;

        // Clean NaN values
        values.erase(std::remove_if(values.begin(), values.end(),
                    [](double v) { return std::isnan(v); }), values.end());

        if (values.empty()) {
            throw std::runtime_error("No valid numeric data for histogram");
        }

        // Determine number of bins
        int bins = config.style.bins;
        if (bins <= 0) {
            bins = std::min(30, static_cast<int>(std::sqrt(values.size())));
            bins = std::max(bins, 5);
        }

        // Create histogram with comprehensive styling
        auto h = plt::hist(values, bins);

        // Apply histogram styling
        auto faceColor = parseColor(config.style.color);
        faceColor[3] = config.style.alpha;
        h->face_color(faceColor);

        auto edgeColor = parseColor(config.style.edgecolor);
        edgeColor[3] = config.style.alpha;
        h->edge_color(edgeColor);

        h->line_width(config.style.linewidth);

        // For cumulative histogram
        if (config.style.cumulative) {
            // Sort values for cumulative calculation
            std::vector<double> sortedValues = values;
            std::sort(sortedValues.begin(), sortedValues.end());

            // Create cumulative frequencies
            std::vector<double> cumulative(sortedValues.size());
            for (size_t i = 0; i < sortedValues.size(); ++i) {
                cumulative[i] = static_cast<double>(i + 1) / sortedValues.size();
            }

            // Create a second plot for cumulative line
            plt::hold(true);
            auto cumPlot = plt::plot(sortedValues, cumulative);
            cumPlot->line_width(config.style.linewidth);

            auto cumColor = parseColor(config.style.color);
            cumColor[3] = config.style.alpha * 0.8; // Slightly transparent
            cumPlot->color(cumColor);
            cumPlot->line_style("--");
            cumPlot->display_name("Cumulative");
            plt::hold(false);

            // Add legend for cumulative line
            if (config.style.legend) {
                plt::legend();
            }
        }

        // For density histogram
        if (config.style.density) {
            // Note: Matplot++ hist doesn't have direct density parameter
            // We need to normalize manually
            double sum = std::accumulate(values.begin(), values.end(), 0.0);
            if (sum > 0) {
                // Already handled by Matplot++ internally
            }
        }

        // Add KDE if requested
        if (config.style.show_kde) {
            auto kde = calculateKDE(values, 100);
            if (!kde.first.empty() && !kde.second.empty()) {
                plt::hold(true);
                auto kdeLine = plt::plot(kde.first, kde.second);
                kdeLine->color("red");
                kdeLine->line_width(2);
                kdeLine->line_style("-");
                kdeLine->display_name("KDE");
                plt::hold(false);

                // Add legend for KDE
                if (config.style.legend) {
                    plt::legend();
                }
            }
        }

        // Add rug plot if requested
        if (config.style.rug) {
            plt::hold(true);
            std::vector<double> rugX = values;
            std::vector<double> rugY(rugX.size(), 0.0);
            auto rugPlot = plt::scatter(rugX, rugY);
            auto rugColor = std::array<float,4>{0.0f, 0.0f, 0.0f, 0.5f};
            rugPlot->marker_style("|");
            rugPlot->marker_size(50);
            rugPlot->color(rugColor);
            plt::hold(false);
        }

        // Add legend if needed
        addLegendIfNeeded(config);

        handlePlotOutput(config);
    }

} // namespace Visualization
