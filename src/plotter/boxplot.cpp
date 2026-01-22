#include "plotter_includes/plotter.h"
#include <matplot/matplot.h>
#include <cstdlib>
#include <ctime>

namespace plt = matplot;

namespace Visualization {

    // Enhanced Box Plot with comprehensive styling
    void Plotter::plotBoxPlot(const PlotData& data, const PlotConfig& config) {
        validatePlotData(data, config);
        setupFigure(config);

        std::vector<std::vector<double>> boxData;
        std::vector<std::string> labels;

        // Extract numeric columns for box plot
        size_t count = 0;
        for (const auto& [colName, values] : data.numericData) {
            if (count >= 10) break;

            std::vector<double> cleanValues;
            for (double v : values) {
                if (!std::isnan(v)) {
                    cleanValues.push_back(v);
                }
            }

            if (!cleanValues.empty()) {
                boxData.push_back(cleanValues);
                labels.push_back(colName);
                count++;
            }
        }

        if (boxData.empty()) {
            throw std::runtime_error("No valid numeric data for box plot");
        }

        // Create box plot with comprehensive styling
        auto bx = plt::boxplot(boxData);

        // Apply box plot styling
        auto boxColor = parseColor(config.style.color);
        boxColor[3] = config.style.alpha; // Alpha in color
        bx->face_color(boxColor);

        bx->edge_width(config.style.linewidth);

        // Configure outlier display
        if (!config.style.showfliers) {
            // Matplot++ doesn't have direct control for hiding outliers
            // We need to filter them manually
        } else {
            // Style outliers if shown
            // Note: Matplot++ doesn't have direct outlier styling control
        }

        // Set labels
        std::vector<double> x_ticks;
        for (size_t i = 1; i <= labels.size(); ++i) {
            x_ticks.push_back(i);
        }
        plt::xticks(x_ticks);
        plt::xticklabels(labels);

        if (config.style.xtick_rotation != 0.0) {
            plt::xtickangle(config.style.xtick_rotation);
        }

        // Add jittered points for individual data if requested
        if (config.style.showfliers) {
            plt::hold(true);
            for (size_t i = 0; i < boxData.size(); ++i) {
                std::vector<double> x_jitter(boxData[i].size());
                std::generate(x_jitter.begin(), x_jitter.end(),
                            [i]() { return i + 1 + (rand() / (RAND_MAX + 1.0) * 0.4 - 0.2); });

                auto s = plt::scatter(x_jitter, boxData[i]);
                std::string flierMarker = parseMarker(config.style.fliermarker);
                auto jitterColor = parseColor("red");
                jitterColor[3] = 0.5f;
                s->marker_style(flierMarker);
                s->marker_size(config.style.fliersize);
                s->color(jitterColor);
                s->marker_face(true);
                s->marker_face_color({0.5, 0.5, 0.5, 0.3});
                s->line_width(0.5);
            }
            plt::hold(false);
        }

        // Add legend if needed
        addLegendIfNeeded(config);

        handlePlotOutput(config);
    }

} // namespace Visualization
