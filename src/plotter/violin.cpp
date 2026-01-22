#include "plotter_includes/plotter.h"
#include <matplot/matplot.h>

namespace plt = matplot;

namespace Visualization {

    void Plotter::plotViolin(const PlotData& data, const PlotConfig& config) {
        // Violin plot implementation
        validatePlotData(data, config);
        setupFigure(config);

        // Note: Matplot++ doesn't have native violin plot support
        // We can simulate it with combination of KDE and box plot

        std::vector<std::vector<double>> violinData;
        std::vector<std::string> labels;

        for (const auto& [colName, values] : data.numericData) {
            std::vector<double> cleanValues;
            for (double v : values) {
                if (!std::isnan(v)) {
                    cleanValues.push_back(v);
                }
            }

            if (!cleanValues.empty()) {
                violinData.push_back(cleanValues);
                labels.push_back(colName);
            }
        }

        if (violinData.empty()) {
            throw std::runtime_error("No valid numeric data for violin plot");
        }

        // Create multiple KDE plots side by side
        plt::hold(true);
        for (size_t i = 0; i < violinData.size(); ++i) {
            auto kde = calculateKDE(violinData[i], 100);
            if (!kde.first.empty() && !kde.second.empty()) {
                // Scale KDE to fit in violin width
                double scale = 0.3;
                std::vector<double> xLeft, xRight;
                for (size_t j = 0; j < kde.second.size(); ++j) {
                    xLeft.push_back(i + 1 - kde.second[j] * scale);
                    xRight.push_back(i + 1 + kde.second[j] * scale);
                }

                // Create filled violin shape
                std::vector<double> violinX = xLeft;
                violinX.insert(violinX.end(), xRight.rbegin(), xRight.rend());

                std::vector<double> violinY = kde.first;
                violinY.insert(violinY.end(), kde.first.rbegin(), kde.first.rend());

                auto fill = plt::fill(violinX, violinY);
                auto fillColor = parseColor(config.style.facecolor);
                fillColor[3] = config.style.alpha * 0.5f;
                fill->color(fillColor);
                fill->line_width(config.style.linewidth);

                // Add median line
                std::sort(violinData[i].begin(), violinData[i].end());
                double median = violinData[i][violinData[i].size() / 2];
                auto medianLine = plt::plot({i + 1 - 0.15, i + 1 + 0.15}, {median, median});
                medianLine->color("white");
                medianLine->line_width(2);
            }
        }
        plt::hold(false);

        // Set labels
        std::vector<double> x_ticks;
        for (size_t i = 1; i <= labels.size(); ++i) {
            x_ticks.push_back(i);
        }
        plt::xticks(x_ticks);
        plt::xticklabels(labels);

        handlePlotOutput(config);
    }

} // namespace Visualization
