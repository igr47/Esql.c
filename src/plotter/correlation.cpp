#include "plotter_includes/plotter.h"
#include <matplot/matplot.h>

namespace plt = matplot;

namespace Visualization {

    void Plotter::plotCorrelationMatrix(const PlotData& data) {
        if (data.numericData.size() < 2) {
            throw std::runtime_error("Need at least 2 numeric columns for correlation matrix");
        }

        // Extract numeric columns
        std::vector<std::string> colNames;
        std::vector<std::vector<double>> numericColumns;

        for (const auto& [colName, values] : data.numericData) {
            colNames.push_back(colName);

            // Clean NaN values
            std::vector<double> cleanValues;
            for (double v : values) {
                if (!std::isnan(v)) {
                    cleanValues.push_back(v);
                }
            }
            numericColumns.push_back(cleanValues);
        }

        // Calculate correlation matrix
        size_t n = numericColumns.size();
        std::vector<std::vector<double>> corrMatrix(n, std::vector<double>(n, 0.0));

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                if (i == j) {
                    corrMatrix[i][j] = 1.0;
                } else {
                    const auto& x = numericColumns[i];
                    const auto& y = numericColumns[j];

                    if (x.size() != y.size() || x.empty()) {
                        corrMatrix[i][j] = 0.0;
                        continue;
                    }

                    double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0;
                    double sum_x2 = 0.0, sum_y2 = 0.0;

                    for (size_t k = 0; k < x.size(); ++k) {
                        sum_x += x[k];
                        sum_y += y[k];
                        sum_xy += x[k] * y[k];
                        sum_x2 += x[k] * x[k];
                        sum_y2 += y[k] * y[k];
                    }

                    double n_val = static_cast<double>(x.size());
                    double numerator = n_val * sum_xy - sum_x * sum_y;
                    double denominator = sqrt((n_val * sum_x2 - sum_x * sum_x) *
                                             (n_val * sum_y2 - sum_y * sum_y));

                    if (denominator != 0.0) {
                        corrMatrix[i][j] = numerator / denominator;
                    } else {
                        corrMatrix[i][j] = 0.0;
                    }
                }
            }
        }

        // Create heatmap visualization
        plt::figure(true);
        plt::figure()->size(800, 600);
        plt::title("Correlation Matrix");

        auto h = plt::heatmap(corrMatrix);

        // Add colorbar
        plt::colorbar();

        // Add labels
        std::vector<double> x_ticks, y_ticks;
        for (size_t i = 0; i < n; ++i) {
            x_ticks.push_back(i + 0.5);
            y_ticks.push_back(i + 0.5);
        }
        plt::xticks(x_ticks);
        plt::xticklabels(colNames);
        plt::yticks(y_ticks);
        plt::yticklabels(colNames);
        plt::xtickangle(45);

        // Add correlation values as text
        plt::hold(true);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                std::stringstream ss;
                ss << std::fixed << std::setprecision(2) << corrMatrix[i][j];
                plt::text(j + 0.5, i + 0.5, ss.str());
            }
        }
        plt::hold(false);

        handlePlotOutput(PlotConfig());
    }
}
