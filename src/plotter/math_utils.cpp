#include "plotter_includes/plotter.h"
#include <matplot/matplot.h>
#include <cmath>
#include <numeric>
#include <algorithm>

namespace plt = matplot;

namespace Visualization {

    // Statistical calculations
    std::pair<std::vector<double>, std::vector<double>> Plotter::calculateKDE(const std::vector<double>& data, int gridPoints) {
        std::pair<std::vector<double>, std::vector<double>> result;

        if (data.empty()) return result;

        // Create grid
        double min_val = *std::min_element(data.begin(), data.end());
        double max_val = *std::max_element(data.begin(), data.end());
        double range = max_val - min_val;

        // Add padding
        min_val -= range * 0.1;
        max_val += range * 0.1;

        std::vector<double> grid = linspace(min_val, max_val, gridPoints);

        // Calculate bandwidth using Silverman's rule of thumb
        double n = static_cast<double>(data.size());
        double stddev = 0.0;
        double mean = std::accumulate(data.begin(), data.end(), 0.0) / n;
        for (double v : data) {
            stddev += (v - mean) * (v - mean);
        }
        stddev = sqrt(stddev / n);

        double bandwidth = 1.06 * stddev * pow(n, -0.2);
        if (bandwidth < 0.1) bandwidth = 0.1; // Minimum bandwidth

        // Calculate KDE
        std::vector<double> kde(grid.size(), 0.0);
        for (size_t i = 0; i < grid.size(); ++i) {
            double sum = 0.0;
            for (double v : data) {
                double u = (grid[i] - v) / bandwidth;
                sum += exp(-0.5 * u * u) / sqrt(2.0 * M_PI);
            }
            kde[i] = sum / (n * bandwidth);
        }

        result.first = grid;
        result.second = kde;
        return result;
    }

    void Plotter::setupFigure(const PlotConfig& config) {
        plt::figure(true);
        plt::figure()->size(config.style.figwidth * 100, config.style.figheight * 100);

        // Set title
        if (!config.title.empty()) {
            plt::title(config.title);
        }

        // Set labels
        if (!config.xLabel.empty()) {
            plt::xlabel(config.xLabel);
        } else if (!config.style.xlabel.empty()) {
            plt::xlabel(config.style.xlabel);
        }

	if (!config.yLabel.empty()) {
            plt::ylabel(config.yLabel);
        } else if (!config.style.ylabel.empty()) {
            plt::ylabel(config.style.ylabel);
        }

        // Set grid
        if (config.style.grid) {
            plt::grid(true);
        }

        // Set axis limits if specified
        if (!std::isnan(config.style.xmin) && !std::isnan(config.style.xmax)) {
            plt::xlim({config.style.xmin, config.style.xmax});
        }

        if (!std::isnan(config.style.ymin) && !std::isnan(config.style.ymax)) {
            plt::ylim({config.style.ymin, config.style.ymax});
        }

	        // Set tick rotation
        if (config.style.xtick_rotation != 0.0) {
            plt::xtickangle(config.style.xtick_rotation);
        }

        if (config.style.ytick_rotation != 0.0) {
            plt::ytickangle(config.style.ytick_rotation);
        }
    }

    void Plotter::addLegendIfNeeded(const PlotConfig& config) {
        if (config.style.legend && !config.seriesNames.empty()) {
            plt::legend(config.seriesNames);
        }
    }

    void Plotter::handlePlotOutput(const PlotConfig& config) {
        if (!config.outputFile.empty()) {
            // Determine format from filename extension
            std::string filename = config.outputFile;
            std::string format = config.style.save_format;

            // Override format based on extension if not specified
            if (format.empty()) {
                size_t dotPos = filename.find_last_of('.');
                if (dotPos != std::string::npos) {
                    format = filename.substr(dotPos + 1);
                } else {
                    format = "png";
                }
            }

            savePlot(filename, format);
        } else {
            showPlot();
        }
    }

    void Plotter::validatePlotData(const PlotData& data, const PlotConfig& config) {
        if (data.rows.empty()) {
            throw std::runtime_error("No data to plot");
        }

        switch (config.type) {
            case PlotType::LINE:
            case PlotType::SCATTER:
                if (data.numericData.size() < 2) {
                    throw std::runtime_error("Need at least 2 numeric columns for line/scatter plot");
                }
                break;
            case PlotType::BAR:
                if (data.categoricalData.empty() || data.numericData.empty()) {
                    throw std::runtime_error("Need both categorical and numeric data for bar plot");
                }
                break;
            case PlotType::HISTOGRAM:
		if (data.numericData.empty()) {
                    throw std::runtime_error("Numeric data required for histogram");
                }
                break;
            case PlotType::BOXPLOT:
                if (data.numericData.empty()) {
                    throw std::runtime_error("Numeric data required for box plot");
                }
                break;
            case PlotType::PIE:
                if (data.categoricalData.empty() || data.numericData.empty()) {
                    throw std::runtime_error("Need both categorical and numeric data for pie chart");
                }
                break;
            case PlotType::HEATMAP:
                if (data.numericData.empty()) {
                    throw std::runtime_error("Numeric data required for heatmap");
		}
		break;
	    case PlotType::STACKED_BAR:
            case PlotType::MULTI_LINE:
                if (data.numericData.size() < 2) {
                    throw std::runtime_error("Need at least 2 numeric columns for this plot type");
                }
                break;
            case PlotType::AREA:
                if (data.numericData.size() < 2) {
                    throw std::runtime_error("Need at least 2 numeric columns for area plot");
                }
                break;
            case PlotType::CONTOUR:
            case PlotType::SURFACE:
            case PlotType::WIREFRAME:
                if (data.numericData.size() < 3) {
                    throw std::runtime_error("Need at least 3 numeric columns for 3D plot");
                }
                break;
            case PlotType::HISTOGRAM_2D:
                if (data.numericData.size() < 2) {
			throw std::runtime_error("Need at least 2 numeric columns for 2D histogram");
                }
                break;
            default:
                break;
        }
    }


    std::pair<double, double> Plotter::linearRegression(const std::vector<double>& x,
                                                       const std::vector<double>& y) {
        if (x.size() != y.size() || x.empty()) {
            return {0.0, 0.0};
        }

        double n = static_cast<double>(x.size());
        double sum_x = std::accumulate(x.begin(), x.end(), 0.0);
        double sum_y = std::accumulate(y.begin(), y.end(), 0.0);
        double sum_xy = std::inner_product(x.begin(), x.end(), y.begin(), 0.0);
        double sum_x2 = std::inner_product(x.begin(), x.end(), x.begin(), 0.0);

        double slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        double intercept = (sum_y - slope * sum_x) / n;

        return {slope, intercept};
    }

    double Plotter::calculateRSquared(const std::vector<double>& x,
                                     const std::vector<double>& y,
                                     double slope, double intercept) {
        if (x.size() != y.size() || x.empty()) {
            return 0.0;
        }

        double n = static_cast<double>(x.size());
        double y_mean = std::accumulate(y.begin(), y.end(), 0.0) / n;

        double ss_tot = 0.0, ss_res = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
            double y_pred = slope * x[i] + intercept;
            ss_tot += (y[i] - y_mean) * (y[i] - y_mean);
            ss_res += (y[i] - y_pred) * (y[i] - y_pred);
        }

        if (ss_tot == 0.0) return 1.0;
        return 1.0 - (ss_res / ss_tot);
    }

    std::vector<std::vector<double>> Plotter::calculateCorrelationMatrix(const PlotData& data) {
        std::vector<std::vector<double>> corrMatrix;

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

        size_t n = numericColumns.size();
        corrMatrix.resize(n, std::vector<double>(n, 0.0));

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

        return corrMatrix;
    }

    // Data transformations
    std::vector<double> Plotter::normalize(const std::vector<double>& data) {
        std::vector<double> result = data;

        if (result.empty()) return result;

        double min_val = *std::min_element(result.begin(), result.end());
        double max_val = *std::max_element(result.begin(), result.end());

        if (max_val > min_val) {
            for (auto& v : result) {
                v = (v - min_val) / (max_val - min_val);
            }
        }

        return result;
    }

    std::vector<double> Plotter::standardize(const std::vector<double>& data) {
        std::vector<double> result = data;

        if (result.empty()) return result;

        double n = static_cast<double>(result.size());
        double mean = std::accumulate(result.begin(), result.end(), 0.0) / n;

        double variance = 0.0;
        for (double v : result) {
            variance += (v - mean) * (v - mean);
        }
        variance /= n;
        double stddev = sqrt(variance);

        if (stddev > 0.0) {
            for (auto& v : result) {
                v = (v - mean) / stddev;
            }
        }

        return result;
    }

    std::vector<double> Plotter::logTransform(const std::vector<double>& data) {
        std::vector<double> result = data;

        for (auto& v : result) {
            if (v > 0.0) {
                v = log(v);
            } else {
                v = 0.0;
            }
        }

        return result;
    }

    // Utility functions
    std::vector<double> Plotter::linspace(double start, double end, size_t num) {
        std::vector<double> result(num);
        if (num == 0) return result;
        if (num == 1) {
            result[0] = start;
            return result;
        }

        double step = (end - start) / (num - 1);
        for (size_t i = 0; i < num; ++i) {
            result[i] = start + i * step;
        }

        return result;
    }

    double Plotter::erfinv(double x) {
        // Inverse error function approximation
        double a = 0.147;
        double ln1x2 = log(1 - x * x);
        double term1 = 2 / (M_PI * a) + ln1x2 / 2;
        double term2 = ln1x2 / a;
        double result = sqrt(sqrt(term1 * term1 - term2) - term1);

        if (x < 0) result = -result;
        return result;
    }

    std::string PlotUtils::formatNumber(double value, int precision) {
        std::stringstream ss;
        ss << std::fixed << std::setprecision(precision) << value;
        return ss.str();
    }



} // namespace Visualization
