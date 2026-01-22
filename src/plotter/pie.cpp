#include "plotter_includes/plotter.h"
#include <matplot/matplot.h>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace plt = matplot;

namespace Visualization {

    void Plotter::plotPie(const PlotData& data, const PlotConfig& config) {
        validatePlotData(data, config);
        setupFigure(config);

        if (data.categoricalData.empty() || data.numericData.empty()) {
            throw std::runtime_error("Need both categorical and numeric data for pie chart");
        }

        auto catIt = data.categoricalData.begin();
        auto numIt = data.numericData.begin();

        std::vector<std::string> categories = catIt->second;
        std::vector<double> values = numIt->second;

        size_t n = std::min(categories.size(), values.size());
        categories.resize(n);
        values.resize(n);

        // Filter out zero and negative values
        std::vector<double> nonZeroValues;
        std::vector<std::string> nonZeroCategories;
        std::vector<size_t> originalIndices;

        for (size_t i = 0; i < values.size(); ++i) {
            if (values[i] > 0) {
                nonZeroValues.push_back(values[i]);
                nonZeroCategories.push_back(categories[i]);
                originalIndices.push_back(i);
            }
        }

        if (nonZeroValues.empty()) {
            throw std::runtime_error("No positive values for pie chart");
        }

        // Create vectors for colors
        std::vector<std::array<float, 4>> sliceColors;

        // Get colors based on configuration
        if (!config.style.colors.empty()) {
            // Use specified colors
            for (size_t i = 0; i < nonZeroValues.size(); ++i) {
                if (i < config.style.colors.size()) {
                    sliceColors.push_back(parseColor(config.style.colors[i]));
                } else {
                    // Fall back to default palette
                    auto defaultColors = getColorPalette(nonZeroValues.size());
                    sliceColors.push_back(parseColor(defaultColors[i % defaultColors.size()]));
                }
            }
        } else {
            // Use default palette
            auto defaultColors = getColorPalette(nonZeroValues.size());
            for (size_t i = 0; i < nonZeroValues.size(); ++i) {
                sliceColors.push_back(parseColor(defaultColors[i % defaultColors.size()]));
            }
        }

        // Apply explode if specified
        std::vector<double> explodeVec(nonZeroValues.size(), 0.0);
        if (!config.style.explode.empty() && config.style.explode.size() >= nonZeroValues.size()) {
            for (size_t i = 0; i < nonZeroValues.size(); ++i) {
                explodeVec[i] = config.style.explode[i];
            }
        }

        // Create pie chart using Matplot++ API
        plt::hold(true);

        // Calculate total for percentages
        double total = std::accumulate(nonZeroValues.begin(), nonZeroValues.end(), 0.0);

        // Create custom pie chart using patches (Matplot++ doesn't have a direct pie function that works well)
        // We'll create it using wedges
        double startAngle = 0.0;
        if (!config.style.startangle.empty()) {
            try {
                startAngle = std::stod(config.style.startangle);
            } catch (...) {
                startAngle = 0.0;
            }
        }

        // Convert to radians
        startAngle = startAngle * M_PI / 180.0;

        // Create pie chart wedges
        for (size_t i = 0; i < nonZeroValues.size(); ++i) {
            double value = nonZeroValues[i];
            double percentage = value / total;
            double angle = 2.0 * M_PI * percentage;

            // Skip very small slices
            if (percentage < 0.001) continue;

            // Calculate wedge vertices
            std::vector<double> wedgeX, wedgeY;
            wedgeX.push_back(0.5); // Center X
            wedgeY.push_back(0.5); // Center Y

            // Add points along the arc
            int segments = 50; // Resolution of the arc
            double explode = explodeVec[i] * 0.1; // Scale explosion

            for (int j = 0; j <= segments; ++j) {
                double theta = startAngle + (angle * j / segments);
                double x = 0.5 + (0.4 + explode) * cos(theta);
                double y = 0.5 + (0.4 + explode) * sin(theta);
                wedgeX.push_back(x);
                wedgeY.push_back(y);
            }

            // Fill the wedge
            auto wedge = plt::fill(wedgeX, wedgeY);
            wedge->color(sliceColors[i]);
            wedge->line_width(1.0);

            // Add edge color
            auto edgeColor = parseColor(config.style.edgecolor);
            wedge->color(edgeColor);

            // Add label
            double labelAngle = startAngle + angle / 2.0;
            double labelDistance = 0.6 + explode * 2.0; // Position outside the wedge

            std::stringstream labelText;
            if (config.style.autopct) {
                labelText << nonZeroCategories[i] << "\n"
                         << std::fixed << std::setprecision(1) << (percentage * 100) << "%";
            } else {
                labelText << nonZeroCategories[i] << "\n" << value;
            }

            double labelX = 0.5 + labelDistance * cos(labelAngle);
            double labelY = 0.5 + labelDistance * sin(labelAngle);

            plt::text(labelX, labelY, labelText.str());

            startAngle += angle;
        }

        plt::hold(false);

        // Set axis properties
        plt::xlim({0.0, 1.0});
        plt::ylim({0.0, 1.0});
        plt::axis("equal");
        plt::axis("off");

        // Set title
        if (!config.title.empty()) {
            plt::title(config.title);
        }

        // Add legend if requested
        if (config.style.legend) {
            std::vector<std::string> legendEntries;
            for (size_t i = 0; i < nonZeroCategories.size(); ++i) {
                std::stringstream entry;
                if (config.style.autopct) {
                    double percentage = (nonZeroValues[i] / total) * 100.0;
                    entry << nonZeroCategories[i] << " ("
                         << std::fixed << std::setprecision(1) << percentage << "%)";
                } else {
                    entry << nonZeroCategories[i] << ": " << nonZeroValues[i];
                }
                legendEntries.push_back(entry.str());
            }

            plt::legend(legendEntries);
        }

        handlePlotOutput(config);
    }

} // namespace Visualization
