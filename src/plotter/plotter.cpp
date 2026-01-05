#include "plotter_includes/plotter.h"
#include <matplot/matplot.h>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace plt = matplot;

namespace Visualization {

    Plotter::Plotter() : plotterInitialized(false) {
        initializePlotter();
    }

    Plotter::~Plotter() {
        finalizePlotter();
    }

    void Plotter::initializePlotter() {
        if (!plotterInitialized) {
            // Initialize Matplot++
            //plt::backend("Agg"); // Non-interactive backend for file output
            plotterInitialized = true;
        }
    }

    void Plotter::finalizePlotter() {
        if (plotterInitialized) {
            //plt::figure()->close();
            plotterInitialized = false;
        }
    }

    PlotData Plotter::convertToPlotData(const ExecutionEngine::ResultSet& result,
                                       const std::vector<std::string>& xColumns,
                                       const std::vector<std::string>& yColumns) {
        PlotData data;
        data.columns = result.columns;
        data.rows = result.rows;

        // Extract column indices
        std::map<std::string, size_t> columnIndices;
        for (size_t i = 0; i < result.columns.size(); ++i) {
            columnIndices[result.columns[i]] = i;
        }

        // Process each column
        for (const auto& colName : result.columns) {
            if (columnIndices.find(colName) == columnIndices.end()) {
                continue;
            }

            size_t colIdx = columnIndices[colName];
            std::vector<std::string> columnValues;

            // Extract column values
            for (const auto& row : result.rows) {
                if (colIdx < row.size()) {
                    columnValues.push_back(row[colIdx]);
                }
            }

            // Check if column is numeric
            if (isNumericColumn(columnValues)) {
                data.numericData[colName] = convertToNumeric(columnValues);
            } else {
                data.categoricalData[colName] = columnValues;
            }
        }

        return data;
    }

    bool Plotter::isNumericColumn(const std::vector<std::string>& values) {
        if (values.empty()) return false;

        size_t numericCount = 0;
        for (const auto& val : values) {
            // Try to convert to double
            try {
                std::stod(val);
                numericCount++;
            } catch (...) {
                // Not numeric
            }
        }

        // Consider column numeric if > 80% values are numeric
        return (static_cast<double>(numericCount) / values.size()) > 0.8;
    }

    std::vector<double> Plotter::convertToNumeric(const std::vector<std::string>& values) {
        std::vector<double> numericValues;
        numericValues.reserve(values.size());

        for (const auto& val : values) {
            try {
                numericValues.push_back(std::stod(val));
            } catch (...) {
                numericValues.push_back(std::numeric_limits<double>::quiet_NaN());
            }
        }

        return numericValues;
    }

    void Plotter::validatePlotData(const PlotData& data, const PlotConfig& config) {
        if (data.rows.empty()) {
            throw std::runtime_error("No data to plot");
        }

        switch (config.type) {
            case PlotType::LINE:
            case PlotType::SCATTER:
                if (data.numericData.empty()) {
                    throw std::runtime_error("Numeric data required for line/scatter plot");
                }
                break;
            case PlotType::BAR:
            case PlotType::PIE:
                // Need at least one categorical column
                break;
            case PlotType::HISTOGRAM:
                if (data.numericData.empty()) {
                    throw std::runtime_error("Numeric data required for histogram");
                }
                break;
            default:
                break;
        }
    }

    void Plotter::setupFigure(const PlotConfig& config) {
        plt::figure(true);  // Set to true for immediate display
        plt::figure()->size(config.width * 100, config.height * 100);

        if (!config.title.empty()) {
            plt::title(config.title);
        }

        if (!config.xLabel.empty()) {
            plt::xlabel(config.xLabel);
        }

        if (!config.yLabel.empty()) {
            plt::ylabel(config.yLabel);
        }

        if (config.grid) {
            plt::grid(true);
        }
    }

    std::vector<std::string> Plotter::getColorPalette(int n) {
        static const std::vector<std::string> defaultPalette = {
            "b", "g", "r", "c", "m", "y", "k",
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
        };

        std::vector<std::string> colors;
        for (int i = 0; i < n; ++i) {
            colors.push_back(defaultPalette[i % defaultPalette.size()]);
        }
        return colors;
    }

    // Line Plot with advanced features
    void Plotter::plotLine(const PlotData& data, const PlotConfig& config) {
        validatePlotData(data, config);
        setupFigure(config);

        if (data.numericData.size() < 2) {
            throw std::runtime_error("Need at least 2 numeric columns for line plot");
        }

        auto itX = data.numericData.begin();
        auto itY = std::next(data.numericData.begin());

        std::vector<double> xData = itX->second;
        std::vector<double> yData = itY->second;

        // Remove NaN values
        std::vector<double> cleanX, cleanY;
        for (size_t i = 0; i < xData.size(); ++i) {
            if (!std::isnan(xData[i]) && !std::isnan(yData[i])) {
                cleanX.push_back(xData[i]);
                cleanY.push_back(yData[i]);
            }
        }

        if (cleanX.empty() || cleanY.empty()) {
            throw std::runtime_error("No valid data points for line plot");
        }

        // Sort by x values for line plot
        std::vector<size_t> indices(cleanX.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
                 [&cleanX](size_t a, size_t b) { return cleanX[a] < cleanX[b]; });

        std::vector<double> sortedX, sortedY;
        for (auto idx : indices) {
            sortedX.push_back(cleanX[idx]);
            sortedY.push_back(cleanY[idx]);
        }

        // Create plot
        auto p = plt::plot(sortedX, sortedY);
        
        // Apply style
        if (config.style.find("color") != config.style.end()) {
            p->color(config.style.at("color"));
        }
        
        if (config.style.find("linewidth") != config.style.end()) {
            try {
                double lw = std::stod(config.style.at("linewidth"));
                p->line_width(lw);
            } catch (...) {}
        }
        
        if (config.style.find("linestyle") != config.style.end()) {
            std::string ls = config.style.at("linestyle");
            p->line_style(ls);
        }
        
        if (config.style.find("marker") != config.style.end()) {
            std::string marker = config.style.at("marker");
            p->marker_style(marker);
            p->marker_size(8);
        }

        // Add legend if needed
        addLegendIfNeeded(config);

        // Save or show
        handlePlotOutput(config);
    }

    // Scatter Plot with advanced features
    void Plotter::plotScatter(const PlotData& data, const PlotConfig& config) {
        validatePlotData(data, config);
        setupFigure(config);

        if (data.numericData.size() < 2) {
            throw std::runtime_error("Need at least 2 numeric columns for scatter plot");
        }

        auto itX = data.numericData.begin();
        auto itY = std::next(data.numericData.begin());

        std::vector<double> xData = itX->second;
        std::vector<double> yData = itY->second;

        // Clean data
        std::vector<double> cleanX, cleanY;
        for (size_t i = 0; i < xData.size(); ++i) {
            if (!std::isnan(xData[i]) && !std::isnan(yData[i])) {
                cleanX.push_back(xData[i]);
                cleanY.push_back(yData[i]);
            }
        }

        if (cleanX.empty() || cleanY.empty()) {
            throw std::runtime_error("No valid data points for scatter plot");
        }

        // Create scatter plot
        auto s = plt::scatter(cleanX, cleanY);
        
        // Apply style
        if (config.style.find("color") != config.style.end()) {
            s->color(config.style.at("color"));
        }
        
        if (config.style.find("marker") != config.style.end()) {
            std::string marker = config.style.at("marker");
            s->marker_style(marker);
        }
        
        s->marker_size(50);
        s->marker_face(true);
        s->marker_face_color("white");
        s->line_width(1);

        // Add legend if needed
        addLegendIfNeeded(config);

        handlePlotOutput(config);
    }

    // Bar Plot with advanced features
    void Plotter::plotBar(const PlotData& data, const PlotConfig& config) {
        validatePlotData(data, config);
        setupFigure(config);

        // For bar plot, we need categorical x and numeric y
        if (data.categoricalData.empty() || data.numericData.empty()) {
            throw std::runtime_error("Need both categorical and numeric data for bar plot");
        }

        auto catIt = data.categoricalData.begin();
        auto numIt = data.numericData.begin();

        std::vector<std::string> categories = catIt->second;
        std::vector<double> values = numIt->second;

        // Take first min(n_categories, n_values) elements
        size_t n = std::min(categories.size(), values.size());
        categories.resize(n);
        values.resize(n);

        // Create bar plot
        auto b = plt::bar(values);
        
        // Apply style
        if (config.style.find("color") != config.style.end()) {
            std::string color = config.style.at("color");
            b->face_color(color);
            b->edge_color("black");
            b->line_width(1);
        }
        
        // Set x ticks to category labels
        std::vector<double> x_ticks;
        for (size_t i = 0; i < categories.size(); ++i) {
            x_ticks.push_back(i + 0.5);
        }
        plt::xticks(x_ticks);
        plt::xticklabels(categories);
        
        // Rotate labels if many categories
        if (n > 5) {
            plt::xtickangle(45);
        }

        // Add legend if needed
        addLegendIfNeeded(config);

        handlePlotOutput(config);
    }

    // Histogram with advanced features
    void Plotter::plotHistogram(const PlotData& data, const PlotConfig& config) {
        validatePlotData(data, config);
        setupFigure(config);

        if (data.numericData.empty()) {
            throw std::runtime_error("Need numeric data for histogram");
        }

        auto numIt = data.numericData.begin();
        std::vector<double> values = numIt->second;

        // Remove NaN values
        values.erase(std::remove_if(values.begin(), values.end(),
                    [](double v) { return std::isnan(v); }), values.end());

        if (values.empty()) {
            throw std::runtime_error("No valid numeric data for histogram");
        }

        // Auto-determine number of bins
        int bins = std::min(30, static_cast<int>(std::sqrt(values.size())));
        bins = std::max(bins, 5);

        // Create histogram
        auto h = plt::hist(values, bins);
        
        // Apply style
        if (config.style.find("color") != config.style.end()) {
            std::string color = config.style.at("color");
            h->face_color(color);
            h->edge_color("black");
            h->line_width(1);
        }
        
        // Add density curve
        plt::hold(true);
        auto kde_line = plt::hist(values, bins);
        // Matplot++ doesn't have direct KDE support, so we'll use a simplified approach
        plt::hold(false);

        // Add legend if needed
        addLegendIfNeeded(config);

        handlePlotOutput(config);
    }

    // Box Plot with advanced features
    void Plotter::plotBoxPlot(const PlotData& data, const PlotConfig& config) {
        validatePlotData(data, config);
        setupFigure(config);

        std::vector<std::vector<double>> boxData;
        std::vector<std::string> labels;

        // Extract numeric columns for box plot
        size_t count = 0;
        for (const auto& [colName, values] : data.numericData) {
            if (count >= 10) break;

            // Clean NaN values
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

        // Create box plot
        auto bx = plt::boxplot(boxData);
        
        // Apply style
        if (config.style.find("color") != config.style.end()) {
            std::string color = config.style.at("color");
	    bx->face_color(parseColor(color));
        }
        
        // Set labels
        std::vector<double> x_ticks;
        for (size_t i = 1; i <= labels.size(); ++i) {
            x_ticks.push_back(i);
        }
        plt::xticks(x_ticks);
        plt::xticklabels(labels);
        
        if (labels.size() > 5) {
            plt::xtickangle(45);
        }

        // Add jittered points for individual data
        plt::hold(true);
        for (size_t i = 0; i < boxData.size(); ++i) {
            std::vector<double> x_jitter(boxData[i].size());
            std::generate(x_jitter.begin(), x_jitter.end(), 
                         [i]() { return i + 1 + (rand() / (RAND_MAX + 1.0) * 0.4 - 0.2); });
            
            auto s = plt::scatter(x_jitter, boxData[i]);
            s->marker_size(10);
            s->color("red");
            s->marker_face(true);
            s->marker_face_color({0.5, 0.5, 0.5, 0.3});
            s->line_width(0.5);
        }
        plt::hold(false);

        // Add legend if needed
        addLegendIfNeeded(config);

        handlePlotOutput(config);
    }

    // Pie Chart
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

        // Take first min(n_categories, n_values) elements
        size_t n = std::min(categories.size(), values.size());
        categories.resize(n);
        values.resize(n);

        // Filter out zero values
        std::vector<double> nonZeroValues;
        std::vector<std::string> nonZeroCategories;
        for (size_t i = 0; i < values.size(); ++i) {
            if (values[i] > 0) {
                nonZeroValues.push_back(values[i]);
                nonZeroCategories.push_back(categories[i]);
            }
        }

        if (nonZeroValues.empty()) {
            throw std::runtime_error("No positive values for pie chart");
        }

        // Create pie chart
        auto p = plt::pie(nonZeroValues);
        
        // Apply labels and style
        // Matplot++ handles pie charts differently - labels are set separately
        // Add labels manually
        std::stringstream legend_text;
        for (size_t i = 0; i < nonZeroCategories.size(); ++i) {
            legend_text << nonZeroCategories[i] << ": " << nonZeroValues[i];
            if (i < nonZeroCategories.size() - 1) {
                legend_text << "\n";
            }
        }
        plt::text(1.5, 0, legend_text.str());

        handlePlotOutput(config);
    }

    // Heatmap
    void Plotter::plotHeatmap(const PlotData& data, const PlotConfig& config) {
        validatePlotData(data, config);
        setupFigure(config);

        // Convert data to matrix format
        std::vector<std::vector<double>> matrix;
        
        if (!data.numericData.empty()) {
            // Use numeric data as matrix
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

        // Create heatmap
        auto h = plt::heatmap(matrix);
        
        // Add colorbar
        plt::colorbar();
        
        // Add labels if available
        if (!data.columns.empty()) {
            std::vector<double> x_ticks, y_ticks;
            for (size_t i = 0; i < data.columns.size(); ++i) {
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

    // Area Plot
    void Plotter::plotArea(const PlotData& data, const PlotConfig& config) {
        validatePlotData(data, config);
        setupFigure(config);

        if (data.numericData.size() < 2) {
            throw std::runtime_error("Need at least 2 numeric columns for area plot");
        }

        auto itX = data.numericData.begin();
        auto itY = std::next(data.numericData.begin());

        std::vector<double> xData = itX->second;
        std::vector<double> yData = itY->second;

        // Remove NaN values
        std::vector<double> cleanX, cleanY;
        for (size_t i = 0; i < xData.size(); ++i) {
            if (!std::isnan(xData[i]) && !std::isnan(yData[i])) {
                cleanX.push_back(xData[i]);
                cleanY.push_back(yData[i]);
            }
        }

        if (cleanX.empty() || cleanY.empty()) {
            throw std::runtime_error("No valid data points for area plot");
        }

        // Sort data
        std::vector<size_t> indices(cleanX.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
                 [&cleanX](size_t a, size_t b) { return cleanX[a] < cleanX[b]; });

        std::vector<double> sortedX, sortedY;
        for (auto idx : indices) {
            sortedX.push_back(cleanX[idx]);
            sortedY.push_back(cleanY[idx]);
        }

        // Create area plot
        plt::hold(true);
        auto line = plt::plot(sortedX, sortedY);
        line->line_width(2);
        
        // Fill area (Matplot++ doesn't have direct fill_between like matplotlib)
        // We'll create a polygon for the filled area
        std::vector<double> fill_x = sortedX;
        std::vector<double> fill_y = sortedY;
        fill_x.push_back(sortedX.back());
        fill_y.push_back(0);
        fill_x.push_back(sortedX.front());
        fill_y.push_back(0);
        
        auto fill_area = plt::fill(fill_x, fill_y);
	fill_area->fill(true);
        fill_area->color({0.3, 0.3, 0.8});  
	fill_area->color({0.2, 0.4, 0.8, 0.3});  // r, g, b, alpha
        //fill_area.face_color({0.3, 0.3, 0.8, 0.3});
        //fill_area.edge_color("none");
        plt::hold(false);

        // Add legend if needed
        addLegendIfNeeded(config);

        handlePlotOutput(config);
    }

    // Stacked Bar Plot
    void Plotter::plotStackedBar(const PlotData& data, const PlotConfig& config) {
        validatePlotData(data, config);
        setupFigure(config);

        if (data.numericData.size() < 2) {
            throw std::runtime_error("Need at least 2 numeric columns for stacked bar plot");
        }

        // Convert numeric data to matrix
        std::vector<std::vector<double>> matrix;
        size_t n_rows = data.numericData.begin()->second.size();
        
        for (size_t i = 0; i < n_rows; ++i) {
            std::vector<double> row;
            for (const auto& [_, values] : data.numericData) {
                if (i < values.size()) {
                    row.push_back(values[i]);
                }
            }
            matrix.push_back(row);
        }

        // Create stacked bar plot (Matplot++ bar function handles stacked bars)
        auto b = plt::bar(matrix);
        b->bar_width(0.8);

        // Add legend
        std::vector<std::string> legendLabels;
        for (const auto& [name, _] : data.numericData) {
            legendLabels.push_back(name);
        }
        plt::legend(legendLabels);

        handlePlotOutput(config);
    }

    // Multi-Line Plot
    void Plotter::plotMultiLine(const PlotData& data, const PlotConfig& config) {
        validatePlotData(data, config);
        setupFigure(config);

        if (data.numericData.size() < 2) {
            throw std::runtime_error("Need at least 2 numeric columns for multi-line plot");
        }

        // Get x data (first column)
        auto xIt = data.numericData.begin();
        std::vector<double> xData = xIt->second;

        // Remove NaN from x
        xData.erase(std::remove_if(xData.begin(), xData.end(),
                    [](double v) { return std::isnan(v); }), xData.end());

        if (xData.empty()) {
            throw std::runtime_error("No valid x data for multi-line plot");
        }

        // Plot each y column
        auto colors = getColorPalette(data.numericData.size() - 1);
        size_t color_idx = 0;
        
        plt::hold(true);
        for (auto it = std::next(data.numericData.begin()); 
             it != data.numericData.end(); ++it) {
            
            std::vector<double> yData = it->second;
            
            // Match length with x data
            if (yData.size() > xData.size()) {
                yData.resize(xData.size());
            } else if (yData.size() < xData.size()) {
                xData.resize(yData.size());
            }
            
            // Remove pairs with NaN
            std::vector<double> cleanX, cleanY;
            for (size_t i = 0; i < xData.size() && i < yData.size(); ++i) {
                if (!std::isnan(xData[i]) && !std::isnan(yData[i])) {
                    cleanX.push_back(xData[i]);
                    cleanY.push_back(yData[i]);
                }
            }
            
            if (!cleanX.empty() && !cleanY.empty()) {
                auto line = plt::plot(cleanX, cleanY);
                line->color(colors[color_idx++ % colors.size()]);
                line->line_width(2);
                line->display_name(it->first);
            }
        }
        plt::hold(false);

        // Add legend
        plt::legend();

        handlePlotOutput(config);
    }

    // Auto-plot based on data characteristics
    void Plotter::autoPlot(const PlotData& data, const std::string& title) {
        PlotConfig config;
        config.title = title.empty() ? "Auto-generated Plot" : title;
        config.grid = true;

        // Decide plot type based on data characteristics
        if (data.numericData.size() >= 3) {
            // Good for heatmap or stacked bar
            if (data.rows.size() > 50) {
                config.type = PlotType::HEATMAP;
                plotHeatmap(data, config);
            } else {
                config.type = PlotType::STACKED_BAR;
                plotStackedBar(data, config);
            }
        } else if (data.numericData.size() == 2) {
            // Good for scatter or line plot
            if (data.rows.size() > 100) {
                config.type = PlotType::SCATTER;
                plotScatter(data, config);
            } else {
                config.type = PlotType::LINE;
                plotLine(data, config);
            }
        } else if (!data.categoricalData.empty() && !data.numericData.empty()) {
            // Good for bar or pie plot
            if (data.categoricalData.begin()->second.size() <= 8) {
                config.type = PlotType::PIE;
                plotPie(data, config);
            } else {
                config.type = PlotType::BAR;
                plotBar(data, config);
            }
        } else if (!data.numericData.empty()) {
            // Good for histogram or box plot
            if (data.numericData.begin()->second.size() > 30) {
                config.type = PlotType::HISTOGRAM;
                plotHistogram(data, config);
            } else {
                config.type = PlotType::BOXPLOT;
                plotBoxPlot(data, config);
            }
        } else {
            throw std::runtime_error("Cannot auto-determine plot type from data");
        }
    }

    void Plotter::showPlot() {
        plt::show();
    }

    void Plotter::savePlot(const std::string& filename) {
        plt::save(filename);
    }

    void Plotter::clearPlot() {
        plt::cla();
    }

    // Statistical plotting functions
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
                    // Calculate Pearson correlation
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
        
        plt::show();
    }

    void Plotter::plotDistribution(const PlotData& data, const std::string& column) {
        if (data.numericData.find(column) == data.numericData.end()) {
            throw std::runtime_error("Column not found or not numeric: " + column);
        }

        const std::vector<double>& values = data.numericData.at(column);

        // Create subplots
        plt::tiledlayout(1, 2);
        plt::nexttile();
        
        // Subplot 1: Histogram
        auto h = plt::hist(values, 20);
        h->face_color("skyblue");
        h->edge_color("black");
        h->line_width(1);
        
        plt::title("Histogram: " + column);
        plt::xlabel(column);
        plt::ylabel("Frequency");
        plt::grid(true);

        // Subplot 2: Box plot
        plt::nexttile();
        
        // Box plot
        //auto bx = plt::boxplot({values});
	auto bx = plt::boxplot(values); 
        //bx->face_color("blue");
	bx->face_color({0, 0, 1, 1}); // RGBA: red, green, blue, alpha
        
        plt::title("Box Plot: " + column);
        plt::grid(true);

        plt::show();
    }

    
    std::array<float, 4> Plotter::parseColor(const std::string& colorStr) {
        // Parse common color names and hex codes
        static const std::map<std::string, std::array<float, 4>> colorMap = {
            {"blue", {0.0f, 0.0f, 1.0f, 1.0f}},
            {"red", {1.0f, 0.0f, 0.0f, 1.0f}},
            {"green", {0.0f, 1.0f, 0.0f, 1.0f}},
            {"black", {0.0f, 0.0f, 0.0f, 1.0f}},
            {"white", {1.0f, 1.0f, 1.0f, 1.0f}},
            {"yellow", {1.0f, 1.0f, 0.0f, 1.0f}},
            {"cyan", {0.0f, 1.0f, 1.0f, 1.0f}},
            {"magenta", {1.0f, 0.0f, 1.0f, 1.0f}},
            {"gray", {0.5f, 0.5f, 0.5f, 1.0f}},
            {"lightblue", {0.68f, 0.85f, 0.9f, 1.0f}},
            {"skyblue", {0.53f, 0.81f, 0.92f, 1.0f}},
            {"orange", {1.0f, 0.65f, 0.0f, 1.0f}},
            {"purple", {0.5f, 0.0f, 0.5f, 1.0f}}
        };

        auto it = colorMap.find(colorStr);
        if (it != colorMap.end()) {
            return it->second;
        }

        // Try to parse hex color
        if (colorStr[0] == '#') {
            // Parse hex color - simple implementation
            // You might want to expand this for full hex parsing
            return {0.0f, 0.0f, 1.0f, 1.0f}; // Default to blue
        }

        // Default to blue
        return {0.0f, 0.0f, 1.0f, 1.0f};
    }

    void Plotter::plotTrendLine(const PlotData& data, const std::string& xColumn,
                               const std::string& yColumn) {
        if (data.numericData.find(xColumn) == data.numericData.end() ||
            data.numericData.find(yColumn) == data.numericData.end()) {
            throw std::runtime_error("Columns not found or not numeric");
        }

        const std::vector<double>& xData = data.numericData.at(xColumn);
        const std::vector<double>& yData = data.numericData.at(yColumn);

        // Clean data
        std::vector<double> cleanX, cleanY;
        for (size_t i = 0; i < std::min(xData.size(), yData.size()); ++i) {
            if (!std::isnan(xData[i]) && !std::isnan(yData[i])) {
                cleanX.push_back(xData[i]);
                cleanY.push_back(yData[i]);
            }
        }

        if (cleanX.empty() || cleanY.empty()) {
            throw std::runtime_error("No valid data points for trend line");
        }

        // Calculate linear regression
        double n = static_cast<double>(cleanX.size());
        double sum_x = std::accumulate(cleanX.begin(), cleanX.end(), 0.0);
        double sum_y = std::accumulate(cleanY.begin(), cleanY.end(), 0.0);
        double sum_xy = std::inner_product(cleanX.begin(), cleanX.end(), cleanY.begin(), 0.0);
        double sum_x2 = std::inner_product(cleanX.begin(), cleanX.end(), cleanX.begin(), 0.0);
        
        double slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        double intercept = (sum_y - slope * sum_x) / n;
        
        // Calculate R-squared
        double y_mean = sum_y / n;
        double ss_tot = 0.0, ss_res = 0.0;
        for (size_t i = 0; i < cleanX.size(); ++i) {
            double y_pred = slope * cleanX[i] + intercept;
            ss_tot += (cleanY[i] - y_mean) * (cleanY[i] - y_mean);
            ss_res += (cleanY[i] - y_pred) * (cleanY[i] - y_pred);
        }
        double r_squared = 1.0 - (ss_res / ss_tot);

        // Create plot
        plt::figure(true);
        plt::figure()->size(800, 600);
        plt::title("Trend Line Analysis");
        
        // Scatter plot
        auto s = plt::scatter(cleanX, cleanY);
        s->marker_size(30);
        s->marker_face(true);
        s->marker_face_color("blue");
        s->color("white");
        s->line_width(1);
        
        // Trend line
        double x_min = *std::min_element(cleanX.begin(), cleanX.end());
        double x_max = *std::max_element(cleanX.begin(), cleanX.end());
        std::vector<double> trend_x = {x_min, x_max};
        std::vector<double> trend_y = {slope * x_min + intercept, slope * x_max + intercept};
        
        plt::hold(true);
        auto trend_line = plt::plot(trend_x, trend_y);
        trend_line->color("red");
        trend_line->line_width(2);
        
        // Add equation and R²
        std::stringstream eq_ss;
        eq_ss << "y = " << std::fixed << std::setprecision(3) << slope << "x + " << intercept;
        eq_ss << "\nR² = " << std::fixed << std::setprecision(3) << r_squared;
        
        plt::text(x_min + (x_max - x_min) * 0.05,
                 *std::max_element(cleanY.begin(), cleanY.end()) * 0.9,
                 eq_ss.str());
        
        plt::hold(false);
        
        plt::xlabel(xColumn);
        plt::ylabel(yColumn);
        plt::grid(true);
        
        plt::show();
    }

    void Plotter::addLegendIfNeeded(const PlotConfig& config) {
        if (config.legend && !config.seriesNames.empty()) {
            plt::legend(config.seriesNames);
        }
    }

    void Plotter::handlePlotOutput(const PlotConfig& config) {
        if (!config.outputFile.empty()) {
            plt::save(config.outputFile);
        } else {
            plt::show();
        }
    }

    PlotData Plotter::convertToPlotData(const ExecutionEngine::ResultSet& result) {
    PlotData data;
    data.columns = result.columns;
    data.rows = result.rows;

    // Extract column indices
    std::map<std::string, size_t> columnIndices;
    for (size_t i = 0; i < result.columns.size(); ++i) {
        columnIndices[result.columns[i]] = i;
    }

    // Process each column
    for (const auto& colName : result.columns) {
        if (columnIndices.find(colName) == columnIndices.end()) {
            continue;
        }

        size_t colIdx = columnIndices[colName];
        std::vector<std::string> columnValues;

        // Extract column values
        for (const auto& row : result.rows) {
            if (colIdx < row.size()) {
                columnValues.push_back(row[colIdx]);
            }
        }

        // Check if column is numeric
        if (isNumericColumn(columnValues)) {
            data.numericData[colName] = convertToNumeric(columnValues);
        } else if (isIntegerColumn(columnValues)) {
            data.integerData[colName] = convertToInteger(columnValues);
        } else if (isBooleanColumn(columnValues)) {
            data.booleanData[colName] = convertToBoolean(columnValues);
        } else {
            data.categoricalData[colName] = columnValues;
        }
    }

    return data;
}

bool Plotter::isIntegerColumn(const std::vector<std::string>& values) {
    if (values.empty()) return false;

    size_t integerCount = 0;
    for (const auto& val : values) {
        try {
            std::stoi(val);
            integerCount++;
        } catch (...) {
            // Not integer
        }
    }

    return (static_cast<double>(integerCount) / values.size()) > 0.8;
}

bool Plotter::isBooleanColumn(const std::vector<std::string>& values) {
    if (values.empty()) return false;

    size_t booleanCount = 0;
    for (const auto& val : values) {
        std::string lowerVal = val;
        std::transform(lowerVal.begin(), lowerVal.end(), lowerVal.begin(), ::tolower);
        if (lowerVal == "true" || lowerVal == "false" ||
            lowerVal == "1" || lowerVal == "0" ||
            lowerVal == "yes" || lowerVal == "no") {
            booleanCount++;
        }
    }

    return (static_cast<double>(booleanCount) / values.size()) > 0.8;
}

std::vector<int> Plotter::convertToInteger(const std::vector<std::string>& values) {
    std::vector<int> intValues;
    intValues.reserve(values.size());

    for (const auto& val : values) {
        try {
            intValues.push_back(std::stoi(val));
        } catch (...) {
            intValues.push_back(0); // Default for invalid integers
        }
    }

    return intValues;
}

std::vector<bool> Plotter::convertToBoolean(const std::vector<std::string>& values) {
    std::vector<bool> boolValues;
    boolValues.reserve(values.size());

    for (const auto& val : values) {
        std::string lowerVal = val;
        std::transform(lowerVal.begin(), lowerVal.end(), lowerVal.begin(), ::tolower);

        if (lowerVal == "true" || lowerVal == "1" || lowerVal == "yes") {
            boolValues.push_back(true);
        } else {
            boolValues.push_back(false);
        }
    }

    return boolValues;
}

} // namespace Visualization
