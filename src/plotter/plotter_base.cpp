#include "plotter_includes/plotter.h"
#include <matplot/matplot.h>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <map>
#include <regex>
#include <chrono>

namespace plt = matplot;

namespace Visualization {

    Plotter::Plotter() : plotterInitialized(false), currentFigureId(0), currentHistoryIndex(0) {
        initializePlotter();
    }

    Plotter::~Plotter() {
        finalizePlotter();
    }

    void Plotter::initializePlotter() {
        if (!plotterInitialized) {
            plotterInitialized = true;
        }
    }

    void Plotter::finalizePlotter() {
        if (plotterInitialized) {
            plotterInitialized = false;
        }
    }

    // Data conversion and preparation
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
		data.columnTypes[colName] = "numeric";
            } else if (isIntegerColumn(columnValues)) {
                data.integerData[colName] = convertToInteger(columnValues);
		data.columnTypes[colName] = "integer";
            } else if (isBooleanColumn(columnValues)) {
                data.booleanData[colName] = convertToBoolean(columnValues);
		data.columnTypes[colName] = "boolean";
            } else if (isDateColumn(columnValues)) {
                // Store as categorical for now
                data.categoricalData[colName] = columnValues;
		data.columnTypes[colName] = "date";
            } else if (isDateTimeColumn(columnValues)) {
                // Store as categorical for now
                data.categoricalData[colName] = columnValues;
		data.columnTypes[colName] = "categorical";
	    } else if (isLatitudeColumn(columnValues)) {
                data.latitudeData[colName] = convertToLatitude(columnValues);
                data.columnTypes[colName] = "latitude";
            } else if (isLongitudeColumn(columnValues)) {
                data.longitudeData[colName] = convertToLongitude(columnValues);
                data.columnTypes[colName] = "longitude";
            } else {
                data.categoricalData[colName] = columnValues;
		data.columnTypes[colName] = "categorical";

                // Check if it might be region data (country names, state names, etc.)
                // Simple heuristic: if column name contains "country", "state", "region", "city"
                std::string lowerName = colName;
                std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), ::tolower);
                if (lowerName.find("country") != std::string::npos ||
                    lowerName.find("state") != std::string::npos ||
                    lowerName.find("region") != std::string::npos ||
                    lowerName.find("city") != std::string::npos ||
                    lowerName.find("county") != std::string::npos) {
                    data.regionData[colName] = columnValues;
		}
            }

            // Store column type info
            if (data.numericData.find(colName) != data.numericData.end()) {
                data.columnTypes[colName] = "numeric";
            } else if (data.categoricalData.find(colName) != data.categoricalData.end()) {
                data.columnTypes[colName] = "categorical";
            } else if (data.integerData.find(colName) != data.integerData.end()) {
                data.columnTypes[colName] = "integer";
            } else if (data.booleanData.find(colName) != data.booleanData.end()) {
                data.columnTypes[colName] = "boolean";
            }
        }

        return data;
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
            } else if (isDateColumn(columnValues)) {
                // Store as categorical for now
                data.categoricalData[colName] = columnValues;
            } else if (isDateTimeColumn(columnValues)) {
                // Store as categorical for now
                data.categoricalData[colName] = columnValues;
            } else {
                data.categoricalData[colName] = columnValues;
            }

            // Store column type info
            if (data.numericData.find(colName) != data.numericData.end()) {
                data.columnTypes[colName] = "numeric";
            } else if (data.categoricalData.find(colName) != data.categoricalData.end()) {
                data.columnTypes[colName] = "categorical";
            } else if (data.integerData.find(colName) != data.integerData.end()) {
                data.columnTypes[colName] = "integer";
            } else if (data.booleanData.find(colName) != data.booleanData.end()) {
                data.columnTypes[colName] = "boolean";
            }
        }

        return data;
    }

    // Advanced data processing
    void Plotter::detectColumnTypes(PlotData& data) {
        // This is already done in convertToPlotData
    }

    void Plotter::cleanData(PlotData& data) {
        // Remove NaN values from numeric data
        for (auto& [colName, values] : data.numericData) {
            values.erase(std::remove_if(values.begin(), values.end(),
                        [](double v) { return std::isnan(v); }), values.end());
        }

        // Remove empty strings from categorical data
        for (auto& [colName, values] : data.categoricalData) {
            values.erase(std::remove_if(values.begin(), values.end(),
                        [](const std::string& v) { return v.empty(); }), values.end());
        }
    }

    void Plotter::normalizeData(PlotData& data) {
        for (auto& [colName, values] : data.numericData) {
            if (values.empty()) continue;

            // Find min and max
            double min_val = *std::min_element(values.begin(), values.end());
            double max_val = *std::max_element(values.begin(), values.end());

            if (max_val > min_val) {
                // Normalize to [0, 1]
                for (auto& v : values) {
                    v = (v - min_val) / (max_val - min_val);
                }
            }
        }
    }

    void Plotter::extractFeatures(PlotData& data) {
        // Extract statistical features from numeric data
        for (const auto& [colName, values] : data.numericData) {
            if (values.size() < 2) continue;

            // Calculate basic statistics
            double sum = std::accumulate(values.begin(), values.end(), 0.0);
            double mean = sum / values.size();

            double variance = 0.0;
            for (double v : values) {
                variance += (v - mean) * (v - mean);
            }
            variance /= values.size();
            double stddev = sqrt(variance);

            // Store as metadata (could be added to PlotData structure)
            // For now, we just calculate but don't store
        }
    }

    // Output control
    void Plotter::showPlot() {
        plt::show();
    }

    void Plotter::savePlot(const std::string& filename) {
        plt::save(filename);
    }

    void Plotter::savePlot(const std::string& filename, const std::string& format) {
        std::string fullFilename = filename;
        if (!format.empty()) {
            // Ensure filename has correct extension
            size_t dotPos = fullFilename.find_last_of('.');
            if (dotPos == std::string::npos) {
                fullFilename += "." + format;
            }
        }
        plt::save(fullFilename);
    }

    void Plotter::clearPlot() {
        plt::cla();
    }

    void Plotter::closeAll() {

    }

    void Plotter::setFont(const std::string& fontName, int size) {
        // Note: Matplot++ has limited font control
        // This is a placeholder for future implementation
    }

    // Style management
    void Plotter::setStyle(const std::string& styleName) {
        // Set plotting style (similar to matplotlib styles)
        currentStyle = styleName;

        // Note: Matplot++ has limited built-in styles compared to matplotlib
        // We can implement custom style presets here
        if (styleName == "default" || styleName == "classic") {
            // Default style - no changes needed
        } else if (styleName == "ggplot") {
            // ggplot-like style
            plt::gca()->box(true);
        } else if (styleName == "seaborn" || styleName == "dark_background") {
            // Dark background style
            plt::gcf()->color({0.15f, 0.15f, 0.15f, 1.0f});
            plt::gca()->color({0.2f, 0.2f, 0.2f, 1.0f});
        } else if (styleName == "fivethirtyeight") {
            // FiveThirtyEight style
            plt::gca()->box(true);
        }
    }

    void Plotter::setColorPalette(const std::string& paletteName) {
        currentPalette = paletteName;

        // Note: Matplot++ doesn't have direct palette management like matplotlib
        // We store the palette name for use in getColorPalette()
    }

    // Auto-plot based on data characteristics
    void Plotter::autoPlot(const PlotData& data, const std::string& title) {
        PlotConfig config;
        config.title = title.empty() ? "Auto-generated Plot" : title;
        config.style.grid = true;
        config.style.legend = true;

        // Analyze data characteristics
        size_t numericCols = data.numericData.size();
        size_t categoricalCols = data.categoricalData.size();
        size_t totalRows = data.rows.size();

        if (numericCols >= 3 && totalRows > 50) {
            // Good for heatmap
            config.type = PlotType::HEATMAP;
            plotHeatmap(data, config);
        } else if (numericCols >= 3 && totalRows <= 50) {
            // Good for stacked bar
            config.type = PlotType::STACKED_BAR;
            plotStackedBar(data, config);
        } else if (numericCols == 2) {
            // Good for scatter or line plot
            if (totalRows > 100) {
                config.type = PlotType::SCATTER;
                plotScatter(data, config);
            } else {
                config.type = PlotType::LINE;
                plotLine(data, config);
            }
        } else if (categoricalCols > 0 && numericCols > 0) {
            // Good for bar or pie plot
            if (data.categoricalData.begin()->second.size() <= 8) {
                config.type = PlotType::PIE;
                plotPie(data, config);
            } else {
                config.type = PlotType::BAR;
                plotBar(data, config);
            }
        } else if (numericCols == 1) {
            // Good for histogram or box plot
            if (totalRows > 30) {
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

    void Plotter::smartPlot(const PlotData& data, const std::string& title) {
        // Enhanced auto-plot with AI-like recommendations
        PlotConfig config;
        config.title = title.empty() ? "Smart Plot" : title;

         // Create a copy of the data for analysis
         PlotData dataCopy = data;

        // Analyze data more thoroughly
        detectColumnTypes(dataCopy);
        cleanData(dataCopy);

        // Choose plot type based on comprehensive analysis
        size_t numericCols = data.numericData.size();
        size_t categoricalCols = data.categoricalData.size();
        size_t totalRows = data.rows.size();

        // Check for time series data
        bool hasDates = false;
        for (const auto& [colName, values] : data.categoricalData) {
            if (isDateColumn(values) || isDateTimeColumn(values)) {
                hasDates = true;
                config.type = PlotType::LINE; // Default to line for time series
                break;
            }
        }

        if (hasDates && numericCols >= 1) {
            // Time series plot
            std::string timeCol;
            std::string valueCol;

            // Find date/time column
            for (const auto& [colName, values] : data.categoricalData) {
                if (isDateColumn(values) || isDateTimeColumn(values)) {
                    timeCol = colName;
                    break;
                }
            }

            // Find first numeric column
            if (!data.numericData.empty()) {
                valueCol = data.numericData.begin()->first;
            }

            if (!timeCol.empty() && !valueCol.empty()) {
                plotTimeSeries(data, timeCol, valueCol, config);
                return;
            }
        }

        // Fall back to regular auto-plot
        autoPlot(data, title);
    }

    // Error handling and performance tracking
    void Plotter::logError(const std::string& error) {
        errorLog.push_back(error);
    }

    void Plotter::clearErrors() {
        errorLog.clear();
    }

    void Plotter::startTimer() {
        currentMetrics.startTime = std::chrono::steady_clock::now();
    }

    void Plotter::stopTimer() {
        auto endTime = std::chrono::steady_clock::now();
        currentMetrics.renderTime = std::chrono::duration<double>(endTime - currentMetrics.startTime).count();
    }

    void Plotter::printMetrics() const {
        std::cout << "Plotter Metrics:" << std::endl;
        std::cout << "  Data points processed: " << currentMetrics.dataPointsProcessed << std::endl;
        std::cout << "  Render time: " << currentMetrics.renderTime << " seconds" << std::endl;
        std::cout << "  Data conversion time: " << currentMetrics.dataConversionTime << " seconds" << std::endl;
    }

} // namespace Visualization
