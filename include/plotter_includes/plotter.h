#ifndef PLOTTER_H
#define PLOTTER_H

#include "parser.h"
#include "executionengine_main.h"
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <functional>
#include <stdexcept>


namespace Visualization {

    // Enhanced Plot types supported by Matplot++
    enum class PlotType {
        LINE,
        SCATTER,
        BAR,
        HISTOGRAM,
        BOXPLOT,
        PIE,
        HEATMAP,
        AREA,
        STACKED_BAR,
        MULTI_LINE,
        VIOLIN,
        CONTOUR,
        SURFACE,
        WIREFRAME,
        HISTOGRAM_2D,
        PARALLEL_COORDINATES,
        RADAR,
        QUIVER,
        STREAMPLOT
    };

    // Enhanced Plot configuration for Matplot++
    struct PlotConfig {
        PlotType type;
        std::string title;
        std::string xLabel;
        std::string yLabel;
        std::string zLabel; // For 3D plots
        std::vector<std::string> seriesNames;
        std::map<std::string, std::string> style = {
            {"color", "blue"},
            {"linewidth", "2"},
            {"marker", "o"},
            {"linestyle", "-"},
            {"markersize", "8"},
            {"facecolor", "lightblue"},
            {"edgecolor", "black"},
            {"colormap", "viridis"},
            {"alpha", "1.0"},
            {"barwidth", "0.8"},
            {"bins", "auto"}
        };
        bool grid = true;
        bool legend = true;
        bool show_values = false;
        bool normalized = false;
        bool stacked = false;
        bool filled = true;
        std::string outputFile = ""; // Empty for display, path for save
        int width = 12; // inches
        int height = 8; // inches
        double dpi = 100;
        
        // 3D plot settings
        bool threeD = false;
        double azimuth = 30.0;
        double elevation = 30.0;
        
        // Statistical plot settings
        bool show_kde = true;
        bool show_outliers = true;
        double confidence_interval = 0.95;
    };

    // Enhanced Data structure for plotting
    struct PlotData {
        std::vector<std::string> columns;
        std::vector<std::vector<std::string>> rows;
        
        // Data organized by column for easy access
        std::map<std::string, std::vector<double>> numericData; // Column -> numeric values
        std::map<std::string, std::vector<std::string>> categoricalData; // Column -> categorical labels
        std::map<std::string, std::vector<int>> integerData; // Column -> integer values
        std::map<std::string, std::vector<bool>> booleanData; // Column -> boolean values
        
        // Metadata for data types
        std::map<std::string, std::string> columnTypes; // Column -> type info
        
        // Timestamp data support
        std::map<std::string, std::vector<std::chrono::system_clock::time_point>> timestampData;
        
        // 3D data support
        std::map<std::string, std::vector<std::vector<double>>> matrixData; // For 2D arrays
        
        // Clear all data
        void clear() {
            columns.clear();
            rows.clear();
            numericData.clear();
            categoricalData.clear();
            integerData.clear();
            booleanData.clear();
            columnTypes.clear();
            timestampData.clear();
            matrixData.clear();
        }
        
        // Check if data is empty
        bool empty() const {
            return rows.empty() && numericData.empty() && categoricalData.empty();
        }
        
        // Get column by name with type checking
        template<typename T>
        const std::vector<T>& getColumn(const std::string& name) const {
            if constexpr (std::is_same_v<T, double>) {
                auto it = numericData.find(name);
                if (it != numericData.end()) return it->second;
            } else if constexpr (std::is_same_v<T, std::string>) {
                auto it = categoricalData.find(name);
                if (it != categoricalData.end()) return it->second;
            } else if constexpr (std::is_same_v<T, int>) {
                auto it = integerData.find(name);
                if (it != integerData.end()) return it->second;
            } else if constexpr (std::is_same_v<T, bool>) {
                auto it = booleanData.find(name);
                if (it != booleanData.end()) return it->second;
            }
            throw std::runtime_error("Column not found or wrong type: " + name);
        }
        
        // Get column names by type
        std::vector<std::string> getNumericColumns() const {
            std::vector<std::string> result;
            for (const auto& [name, _] : numericData) {
                result.push_back(name);
            }
            return result;
        }
        
        std::vector<std::string> getCategoricalColumns() const {
            std::vector<std::string> result;
            for (const auto& [name, _] : categoricalData) {
                result.push_back(name);
            }
            return result;
        }
    };

    // Main plotter class using Matplot++
    class Plotter {
    public:
        Plotter();
        ~Plotter();

        // Initialize and cleanup
        void initializePlotter();
        void finalizePlotter();
        
        // Data conversion and preparation
        PlotData convertToPlotData(const ExecutionEngine::ResultSet& result,
                                  const std::vector<std::string>& xColumns,
                                  const std::vector<std::string>& yColumns);
        
        PlotData convertToPlotData(const ExecutionEngine::ResultSet& result);
        
        // Advanced data processing
        void detectColumnTypes(PlotData& data);
        void cleanData(PlotData& data);
        void normalizeData(PlotData& data);
        void extractFeatures(PlotData& data);
        
        // Basic plot functions
        void plotLine(const PlotData& data, const PlotConfig& config);
        void plotScatter(const PlotData& data, const PlotConfig& config);
        void plotBar(const PlotData& data, const PlotConfig& config);
        void plotHistogram(const PlotData& data, const PlotConfig& config);
        void plotBoxPlot(const PlotData& data, const PlotConfig& config);
        void plotPie(const PlotData& data, const PlotConfig& config);
        void plotHeatmap(const PlotData& data, const PlotConfig& config);
        void plotArea(const PlotData& data, const PlotConfig& config);
        void plotStackedBar(const PlotData& data, const PlotConfig& config);
        void plotMultiLine(const PlotData& data, const PlotConfig& config);
        
        // Advanced plot functions (Matplot++ specific)
        void plotViolin(const PlotData& data, const PlotConfig& config);
        void plotContour(const PlotData& data, const PlotConfig& config);
        void plotSurface(const PlotData& data, const PlotConfig& config);
        void plotWireframe(const PlotData& data, const PlotConfig& config);
        void plotHistogram2D(const PlotData& data, const PlotConfig& config);
        void plotParallelCoordinates(const PlotData& data, const PlotConfig& config);
        void plotRadar(const PlotData& data, const PlotConfig& config);
        void plotQuiver(const PlotData& data, const PlotConfig& config);
        void plotStreamplot(const PlotData& data, const PlotConfig& config);
        
        // Statistical plotting
        void plotCorrelationMatrix(const PlotData& data);
        void plotDistribution(const PlotData& data, const std::string& column);
        void plotTrendLine(const PlotData& data, const std::string& xColumn,
                          const std::string& yColumn);
        void plotQQPlot(const PlotData& data, const std::string& column);
        void plotResiduals(const PlotData& data, const std::string& xColumn,
                          const std::string& yColumn);
        
        // Time series plotting
        void plotTimeSeries(const PlotData& data, const std::string& timeColumn,
                           const std::string& valueColumn, const PlotConfig& config);
        void plotCandlestick(const PlotData& data, const PlotConfig& config);
        
        // Geographical plotting
        void plotGeoMap(const PlotData& data, const std::string& latColumn,
                       const std::string& lonColumn, const PlotConfig& config);
        
        // Interactive plotting
        void createInteractivePlot(const PlotData& data, const PlotConfig& config);
        void addWidgets(const PlotConfig& config);
        
        // Multi-plot layouts
        void createDashboard(const std::vector<PlotData>& datasets,
                            const std::vector<PlotConfig>& configs,
                            int rows, int cols);
        
        // Animation support
        void createAnimation(const std::vector<PlotData>& frames,
                            const PlotConfig& config, int fps = 10);
        
        // Auto-plot based on data characteristics with AI-like detection
        void autoPlot(const PlotData& data, const std::string& title = "");
        
        // Plot with AI-recommended settings
        void smartPlot(const PlotData& data, const std::string& title = "");
        
        // Output control
        void showPlot();
        void savePlot(const std::string& filename);
        void savePlot(const std::string& filename, const std::string& format);
        void clearPlot();
        void closeAll();
        
        // Style management
        void setStyle(const std::string& styleName);
        void setColorPalette(const std::string& paletteName);
        void setFont(const std::string& fontName, int size = 12);
	std::array<float, 4> parseColor(const std::string& colorStr);
        
        // Utility functions
        void addLegendIfNeeded(const PlotConfig& config);
        void handlePlotOutput(const PlotConfig& config);
        
    private:
        // Helper methods
        bool isNumericColumn(const std::vector<std::string>& values);
        bool isIntegerColumn(const std::vector<std::string>& values);
        bool isBooleanColumn(const std::vector<std::string>& values);
        bool isDateColumn(const std::vector<std::string>& values);
        bool isDateTimeColumn(const std::vector<std::string>& values);
        
        std::vector<double> convertToNumeric(const std::vector<std::string>& values);
        std::vector<int> convertToInteger(const std::vector<std::string>& values);
        std::vector<bool> convertToBoolean(const std::vector<std::string>& values);
        
        // Data validation and cleaning
        void validatePlotData(const PlotData& data, const PlotConfig& config);
        void validateNumericData(const std::vector<double>& data, const std::string& columnName);
        void validateCategoricalData(const std::vector<std::string>& data, const std::string& columnName);
        
        // Plot setup
        void setupFigure(const PlotConfig& config);
        void setup3DAxes(const PlotConfig& config);
        void applyStyle(const PlotConfig& config);
        
        // Color management
        std::vector<std::string> getColorPalette(int n);
        std::vector<std::string> getSequentialPalette(int n);
        std::vector<std::string> getDivergingPalette(int n);
        std::vector<std::string> getQualitativePalette(int n);
        
        // Statistical calculations
        std::vector<double> calculateKDE(const std::vector<double>& data, int gridPoints = 100);
        std::pair<double, double> linearRegression(const std::vector<double>& x, 
                                                  const std::vector<double>& y);
        double calculateRSquared(const std::vector<double>& x, 
                               const std::vector<double>& y,
                               double slope, double intercept);
        std::vector<std::vector<double>> calculateCorrelationMatrix(const PlotData& data);
        
        // Data transformations
        std::vector<double> normalize(const std::vector<double>& data);
        std::vector<double> standardize(const std::vector<double>& data);
        std::vector<double> logTransform(const std::vector<double>& data);
        
        // Plotter state
        bool plotterInitialized;
        int currentFigureId;
        
        // Configuration cache
        std::string currentStyle;
        std::string currentPalette;
        
        // Plot history for undo/redo
        struct PlotState {
            PlotData data;
            PlotConfig config;
            std::string timestamp;
        };
        
        std::vector<PlotState> plotHistory;
        size_t currentHistoryIndex;
        
        // Error handling
        std::vector<std::string> errorLog;
        void logError(const std::string& error);
        void clearErrors();
        
        // Performance tracking
        struct PerformanceMetrics {
            size_t dataPointsProcessed;
            double renderTime;
            double dataConversionTime;
            std::chrono::steady_clock::time_point startTime;
        };
        
        PerformanceMetrics currentMetrics;
        void startTimer();
        void stopTimer();
        void printMetrics() const;
    };

    // Enhanced Plot query extension for AST with Matplot++ support
    class PlotStatement : public AST::Statement {
    public:
        enum class PlotSubType {
            STANDARD,
            CORRELATION,
            DISTRIBUTION,
            TREND,
            TIME_SERIES,
            QQ_PLOT,
            RESIDUALS,
            ANIMATION,
            INTERACTIVE,
            DASHBOARD,
            GEO_MAP,
            CANDLESTICK
        };
        
        enum class OutputFormat {
            DISPLAY,
            PNG,
            PDF,
            SVG,
            EPS,
            JPG,
            GIF,
            MP4,
            HTML
        };

        PlotSubType subType = PlotSubType::STANDARD;
        OutputFormat outputFormat = OutputFormat::DISPLAY;
        std::unique_ptr<AST::SelectStatement> query;
        PlotConfig config;
        std::vector<std::string> xColumns;
        std::vector<std::string> yColumns;
        std::vector<std::string> zColumns; // For 3D plots
        std::string targetColumn; // For distribution/trend plots
        std::string timeColumn; // For time series
        std::string groupColumn; // For grouping
        std::string animationColumn; // For animations
        int animationFPS = 10;
        int dashboardRows = 2;
        int dashboardCols = 2;
        
        // Interactive controls
        struct Control {
            std::string type; // "slider", "dropdown", "checkbox", "button"
            std::string name;
            std::string label;
            double minValue = 0.0;
            double maxValue = 1.0;
            double step = 0.1;
            double defaultValue = 0.5;
            std::vector<std::string> options;
        };
        
        std::vector<Control> controls;

        PlotStatement() = default;
        
        // Helper methods
        bool is3DPlot() const {
            return config.threeD || !zColumns.empty();
        }
        
        bool isAnimated() const {
            return subType == PlotSubType::ANIMATION || !animationColumn.empty();
        }
        
        bool isInteractive() const {
            return subType == PlotSubType::INTERACTIVE || !controls.empty();
        }
        
        std::string getOutputFileExtension() const {
            switch (outputFormat) {
                case OutputFormat::PNG: return ".png";
                case OutputFormat::PDF: return ".pdf";
                case OutputFormat::SVG: return ".svg";
                case OutputFormat::EPS: return ".eps";
                case OutputFormat::JPG: return ".jpg";
                case OutputFormat::GIF: return ".gif";
                case OutputFormat::MP4: return ".mp4";
                case OutputFormat::HTML: return ".html";
                default: return "";
            }
        }
    };

    // Utility functions for plotting
    namespace PlotUtils {
        
        // Color conversion
        std::string rgbToHex(int r, int g, int b);
        std::tuple<int, int, int> hexToRgb(const std::string& hex);
        
        // Data sampling for large datasets
        PlotData sampleData(const PlotData& data, size_t maxPoints = 10000, 
                           bool random = true);
        
        // Data aggregation
        PlotData aggregateData(const PlotData& data, const std::string& groupColumn,
                              const std::vector<std::string>& aggColumns,
                              const std::vector<std::string>& aggFunctions);
        
        // Outlier detection
        std::vector<bool> detectOutliers(const std::vector<double>& data, 
                                        double threshold = 3.0);
        
        // Data smoothing
        std::vector<double> smoothData(const std::vector<double>& data, 
                                      int windowSize = 5);
        
        // Interpolation
        std::vector<double> interpolateData(const std::vector<double>& x,
                                           const std::vector<double>& y,
                                           const std::vector<double>& newX);
        
        // Statistical summaries
        struct SummaryStats {
            double mean;
            double median;
            double stddev;
            double min;
            double max;
            double q1;
            double q3;
            double iqr;
            size_t count;
            size_t missing;
        };
        
        SummaryStats calculateStats(const std::vector<double>& data);
        
        // Data binning
        std::vector<double> createBins(const std::vector<double>& data, 
                                      int numBins = 10);
        
        // Formatting
        std::string formatNumber(double value, int precision = 3);
        std::string formatDateTime(const std::chrono::system_clock::time_point& tp);
        
        // File operations
        bool savePlotData(const PlotData& data, const std::string& filename);
        PlotData loadPlotData(const std::string& filename);
        
        // Validation
        bool validatePlotConfiguration(const PlotConfig& config);
        bool validatePlotDataForType(const PlotData& data, PlotType type);
    };

} // namespace Visualization

#endif // PLOTTER_H
