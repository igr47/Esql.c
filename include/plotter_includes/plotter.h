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
#include <array>
#include <chrono>
#include <limits>

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
        
        // Enhanced style options
        struct Style {
            // Color options
            std::string color = "blue";
            std::string facecolor = "lightblue";
            std::string edgecolor = "black";
            std::string markercolor = "blue";
            std::string markerfacecolor = "white";
	    std::string xlabel;
	    std::string ylabel;
            std::vector<std::string> colors; // For multiple series
            
            // Line options
            double linewidth = 2.0;
            std::string linestyle = "-";
            double alpha = 1.0;
            
            // Marker options
            std::string marker = "o";
            double markersize = 8.0;
            
            // Bar/Histogram options
            double barwidth = 0.8;
            std::string baralign = "center";
            bool stacked = false;
            
            // Histogram specific
            int bins = 30;
            std::string histtype = "bar";
            bool cumulative = false;
            bool density = false;
            
            // Box plot specific
            bool showfliers = true; // Show outliers
            double fliersize = 5.0;
            std::string fliermarker = "+";
            double whiskerwidth = 0.5;
            
            // Pie chart specific
            std::vector<double> explode;
            bool autopct = false;
            std::string startangle = "0";
            bool shadow = false;
            
            // Heatmap specific
            std::string colormap = "viridis";
            bool annotate = false;
            std::string fmt = ".2f";
            
            // Grid and layout
            bool grid = true;
            std::string gridstyle = "-";
            double gridalpha = 0.3;
            std::string gridcolor = "gray";
            
            // Legend
            bool legend = true;
            std::string legend_loc = "best";
            int legend_ncol = 1;
            double legend_fontsize = 10.0;
            
            // Figure size
            double figwidth = 12.0;
            double figheight = 8.0;
            double dpi = 100.0;
            
            // Axis limits
            double xmin = std::numeric_limits<double>::quiet_NaN();
            double xmax = std::numeric_limits<double>::quiet_NaN();
            double ymin = std::numeric_limits<double>::quiet_NaN();
            double ymax = std::numeric_limits<double>::quiet_NaN();
            double zmin = std::numeric_limits<double>::quiet_NaN();
            double zmax = std::numeric_limits<double>::quiet_NaN();
            
            // Tick parameters
            double xtick_rotation = 0.0;
            double ytick_rotation = 0.0;
            double tick_fontsize = 10.0;
            
            // Title and label font sizes
            double title_fontsize = 14.0;
            double xlabel_fontsize = 12.0;
            double ylabel_fontsize = 12.0;
            
            // 3D plot settings
            double azimuth = 30.0;
            double elevation = 30.0;
            bool view_init_set = false;
            
            // Statistical settings
            double confidence_interval = 0.95;
            bool show_kde = true;
            bool rug = false;
            
            // Animation settings
            int fps = 10;
            bool repeat = true;
            
            // Interactive settings
            bool interactive = false;
            std::string toolbar = "toolbar2";
            
            // Save settings
            std::string save_format = "png";
            bool bbox_inches_tight = true;
            double pad_inches = 0.1;
            
            // Custom style parser
            void parseFromMap(const std::map<std::string, std::string>& styleMap);
        };
        
        Style style;
        std::string outputFile = "";
        bool threeD = false;
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
        
        // Statistical plotting
        void plotCorrelationMatrix(const PlotData& data);
        void plotDistribution(const PlotData& data, const std::string& column);
        void plotTrendLine(const PlotData& data, const std::string& xColumn,
                          const std::string& yColumn);
        void plotQQPlot(const PlotData& data, const std::string& column);
        //void plotResiduals(const PlotData& data, const std::string& xColumn,const std::string& yColumn);
        
        // Time series plotting
        void plotTimeSeries(const PlotData& data, const std::string& timeColumn,
                           const std::string& valueColumn, const PlotConfig& config);
        
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
        std::pair<std::vector<double>, std::vector<double>> calculateKDE(const std::vector<double>& data, int gridPoints = 100);
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
        
        // Utility functions
        std::string parseLineStyle(const std::string& styleStr);
        std::string parseMarker(const std::string& markerStr);
        std::vector<double> linspace(double start, double end, size_t num);
        double erfinv(double x);
        
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
