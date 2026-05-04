#ifndef REAL_TIME_PLOTTER_PARSER_H
#define REAL_TIME_PLOTTER_PARSER_H
#include "real_time_plotter_parser.h"
#include "parser.h"
#include <algorithm>
#include <string>
#include <vector>
#include <memory>

namespace AST {
class RealTimePlotStatement : public Statement {
public:
    // Plot type enumeration
    enum class PlotType {
        LINE,
        SCATTER,
        BAR,
        AREA,
        HISTOGRAM,
        MULTI_LINE,
        STACKED_BAR
    };
    
    // Plot configuration
    PlotType type = PlotType::LINE;
    std::string title;
    std::string xLabel = "Time";
    std::string yLabel = "Value";
    std::string outputFile;
    
    // Real-time configuration
    int refreshIntervalMs = 100;    // Milliseconds between updates
    int historyWindowSec = 60;       // How many seconds of history to show
    int maxDataPoints = 1000;        // Max points to keep in memory
    
    // Column mappings
    std::string xColumn;             // X-axis column (usually timestamp)
    std::vector<std::string> yColumns;  // Y-axis columns to plot
    std::string groupColumn;         // Optional grouping column for multi-line
    
    // Visual settings
    std::vector<std::string> colors; // Custom colors per series
    bool showGrid = true;
    bool showLegend = true;
    bool autoRangeY = true;
    double yMin = 0;
    double yMax = 100;
    bool autoRangeX = true;
    
    // Auto-detect columns (for tables with standard names)
    bool autoDetectColumns = true;
    
    // The query that provides the data stream
    std::unique_ptr<AST::SelectStatement> query;
    
    // Sampling configuration
    enum class SamplingMethod {
        NONE,           // Every point
        EVERY_N,        // Every Nth point
        AVERAGE,        // Average over interval
        MAX,            // Max over interval
        MIN,            // Min over interval
        LAST            // Last value in interval
    };
    
    SamplingMethod samplingMethod = SamplingMethod::NONE;
    int samplingInterval = 10;       // For EVERY_N method
    int samplingWindowMs = 1000;     // For aggregation methods
    
    RealTimePlotStatement() = default;
    
    bool isMultiSeries() const {
        return yColumns.size() > 1 || !groupColumn.empty();
    }
    
    std::string getPlotTypeString() const {
        switch (type) {
            case PlotType::LINE: return "Line";
            case PlotType::SCATTER: return "Scatter";
            case PlotType::BAR: return "Bar";
            case PlotType::AREA: return "Area";
            case PlotType::HISTOGRAM: return "Histogram";
            case PlotType::MULTI_LINE: return "Multi-Line";
            case PlotType::STACKED_BAR: return "Stacked Bar";
            default: return "Line";
        }
    }
};
}

#endif // !REAL_TIME_PLOTTER_PARSER_H
