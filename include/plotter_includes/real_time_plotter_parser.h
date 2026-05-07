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

class RealTimeCandlestickStatement : public Statement {
public:
    // Candle configuration
    int intervalSeconds = 5;  // Default 5 seconds
    
    // Column mappings
    std::string openColumn;
    std::string closeColumn;
    std::string highColumn;
    std::string lowColumn;
    std::string volumeColumn;  // Optional
    
    // Plot configuration
    std::string title;
    std::string xLabel = "Time";
    std::string yLabel = "Value";
    std::string outputFile;
    int maxCandles = 100;  // Number of candles to keep in view
    
    // Color configuration
    std::string bullishColor = "#00ff00";  // Green
    std::string bearishColor = "#ff0000";  // Red
    
    // Auto-detect column mapping (for market data tables)
    bool autoDetectColumns = true;
    
    // The query that provides the data stream
    std::unique_ptr<SelectStatement> query;
    
    RealTimeCandlestickStatement() = default;
    
    bool hasVolume() const { return !volumeColumn.empty(); }
    
    // Helper to check if columns are properly configured
    bool isConfigured() const {
        if (autoDetectColumns) return true;
        return !openColumn.empty() && !closeColumn.empty() && 
               !highColumn.empty() && !lowColumn.empty();
    }
    
    // Get column mapping as a struct for execution
    struct ColumnMapping {
        std::string open;
        std::string close;
        std::string high;
        std::string low;
        std::string volume;
        bool useOpen = false;
        bool useClose = false;
        bool useHigh = false;
        bool useLow = false;
        bool useVolume = false;
        
        bool hasAllRequired() const {
            return useOpen && useClose && useHigh && useLow;
        }
    };
    
    ColumnMapping getMapping() const {
        ColumnMapping mapping;
        mapping.open = openColumn;
        mapping.close = closeColumn;
        mapping.high = highColumn;
        mapping.low = lowColumn;
        mapping.volume = volumeColumn;
        mapping.useOpen = !openColumn.empty();
        mapping.useClose = !closeColumn.empty();
        mapping.useHigh = !highColumn.empty();
        mapping.useLow = !lowColumn.empty();
        mapping.useVolume = !volumeColumn.empty();
        return mapping;
    }
};

}

#endif // !REAL_TIME_PLOTTER_PARSER_H
