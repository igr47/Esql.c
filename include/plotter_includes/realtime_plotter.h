#ifndef REALTIME_PLOTTER_H
#define REALTIME_PLOTTER_H

#include "plotter.h"
#include "execution_engine_includes/executionengine_main.h"
#include "real_time_plotter_parser.h"  // Add this include
#include <thread>
#include <atomic>
#include <chrono>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <queue>
#include <limits>

// Forward declare ImGui/ImPlot types
struct ImVec4;
struct ImVec2;

namespace Visualization {

// Data point structure
struct RealTimeDataPoint {
    double x;                           // X value (usually timestamp)
    double y;                           // Y value
    std::string series;                 // Series name (for multi-series)
    std::chrono::system_clock::time_point timestamp;
    
    RealTimeDataPoint() : x(0), y(0) {}
    RealTimeDataPoint(double x_val, double y_val, const std::string& s = "")
        : x(x_val), y(y_val), series(s), timestamp(std::chrono::system_clock::now()) {}
};

// Series data structure
struct RealTimeSeries {
    std::string name;
    std::deque<RealTimeDataPoint> points;
    // Store color as separate components to avoid ImVec4 dependency in header
    float color_r, color_g, color_b, color_a;
    double minY = std::numeric_limits<double>::max();
    double maxY = std::numeric_limits<double>::lowest();
    
    void setColor(float r, float g, float b, float a = 1.0f) {
        color_r = r; color_g = g; color_b = b; color_a = a;
    }
    
    void addPoint(const RealTimeDataPoint& point) {
        points.push_back(point);
        minY = std::min(minY, point.y);
        maxY = std::max(maxY, point.y);
    }
    
    void clear() {
        points.clear();
        minY = std::numeric_limits<double>::max();
        maxY = std::numeric_limits<double>::lowest();
    }
    
    size_t size() const { return points.size(); }
    
    double getLastY() const {
        return points.empty() ? 0 : points.back().y;
    }
};

class RealTimePlotter {
public:
    RealTimePlotter();
    ~RealTimePlotter();
    
    // Initialize with configuration
    void initialize(const AST::RealTimePlotStatement& config);
    
    // Start/stop plotting
    void start();
    void stop();
    
    // Add data points (called from data provider)
    void addDataPoint(const RealTimeDataPoint& point);
    void addDataPoints(const std::vector<RealTimeDataPoint>& points);
    
    // Bulk data loading from query results
    void loadInitialData(const ExecutionEngine::ResultSet& result);
    
    // Streaming update (for continuous queries)
    void streamUpdate(const std::vector<std::unordered_map<std::string, std::string>>& newRows);
    
    // Check status
    bool isRunning() const { return running_; }
    
    // Get statistics
    struct Statistics {
        size_t totalPointsReceived;
        size_t totalPointsPlotted;
        size_t activeSeries;
        double currentFPS;
        double avgLatencyMs;
        std::chrono::seconds elapsedTime;
    };
    
    Statistics getStatistics() const;
    
    // Pause/resume
    void pause() { paused_ = true; }
    void resume() { paused_ = false; }
    bool isPaused() const { return paused_; }
    
    // Clear all data
    void clear();

private:
    // Plotting thread function
    void plotThreadFunction();
    
    // Rendering functions
    void renderPlot();
    void renderLinePlot();
    void renderScatterPlot();
    void renderBarPlot();
    void renderAreaPlot();
    void renderHistogram();
    void renderMultiLinePlot();
    void renderStackedBarPlot();
    
    void renderLegend();
    void renderStatisticsPanel();
    void renderControls();
    
    // Data management
    void updateYRange();
    void updateXRange();
    void pruneOldData();
    void resampleData();
    void processPendingPoints();  // Add this declaration
    
    // Data sampling
    std::vector<RealTimeDataPoint> sampleData(const std::deque<RealTimeDataPoint>& data);
    std::vector<RealTimeDataPoint> averageSample(const std::deque<RealTimeDataPoint>& data, int windowMs);
    std::vector<RealTimeDataPoint> maxSample(const std::deque<RealTimeDataPoint>& data, int windowMs);
    std::vector<RealTimeDataPoint> minSample(const std::deque<RealTimeDataPoint>& data, int windowMs);
    std::vector<RealTimeDataPoint> lastSample(const std::deque<RealTimeDataPoint>& data, int windowMs);
    
    // Helper functions
    std::string formatTimestamp(double timestamp);
    
    // Configuration
    AST::RealTimePlotStatement::PlotType plotType_;
    std::string title_;
    std::string xLabel_;
    std::string yLabel_;
    std::string outputFile_;
    int refreshIntervalMs_;
    int historyWindowSec_;
    int maxDataPoints_;
    bool showGrid_;
    bool showLegend_;
    bool autoRangeY_;
    double yMin_;
    double yMax_;
    bool autoRangeX_;
    std::vector<std::string> colors_;
    
    // Column mappings
    std::string xColumn_;
    std::vector<std::string> yColumns_;
    std::string groupColumn_;
    
    // Sampling configuration
    AST::RealTimePlotStatement::SamplingMethod samplingMethod_;
    int samplingInterval_;
    int samplingWindowMs_;
    
    // Data storage
    std::unordered_map<std::string, RealTimeSeries> series_;
    mutable std::mutex dataMutex_;
    
    // X-axis range tracking
    double xMin_;
    double xMax_;
    double xRange_;
    
    // Performance tracking
    std::chrono::steady_clock::time_point lastDataPointTime_;
    std::chrono::steady_clock::time_point lastRenderTime_;
    
    // Thread control
    std::atomic<bool> running_;
    std::atomic<bool> shouldStop_;
    std::atomic<bool> paused_;
    std::thread plotThread_;
    std::condition_variable cv_;
    std::mutex cvMutex_;
    
    // Statistics
    struct {
        size_t totalPointsReceived = 0;
        size_t totalPointsPlotted = 0;
        size_t activeSeries = 0;
        double currentFPS = 0;
        double avgLatencyMs = 0;
        std::chrono::steady_clock::time_point startTime;
    } stats_;
    mutable std::mutex statsMutex_;
    
    // Data pending for processing
    std::queue<RealTimeDataPoint> pendingPoints_;
    std::mutex pendingMutex_;
    
    // Helper to get color as ImVec4 (implemented in .cpp)
    struct Color {
        float r, g, b, a;
        Color() : r(0), g(0), b(0), a(0) {}
        Color(float r_, float g_, float b_, float a_ = 1.0f) : r(r_), g(g_), b(b_), a(a_) {}
    };
    
    Color defaultColors_[10];
    void initDefaultColors();
};

} // namespace Visualization

#endif // REALTIME_PLOTTER_H
