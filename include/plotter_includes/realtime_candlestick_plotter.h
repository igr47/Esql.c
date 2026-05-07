#ifndef REALTIME_CANDLESTICK_PLOTTER_H
#define REALTIME_CANDLESTICK_PLOTTER_H

#include "plotter.h"
#include "real_time_plotter_parser.h"
#include "execution_engine_includes/executionengine_main.h"
#include <thread>
#include <atomic>
#include <chrono>
#include <deque>
#include <mutex>
#include <condition_variable>

typedef unsigned int ImU32;
struct ImVec4;

namespace Visualization {

// Structure to hold candlestick data
struct RealTimeCandlestick {
    double open;
    double high;
    double low;
    double close;
    double volume;
    std::chrono::system_clock::time_point timestamp;
    int index;
    
    RealTimeCandlestick() : open(0), high(0), low(0), close(0), volume(0), index(0) {}
    
    RealTimeCandlestick(double o, double h, double l, double c, double v, 
                        std::chrono::system_clock::time_point ts, int idx)
        : open(o), high(h), low(l), close(c), volume(v), timestamp(ts), index(idx) {}
    
    bool isBullish() const { return close > open; }
    double getBodyHigh() const { return std::max(open, close); }
    double getBodyLow() const { return std::min(open, close); }
    double getBodyLength() const { return std::abs(close - open); }
    double getTotalRange() const { return high - low; }
    double getUpperWick() const { return high - getBodyHigh(); }
    double getLowerWick() const { return getBodyLow() - low; }
};

class RealTimeCandlestickPlotter {
public:
    RealTimeCandlestickPlotter();
    ~RealTimeCandlestickPlotter();
    
    // Initialize the plotter with configuration
    void initialize(const AST::RealTimeCandlestickStatement& config);
    
    // Start real-time plotting
    void start();
    
    // Stop real-time plotting
    void stop();
    
    // Add a new candlestick (called from data provider)
    void addCandlestick(const RealTimeCandlestick& candle);
    
    // Update the current in-progress candle (for streaming data)
    void updateCurrentCandle(double price, double volume = 0);
    
    // Check if running
    bool isRunning() const { return running_; }
    
    // Get current statistics
    struct Statistics {
        size_t totalCandles;
        double highestHigh;
        double lowestLow;
        double totalVolume;
        double avgBodyLength;
        size_t bullishCount;
        size_t bearishCount;
        std::chrono::seconds elapsedTime;
    };
    
    Statistics getStatistics() const;

    bool isPaused() const { return paused_; }
    void pause() { paused_ = true; }
    void resume() { paused_ = false; }
    size_t getCandleCount() const { 
        std::lock_guard<std::mutex> lock(dataMutex_);
        return candles_.size(); 
    }

private:
    // Plotting thread function
    void plotThreadFunction();
    
    // Render the candlestick chart
    void renderCandlestickChart();
    
    // Render statistics panel
    void renderStatisticsPanel();
    
    // Update price range for auto-scaling
    void updatePriceRange();
    
    // Calculate moving averages
    void calculateMovingAverages();
    
    // Configuration
    std::string title_;
    std::string xLabel_;
    std::string yLabel_;
    std::string bullishColor_;
    std::string bearishColor_;
    int maxCandles_;
    int intervalSeconds_;
    std::string outputFile_;
    
    // Column mappings
    AST::RealTimeCandlestickStatement::ColumnMapping mapping_;
    
    // Data storage
    std::deque<RealTimeCandlestick> candles_;
    mutable std::mutex dataMutex_;
    
    // Current candle (being built from streaming data)
    RealTimeCandlestick currentCandle_;
    bool hasCurrentCandle_;
    std::chrono::steady_clock::time_point candleStartTime_;
    
    // Price range
    double priceMin_;
    double priceMax_;
    mutable std::mutex rangeMutex_;
    
    // Moving averages
    std::vector<double> ma5_;
    std::vector<double> ma10_;
    std::vector<double> ma20_;

    std::atomic<bool> paused_{false};
    
    // Thread control
    std::atomic<bool> running_;
    std::atomic<bool> shouldStop_;
    std::thread plotThread_;
    std::condition_variable cv_;
    std::mutex cvMutex_;
    
    // Timing
    std::chrono::steady_clock::time_point lastRenderTime_;
    double targetFrameRate_ = 60.0;
    
    // Statistics
    Statistics stats_;
    mutable std::mutex statsMutex_;
    
    // Helper functions
    std::array<float, 4> parseColor(const std::string& colorStr);
    ImU32 toImU32(const std::string& colorStr);
};

} // namespace Visualization

#endif // REALTIME_CANDLESTICK_PLOTTER_H
