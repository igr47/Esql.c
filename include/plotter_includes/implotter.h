#ifndef IMPLOTTER_H
#define IMPLOTTER_H

#include "plotter.h"
#include "market_simulator.h"
#include <memory>
#include <mutex>
#include <atomic>
#include <thread>
#include <chrono>
#include <vector>
#include <string>
#include <deque>
#include <map>
#include <imgui.h>
#include <implot.h>

struct GLFWwindow;

namespace Visualization {

class ImPlotSimulationPlotter {
public:
    ImPlotSimulationPlotter();
    ~ImPlotSimulationPlotter();

    // Core functions
    void initialize();
    void shutdown();
    void setupWindow(const PlotStatement::SimulationPlotConfig& config);
    void plotSimulationCandlestick(
        const std::shared_ptr<esql::ai::SimulationPath>& path,
        const PlotStatement::SimulationPlotConfig& config,
        size_t current_step);
    void renderLoop();
    bool isWindowClosed() const;

    // Animation control
    void startAnimation();
    void stopAnimation();
    void pauseAnimation();
    void resumeAnimation();
    void setPlaybackSpeed(double speed);
    float getPlaybackSpeed() const { return playback_speed_.load(); }

    // Input handling
    void handleInput();

private:
    // Enhanced plot data buffer with scrolling support
    struct PlotBuffer {
        static constexpr size_t MAX_VISIBLE_POINTS = 200; // Show last 200 candles

        std::deque<double> times;
        std::deque<double> opens;
        std::deque<double> highs;
        std::deque<double> lows;
        std::deque<double> closes;
        std::deque<double> volumes;
        std::deque<esql::ai::MarketRegime> regimes;

        // Technical indicators
        std::deque<double> sma_20;      // Simple Moving Average 20
        std::deque<double> sma_50;      // Simple Moving Average 50
        std::deque<double> ema_12;       // Exponential MA 12
        std::deque<double> ema_26;       // Exponential MA 26
        std::deque<double> upper_band;   // Bollinger Upper
        std::deque<double> lower_band;   // Bollinger Lower
        std::deque<double> rsi;          // RSI values

        // Trend analysis
        struct TrendLine {
            double start_x;
            double start_y;
            double end_x;
            double end_y;
            ImVec4 color;
            std::string label;
        };
        std::vector<TrendLine> trendlines;

        // Support/Resistance levels
        struct SRLvl {
            double price;
            ImVec4 color;
            bool is_support;
        };
        std::vector<SRLvl> support_resistance;

        std::mutex mutex;

        void addDataPoint(double time, double open, double high, double low,
                          double close, double volume, esql::ai::MarketRegime regime) {
            std::lock_guard<std::mutex> lock(mutex);

            times.push_back(time);
            opens.push_back(open);
            highs.push_back(high);
            lows.push_back(low);
            closes.push_back(close);
            volumes.push_back(volume);
            regimes.push_back(regime);

            // Maintain max size
            while (times.size() > MAX_VISIBLE_POINTS) {
                times.pop_front();
                opens.pop_front();
                highs.pop_front();
                lows.pop_front();
                closes.pop_front();
                volumes.pop_front();
                regimes.pop_front();
            }
        }

        void clear() {
            std::lock_guard<std::mutex> lock(mutex);
            times.clear();
            opens.clear();
            highs.clear();
            lows.clear();
            closes.clear();
            volumes.clear();
            regimes.clear();
            sma_20.clear();
            sma_50.clear();
            ema_12.clear();
            ema_26.clear();
            upper_band.clear();
            lower_band.clear();
            rsi.clear();
            trendlines.clear();
            support_resistance.clear();
        }
    } buffer_;

    // Window state
    GLFWwindow* window_;
    bool glfw_initialized_;
    bool imgui_initialized_;
    std::atomic<bool> window_closed_;
    static std::mutex glfw_mutex_;

    // Animation state
    std::atomic<bool> animation_running_;
    std::atomic<bool> is_playing_;
    std::atomic<float> playback_speed_;
    std::thread render_thread_;
    std::chrono::steady_clock::time_point last_update_;

    // Plot configuration
    std::string window_title_;
    int window_width_;
    int window_height_;

    // Viewport state for scrolling
    struct ViewportState {
        double x_min = 0;
        double x_max = 200;
        double y_min = 0;
        double y_max = 100;
        bool auto_scroll = true;
    } viewport_;

    // UI state
    bool show_volume_ = true;
    bool show_indicators_ = true;
    bool show_trendlines_ = true;
    bool show_sr_levels_ = true;
    bool show_grid_ = true;
    int selected_indicator_ = 0;
    std::vector<const char*> indicator_names_ = {"SMA 20", "SMA 50", "EMA 12/26", "Bollinger Bands", "RSI"};

    // Mutex for thread safety
    std::mutex plot_mutex_;

    // Helper functions
    void drawCandlestick(double time, double open, double high, double low,
                        double close, const ImVec4& color);
    void drawVolumeBars();
    void drawTechnicalIndicators();
    void drawTrendlines();
    void drawSupportResistance();
    void calculateIndicators();
    void updateViewport();
    void drawToolbar();
    void drawStatisticsPanel();

    ImVec4 hexToImColor(const std::string& hex, float alpha = 1.0f);
    void setupPlotStyle(const PlotStatement::SimulationPlotConfig& config);

    void calculateSMA(const std::deque<double>& data, std::deque<double>& output, int period);
    void calculateEMA(const std::deque<double>& data, std::deque<double>& output, int period);
    void calculateBollingerBands();
    void calculateRSI();
    void detectTrendlines();
    void detectSupportResistance();
};

} // namespace Visualization

#endif // IMPLOTTER_H
