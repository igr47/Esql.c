#ifndef IMPLOT_PLOTTER_H
#define IMPLOT_PLOTTER_H

#include "plotter_includes/plotter.h"
#include "market_simulator.h"
#include <imgui.h>
#include <implot.h>
#include <implot_internal.h>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <queue>

namespace Visualization {

// ImPlot-based real-time plotter for simulations
class ImPlotSimulationPlotter {
public:
    ImPlotSimulationPlotter();
    ~ImPlotSimulationPlotter();

    // Initialize ImPlot context
    void initialize();

    // Setup plot window
    void setupWindow(const PlotStatement::SimulationPlotConfig& config);

    // Plot simulation data in real-time
    void plotSimulationCandlestick(
        const std::shared_ptr<esql::ai::SimulationPath>& path,
        const PlotStatement::SimulationPlotConfig& config,
        size_t current_step
    );

    // Plot multiple simulation paths
    void plotMultiplePaths(
        const std::vector<std::shared_ptr<esql::ai::SimulationPath>>& paths,
        const PlotStatement::SimulationPlotConfig& config,
        size_t current_step
    );

    // Plot order book for microstructure simulation
    void plotOrderBook(
        const std::vector<esql::ai::OrderBook>& order_books,
        const PlotStatement::SimulationPlotConfig& config
    );

    // Plot technical indicators
    void plotIndicators(
        const esql::ai::TechnicalIndicators& indicators,
        const PlotStatement::SimulationPlotConfig& config
    );

    // Plot volume histogram
    void plotVolume(
        const std::vector<double>& volumes,
        const std::vector<double>& prices,
        const std::vector<esql::ai::MarketRegime>& regimes,
        const PlotStatement::SimulationPlotConfig& config,
        size_t current_step
    );

    // Animation control
    void updatePlayback();
    void startAnimation();
    void pauseAnimation();
    void resumeAnimation();
    void stopAnimation();
    void setPlaybackSpeed(double speed);
    
    // Get user input for interactive controls
    bool isWindowClosed() const { return window_closed_; }
    void handleInput();

    // Main render loop
    void renderLoop();

private:
    // Helper methods for ImPlot styling
    void setupPlotStyle(const PlotStatement::SimulationPlotConfig& config);
    void setupAxis(const PlotStatement::SimulationPlotConfig& config, 
                   const std::vector<double>& prices,
                   size_t current_step);
    
    // Color conversion
    ImVec4 hexToImColor(const std::string& hex, float alpha = 1.0f);
    
    // Candlestick drawing
    void drawCandlestick(
        double time,
        double open,
        double high,
        double low,
        double close,
        const ImVec4& color
    );

    // Data buffering for smooth scrolling
    struct PlotBuffer {
        std::vector<double> times;
        std::vector<double> opens;
        std::vector<double> highs;
        std::vector<double> lows;
        std::vector<double> closes;
        std::vector<double> volumes;
        std::vector<esql::ai::MarketRegime> regimes;
        std::mutex mutex;
    };
    
    PlotBuffer buffer_;
    std::atomic<bool> animation_running_{false};
    std::atomic<bool> window_closed_{false};
    std::thread render_thread_;
    
    // State
    std::chrono::steady_clock::time_point last_update_;
    float playback_speed_ = 1.0;
    bool is_playing_ = true;
    
    // Window state
    std::string window_title_;
    int window_width_;
    int window_height_;
    ImGuiID dockspace_id_;
};

} // namespace Visualization

#endif // IMPLOT_PLOTTER_H
