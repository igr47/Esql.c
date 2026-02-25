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
//#include <GLFW/glfw3.h>
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

private:
    // Plot data buffer
    struct PlotBuffer {
        std::vector<double> times;
        std::vector<double> opens;
        std::vector<double> highs;
        std::vector<double> lows;
        std::vector<double> closes;
        std::vector<double> volumes;
        std::vector<esql::ai::MarketRegime> regimes;
        std::mutex mutex;
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
    //float playback_speed_;
    std::thread render_thread_;
    std::chrono::steady_clock::time_point last_update_;
    
    // Plot configuration
    std::string window_title_;
    int window_width_;
    int window_height_;
    
    // Mutex for thread safety
    std::mutex plot_mutex_;
    
    // Helper functions
    void drawCandlestick(double time, double open, double high, double low, double close, const ImVec4& color);
    ImVec4 hexToImColor(const std::string& hex, float alpha = 1.0f);
    void setupPlotStyle(const PlotStatement::SimulationPlotConfig& config);
    void handleInput();
};

} // namespace Visualization

#endif // IMPLOTTER_H
