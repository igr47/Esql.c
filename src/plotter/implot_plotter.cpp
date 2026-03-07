#include "plotter_includes/implotter.h"

// IMPORTANT: Remove GLEW include, only use GLAD
#include <glad/glad.h>  // This should be the FIRST OpenGL include

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <implot.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <cstdio>
#include <atomic>
#include <deque>
#include <vector>

namespace Visualization {

std::mutex Visualization::ImPlotSimulationPlotter::glfw_mutex_;

ImPlotSimulationPlotter::ImPlotSimulationPlotter()
    : window_(nullptr)
    , glfw_initialized_(false)
    , imgui_initialized_(false)
    , window_closed_(false)
    , animation_running_(false)
    , is_playing_(false)
    , playback_speed_(1.0f)
    , window_width_(1600)
    , window_height_(900) {
}

ImPlotSimulationPlotter::~ImPlotSimulationPlotter() {
    stopAnimation();
    if (render_thread_.joinable()) {
        render_thread_.join();
    }
    shutdown();
}

void ImPlotSimulationPlotter::initialize() {
    std::lock_guard<std::mutex> lock(glfw_mutex_);
    std::cout << "[ImPlot] Initializing with GLAD..." << std::endl;

    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "[ImPlot] Failed to initialize GLFW" << std::endl;
        return;
    }
    glfw_initialized_ = true;

    // Set GLFW error callback
    glfwSetErrorCallback([](int error, const char* description) {
        std::cerr << "[GLFW Error] " << error << ": " << description << std::endl;
    });

    // GLFW window hints for OpenGL 3.3 Core profile
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    std::cout << "[ImPlot] GLFW initialized successfully" << std::endl;
}

void ImPlotSimulationPlotter::shutdown() {
    std::lock_guard<std::mutex> lock(glfw_mutex_);
    std::cout << "[ImPlot] Shutting down..." << std::endl;

    // Cleanup ImGui
    if (imgui_initialized_) {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImPlot::DestroyContext();
        ImGui::DestroyContext();
        imgui_initialized_ = false;
    }

    // Cleanup GLFW window
    if (window_) {
        glfwDestroyWindow(window_);
        window_ = nullptr;
    }

    // Terminate GLFW
    if (glfw_initialized_) {
        glfwTerminate();
        glfw_initialized_ = false;
    }

    window_closed_ = true;
    std::cout << "[ImPlot] Shutdown complete" << std::endl;
}

void ImPlotSimulationPlotter::setupWindow(const PlotStatement::SimulationPlotConfig& config) {
    std::cout << "[ImPlot] Setting up window: " << config.window_title << std::endl;

    window_title_ = config.window_title;
    window_width_ = config.window_width;
    window_height_ = config.window_height;

    // Create window
    window_ = glfwCreateWindow(window_width_, window_height_,
                               window_title_.c_str(), NULL, NULL);
    if (!window_) {
        std::cerr << "[ImPlot] Failed to create GLFW window" << std::endl;
        return;
    }

    // Make context current in THIS thread
    glfwMakeContextCurrent(window_);

    // Initialize GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "[ImPlot] Failed to initialize GLAD" << std::endl;
        glfwDestroyWindow(window_);
        window_ = nullptr;
        return;
    }

    glfwSwapInterval(1); // Enable vsync

    // Verify OpenGL context is working
    std::cout << "[ImPlot] OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "[ImPlot] GLSL Version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();

    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls

    // Check if docking is available in this ImGui version
    #ifdef ImGuiConfigFlags_DockingEnable
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    #endif

    #ifdef ImGuiConfigFlags_ViewportsEnable
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
    #endif

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // Initialize ImGui backends
    if (!ImGui_ImplGlfw_InitForOpenGL(window_, true)) {
        std::cerr << "[ImPlot] Failed to initialize ImGui GLFW backend" << std::endl;
        ImPlot::DestroyContext();
        ImGui::DestroyContext();
        glfwDestroyWindow(window_);
        window_ = nullptr;
        return;
    }

    if (!ImGui_ImplOpenGL3_Init("#version 330 core")) {
        std::cerr << "[ImPlot] Failed to initialize ImGui OpenGL backend" << std::endl;
        ImGui_ImplGlfw_Shutdown();
        ImPlot::DestroyContext();
        ImGui::DestroyContext();
        glfwDestroyWindow(window_);
        window_ = nullptr;
        return;
    }

    imgui_initialized_ = true;
    window_closed_ = false;

    std::cout << "[ImPlot] Window setup complete" << std::endl;
}

void ImPlotSimulationPlotter::plotSimulationCandlestick(
    const std::shared_ptr<esql::ai::SimulationPath>& path,
    const PlotStatement::SimulationPlotConfig& config,
    size_t current_step) {

    if (!path || path->prices.empty()) return;
    if (!window_ || window_closed_) return;

    std::lock_guard<std::mutex> lock(plot_mutex_);

    // Update buffer with new data
    {
        std::lock_guard<std::mutex> buffer_lock(buffer_.mutex);

        double time = static_cast<double>(current_step);
        double price = path->prices[current_step];
        double prev_price = (current_step > 0) ? path->prices[current_step-1] : price;

        double volatility = (current_step < path->volatilities.size()) ?
                            path->volatilities[current_step] : 0.01;
        double high = price * (1.0 + volatility * 0.5);
        double low = price * (1.0 - volatility * 0.5);
        double volume = (current_step < path->volumes.size()) ?
                        path->volumes[current_step] : 1000.0;

        esql::ai::MarketRegime regime = (current_step < path->regimes.size()) ?
                                        path->regimes[current_step] :
                                        esql::ai::MarketRegime::SIDEWAYS;

        buffer_.addDataPoint(time, prev_price, high, low, price, volume, regime);

        // Calculate indicators for new data
        calculateIndicators();
        detectTrendlines();
        detectSupportResistance();
    }

    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Create main dock space (if docking is available)
    #ifdef ImGuiConfigFlags_DockingEnable
    ImGuiID dockspace_id = ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());
    #endif

    // Draw toolbar
    drawToolbar();

    // Create main chart window
    ImGui::Begin("Market Simulation", nullptr, ImGuiWindowFlags_NoCollapse);

    setupPlotStyle(config);

    // Update viewport for scrolling
    updateViewport();

    // Main price chart with candlesticks and indicators
    float chart_height = show_volume_ ? ImGui::GetContentRegionAvail().y - 200 : -1;

    if (ImPlot::BeginPlot("Price Chart", ImVec2(-1, chart_height))) {
        ImPlot::SetupAxis(ImAxis_X1, "Step");
        ImPlot::SetupAxis(ImAxis_Y1, "Price");

        // Set viewport limits for scrolling
        ImPlot::SetupAxisLimits(ImAxis_X1, viewport_.x_min, viewport_.x_max,
                                viewport_.auto_scroll ? ImGuiCond_Always : ImGuiCond_Once);
        ImPlot::SetupAxisLimits(ImAxis_Y1, viewport_.y_min, viewport_.y_max,
                                ImGuiCond_Once);

        std::lock_guard<std::mutex> buffer_lock(buffer_.mutex);

        // Convert deque to vector for ImPlot (ImPlot requires contiguous memory)
        std::vector<double> times_vec(buffer_.times.begin(), buffer_.times.end());
        std::vector<double> opens_vec(buffer_.opens.begin(), buffer_.opens.end());
        std::vector<double> highs_vec(buffer_.highs.begin(), buffer_.highs.end());
        std::vector<double> lows_vec(buffer_.lows.begin(), buffer_.lows.end());
        std::vector<double> closes_vec(buffer_.closes.begin(), buffer_.closes.end());

        // Draw candlesticks (simplified - just draw lines for now)
        // For proper candlesticks, you'd need custom rendering
        //ImPlot::SetNexMarkerStyle(ImPlotMarker_Circle, 0); // Hide markers
        //ImPlot::SetNextLineStyle(ImVec4(0, 1, 0, 1), 1.5f);
        ImPlot::PlotLine("Price", times_vec.data(), closes_vec.data(), times_vec.size());

        // Draw technical indicators
        if (show_indicators_) {
            drawTechnicalIndicators();
        }

        // Draw trendlines
        if (show_trendlines_) {
            drawTrendlines();
        }

        // Draw support/resistance levels
        if (show_sr_levels_) {
            drawSupportResistance();
        }

        ImPlot::EndPlot();
    }

    // Volume chart
    if (show_volume_ && !buffer_.volumes.empty()) {
        drawVolumeBars();
    }

    ImGui::End();

    // Statistics panel
    drawStatisticsPanel();

    // Rendering
    ImGui::Render();
    int display_w, display_h;
    glfwGetFramebufferSize(window_, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    // Update and render additional Platform Windows (if viewports are enabled)
    #ifdef ImGuiConfigFlags_ViewportsEnable
    if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
        GLFWwindow* backup_current_context = glfwGetCurrentContext();
        ImGui::UpdatePlatformWindows();
        ImGui::RenderPlatformWindowsDefault();
        glfwMakeContextCurrent(backup_current_context);
    }
    #endif

    glfwSwapBuffers(window_);
    glfwPollEvents();

    // Check if window should close
    if (glfwWindowShouldClose(window_)) {
        window_closed_ = true;
    }
}

void ImPlotSimulationPlotter::drawCandlestick(
    double time, double open, double high, double low, double close, const ImVec4& color) {
    // This is a simplified version - full candlestick rendering would require custom ImPlot integration
    // For now, we'll just use the line plot in the main function
}

void ImPlotSimulationPlotter::drawVolumeBars() {
    std::lock_guard<std::mutex> buffer_lock(buffer_.mutex);

    if (buffer_.volumes.empty()) return;

    if (ImPlot::BeginPlot("Volume", ImVec2(-1, 150))) {
        ImPlot::SetupAxis(ImAxis_X1, "Step");
        ImPlot::SetupAxis(ImAxis_Y1, "Volume");

        std::vector<double> times_vec(buffer_.times.begin(), buffer_.times.end());
        std::vector<double> volumes_vec(buffer_.volumes.begin(), buffer_.volumes.end());

        ImPlot::PlotBars("Volume",
                         times_vec.data(),
                         volumes_vec.data(),
                         volumes_vec.size(),
                         0.8);

        ImPlot::EndPlot();
    }
}

void ImPlotSimulationPlotter::drawTechnicalIndicators() {
    if (buffer_.times.empty()) return;

    std::vector<double> times_vec(buffer_.times.begin(), buffer_.times.end());

    // Draw SMA lines
    if (!buffer_.sma_20.empty() && buffer_.sma_20.size() == buffer_.times.size()) {
        std::vector<double> sma20_vec(buffer_.sma_20.begin(), buffer_.sma_20.end());
        ImPlot::PlotLine("SMA 20", times_vec.data(), sma20_vec.data(), sma20_vec.size());
    }

    if (!buffer_.sma_50.empty() && buffer_.sma_50.size() == buffer_.times.size()) {
        std::vector<double> sma50_vec(buffer_.sma_50.begin(), buffer_.sma_50.end());
        ImPlot::PlotLine("SMA 50", times_vec.data(), sma50_vec.data(), sma50_vec.size());
    }
}

void ImPlotSimulationPlotter::drawTrendlines() {
    std::lock_guard<std::mutex> buffer_lock(buffer_.mutex);

    for (const auto& trendline : buffer_.trendlines) {
        double x[2] = {trendline.start_x, trendline.end_x};
        double y[2] = {trendline.start_y, trendline.end_y};
        ImPlot::PlotLine(trendline.label.c_str(), x, y, 2);
    }
}

void ImPlotSimulationPlotter::drawSupportResistance() {
    std::lock_guard<std::mutex> buffer_lock(buffer_.mutex);

    for (const auto& level : buffer_.support_resistance) {
        double x[2] = {buffer_.times.front(), buffer_.times.back()};
        double y[2] = {level.price, level.price};
        ImPlot::PlotLine(level.is_support ? "Support" : "Resistance", x, y, 2);
    }
}

void ImPlotSimulationPlotter::drawToolbar() {
    ImGui::Begin("Controls", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

    // Playback controls
    bool playing = is_playing_.load();
    if (playing) {
        if (ImGui::Button("Pause")) {
            is_playing_ = false;
        }
    } else {
        if (ImGui::Button("Play")) {
            is_playing_ = true;
        }
    }

    ImGui::SameLine();
    if (ImGui::Button("Stop")) {
        is_playing_ = false;
    }

    ImGui::SameLine();
    float speed = playback_speed_.load();
    if (ImGui::SliderFloat("Speed", &speed, 0.1f, 5.0f, "%.1fx")) {
        playback_speed_ = speed;
    }

    ImGui::Separator();

    // View options
    ImGui::Checkbox("Auto-scroll", &viewport_.auto_scroll);
    ImGui::SameLine();
    ImGui::Checkbox("Show Volume", &show_volume_);
    ImGui::SameLine();
    ImGui::Checkbox("Show Indicators", &show_indicators_);

    if (show_indicators_) {
        ImGui::Combo("Indicator", &selected_indicator_,
                     indicator_names_.data(), indicator_names_.size());
    }

    ImGui::SameLine();
    ImGui::Checkbox("Show Trendlines", &show_trendlines_);
    ImGui::SameLine();
    ImGui::Checkbox("Show S/R Levels", &show_sr_levels_);

    ImGui::End();
}

void ImPlotSimulationPlotter::drawStatisticsPanel() {
    std::lock_guard<std::mutex> buffer_lock(buffer_.mutex);

    ImGui::Begin("Statistics", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

    if (!buffer_.closes.empty()) {
        double current_price = buffer_.closes.back();
        double prev_price = buffer_.closes.size() > 1 ? buffer_.closes[buffer_.closes.size()-2] : current_price;
        double change = current_price - prev_price;
        double change_pct = (change / prev_price) * 100;

        ImGui::Text("Price: %.4f", current_price);
        ImGui::Text("Change: %.4f (%.2f%%)", change, change_pct);

        if (!buffer_.rsi.empty()) {
            ImGui::Text("RSI: %.1f", buffer_.rsi.back());
        }

        // Calculate some basic statistics
        if (buffer_.closes.size() >= 20) {
            double sum = 0.0;
            for (size_t i = buffer_.closes.size() - 20; i < buffer_.closes.size(); ++i) {
                sum += buffer_.closes[i];
            }
            double sma20 = sum / 20.0;
            ImGui::Text("SMA20: %.4f", sma20);

            // Calculate volatility
            std::vector<double> returns;
            for (size_t i = buffer_.closes.size() - 20; i < buffer_.closes.size() - 1; ++i) {
                returns.push_back((buffer_.closes[i+1] - buffer_.closes[i]) / buffer_.closes[i]);
            }
            if (!returns.empty()) {
                double mean = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
                double sq_sum = std::inner_product(returns.begin(), returns.end(), returns.begin(), 0.0);
                double volatility = std::sqrt(sq_sum / returns.size() - mean * mean);
                ImGui::Text("Volatility: %.4f", volatility);
            }
        }
    }

    ImGui::End();
}

void ImPlotSimulationPlotter::updateViewport() {
    std::lock_guard<std::mutex> buffer_lock(buffer_.mutex);

    if (buffer_.times.empty()) return;

    if (viewport_.auto_scroll) {
        // Scroll to show latest data
        double latest_time = buffer_.times.back();
        viewport_.x_min = std::max(0.0, latest_time - buffer_.MAX_VISIBLE_POINTS);
        viewport_.x_max = latest_time + 10; // Add some padding
    }

    // Update Y-axis limits based on visible data
    if (!buffer_.highs.empty() && !buffer_.lows.empty()) {
        double min_price = *std::min_element(buffer_.lows.begin(), buffer_.lows.end()) * 0.995;
        double max_price = *std::max_element(buffer_.highs.begin(), buffer_.highs.end()) * 1.005;
        viewport_.y_min = min_price;
        viewport_.y_max = max_price;
    }
}

void ImPlotSimulationPlotter::calculateIndicators() {
    if (buffer_.closes.size() < 50) return;

    std::lock_guard<std::mutex> lock(buffer_.mutex);

    // Calculate SMAs
    calculateSMA(buffer_.closes, buffer_.sma_20, 20);
    calculateSMA(buffer_.closes, buffer_.sma_50, 50);

    // Calculate EMAs
    calculateEMA(buffer_.closes, buffer_.ema_12, 12);
    calculateEMA(buffer_.closes, buffer_.ema_26, 26);

    // Calculate Bollinger Bands
    calculateBollingerBands();

    // Calculate RSI
    calculateRSI();
}

void ImPlotSimulationPlotter::calculateSMA(const std::deque<double>& data,
                                           std::deque<double>& output, int period) {
    if (data.size() < static_cast<size_t>(period)) return;

    output.clear();

    for (size_t i = period - 1; i < data.size(); ++i) {
        double sum = 0.0;
        for (int j = 0; j < period; ++j) {
            sum += data[i - j];
        }
        output.push_back(sum / period);
    }
}

void ImPlotSimulationPlotter::calculateEMA(const std::deque<double>& data,
                                           std::deque<double>& output, int period) {
    if (data.empty()) return;

    double multiplier = 2.0 / (period + 1);
    double ema = data[0];

    output.clear();
    output.push_back(ema);

    for (size_t i = 1; i < data.size(); ++i) {
        ema = (data[i] - ema) * multiplier + ema;
        output.push_back(ema);
    }
}

void ImPlotSimulationPlotter::calculateBollingerBands() {
    if (buffer_.closes.size() < 20) return;

    buffer_.upper_band.clear();
    buffer_.lower_band.clear();

    for (size_t i = 19; i < buffer_.closes.size(); ++i) {
        double sum = 0.0;
        for (int j = 0; j < 20; ++j) {
            sum += buffer_.closes[i - j];
        }
        double mean = sum / 20.0;

        double sq_sum = 0.0;
        for (int j = 0; j < 20; ++j) {
            sq_sum += std::pow(buffer_.closes[i - j] - mean, 2);
        }
        double stddev = std::sqrt(sq_sum / 20.0);

        buffer_.upper_band.push_back(mean + 2 * stddev);
        buffer_.lower_band.push_back(mean - 2 * stddev);
    }
}

void ImPlotSimulationPlotter::calculateRSI() {
    if (buffer_.closes.size() < 15) return;

    buffer_.rsi.clear();

    for (size_t i = 14; i < buffer_.closes.size(); ++i) {
        double gain_sum = 0.0, loss_sum = 0.0;

        for (int j = 0; j < 14; ++j) {
            double change = buffer_.closes[i - j] - buffer_.closes[i - j - 1];
            if (change > 0) {
                gain_sum += change;
            } else {
                loss_sum -= change;
            }
        }

        double avg_gain = gain_sum / 14.0;
        double avg_loss = loss_sum / 14.0;

        if (avg_loss == 0) {
            buffer_.rsi.push_back(100.0);
        } else {
            double rs = avg_gain / avg_loss;
            buffer_.rsi.push_back(100.0 - (100.0 / (1.0 + rs)));
        }
    }
}

void ImPlotSimulationPlotter::detectTrendlines() {
    // Simple trendline detection based on pivot points
    if (buffer_.closes.size() < 20) return;

    buffer_.trendlines.clear();

    // Find local minima and maxima
    std::vector<size_t> peaks;
    std::vector<size_t> troughs;

    for (size_t i = 5; i < buffer_.closes.size() - 5; ++i) {
        bool is_peak = true;
        bool is_trough = true;

        for (int j = -5; j <= 5; ++j) {
            if (j == 0) continue;
            if (buffer_.closes[i] < buffer_.closes[i + j]) is_peak = false;
            if (buffer_.closes[i] > buffer_.closes[i + j]) is_trough = false;
        }

        if (is_peak) peaks.push_back(i);
        if (is_trough) troughs.push_back(i);
    }

    // Connect peaks for resistance line
    if (peaks.size() >= 2) {
        size_t last_peak = peaks.back();
        size_t prev_peak = peaks[peaks.size() - 2];

        PlotBuffer::TrendLine resistance;
        resistance.start_x = buffer_.times[prev_peak];
        resistance.start_y = buffer_.closes[prev_peak];
        resistance.end_x = buffer_.times[last_peak];
        resistance.end_y = buffer_.closes[last_peak];
        resistance.color = ImVec4(1, 0, 0, 1); // Red
        resistance.label = "Resistance";
        buffer_.trendlines.push_back(resistance);
    }

    // Connect troughs for support line
    if (troughs.size() >= 2) {
        size_t last_trough = troughs.back();
        size_t prev_trough = troughs[troughs.size() - 2];

        PlotBuffer::TrendLine support;
        support.start_x = buffer_.times[prev_trough];
        support.start_y = buffer_.closes[prev_trough];
        support.end_x = buffer_.times[last_trough];
        support.end_y = buffer_.closes[last_trough];
        support.color = ImVec4(0, 1, 0, 1); // Green
        support.label = "Support";
        buffer_.trendlines.push_back(support);
    }
}

void ImPlotSimulationPlotter::detectSupportResistance() {
    if (buffer_.closes.size() < 20) return;

    buffer_.support_resistance.clear();

    // Find price levels where price reversed multiple times
    std::map<double, int> level_hits;
    double tolerance = (buffer_.highs.back() - buffer_.lows.back()) * 0.01; // 1% tolerance

    for (size_t i = 5; i < buffer_.closes.size() - 5; ++i) {
        // Check if this is a reversal point
        if ((buffer_.highs[i] > buffer_.highs[i-1] && buffer_.highs[i] > buffer_.highs[i+1]) ||
            (buffer_.lows[i] < buffer_.lows[i-1] && buffer_.lows[i] < buffer_.lows[i+1])) {

            double price = (buffer_.highs[i] + buffer_.lows[i]) / 2;

            // Group nearby levels
            bool found = false;
            for (auto& [level, hits] : level_hits) {
                if (std::abs(price - level) / level < tolerance) {
                    hits++;
                    found = true;
                    break;
                }
            }
            if (!found) {
                level_hits[price] = 1;
            }
        }
    }

    // Add levels that were hit multiple times
    for (const auto& [price, hits] : level_hits) {
        if (hits >= 3) {
            PlotBuffer::SRLvl level;
            level.price = price;
            // Determine if support or resistance based on price position
            level.is_support = price < buffer_.closes.back();
            level.color = level.is_support ? ImVec4(0, 1, 0, 0.5) : ImVec4(1, 0, 0, 0.5);
            buffer_.support_resistance.push_back(level);
        }
    }
}

ImVec4 ImPlotSimulationPlotter::hexToImColor(const std::string& hex, float alpha) {
    unsigned int r = 0, g = 0, b = 0;
    if (hex[0] == '#') {
        sscanf(hex.c_str(), "#%02x%02x%02x", &r, &g, &b);
    } else {
        sscanf(hex.c_str(), "%02x%02x%02x", &r, &g, &b);
    }
    return ImVec4(r/255.0f, g/255.0f, b/255.0f, alpha);
}

void ImPlotSimulationPlotter::setupPlotStyle(const PlotStatement::SimulationPlotConfig& config) {
    ImPlotStyle& style = ImPlot::GetStyle();

    if (config.show_grid) {
        style.Colors[ImPlotCol_AxisGrid] = ImVec4(0.5f, 0.5f, 0.5f, 0.25f);
    } else {
        style.Colors[ImPlotCol_AxisGrid] = ImVec4(0, 0, 0, 0);
    }

    if (config.show_legend) {
        style.Colors[ImPlotCol_LegendBg] = ImVec4(0.1f, 0.1f, 0.1f, 0.8f);
    }

    // Set professional trading style
    //style.Colors[ImPlotCol_Line] = ImVec4(0.8f, 0.8f, 0.8f, 1.0f);
    style.Colors[ImPlotCol_FrameBg] = ImVec4(0.1f, 0.1f, 0.1f, 1.0f);
    style.Colors[ImPlotCol_PlotBg] = ImVec4(0.05f, 0.05f, 0.05f, 1.0f);
    style.Colors[ImPlotCol_PlotBorder] = ImVec4(0.3f, 0.3f, 0.3f, 1.0f);

    style.Colors[ImPlotCol_AxisBg] = ImVec4(0.1f, 0.1f, 0.1f, 1.0f);
    style.Colors[ImPlotCol_AxisBgHovered] = ImVec4(0.15f, 0.15f, 0.15f, 1.0f);
    style.Colors[ImPlotCol_AxisBgActive] = ImVec4(0.2f, 0.2f, 0.2f, 1.0f);
    //style.Colors[ImPlotCol_AxisLabel] = ImVec4(0.8f, 0.8f, 0.8f, 1.0f);
    style.Colors[ImPlotCol_AxisTick] = ImVec4(0.8f, 0.8f, 0.8f, 1.0f);
    //style.Colors[ImPlotCol_AxisTickLabels] = ImVec4(0.8f, 0.8f, 0.8f, 1.0f);
}

void ImPlotSimulationPlotter::handleInput() {
    ImGuiIO& io = ImGui::GetIO();

    // Handle mouse wheel for zooming
    if (io.MouseWheel != 0) {
        viewport_.auto_scroll = false;

        double zoom_factor = 1.0 - io.MouseWheel * 0.1;
        double center = (viewport_.x_min + viewport_.x_max) / 2;
        double range = (viewport_.x_max - viewport_.x_min) * zoom_factor;
        viewport_.x_min = center - range / 2;
        viewport_.x_max = center + range / 2;
    }

    // Handle arrow keys for panning
    if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow)) {
        viewport_.auto_scroll = false;
        double range = viewport_.x_max - viewport_.x_min;
        viewport_.x_min -= range * 0.1;
        viewport_.x_max -= range * 0.1;
    }
    if (ImGui::IsKeyPressed(ImGuiKey_RightArrow)) {
        viewport_.auto_scroll = false;
        double range = viewport_.x_max - viewport_.x_min;
        viewport_.x_min += range * 0.1;
        viewport_.x_max += range * 0.1;
    }
}

void ImPlotSimulationPlotter::renderLoop() {
    if (!window_ || window_closed_) return;

    glfwPollEvents();

    if (glfwWindowShouldClose(window_)) {
        window_closed_ = true;
    }
}

bool ImPlotSimulationPlotter::isWindowClosed() const {
    return window_closed_ || (window_ && glfwWindowShouldClose(window_));
}

void ImPlotSimulationPlotter::startAnimation() {
    animation_running_ = true;
    is_playing_ = true;
    last_update_ = std::chrono::steady_clock::now();
}

void ImPlotSimulationPlotter::stopAnimation() {
    animation_running_ = false;
    is_playing_ = false;
}

void ImPlotSimulationPlotter::pauseAnimation() {
    is_playing_ = false;
}

void ImPlotSimulationPlotter::resumeAnimation() {
    is_playing_ = true;
}

void ImPlotSimulationPlotter::setPlaybackSpeed(double speed) {
    playback_speed_ = static_cast<float>(std::max(0.1, std::min(10.0, speed)));
}

} // namespace Visualization
