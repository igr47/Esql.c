#include "plotter_includes/implotter.h"
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <GLFW/glfw3.h>
#include <algorithm>
#include <cmath>
#include <iomanip>

namespace Visualization {

ImPlotSimulationPlotter::ImPlotSimulationPlotter() {
    // Constructor
}

ImPlotSimulationPlotter::~ImPlotSimulationPlotter() {
    stopAnimation();
    if (render_thread_.joinable()) {
        render_thread_.join();
    }
}

void ImPlotSimulationPlotter::initialize() {
    // Setup ImGui and ImPlot context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    
    // Setup platform/renderer backends
    ImGui_ImplGlfw_InitForOpenGL(glfwGetCurrentContext(), true);
    ImGui_ImplOpenGL3_Init("#version 130");
    
    // Setup ImPlot style
    ImPlotStyle& style = ImPlot::GetStyle();
    style.UseLocalTime = true;
    style.UseISO8601 = true;
    style.Colormap = ImPlotColormap_Viridis;
}

void ImPlotSimulationPlotter::setupWindow(const PlotStatement::SimulationPlotConfig& config) {
    window_title_ = config.window_title;
    window_width_ = config.window_width;
    window_height_ = config.window_height;
    
    // Create window with docking
    glfwWindowHint(GLFW_RESIZABLE, config.window_resizable ? GLFW_TRUE : GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow(window_width_, window_height_, 
                                          window_title_.c_str(), NULL, NULL);
    glfwMakeContextCurrent(window);
    
    // Setup ImGui for this window
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");
    
    // Create dockspace
    ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());
    //dockspace_id_ = ImGui::DockSpaceOverViewport(ImGui::GetMainViewport()->ID);
}

void ImPlotSimulationPlotter::plotSimulationCandlestick(
    const std::shared_ptr<esql::ai::SimulationPath>& path,
    const PlotStatement::SimulationPlotConfig& config,
    size_t current_step) {
    
    if (!path || path->prices.empty()) return;
    
    std::lock_guard<std::mutex> lock(buffer_.mutex);
    
    // Update buffer with new data up to current_step
    buffer_.times.clear();
    buffer_.opens.clear();
    buffer_.highs.clear();
    buffer_.lows.clear();
    buffer_.closes.clear();
    buffer_.volumes.clear();
    buffer_.regimes.clear();
    
    size_t start = (current_step > config.max_points_display) ? 
                   current_step - config.max_points_display : 0;
    
    for (size_t i = start; i <= current_step && i < path->prices.size(); ++i) {
        // Simulate OHLC data from price series (simplified - in reality you'd have proper OHLC)
        double price = path->prices[i];
        double prev_price = (i > 0) ? path->prices[i-1] : price;
        
        buffer_.times.push_back(static_cast<double>(i));
        buffer_.opens.push_back(prev_price);
        buffer_.highs.push_back(price * (1.0 + path->volatilities[i] * 0.5));
        buffer_.lows.push_back(price * (1.0 - path->volatilities[i] * 0.5));
        buffer_.closes.push_back(price);
        
        if (i < path->volumes.size()) {
            buffer_.volumes.push_back(path->volumes[i]);
        }
        
        if (i < path->regimes.size()) {
            buffer_.regimes.push_back(path->regimes[i]);
        }
    }
    
    // Create ImPlot window
    if (ImGui::Begin("Market Simulation", nullptr, ImGuiWindowFlags_NoCollapse)) {
        setupPlotStyle(config);
        
        // Main price chart
        if (ImPlot::BeginPlot("Price Chart", ImVec2(-1, config.show_volume ? -200 : -1))) {
            ImPlot::SetupAxis(ImAxis_X1, "Step");
            ImPlot::SetupAxis(ImAxis_Y1, "Price");
            
            if (config.auto_fit) {
                ImPlot::SetupAxesLimits(
                    static_cast<double>(start),
                    static_cast<double>(current_step + 1),
                    *std::min_element(buffer_.lows.begin(), buffer_.lows.end()) * 0.99,
                    *std::max_element(buffer_.highs.begin(), buffer_.highs.end()) * 1.01,
                    ImGuiCond_Always
                );
            }
            
            // Draw candlesticks
            for (size_t i = 0; i < buffer_.times.size(); ++i) {
                ImVec4 color = (buffer_.closes[i] >= buffer_.opens[i]) ?
                               hexToImColor(config.bull_color) :
                               hexToImColor(config.bear_color);
                
                drawCandlestick(
                    buffer_.times[i],
                    buffer_.opens[i],
                    buffer_.highs[i],
                    buffer_.lows[i],
                    buffer_.closes[i],
                    color
                );
            }
            
            ImPlot::EndPlot();
        }
        
        // Volume chart
        if (config.show_volume && !buffer_.volumes.empty()) {
            if (ImPlot::BeginPlot("Volume", ImVec2(-1, 150))) {
                ImPlot::SetupAxis(ImAxis_X1, "Step");
                ImPlot::SetupAxis(ImAxis_Y1, "Volume");
                
                if (config.auto_fit) {
                    ImPlot::SetupAxesLimits(
                        static_cast<double>(start),
                        static_cast<double>(current_step + 1),
                        0,
                        *std::max_element(buffer_.volumes.begin(), buffer_.volumes.end()) * 1.1,
                        ImGuiCond_Always
                    );
                }
                
                ImPlot::PlotBars("Volume", 
                                 buffer_.times.data(), 
                                 buffer_.volumes.data(), 
                                 buffer_.volumes.size(), 
                                 0.8);
                
                ImPlot::EndPlot();
            }
        }

if (config.show_indicators && !path->indicators.sma_20.empty()) {
    if (ImGui::Begin("Indicators", nullptr, ImGuiWindowFlags_NoCollapse)) {
        // RSI
        if (ImPlot::BeginPlot("RSI", ImVec2(-1, 150))) {
            ImPlot::SetupAxis(ImAxis_X1, "Step");
            ImPlot::SetupAxis(ImAxis_Y1, "RSI");
            ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 100);

            // Plot RSI line first
            ImPlot::SetNextLineStyle(ImVec4(1, 0.5f, 0, 1));
            ImPlot::PlotLine("RSI",
                             buffer_.times.data(),
                             path->indicators.rsi.data() + start,
                             buffer_.times.size());

            // Add horizontal lines for overbought/oversold
            // PlotHLines takes an array of y-values where to draw horizontal lines
            double overbought[] = {70.0};
            double oversold[] = {30.0};

            // Overbought line (70)
            ImPlot::SetNextLineStyle(ImVec4(1, 0, 0, 0.3));
            ImPlot::PlotLine("Overbought", overbought, 1);

            // Oversold line (30)
            ImPlot::SetNextLineStyle(ImVec4(0, 1, 0, 0.3));
            ImPlot::PlotLine("Oversold", oversold, 1);

            ImPlot::EndPlot();
        }

        ImGui::End();
    }
}
        
        /*if (config.show_indicators && !path->indicators.sma_20.empty()) {
            if (ImGui::Begin("Indicators", nullptr, ImGuiWindowFlags_NoCollapse)) {
                // RSI
                if (ImPlot::BeginPlot("RSI", ImVec2(-1, 150))) {
                    ImPlot::SetupAxis(ImAxis_X1, "Step");
                    ImPlot::SetupAxis(ImAxis_Y1, "RSI");
                    ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 100);
                    
                    // Add overbought/oversold lines
                    ImVec4 overbought_color = ImVec4(1, 0, 0, 0.3);
                    ImVec4 oversold_color = ImVec4(0, 1, 0, 0.3);
                    
                    ImPlot::PushStyleColor(ImPlotCol_Line, overbought_color);
                    ImPlot::PlotInfLines("Overbought", ImPlot::GetCurrentPlot()->Axes[ImAxis_Y1].Range.Min,
                                       70, static_cast<int>(buffer_.times.size()));
                    ImPlot::PushStyleColor(ImPlotCol_Line, oversold_color);
                    ImPlot::PlotInfLines("Oversold", ImPlot::GetCurrentPlot()->Axes[ImAxis_Y1].GetPlotMin(),
                                       30, static_cast<int>(buffer_.times.size()));
                    
                    // Plot RSI
                    ImPlot::SetNextLineStyle(ImVec4(1, 0.5, 0, 1));
                    ImPlot::PlotLine("RSI", 
                                     buffer_.times.data(), 
                                     path->indicators.rsi.data() + start, 
                                     buffer_.times.size());
                    
                    ImPlot::EndPlot();
                }
                
                ImGui::End();
            }
        }*/
        
        ImGui::End();
    }
    
    // Handle user input
    handleInput();
}

void ImPlotSimulationPlotter::drawCandlestick(
    double time, double open, double high, double low, double close, const ImVec4& color) {

    ImDrawList* draw_list = ImPlot::GetPlotDrawList();

    // Convert time to plot coordinates
    ImPlotPoint p_low = ImPlot::PlotToPixels(time, low);
    ImPlotPoint p_high = ImPlot::PlotToPixels(time, high);

    // Draw the high-low line (wick)
    draw_list->AddLine(
        ImVec2(p_low.x, p_low.y),
        ImVec2(p_high.x, p_high.y),
        ImGui::ColorConvertFloat4ToU32(color),
        1.0f
    );

    // Draw the open-close body
    double body_bottom = std::min(open, close);
    double body_top = std::max(open, close);
    double half_width = 0.3;

    ImPlotPoint p_body_bottom = ImPlot::PlotToPixels(time - half_width, body_bottom);
    ImPlotPoint p_body_top = ImPlot::PlotToPixels(time + half_width, body_top);

    draw_list->AddRectFilled(
        ImVec2(p_body_bottom.x, p_body_bottom.y),
        ImVec2(p_body_top.x, p_body_top.y),
        ImGui::ColorConvertFloat4ToU32(color)
    );
}

/*void ImPlotSimulationPlotter::drawCandlestick(
    double time, double open, double high, double low, double close, const ImVec4& color) {
    
    ImDrawList* draw_list = ImPlot::GetPlotDrawList();
    ImPlot::GetPlotContext();
    
    ImPlot::PushStyleColor(ImPlotCol_Line, color);
    
    // Draw the high-low line (wick)
    ImPlot::PlotLine("##wick", 
                     std::vector<double>{time, time}.data(),
                     std::vector<double>{high, low}.data(),
                     2);
    
    // Draw the open-close body
    double half_width = 0.3;
    ImPlot::PlotRectangles("##body", 
                           std::vector<double>{time}.data(),
                           std::vector<double>{std::min(open, close)}.data(),
                           std::vector<double>{half_width * 2}.data(),
                           std::vector<double>{std::abs(close - open)}.data());
    
    ImPlot::PopStyleColor();
}*/

ImVec4 ImPlotSimulationPlotter::hexToImColor(const std::string& hex, float alpha) {
    unsigned int r, g, b;
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
}

void ImPlotSimulationPlotter::handleInput() {
    ImGuiIO& io = ImGui::GetIO();
    
    // Playback controls
    if (ImGui::Begin("Controls", nullptr, ImGuiWindowFlags_NoCollapse)) {
        if (is_playing_) {
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
            // Reset to beginning
        }
        
        ImGui::SameLine();
        ImGui::SliderFloat("Speed", &playback_speed_, 0.1f, 5.0f, "%.1fx");
        
        ImGui::End();
    }
}

void ImPlotSimulationPlotter::startAnimation() {
    animation_running_ = true;
    is_playing_ = true;
    last_update_ = std::chrono::steady_clock::now();
    
    render_thread_ = std::thread([this]() {
        while (animation_running_ && !window_closed_) {
            renderLoop();
            
            // Control update rate based on playback speed
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - last_update_).count();
            
            int target_delay = static_cast<int>(100 / playback_speed_);
            if (elapsed < target_delay) {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(target_delay - elapsed)
                );
            }
            
            last_update_ = now;
        }
    });
}

void ImPlotSimulationPlotter::renderLoop() {
    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    
    // Create dockspace
    ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());
    
    // Render plots (called from main simulation loop)
    
    // Rendering
    ImGui::Render();
    int display_w, display_h;
    glfwGetFramebufferSize(glfwGetCurrentContext(), &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    
    glfwSwapBuffers(glfwGetCurrentContext());
    glfwPollEvents();
    
    window_closed_ = glfwWindowShouldClose(glfwGetCurrentContext());
}

void ImPlotSimulationPlotter::stopAnimation() {
    animation_running_ = false;
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
