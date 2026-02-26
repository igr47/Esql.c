#include "plotter_includes/implotter.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <implot.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
//#include <GLFW/glfw3.h>
//#include <GL/glew.h>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <cstdio>
#include <atomic>

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
    // Constructor
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
    std::cout << "[ImPlot] Initializing..." << std::endl;
    
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
    
    // GLFW window hints
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
    } else {
        std::cout << "[Implot] GLFW Window  created successfully. " << std::endl;
    }

    // Make context current in THIS thread
    std::cout << "[Implot] Entering window context creation." << std::endl;
    glfwMakeContextCurrent(window_);

    // Verify context is current
    if (glfwGetCurrentContext() != window_) {
        std::cerr << "[ImPlot] Failed to make OpenGL context current" << std::endl;
        return;
    } else {
        std::cout << "[Implot] Successfully madde OpenGL context current." << std::endl;
    }

    glfwSwapInterval(1); // Enable vsync

    // Initialize GLEW - with better error handling
    glewExperimental = GL_TRUE;

    // Clear any previous errors
    while (glGetError() != GL_NO_ERROR) {}

    GLenum glew_err = glewInit();
    if (glew_err != GLEW_OK) {
       std::cerr << "[ImPlot] Failed to initialize GLEW: "
                  << glewGetErrorString(glew_err) << std::endl;
        //glfwDestroyWindow(window_);
        //window_ = nullptr;
        //return;
    }
    std::cerr << "GLEW error code: " << glew_err << std::endl;

    // Clear the error that GLEW might have generated
    glGetError();

    // Verify OpenGL context is working
    const GLubyte* version = glGetString(GL_VERSION);
    const GLubyte* glsl_version = glGetString(GL_SHADING_LANGUAGE_VERSION);

    if (!version) {
        std::cerr << "[ImPlot] Failed to get OpenGL version - context may be invalid" << std::endl;
        glfwDestroyWindow(window_);
        window_ = nullptr;
        return;
    }

    std::cout << "[ImPlot] OpenGL Version: " << version << std::endl;
    std::cout << "[ImPlot] GLSL Version: " << (glsl_version ? glsl_version : (const GLubyte*)"unknown") << std::endl;
    std::cout << "[ImPlot] GLEW Version: " << glewGetString(GLEW_VERSION) << std::endl;

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();

    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

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

    if (!ImGui_ImplOpenGL3_Init("#version 330")) {
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
    
    // Update buffer with new data up to current_step
    {
        std::lock_guard<std::mutex> buffer_lock(buffer_.mutex);
        
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
            double price = path->prices[i];
            double prev_price = (i > 0) ? path->prices[i-1] : price;
            
            buffer_.times.push_back(static_cast<double>(i));
            buffer_.opens.push_back(prev_price);
            
            double volatility = (i < path->volatilities.size()) ? path->volatilities[i] : 0.01;
            buffer_.highs.push_back(price * (1.0 + volatility * 0.5));
            buffer_.lows.push_back(price * (1.0 - volatility * 0.5));
            buffer_.closes.push_back(price);
            
            if (i < path->volumes.size()) {
                buffer_.volumes.push_back(path->volumes[i]);
            } else {
                buffer_.volumes.push_back(1000.0);
            }
            
            if (i < path->regimes.size()) {
                buffer_.regimes.push_back(path->regimes[i]);
            }
        }
    }
    
    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    
    // Create main dock space
    ImGuiID dockspace_id = ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());
    
    // Create ImGui window for the plot
    ImGui::SetNextWindowDockID(dockspace_id, ImGuiCond_FirstUseEver);
    ImGui::Begin("Market Simulation", nullptr, ImGuiWindowFlags_NoCollapse);
    
    setupPlotStyle(config);
    
    // Main price chart
    if (ImPlot::BeginPlot("Price Chart", ImVec2(-1, config.show_volume ? -200 : -1))) {
        ImPlot::SetupAxis(ImAxis_X1, "Step");
        ImPlot::SetupAxis(ImAxis_Y1, "Price");
        
        if (config.auto_fit && !buffer_.lows.empty() && !buffer_.highs.empty()) {
            double min_price = *std::min_element(buffer_.lows.begin(), buffer_.lows.end()) * 0.99;
            double max_price = *std::max_element(buffer_.highs.begin(), buffer_.highs.end()) * 1.01;
            double min_step = buffer_.times.front();
            double max_step = buffer_.times.back();
            ImPlot::SetupAxesLimits(min_step, max_step, min_price, max_price, ImGuiCond_Always);
        }
        
        // Draw candlesticks
        std::lock_guard<std::mutex> buffer_lock(buffer_.mutex);
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
                double max_volume = *std::max_element(buffer_.volumes.begin(), buffer_.volumes.end()) * 1.1;
                ImPlot::SetupAxisLimits(ImAxis_Y1, 0, max_volume, ImGuiCond_Always);
            }
            
            ImPlot::PlotBars("Volume", 
                             buffer_.times.data(), 
                             buffer_.volumes.data(), 
                             buffer_.volumes.size(), 
                             0.8);
            
            ImPlot::EndPlot();
        }
    }
    
    ImGui::End();
    
    // Controls window
    ImGui::Begin("Controls", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
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
        // Reset to beginning
    }
    
    ImGui::SameLine();
    float speed = playback_speed_.load();
    if (ImGui::SliderFloat("Speed", &speed, 0.1f, 5.0f, "%.1fx")) {
        playback_speed_ = speed;
    }

    ImGui::End();
    
    // Rendering
    ImGui::Render();
    int display_w, display_h;
    glfwGetFramebufferSize(window_, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    
    // Update and render additional Platform Windows
    if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
        GLFWwindow* backup_current_context = glfwGetCurrentContext();
        ImGui::UpdatePlatformWindows();
        ImGui::RenderPlatformWindowsDefault();
        glfwMakeContextCurrent(backup_current_context);
    }
    
    glfwSwapBuffers(window_);
    glfwPollEvents();
    
    // Check if window should close
    if (glfwWindowShouldClose(window_)) {
        window_closed_ = true;
    }
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
}

void ImPlotSimulationPlotter::handleInput() {
    ImGuiIO& io = ImGui::GetIO();
    // Handle additional input if needed
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
