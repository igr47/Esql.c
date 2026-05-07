#include "plotter_includes/realtime_candlestick_plotter.h"
#include "plotter_includes/real_time_plotter_parser.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "implot.h"
#include "implot_internal.h"
#include <GLFW/glfw3.h>
#include <cfloat>
#include <numeric>
#include <algorithm>
#include <ostream>
#include <sstream>
#include <iomanip>

namespace Visualization {

RealTimeCandlestickPlotter::RealTimeCandlestickPlotter()
    : running_(false)
    , shouldStop_(false)
    , paused_(false)
    , hasCurrentCandle_(false)
    , priceMin_(std::numeric_limits<double>::max())
    , priceMax_(std::numeric_limits<double>::lowest())
    , maxCandles_(100)
    , intervalSeconds_(5)
    , targetFrameRate_(60.0)
{
    stats_ = Statistics{0, 0, 0, 0, 0, 0, 0, std::chrono::seconds(0)};
}

RealTimeCandlestickPlotter::~RealTimeCandlestickPlotter() {
    stop();
}

void RealTimeCandlestickPlotter::initialize(const AST::RealTimeCandlestickStatement& config) {
    title_ = config.title;
    xLabel_ = config.xLabel;
    yLabel_ = config.yLabel;
    maxCandles_ = config.maxCandles;
    intervalSeconds_ = config.intervalSeconds;
    outputFile_ = config.outputFile;
    bullishColor_ = config.bullishColor;
    bearishColor_ = config.bearishColor;
    mapping_ = config.getMapping();
}

void RealTimeCandlestickPlotter::start() {
    std::cout << "[DEBUG] Entered start in ploter" << std::endl;
    if (running_) return;
    
    running_ = true;
    shouldStop_ = false;
    
    // Initialize GLFW and ImGui in the main thread
    if (!glfwInit()) {
        running_ = false;
        throw std::runtime_error("Failed to initialize GLFW");
    } else {
        std::cout << "[DEBUG] Initialized GLFW" << std::endl; 
    }
    
    std::cout << "[DEBUG] STARTING PLOTTING THREAD" << std::endl;
    plotThread_ = std::thread(&RealTimeCandlestickPlotter::plotThreadFunction, this);
    std::cout << "[DEBUG] Finished starting plotting thread and now leaving start." << std::endl;
}

void RealTimeCandlestickPlotter::stop() {
    if (!running_) return;
    
    shouldStop_ = true;
    cv_.notify_all();
    
    if (plotThread_.joinable()) {
        plotThread_.join();
    }
    
    running_ = false;
}

void RealTimeCandlestickPlotter::addCandlestick(const RealTimeCandlestick& candle) {
    std::cout << "[DEBUG] Entered add candle stick" << std::endl;
    std::lock_guard<std::mutex> lock(dataMutex_);
    
    candles_.push_back(candle);
    
    // Keep only the last N candles
    while (candles_.size() > static_cast<size_t>(maxCandles_)) {
        candles_.pop_front();
    }
    
    // Update statistics
    {
        std::cout << "[DEBUG] Calculating statistics." << std::endl;
        std::lock_guard<std::mutex> statsLock(statsMutex_);
        stats_.totalCandles = candles_.size();
        stats_.highestHigh = std::max(stats_.highestHigh, candle.high);
        stats_.lowestLow = std::min(stats_.lowestLow, candle.low);
        stats_.totalVolume += candle.volume;
        stats_.avgBodyLength = (stats_.avgBodyLength * (stats_.totalCandles - 1) + candle.getBodyLength()) / stats_.totalCandles;
        
        if (candle.isBullish()) {
            stats_.bullishCount++;
        } else {
            stats_.bearishCount++;
        }
        std::cout << "[DEBUG] Finished calcuating and leaving" << std::endl;
    }
    std::cout << "[DEBUG] Leaving add candle stick" << std::endl;
    updatePriceRangeInternal();
    calculateMovingAveragesInternal();
}

void RealTimeCandlestickPlotter::updateCurrentCandle(double price, double volume) {
    std::lock_guard<std::mutex> lock(dataMutex_);
    std::cout << "[DEBUG] Entered updateCurrentCandle" << std::endl;
    
    auto now = std::chrono::steady_clock::now();
    
    if (!hasCurrentCandle_) {
        // Start a new candle
        currentCandle_ = RealTimeCandlestick();
        currentCandle_.open = price;
        currentCandle_.high = price;
        currentCandle_.low = price;
        currentCandle_.close = price;
        currentCandle_.volume = volume;
        currentCandle_.timestamp = std::chrono::system_clock::now();
        currentCandle_.index = candles_.size();
        candleStartTime_ = now;
        hasCurrentCandle_ = true;
    } else {
        // Check if candle period is complete
        double elapsed = std::chrono::duration<double>(now - candleStartTime_).count();
        
        if (elapsed >= intervalSeconds_) {
            // Complete the current candle and start a new one
            currentCandle_.close = price;
            currentCandle_.high = std::max(currentCandle_.high, price);
            currentCandle_.low = std::min(currentCandle_.low, price);
            currentCandle_.volume += volume;
            
            addCandlestick(currentCandle_);
            
            // Start new candle
            currentCandle_ = RealTimeCandlestick();
            currentCandle_.open = price;
            currentCandle_.high = price;
            currentCandle_.low = price;
            currentCandle_.close = price;
            currentCandle_.volume = volume;
            currentCandle_.timestamp = std::chrono::system_clock::now();
            currentCandle_.index = candles_.size();
            candleStartTime_ = now;
        } else {
            // Update current candle
            currentCandle_.high = std::max(currentCandle_.high, price);
            currentCandle_.low = std::min(currentCandle_.low, price);
            currentCandle_.close = price;
            currentCandle_.volume += volume;
        }
    }
    std::cout << "[DEBUG] Leaving updateCurrentCandle" << std::endl;
}

void RealTimeCandlestickPlotter::updatePriceRange() {
    std::lock_guard<std::mutex> lock(dataMutex_);
    updatePriceRangeInternal();
}

void RealTimeCandlestickPlotter::updatePriceRangeInternal() {
    std::cout << "[DEBUG] Entering the second lock guard" << std::endl;
    std::lock_guard<std::mutex> rangeLock(rangeMutex_);
    std::cout << "[DEBUG] Finished second lock_guard" << std::endl;
    
    if (candles_.empty()) return;
    
    priceMin_ = candles_[0].low;
    priceMax_ = candles_[0].high;
    
    for (const auto& candle : candles_) {
        priceMin_ = std::min(priceMin_, candle.low);
        priceMax_ = std::max(priceMax_, candle.high);
    }
    
    if (hasCurrentCandle_) {
        priceMin_ = std::min(priceMin_, currentCandle_.low);
        priceMax_ = std::max(priceMax_, currentCandle_.high);
    }
    
    // Add 5% padding
    double padding = (priceMax_ - priceMin_) * 0.05;
    priceMin_ -= padding;
    priceMax_ += padding;
    std::cout << "[DEBUG] moving out of updatePriceRange" << std::endl;
}

void RealTimeCandlestickPlotter::calculateMovingAverages() {
    std::lock_guard<std::mutex> lock(dataMutex_);
    calculateMovingAveragesInternal();
}

void RealTimeCandlestickPlotter::calculateMovingAveragesInternal() {
    //std::lock_guard<std::mutex> lock(dataMutex_);
    std::cout << "[DEBUG] Entered calculateMovingAverages" << std::endl;
    
    if (candles_.empty()) return;
    
    // Calculate MA5
    ma5_.clear();
    for (size_t i = 4; i < candles_.size(); i++) {
        double sum = 0;
        for (size_t j = i - 4; j <= i; j++) {
            sum += candles_[j].close;
        }
        ma5_.push_back(sum / 5);
    }
    
    // Calculate MA10
    ma10_.clear();
    for (size_t i = 9; i < candles_.size(); i++) {
        double sum = 0;
        for (size_t j = i - 9; j <= i; j++) {
            sum += candles_[j].close;
        }
        ma10_.push_back(sum / 10);
    }
    
    // Calculate MA20
    ma20_.clear();
    if (candles_.size() >= 20) {
        for (size_t i = 19; i < candles_.size(); i++) {
            double sum = 0;
            for (size_t j = i - 19; j <= i; j++) {
                sum += candles_[j].close;
            }
            ma20_.push_back(sum / 20);
        }
    }
    std::cout << "[DEBUG] Leaving calculateMovingAverages" << std::endl;
}

ImU32 RealTimeCandlestickPlotter::toImU32(const std::string& colorStr) {
    // Parse hex color (#RRGGBB or #RRGGBBAA)
    unsigned int r = 0, g = 0, b = 0, a = 255;
    
    if (colorStr.size() >= 7 && colorStr[0] == '#') {
        std::string hex = colorStr.substr(1);
        if (hex.size() >= 6) {
            r = std::stoi(hex.substr(0, 2), nullptr, 16);
            g = std::stoi(hex.substr(2, 2), nullptr, 16);
            b = std::stoi(hex.substr(4, 2), nullptr, 16);
            if (hex.size() >= 8) {
                a = std::stoi(hex.substr(6, 2), nullptr, 16);
            }
        }
    }
    
    return IM_COL32(r, g, b, a);
}

void RealTimeCandlestickPlotter::renderCandlestickChart() {
    std::lock_guard<std::mutex> lock(dataMutex_);
    std::lock_guard<std::mutex> rangeLock(rangeMutex_);
    
    if (candles_.empty() && !hasCurrentCandle_) {
        ImGui::Text("Waiting for data...");
        return;
    }
    
    ImVec2 plotSize = ImGui::GetContentRegionAvail();
    plotSize.y -= 150; // Reserve space for volume and stats
    
    if (ImPlot::BeginPlot("##CandlestickChart", plotSize)) {
        // Setup axes
        double xMax = std::max(candles_.size(), hasCurrentCandle_ ? candles_.size() + 1 : candles_.size());
        ImPlot::SetupAxisLimits(ImAxis_X1, 0, xMax, ImGuiCond_Always);
        ImPlot::SetupAxisLimits(ImAxis_Y1, priceMin_, priceMax_, ImGuiCond_Always);
        
        ImPlot::SetupAxis(ImAxis_X1, xLabel_.c_str());
        ImPlot::SetupAxis(ImAxis_Y1, yLabel_.c_str());
        
        ImDrawList* drawList = ImPlot::GetPlotDrawList();
        
        // Calculate candle width
        double xRange = priceMax_ - priceMin_;
        float candleWidthPx = std::max(4.0f, std::min(20.0f, 800.0f / std::max(1.0f, (float)candles_.size())));
        
        // Plot completed candles
        for (size_t i = 0; i < candles_.size(); i++) {
            const auto& candle = candles_[i];
            double x = static_cast<double>(i);
            
            // Convert to pixel coordinates
            ImVec2 highPixel = ImPlot::PlotToPixels(ImVec2(x, candle.high));
            ImVec2 lowPixel = ImPlot::PlotToPixels(ImVec2(x, candle.low));
            ImVec2 bodyTopPixel = ImPlot::PlotToPixels(ImVec2(x, candle.getBodyHigh()));
            ImVec2 bodyBottomPixel = ImPlot::PlotToPixels(ImVec2(x, candle.getBodyLow()));
            
            float bodyLeft = highPixel.x - candleWidthPx * 0.5f;
            float bodyRight = highPixel.x + candleWidthPx * 0.5f;
            
            ImU32 color = candle.isBullish() ? toImU32(bullishColor_) : toImU32(bearishColor_);
            
            // Draw wick
            drawList->AddLine(ImVec2(highPixel.x, highPixel.y), 
                            ImVec2(lowPixel.x, lowPixel.y), 
                            color, 1.0f);
            
            // Draw body
            drawList->AddRectFilled(ImVec2(bodyLeft, std::min(bodyTopPixel.y, bodyBottomPixel.y)),
                                  ImVec2(bodyRight, std::max(bodyTopPixel.y, bodyBottomPixel.y)),
                                  color);
            
            // Draw body outline
            drawList->AddRect(ImVec2(bodyLeft, std::min(bodyTopPixel.y, bodyBottomPixel.y)),
                            ImVec2(bodyRight, std::max(bodyTopPixel.y, bodyBottomPixel.y)),
                            IM_COL32(255, 255, 255, 200), 0.0f, 0, 1.0f);
        }
        
        // Plot current candle if active
        if (hasCurrentCandle_) {
            double x = static_cast<double>(candles_.size());
            
            ImVec2 highPixel = ImPlot::PlotToPixels(ImVec2(x, currentCandle_.high));
            ImVec2 lowPixel = ImPlot::PlotToPixels(ImVec2(x, currentCandle_.low));
            ImVec2 openPixel = ImPlot::PlotToPixels(ImVec2(x, currentCandle_.open));
            ImVec2 closePixel = ImPlot::PlotToPixels(ImVec2(x, currentCandle_.close));
            
            float bodyLeft = highPixel.x - candleWidthPx * 0.5f;
            float bodyRight = highPixel.x + candleWidthPx * 0.5f;
            
            bool bullish = currentCandle_.close > currentCandle_.open;
            ImU32 color = bullish ? toImU32(bullishColor_) : toImU32(bearishColor_);
            
            // Draw wick
            drawList->AddLine(ImVec2(highPixel.x, highPixel.y), 
                            ImVec2(lowPixel.x, lowPixel.y), 
                            color, 1.0f);
            
            float top = std::min(openPixel.y, closePixel.y);
            float bottom = std::max(openPixel.y, closePixel.y);
            
            if (std::abs(top - bottom) < 1.0f) bottom = top + 1.0f;
            
            // Add slight transparency for live candle
            color = (color & 0x00FFFFFF) | (0xCC << 24); // ~80% opacity
            
            drawList->AddRectFilled(ImVec2(bodyLeft, top),
                                  ImVec2(bodyRight, bottom),
                                  color);
            
            drawList->AddRect(ImVec2(bodyLeft, top),
                            ImVec2(bodyRight, bottom),
                            IM_COL32(255, 255, 255, 200), 0.0f, 0, 1.0f);
        }
        
        // Plot moving averages if enough data
        if (ma5_.size() > 0) {
            std::vector<double> xValues(ma5_.size());
            for (size_t i = 0; i < ma5_.size(); i++) {
                xValues[i] = i + 4;
            }
            ImPlotSpec ma5_spec;
            ma5_spec.LineColor = ImVec4(1.0f, 1.0f, 0.0f, 1.0f);  // Yellow
            ma5_spec.LineWeight = 1.5f;
            ImPlot::PlotLine("MA5", xValues.data(), ma5_.data(), ma5_.size(),ma5_spec);
        }
        
        if (ma10_.size() > 0) {
            std::vector<double> xValues(ma10_.size());
            for (size_t i = 0; i < ma10_.size(); i++) {
                xValues[i] = i + 9;
            }
            ImPlotSpec ma10_spec;
            ma10_spec.LineColor = ImVec4(0.0f, 1.0f, 1.0f, 1.0f);  // Cyan
            ma10_spec.LineWeight = 1.5f;
            ImPlot::PlotLine("MA10", xValues.data(), ma10_.data(), ma10_.size(),ma10_spec);
        }
        
        if (ma20_.size() > 0) {
            std::vector<double> xValues(ma20_.size());
            for (size_t i = 0; i < ma20_.size(); i++) {
                xValues[i] = i + 19;
            }
            ImPlotSpec ma20_spec;
            ma20_spec.LineColor = ImVec4(1.0f, 0.0f, 1.0f, 1.0f);  // Magenta
            ma20_spec.LineWeight = 1.5f;
            ImPlot::PlotLine("MA20", xValues.data(), ma20_.data(), ma20_.size());
        }
        
        ImPlot::EndPlot();
    }
    
    // Volume subplot
    if (!candles_.empty() && mapping_.useVolume) {
        if (ImPlot::BeginPlot("##VolumeChart", ImVec2(-1, 100))) {
            ImPlot::SetupAxis(ImAxis_X1, "Candle");
            ImPlot::SetupAxis(ImAxis_Y1, "Volume");
            
            for (size_t i = 0; i < candles_.size(); i++) {
                double x = i;
                ImU32 color = candles_[i].isBullish() ? 
                    IM_COL32(0, 100, 0, 150) : IM_COL32(100, 0, 0, 150);
                
                std::vector<double> xs = {x};
                std::vector<double> vols = {candles_[i].volume};

                ImPlotSpec volume_spec;
                volume_spec.FillColor = candles_[i].isBullish() ? 
                    ImVec4(0.0f, 0.5f, 0.0f, 0.6f) : ImVec4(0.5f, 0.0f, 0.0f, 0.6f);
                volume_spec.FillAlpha = 0.6f;
                ImPlot::PlotBars("Volume", xs.data(), vols.data(), xs.size(), 0.8, volume_spec);
            }
            
            ImPlot::EndPlot();
        }
    }
}

void RealTimeCandlestickPlotter::renderStatisticsPanel() {
    std::lock_guard<std::mutex> statsLock(statsMutex_);
    
    ImGui::Begin("Candlestick Statistics");
    
    if (stats_.totalCandles > 0) {
        ImGui::Text("Total Candles: %zu", stats_.totalCandles);
        ImGui::Text("Price Range: %.4f - %.4f", stats_.lowestLow, stats_.highestHigh);
        ImGui::Text("Total Volume: %.2f", stats_.totalVolume);
        ImGui::Text("Avg Body Length: %.4f", stats_.avgBodyLength);
        
        double bullRatio = stats_.totalCandles > 0 ? 
            (static_cast<double>(stats_.bullishCount) / stats_.totalCandles) * 100.0 : 0.0;
        ImGui::Text("Bullish: %zu (%.1f%%)", stats_.bullishCount, bullRatio);
        ImGui::Text("Bearish: %zu (%.1f%%)", stats_.bearishCount, 100.0 - bullRatio);
        
        // Current candle info
        std::lock_guard<std::mutex> lock(dataMutex_);
        if (hasCurrentCandle_) {
            auto elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - candleStartTime_).count();
            double remaining = std::max(0.0, intervalSeconds_ - elapsed);
            
            ImGui::Separator();
            ImGui::Text("Current Candle:");
            ImGui::Text("  Open: %.4f", currentCandle_.open);
            ImGui::Text("  High: %.4f", currentCandle_.high);
            ImGui::Text("  Low: %.4f", currentCandle_.low);
            ImGui::Text("  Close: %.4f", currentCandle_.close);
            ImGui::Text("  Time Remaining: %.1f sec", remaining);
        }
    } else {
        ImGui::Text("No candles yet. Waiting for data...");
    }
    
    ImGui::End();
}

void RealTimeCandlestickPlotter::plotThreadFunction() {
    // Setup GLFW window - FOLLOWING EXACT SAME PATTERN AS RealTimePlotter
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
    GLFWwindow* window = glfwCreateWindow(1280, 720, 
        (title_.empty() ? "Real-Time Candlestick Chart" : title_.c_str()), 
        NULL, NULL);
    
    if (!window) {
        std::cerr << "Failed to create GLFW window for candlestick chart" << std::endl;
        running_ = false;
        return;
    } else {
        std::cout << "[DEBUG] Initialized window" << std::endl;
    }
    
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    
    // Setup ImGui - EXACT SAME PATTERN
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
    
    ImGui::StyleColorsDark();
    
    // Dark theme for ImPlot - SAME AS RealTimePlotter
    ImPlotStyle& plotStyle = ImPlot::GetStyle();
    plotStyle.Colors[ImPlotCol_PlotBg] = ImVec4(25/255.0f, 25/255.0f, 35/255.0f, 1.0f);
    plotStyle.Colors[ImPlotCol_FrameBg] = ImVec4(30/255.0f, 30/255.0f, 40/255.0f, 1.0f);
    plotStyle.Colors[ImPlotCol_PlotBorder] = ImVec4(60/255.0f, 60/255.0f, 80/255.0f, 1.0f);
    
    auto lastFrameTime = std::chrono::steady_clock::now();
    double frameDuration = 1.0 / targetFrameRate_;
    int frameCount = 0;
    auto lastFPSTime = std::chrono::steady_clock::now();
    
    // Main loop - FOLLOWING EXACT SAME PATTERN AS RealTimePlotter
    while (!shouldStop_ && !glfwWindowShouldClose(window)) {
        auto frameStart = std::chrono::steady_clock::now();
        double deltaTime = std::chrono::duration<double>(frameStart - lastFrameTime).count();
        
        if (deltaTime < frameDuration && !paused_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        
        lastFrameTime = frameStart;
        
        // Update FPS - SAME PATTERN
        frameCount++;
        auto now = std::chrono::steady_clock::now();
        double fpsDelta = std::chrono::duration<double>(now - lastFPSTime).count();
        if (fpsDelta >= 1.0) {
            {
                std::lock_guard<std::mutex> statsLock(statsMutex_);
                // We'll track this through stats if needed
            }
            frameCount = 0;
            lastFPSTime = now;
        }
        
        // Update price range periodically - SAME PATTERN
        if (candles_.size() % 10 == 0 || !hasCurrentCandle_) {
            updatePriceRange();
        }
        
        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        
        // Main plot window
        ImGui::Begin("Real-Time Candlestick Chart", nullptr, 
                    ImGuiWindowFlags_NoCollapse);
        
        renderCandlestickChart();
        
        ImGui::End();
        
        // Statistics panel
        renderStatisticsPanel();
        
        // Render - EXACT SAME PATTERN
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.1f, 0.1f, 0.12f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        
        glfwSwapBuffers(window);
        glfwPollEvents();
        
        // Handle close
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            shouldStop_ = true;
        }
    }
    
    // Cleanup - EXACT SAME PATTERN
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    
    glfwDestroyWindow(window);
    
    // Save output if requested
    if (!outputFile_.empty()) {
        // Save the final plot
    }
}

/*void RealTimeCandlestickPlotter::plotThreadFunction() {
    // Setup GLFW window
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
    GLFWwindow* window = glfwCreateWindow(1280, 720, 
        (title_.empty() ? "Real-Time Candlestick Chart" : title_.c_str()), 
        NULL, NULL);
    
    if (!window) {
        running_ = false;
        return;
    }
    
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    
    // Setup ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
    
    ImGui::StyleColorsDark();
    
    auto lastFrameTime = std::chrono::steady_clock::now();
    double frameDuration = 1.0 / targetFrameRate_;
    
    while (!shouldStop_ && !glfwWindowShouldClose(window)) {
        auto frameStart = std::chrono::steady_clock::now();
        double deltaTime = std::chrono::duration<double>(frameStart - lastFrameTime).count();
        
        if (deltaTime < frameDuration) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        
        lastFrameTime = frameStart;
        
        // Update price range periodically
        if (candles_.size() % 10 == 0 || !hasCurrentCandle_) {
            updatePriceRange();
        }
        
        // Render
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        
        // Main plot window
        ImGui::Begin("Real-Time Candlestick Chart", nullptr, 
                    ImGuiWindowFlags_NoCollapse);
        
        renderCandlestickChart();
        
        ImGui::End();
        
        renderStatisticsPanel();
        
        // Render
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        
        glfwSwapBuffers(window);
        glfwPollEvents();
        
        // Handle window close
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            shouldStop_ = true;
        }
    }
    
    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    
    glfwDestroyWindow(window);
    
    // Save output if requested
    if (!outputFile_.empty()) {
        // Save the final plot
        // Implementation depends on your screenshot/save functionality
    }
}*/

RealTimeCandlestickPlotter::Statistics RealTimeCandlestickPlotter::getStatistics() const {
    std::lock_guard<std::mutex> lock(statsMutex_);
    return stats_;
}

} // namespace Visualization
