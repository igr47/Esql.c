#include "plotter_includes/realtime_plotter.h"
#include "plotter_includes/real_time_plotter_parser.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "implot.h"
#include <GLFW/glfw3.h>
#include <numeric>
#include <algorithm>
#include <cmath>

namespace Visualization {

void RealTimePlotter::initDefaultColors() {
    defaultColors_[0] = Color(1.0f, 0.39f, 0.28f, 1.0f);  // Tomato
    defaultColors_[1] = Color(0.24f, 0.70f, 0.44f, 1.0f);  // MediumSeaGreen
    defaultColors_[2] = Color(0.53f, 0.81f, 0.92f, 1.0f);  // SkyBlue
    defaultColors_[3] = Color(1.0f, 0.84f, 0.0f, 1.0f);    // Gold
    defaultColors_[4] = Color(0.58f, 0.44f, 0.86f, 1.0f);  // MediumPurple
    defaultColors_[5] = Color(1.0f, 0.41f, 0.71f, 1.0f);   // HotPink
    defaultColors_[6] = Color(0.27f, 0.51f, 0.71f, 1.0f);  // SteelBlue
    defaultColors_[7] = Color(1.0f, 0.55f, 0.0f, 1.0f);    // DarkOrange
    defaultColors_[8] = Color(0.20f, 0.80f, 0.20f, 1.0f);  // LimeGreen
    defaultColors_[9] = Color(0.78f, 0.08f, 0.52f, 1.0f);  // MediumVioletRed
}

RealTimePlotter::RealTimePlotter()
    : running_(false)
    , shouldStop_(false)
    , paused_(false)
    , refreshIntervalMs_(100)
    , historyWindowSec_(60)
    , maxDataPoints_(1000)
    , showGrid_(true)
    , showLegend_(true)
    , autoRangeY_(true)
    , autoRangeX_(true)
    , xMin_(0)
    , xMax_(100)
    , xRange_(100)
    , samplingMethod_(AST::RealTimePlotStatement::SamplingMethod::NONE)
    , samplingInterval_(10)
    , samplingWindowMs_(1000) {
    stats_.startTime = std::chrono::steady_clock::now();
    lastDataPointTime_ = std::chrono::steady_clock::now();
    lastRenderTime_ = std::chrono::steady_clock::now();
    initDefaultColors();
}

RealTimePlotter::~RealTimePlotter() {
    stop();
}

void RealTimePlotter::initialize(const AST::RealTimePlotStatement& config) {
    plotType_ = config.type;
    title_ = config.title;
    xLabel_ = config.xLabel;
    yLabel_ = config.yLabel;
    outputFile_ = config.outputFile;
    refreshIntervalMs_ = config.refreshIntervalMs;
    historyWindowSec_ = config.historyWindowSec;
    maxDataPoints_ = config.maxDataPoints;
    showGrid_ = config.showGrid;
    showLegend_ = config.showLegend;
    autoRangeY_ = config.autoRangeY;
    yMin_ = config.yMin;
    yMax_ = config.yMax;
    autoRangeX_ = config.autoRangeX;
    colors_ = config.colors;
    xColumn_ = config.xColumn;
    yColumns_ = config.yColumns;
    groupColumn_ = config.groupColumn;
    samplingMethod_ = config.samplingMethod;
    samplingInterval_ = config.samplingInterval;
    samplingWindowMs_ = config.samplingWindowMs;
    
    // Initialize xRange from history window
    xRange_ = static_cast<double>(historyWindowSec_);
    xMax_ = 0;
    xMin_ = -xRange_;
}

void RealTimePlotter::addDataPoint(const RealTimeDataPoint& point) {
    std::lock_guard<std::mutex> lock(pendingMutex_);
    pendingPoints_.push(point);
}

void RealTimePlotter::addDataPoints(const std::vector<RealTimeDataPoint>& points) {
    std::lock_guard<std::mutex> lock(pendingMutex_);
    for (const auto& point : points) {
        pendingPoints_.push(point);
    }
}

void RealTimePlotter::loadInitialData(const ExecutionEngine::ResultSet& result) {
    // Find column indices
    int xIdx = -1;
    std::vector<int> yIdxs;
    int groupIdx = -1;
    
    for (size_t i = 0; i < result.columns.size(); i++) {
        if (result.columns[i] == xColumn_) xIdx = i;
        for (const auto& ycol : yColumns_) {
            if (result.columns[i] == ycol) yIdxs.push_back(i);
        }
        if (result.columns[i] == groupColumn_) groupIdx = i;
    }
    
    // Process all rows
    std::vector<RealTimeDataPoint> points;
    for (const auto& row : result.rows) {
        double x = (xIdx >= 0) ? std::stod(row[xIdx]) : points.size();
        
        if (!groupColumn_.empty() && groupIdx >= 0) {
            // Grouped data
            std::string group = row[groupIdx];
            for (int yIdx : yIdxs) {
                double y = std::stod(row[yIdx]);
                points.emplace_back(x, y, group + ":" + result.columns[yIdx]);
            }
        } else {
            // One series per y-column
            for (int yIdx : yIdxs) {
                double y = std::stod(row[yIdx]);
                points.emplace_back(x, y, result.columns[yIdx]);
            }
        }
    }
    
    addDataPoints(points);
}

void RealTimePlotter::streamUpdate(const std::vector<std::unordered_map<std::string, std::string>>& newRows) {
    std::vector<RealTimeDataPoint> points;
    
    for (const auto& row : newRows) {
        double x = row.count(xColumn_) ? std::stod(row.at(xColumn_)) : points.size();
        
        if (!groupColumn_.empty() && row.count(groupColumn_)) {
            std::string group = row.at(groupColumn_);
            for (const auto& ycol : yColumns_) {
                if (row.count(ycol)) {
                    double y = std::stod(row.at(ycol));
                    points.emplace_back(x, y, group + ":" + ycol);
                }
            }
        } else {
            for (const auto& ycol : yColumns_) {
                if (row.count(ycol)) {
                    double y = std::stod(row.at(ycol));
                    points.emplace_back(x, y, ycol);
                }
            }
        }
    }
    
    addDataPoints(points);
}

void RealTimePlotter::pruneOldData() {
    auto now = std::chrono::steady_clock::now();
    double currentX = std::chrono::duration<double>(now - stats_.startTime).count();
    double cutoffX = currentX - historyWindowSec_;
    
    std::lock_guard<std::mutex> lock(dataMutex_);
    
    for (auto& pair : series_) {
        auto& series = pair.second;
        while (!series.points.empty() && series.points.front().x < cutoffX) {
            series.points.pop_front();
        }
        
        // Update min/max
        series.minY = std::numeric_limits<double>::max();
        series.maxY = std::numeric_limits<double>::lowest();
        for (const auto& point : series.points) {
            series.minY = std::min(series.minY, point.y);
            series.maxY = std::max(series.maxY, point.y);
        }
    }
}

void RealTimePlotter::updateXRange() {
    if (autoRangeX_) {
        xMax_ = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - stats_.startTime).count();
        xMin_ = xMax_ - xRange_;
    }
}

void RealTimePlotter::updateYRange() {
    if (!autoRangeY_) return;
    
    std::lock_guard<std::mutex> lock(dataMutex_);
    
    double globalMin = std::numeric_limits<double>::max();
    double globalMax = std::numeric_limits<double>::lowest();
    
    for (const auto& pair : series_) {
        if (!pair.second.points.empty()) {
            globalMin = std::min(globalMin, pair.second.minY);
            globalMax = std::max(globalMax, pair.second.maxY);
        }
    }
    
    if (globalMin < globalMax) {
        double padding = (globalMax - globalMin) * 0.1;
        yMin_ = globalMin - padding;
        yMax_ = globalMax + padding;
    } else {
        yMin_ = globalMin - 10;
        yMax_ = globalMax + 10;
    }
}

std::vector<RealTimeDataPoint> RealTimePlotter::sampleData(const std::deque<RealTimeDataPoint>& data) {
    if (samplingMethod_ == AST::RealTimePlotStatement::SamplingMethod::NONE) {
        return std::vector<RealTimeDataPoint>(data.begin(), data.end());
    }
    
    if (samplingMethod_ == AST::RealTimePlotStatement::SamplingMethod::EVERY_N) {
        std::vector<RealTimeDataPoint> result;
        for (size_t i = 0; i < data.size(); i += samplingInterval_) {
            result.push_back(data[i]);
        }
        return result;
    }
    
    // For time-based sampling, we need the x values
    if (data.empty()) return {};
    
    double minX = data.front().x;
    double maxX = data.back().x;
    double windowSize = static_cast<double>(samplingWindowMs_) / 1000.0;
    
    int numWindows = static_cast<int>((maxX - minX) / windowSize) + 1;
    std::vector<RealTimeDataPoint> result;
    
    if (samplingMethod_ == AST::RealTimePlotStatement::SamplingMethod::AVERAGE) {
        result = averageSample(data, samplingWindowMs_);
    } else if (samplingMethod_ == AST::RealTimePlotStatement::SamplingMethod::MAX) {
        result = maxSample(data, samplingWindowMs_);
    } else if (samplingMethod_ == AST::RealTimePlotStatement::SamplingMethod::MIN) {
        result = minSample(data, samplingWindowMs_);
    } else if (samplingMethod_ == AST::RealTimePlotStatement::SamplingMethod::LAST) {
        result = lastSample(data, samplingWindowMs_);
    }
    
    return result;
}

std::vector<RealTimeDataPoint> RealTimePlotter::averageSample(const std::deque<RealTimeDataPoint>& data, int windowMs) {
    std::vector<RealTimeDataPoint> result;
    if (data.empty()) return result;
    
    double windowSec = windowMs / 1000.0;
    double currentWindowStart = data.front().x;
    double sum = 0;
    int count = 0;
    
    for (const auto& point : data) {
        if (point.x < currentWindowStart + windowSec) {
            sum += point.y;
            count++;
        } else {
            if (count > 0) {
                result.emplace_back(currentWindowStart + windowSec / 2, sum / count, point.series);
            }
            currentWindowStart = point.x;
            sum = point.y;
            count = 1;
        }
    }
    
    if (count > 0) {
        result.emplace_back(currentWindowStart + windowSec / 2, sum / count, data.back().series);
    }
    
    return result;
}

std::vector<RealTimeDataPoint> RealTimePlotter::maxSample(const std::deque<RealTimeDataPoint>& data, int windowMs) {
    std::vector<RealTimeDataPoint> result;
    if (data.empty()) return result;
    
    double windowSec = windowMs / 1000.0;
    double currentWindowStart = data.front().x;
    double maxVal = data.front().y;
    
    for (const auto& point : data) {
        if (point.x < currentWindowStart + windowSec) {
            maxVal = std::max(maxVal, point.y);
        } else {
            result.emplace_back(currentWindowStart + windowSec / 2, maxVal, point.series);
            currentWindowStart = point.x;
            maxVal = point.y;
        }
    }
    
    result.emplace_back(currentWindowStart + windowSec / 2, maxVal, data.back().series);
    return result;
}

std::vector<RealTimeDataPoint> RealTimePlotter::minSample(const std::deque<RealTimeDataPoint>& data, int windowMs) {
    std::vector<RealTimeDataPoint> result;
    if (data.empty()) return result;
    
    double windowSec = windowMs / 1000.0;
    double currentWindowStart = data.front().x;
    double minVal = data.front().y;
    
    for (const auto& point : data) {
        if (point.x < currentWindowStart + windowSec) {
            minVal = std::min(minVal, point.y);
        } else {
            result.emplace_back(currentWindowStart + windowSec / 2, minVal, point.series);
            currentWindowStart = point.x;
            minVal = point.y;
        }
    }
    
    result.emplace_back(currentWindowStart + windowSec / 2, minVal, data.back().series);
    return result;
}

std::vector<RealTimeDataPoint> RealTimePlotter::lastSample(const std::deque<RealTimeDataPoint>& data, int windowMs) {
    std::vector<RealTimeDataPoint> result;
    if (data.empty()) return result;
    
    double windowSec = windowMs / 1000.0;
    double currentWindowStart = data.front().x;
    double lastVal = data.front().y;
    
    for (const auto& point : data) {
        if (point.x < currentWindowStart + windowSec) {
            lastVal = point.y;
        } else {
            result.emplace_back(currentWindowStart + windowSec / 2, lastVal, point.series);
            currentWindowStart = point.x;
            lastVal = point.y;
        }
    }
    
    result.emplace_back(currentWindowStart + windowSec / 2, lastVal, data.back().series);
    return result;
}

void RealTimePlotter::processPendingPoints() {
    std::queue<RealTimeDataPoint> points;
    {
        std::lock_guard<std::mutex> lock(pendingMutex_);
        std::swap(points, pendingPoints_);
    }
    
    std::lock_guard<std::mutex> lock(dataMutex_);
    
    while (!points.empty()) {
        const auto& point = points.front();
        
        auto it = series_.find(point.series);
        if (it == series_.end()) {
            // Create new series
            RealTimeSeries newSeries;
            newSeries.name = point.series;
            size_t colorIdx = series_.size() % 10;
            
            // Set color
            if (colorIdx < colors_.size() && !colors_[colorIdx].empty()) {
                // Parse hex color
                unsigned int hex;
                if (colors_[colorIdx][0] == '#') {
                    sscanf(colors_[colorIdx].c_str() + 1, "%06x", &hex);
                    newSeries.setColor(
                        ((hex >> 16) & 0xFF) / 255.0f,
                        ((hex >> 8) & 0xFF) / 255.0f,
                        (hex & 0xFF) / 255.0f
                    );
                } else {
                    newSeries.setColor(
                        defaultColors_[colorIdx].r,
                        defaultColors_[colorIdx].g,
                        defaultColors_[colorIdx].b,
                        defaultColors_[colorIdx].a
                    );
                }
            } else {
                newSeries.setColor(
                    defaultColors_[colorIdx].r,
                    defaultColors_[colorIdx].g,
                    defaultColors_[colorIdx].b,
                    defaultColors_[colorIdx].a
                );
            }
            it = series_.emplace(point.series, std::move(newSeries)).first;
        }
        
        it->second.addPoint(point);
        
        // Prune if too many points
        while (it->second.points.size() > static_cast<size_t>(maxDataPoints_)) {
            it->second.points.pop_front();
        }
        
        points.pop();
        
        {
            std::lock_guard<std::mutex> statsLock(statsMutex_);
            stats_.totalPointsReceived++;
        }
    }
}


void RealTimePlotter::renderLinePlot() {
    std::lock_guard<std::mutex> lock(dataMutex_);
    
    for (auto& pair : series_) {
        auto& series = pair.second;
        if (series.points.empty()) continue;
        
        auto sampled = sampleData(series.points);
        if (sampled.empty()) continue;
        
        std::vector<double> xs, ys;
        xs.reserve(sampled.size());
        ys.reserve(sampled.size());
        
        for (const auto& point : sampled) {
            xs.push_back(point.x);
            ys.push_back(point.y);
        }
        
        ImPlotSpec spec;
        spec.LineColor = ImVec4(series.color_r, series.color_g, series.color_b, series.color_a);
        spec.LineWeight = 2.0f;

        ImPlot::PlotLine(series.name.c_str(), xs.data(), ys.data(), xs.size(), spec);
    }
}

void RealTimePlotter::renderScatterPlot() {
    std::lock_guard<std::mutex> lock(dataMutex_);
    
    for (auto& pair : series_) {
        auto& series = pair.second;
        if (series.points.empty()) continue;
        
        auto sampled = sampleData(series.points);
        if (sampled.empty()) continue;
        
        std::vector<double> xs, ys;
        xs.reserve(sampled.size());
        ys.reserve(sampled.size());
        
        for (const auto& point : sampled) {
            xs.push_back(point.x);
            ys.push_back(point.y);
        }
        
        ImPlotSpec spec;
        spec.Marker = ImPlotMarker_Circle;
        spec.MarkerSize = 4.0f;
        spec.MarkerFillColor = ImVec4(series.color_r, series.color_g, series.color_b, series.color_a);
        spec.MarkerLineColor = ImVec4(series.color_r, series.color_g, series.color_b, series.color_a);
        ImPlot::PlotScatter(series.name.c_str(), xs.data(), ys.data(), xs.size(),spec);
    }
}

void RealTimePlotter::renderBarPlot() {
    std::lock_guard<std::mutex> lock(dataMutex_);
    
    // For bar plots, we need to aggregate data by x value
    std::map<double, std::map<std::string, double>> aggregated;
    
    for (const auto& pair : series_) {
        const auto& series = pair.second;
        for (const auto& point : series.points) {
            aggregated[point.x][series.name] = point.y;
        }
    }
    
    // Find all unique x values
    std::vector<double> xs;
    for (const auto& entry : aggregated) {
        xs.push_back(entry.first);
    }
    
    if (xs.empty()) return;
    
    // Plot as grouped bars
    int numSeries = series_.size();
    double barWidth = 0.8 / numSeries;
    
    int idx = 0;
    for (auto& pair : series_) {
        const auto& series = pair.second;
        std::vector<double> ys;
        ys.reserve(xs.size());
        
        for (double x : xs) {
            auto it = aggregated.find(x);
            if (it != aggregated.end()) {
                auto sit = it->second.find(series.name);
                ys.push_back(sit != it->second.end() ? sit->second : 0);
            } else {
                ys.push_back(0);
            }
        }
        
        ImPlotSpec spec;
        spec.FillColor = ImVec4(series.color_r, series.color_g, series.color_b, series.color_a);
        spec.FillAlpha = 0.7f;

        ImPlot::PlotBars(series.name.c_str(), xs.data(), ys.data(), xs.size(), barWidth, spec);
        idx++;
    }
}

void RealTimePlotter::renderAreaPlot() {
    std::lock_guard<std::mutex> lock(dataMutex_);
    
    for (auto& pair : series_) {
        auto& series = pair.second;
        if (series.points.empty()) continue;
        
        auto sampled = sampleData(series.points);
        if (sampled.empty()) continue;
        
        std::vector<double> xs, ys;
        xs.reserve(sampled.size());
        ys.reserve(sampled.size());
        
        for (const auto& point : sampled) {
            xs.push_back(point.x);
            ys.push_back(point.y);
        }
        
        ImPlotSpec spec;
        spec.FillColor = ImVec4(series.color_r, series.color_g, series.color_b, series.color_a);
        spec.FillAlpha = 0.3f;
        spec.LineColor = ImVec4(series.color_r, series.color_g, series.color_b, series.color_a);
        spec.LineWeight = 2.0f;

        ImPlot::PlotShaded(series.name.c_str(), xs.data(), ys.data(), xs.size(), 0, spec);
    }
}

void RealTimePlotter::renderHistogram() {
    std::lock_guard<std::mutex> lock(dataMutex_);
    
    // Collect all y values
    std::vector<double> allValues;
    for (const auto& pair : series_) {
        for (const auto& point : pair.second.points) {
            allValues.push_back(point.y);
        }
    }
    
    if (allValues.empty()) return;
    
    // Calculate histogram
    int numBins = 50;
    double minVal = *std::min_element(allValues.begin(), allValues.end());
    double maxVal = *std::max_element(allValues.begin(), allValues.end());
    double binWidth = (maxVal - minVal) / numBins;
    
    std::vector<double> bins(numBins, 0);
    for (double val : allValues) {
        int bin = static_cast<int>((val - minVal) / binWidth);
        if (bin >= 0 && bin < numBins) {
            bins[bin]++;
        }
    }
    
    std::vector<double> binCenters(numBins);
    for (int i = 0; i < numBins; i++) {
        binCenters[i] = minVal + (i + 0.5) * binWidth;
    }
    
    ImPlotSpec spec;
    spec.FillColor = ImVec4(defaultColors_[0].r, defaultColors_[0].g, defaultColors_[0].b, defaultColors_[0].a);
    spec.FillAlpha = 0.7f;

    ImPlot::PlotBars("Histogram", binCenters.data(), bins.data(), numBins, binWidth * 0.9, spec);
}

void RealTimePlotter::renderMultiLinePlot() {
    std::lock_guard<std::mutex> lock(dataMutex_);
    
    // Find all unique x values
    std::map<double, std::map<std::string, double>> dataMap;
    
    for (const auto& pair : series_) {
        const auto& series = pair.second;
        for (const auto& point : series.points) {
            dataMap[point.x][series.name] = point.y;
        }
    }
    
    if (dataMap.empty()) return;
    
    // Build x array
    std::vector<double> xs;
    for (const auto& entry : dataMap) {
        xs.push_back(entry.first);
    }
    
    // Plot each series
    for (const auto& pair : series_) {
        const auto& series = pair.second;
        std::vector<double> ys;
        ys.reserve(xs.size());
        
        for (double x : xs) {
            auto it = dataMap.find(x);
            if (it != dataMap.end()) {
                auto sit = it->second.find(series.name);
                ys.push_back(sit != it->second.end() ? sit->second : 0);
            } else {
                ys.push_back(0);
            }
        }
        
        ImPlotSpec spec;
        spec.LineColor = ImVec4(series.color_r, series.color_g, series.color_b, series.color_a);
        spec.LineWeight = 2.0f;

        ImPlot::PlotLine(series.name.c_str(), xs.data(), ys.data(), xs.size(), spec);
    }
}

void RealTimePlotter::renderStackedBarPlot() {
    std::lock_guard<std::mutex> lock(dataMutex_);
    
    // Aggregate data by x
    std::map<double, std::map<std::string, double>> aggregated;
    
    for (const auto& pair : series_) {
        const auto& series = pair.second;
        for (const auto& point : series.points) {
            aggregated[point.x][series.name] = point.y;
        }
    }
    
    if (aggregated.empty()) return;
    
    // Build x array
    std::vector<double> xs;
    for (const auto& entry : aggregated) {
        xs.push_back(entry.first);
    }
    
    // Build stacked data
    std::vector<std::string> seriesNames;
    for (const auto& pair : series_) {
        seriesNames.push_back(pair.first);
    }
    
    // Create stacked data array
    std::vector<std::vector<double>> stackedData(seriesNames.size(), std::vector<double>(xs.size(), 0));
    
    for (size_t i = 0; i < xs.size(); i++) {
        double x = xs[i];
        double cumulative = 0;
        
        for (size_t s = 0; s < seriesNames.size(); s++) {
            auto it = aggregated.find(x);
            if (it != aggregated.end()) {
                auto sit = it->second.find(seriesNames[s]);
                double val = sit != it->second.end() ? sit->second : 0;
                stackedData[s][i] = val;
            }
        }
    }
    
    // Plot stacked bars
    for (size_t s = 0; s < seriesNames.size(); s++) {
        auto it = series_.find(seriesNames[s]);
        if (it != series_.end()) {
            ImPlotSpec spec;
            spec.FillColor = ImVec4(it->second.color_r, it->second.color_g, it->second.color_b, it->second.color_a);
            spec.FillAlpha = 0.7f;
            ImPlot::PlotBars(seriesNames[s].c_str(), xs.data(), stackedData[s].data(), xs.size(), 0.8, spec);
        }
    }
}

void RealTimePlotter::renderPlot() {
    updateXRange();
    updateYRange();
    
    ImVec2 plotSize = ImGui::GetContentRegionAvail();
    plotSize.y -= 150; // Reserve space for controls and stats
    
    ImPlotFlags plotFlags = ImPlotFlags_Crosshairs;
    if (!showLegend_) plotFlags |= ImPlotFlags_NoLegend;
    
    if (ImPlot::BeginPlot("##RealTimePlot", plotSize, plotFlags)) {
        // Setup axes
        ImPlot::SetupAxis(ImAxis_X1, xLabel_.c_str());
        ImPlot::SetupAxis(ImAxis_Y1, yLabel_.c_str());
        
        ImPlot::SetupAxisLimits(ImAxis_X1, xMin_, xMax_, ImGuiCond_Always);
        ImPlot::SetupAxisLimits(ImAxis_Y1, yMin_, yMax_, ImGuiCond_Always);
        
        if (!showGrid_) {
            ImPlot::SetupAxis(ImAxis_X1, nullptr, ImPlotAxisFlags_NoGridLines);
            ImPlot::SetupAxis(ImAxis_Y1, nullptr, ImPlotAxisFlags_NoGridLines);
        }
        
        // Render based on plot type
        switch (plotType_) {
            case AST::RealTimePlotStatement::PlotType::LINE:
                renderLinePlot();
                break;
            case AST::RealTimePlotStatement::PlotType::SCATTER:
                renderScatterPlot();
                break;
            case AST::RealTimePlotStatement::PlotType::BAR:
                renderBarPlot();
                break;
            case AST::RealTimePlotStatement::PlotType::AREA:
                renderAreaPlot();
                break;
            case AST::RealTimePlotStatement::PlotType::HISTOGRAM:
                renderHistogram();
                break;
            case AST::RealTimePlotStatement::PlotType::MULTI_LINE:
                renderMultiLinePlot();
                break;
            case AST::RealTimePlotStatement::PlotType::STACKED_BAR:
                renderStackedBarPlot();
                break;
        }
        
        ImPlot::EndPlot();
    }
}

void RealTimePlotter::renderLegend() {
    if (!showLegend_) return;
    
    ImGui::Begin("Legend", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    
    std::lock_guard<std::mutex> lock(dataMutex_);
    
    for (const auto& pair : series_) {
        ImGui::ColorButton(("##" + pair.first).c_str(), ImVec4(pair.second.color_r, pair.second.color_g, pair.second.color_b, pair.second.color_a), 
                          ImGuiColorEditFlags_NoTooltip | ImGuiColorEditFlags_NoDragDrop);
        ImGui::SameLine();
        ImGui::Text("%s", pair.first.c_str());
        ImGui::SameLine(150);
        ImGui::Text("Last: %.4f", pair.second.getLastY());
    }
    
    ImGui::End();
}

void RealTimePlotter::renderStatisticsPanel() {
    ImGui::Begin("Statistics", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    
    std::lock_guard<std::mutex> statsLock(statsMutex_);
    
    ImGui::Text("Total Points Received: %zu", stats_.totalPointsReceived);
    ImGui::Text("Active Series: %zu", series_.size());
    ImGui::Text("FPS: %.1f", stats_.currentFPS);
    ImGui::Text("Avg Latency: %.1f ms", stats_.avgLatencyMs);
    
    auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - stats_.startTime);
    ImGui::Text("Elapsed: %.1f sec", elapsed.count());
    
    ImGui::End();
}

void RealTimePlotter::renderControls() {
    ImGui::Begin("Controls");
    
    if (paused_) {
        if (ImGui::Button("Resume")) {
            resume();
        }
    } else {
        if (ImGui::Button("Pause")) {
            pause();
        }
    }
    
    ImGui::SameLine();
    if (ImGui::Button("Clear")) {
        clear();
    }
    
    ImGui::SameLine();
    if (ImGui::Button("Reset View")) {
        updateXRange();
        updateYRange();
    }
    
    // Statistics
    std::lock_guard<std::mutex> lock(dataMutex_);
    ImGui::Text("Series: %zu", series_.size());
    
    // Sampling method selector
    const char* samplingMethods[] = {"None", "Every N", "Average", "Max", "Min", "Last"};
    int currentMethod = static_cast<int>(samplingMethod_);
    if (ImGui::Combo("Sampling", &currentMethod, samplingMethods, 6)) {
        samplingMethod_ = static_cast<AST::RealTimePlotStatement::SamplingMethod>(currentMethod);
    }
    
    if (samplingMethod_ == AST::RealTimePlotStatement::SamplingMethod::EVERY_N) {
        ImGui::SliderInt("Every N", &samplingInterval_, 2, 100);
    } else if (samplingMethod_ != AST::RealTimePlotStatement::SamplingMethod::NONE) {
        ImGui::SliderInt("Window (ms)", &samplingWindowMs_, 100, 10000);
    }
    
    ImGui::End();
}

void RealTimePlotter::plotThreadFunction() {
    // Setup GLFW window
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
    int windowWidth = 1280;
    int windowHeight = 800;
    
    GLFWwindow* window = glfwCreateWindow(windowWidth, windowHeight, 
        (title_.empty() ? "Real-Time Plotter" : title_.c_str()), 
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
    
    // Dark theme for ImPlot
    ImPlotStyle& plotStyle = ImPlot::GetStyle();
    plotStyle.Colors[ImPlotCol_PlotBg] = ImVec4(25/255.0f, 25/255.0f, 35/255.0f, 1.0f);
    plotStyle.Colors[ImPlotCol_FrameBg] = ImVec4(30/255.0f, 30/255.0f, 40/255.0f, 1.0f);
    plotStyle.Colors[ImPlotCol_PlotBorder] = ImVec4(60/255.0f, 60/255.0f, 80/255.0f, 1.0f);
    
    auto lastFrameTime = std::chrono::steady_clock::now();
    double frameDuration = refreshIntervalMs_ / 1000.0;
    int frameCount = 0;
    auto lastFPSTime = std::chrono::steady_clock::now();
    
    while (!shouldStop_ && !glfwWindowShouldClose(window)) {
        auto frameStart = std::chrono::steady_clock::now();
        double deltaTime = std::chrono::duration<double>(frameStart - lastFrameTime).count();
        
        if (deltaTime < frameDuration && !paused_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        
        lastFrameTime = frameStart;
        
        // Process pending data
        processPendingPoints();
        
        // Prune old data
        if (!paused_) {
            pruneOldData();
        }
        
        // Update FPS
        frameCount++;
        auto now = std::chrono::steady_clock::now();
        double fpsDelta = std::chrono::duration<double>(now - lastFPSTime).count();
        if (fpsDelta >= 1.0) {
            {
                std::lock_guard<std::mutex> statsLock(statsMutex_);
                stats_.currentFPS = frameCount / fpsDelta;
            }
            frameCount = 0;
            lastFPSTime = now;
        }
        
        // Update latency
        if (!pendingPoints_.empty()) {
            std::lock_guard<std::mutex> statsLock(statsMutex_);
            if (stats_.totalPointsReceived > 0) {
                // Simple latency tracking
                stats_.avgLatencyMs = stats_.avgLatencyMs * 0.95 + 
                    (deltaTime * 1000) * 0.05;
            }
        }
        
        // Render
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        
        // Main plot window
        ImGui::Begin("Real-Time Plot", nullptr, ImGuiWindowFlags_NoCollapse);
        renderPlot();
        ImGui::End();
        
        // Controls window
        renderControls();
        
        // Legend window
        if (showLegend_) {
            renderLegend();
        }
        
        // Statistics window
        renderStatisticsPanel();
        
        // Render
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
    
    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    
    glfwDestroyWindow(window);
}

void RealTimePlotter::start() {
    if (running_) return;
    
    running_ = true;
    shouldStop_ = false;
    paused_ = false;
    
    if (!glfwInit()) {
        running_ = false;
        throw std::runtime_error("Failed to initialize GLFW");
    }
    
    plotThread_ = std::thread(&RealTimePlotter::plotThreadFunction, this);
}

void RealTimePlotter::stop() {
    if (!running_) return;
    
    shouldStop_ = true;
    cv_.notify_all();
    
    if (plotThread_.joinable()) {
        plotThread_.join();
    }
    
    running_ = false;
}

void RealTimePlotter::clear() {
    std::lock_guard<std::mutex> lock(dataMutex_);
    for (auto& pair : series_) {
        pair.second.clear();
    }
}

RealTimePlotter::Statistics RealTimePlotter::getStatistics() const {
    std::lock_guard<std::mutex> lock(statsMutex_);
    Statistics stats;
    stats.totalPointsReceived = stats_.totalPointsReceived;
    stats.totalPointsPlotted = stats_.totalPointsPlotted;
    stats.activeSeries = series_.size();
    stats.currentFPS = stats_.currentFPS;
    stats.avgLatencyMs = stats_.avgLatencyMs;
    stats.elapsedTime = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - stats_.startTime);
    return stats;
}

} // namespace Visualization
