#include "execution_engine_includes/executionengine_main.h"
#include "plotter_includes/real_time_plotter_parser.h"
#include "plotter_includes/realtime_plotter.h"
#include "plotter_includes/realtime_candlestick_plotter.h"
#include <thread>
#include <chrono>
#include <atomic>
#include <condition_variable>

// Helper function to stream query data
void ExecutionEngine::streamQueryData(
    AST::SelectStatement& query,
    std::function<void(const std::vector<std::unordered_map<std::string, std::string>>&)> callback,
    int refreshIntervalMs) {
    
    // Track last timestamp to only get new data
    std::string lastTimestamp;
    std::atomic<bool> running{true};
    std::mutex callbackMutex;
    
    // For time-based incremental queries, we need to modify the query to only get new data
    // Store the original WHERE clause to combine with incremental filter
    
    // Create a thread for periodic polling
    std::thread pollingThread([&, refreshIntervalMs]() {
        while (running) {
            try {
                // Execute the query
                auto result = executeSelect(query);
                
                // Convert to map format for callback
                std::vector<std::unordered_map<std::string, std::string>> rows;
                for (const auto& row : result.rows) {
                    std::unordered_map<std::string, std::string> rowMap;
                    for (size_t i = 0; i < result.columns.size(); i++) {
                        rowMap[result.columns[i]] = row[i];
                    }
                    rows.push_back(rowMap);
                }
                
                // Call callback with new data
                {
                    std::lock_guard<std::mutex> lock(callbackMutex);
                    callback(rows);
                }
                
                // Wait for next interval
                std::this_thread::sleep_for(std::chrono::milliseconds(refreshIntervalMs));
                
            } catch (const std::exception& e) {
                // Log error but continue
                std::cerr << "Stream query error: " << e.what() << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(refreshIntervalMs));
            }
        }
    });
    
    // Wait for user to stop (this will be handled by the plotter's stop mechanism)
    // For now, we'll detach and let the plotter manage the thread
    pollingThread.detach();
}

ExecutionEngine::ResultSet ExecutionEngine::executeRealTimePlot(AST::RealTimePlotStatement& stmt) {
    // Validate configuration
    if (!stmt.autoDetectColumns && stmt.yColumns.empty()) {
        throw std::runtime_error("No Y columns specified for real-time plot. Use WITH Y AS column_name");
    }
    
    // If auto-detect is enabled, try to detect columns from the query
    if (stmt.autoDetectColumns) {
        // Execute query once to get schema
        auto sampleResult = executeSelect(*stmt.query);
        
        // Auto-detect timestamp column (look for time/date columns)
        for (const auto& col : sampleResult.columns) {
            std::string colLower = col;
            std::transform(colLower.begin(), colLower.end(), colLower.begin(), ::tolower);
            if (colLower.find("time") != std::string::npos || 
                colLower.find("date") != std::string::npos ||
                colLower.find("timestamp") != std::string::npos) {
                stmt.xColumn = col;
                break;
            }
        }
        
        // If no time column found, use index as X
        if (stmt.xColumn.empty()) {
            // Use row index as X
        }
        
        // Auto-detect numeric columns for Y
        for (const auto& col : sampleResult.columns) {
            if (col == stmt.xColumn) continue;
            
            // Check if column contains numeric data
            bool isNumeric = true;
            for (const auto& row : sampleResult.rows) {
                auto it = std::find(sampleResult.columns.begin(), sampleResult.columns.end(), col);
                if (it != sampleResult.columns.end()) {
                    size_t idx = std::distance(sampleResult.columns.begin(), it);
                    if (!isNumericString(row[idx])) {
                        isNumeric = false;
                        break;
                    }
                }
            }
            
            if (isNumeric) {
                stmt.yColumns.push_back(col);
            }
        }
        
        if (stmt.yColumns.empty()) {
            throw std::runtime_error("No numeric columns found for plotting. Please specify Y columns explicitly.");
        }
    }
    
    // Initialize the plotter
    Visualization::RealTimePlotter plotter;
    plotter.initialize(stmt);
    
    // Load initial data
    auto initialResult = executeSelect(*stmt.query);
    plotter.loadInitialData(initialResult);
    
    // Start the plotter (this will create a window and start the plotting thread)
    plotter.start();
    
    // Set up streaming for new data
    // For now, we'll use a polling approach. In production, you'd want a push-based system.
    std::atomic<bool> streaming{true};
    
    // Store the last row count to only get new rows
    size_t lastRowCount = initialResult.rows.size();
    
    // Create a copy of the query for incremental polling
    //auto pollingQuery = std::make_unique<AST::SelectStatement>();
    //*pollingQuery = *stmt.query;  // Deep copy
    auto* queryPtr = stmt.query.get();
    
    // If there's an ORDER BY with time, we can use it for incremental updates
    bool hasTimeOrder = false;
    if (queryPtr->orderBy && !queryPtr->orderBy->columns.empty()) {
        // Check if ordering is by time column
        if (auto* identifier = dynamic_cast<AST::Identifier*>(queryPtr->orderBy->columns[0].first.get())) {
            if (identifier->token.lexeme == stmt.xColumn) {
                hasTimeOrder = true;
            }
        }
    }
    
    // Polling thread for incremental data
    std::thread pollingThread([&, hasTimeOrder, lastRowCount]() mutable {
        size_t currentLastRowCount = lastRowCount;
        
        while (streaming && plotter.isRunning()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(stmt.refreshIntervalMs));
            
            if (plotter.isPaused()) continue;
            
            try {
                // Execute query to get all data (for now)
                // In production, you'd want to use a more efficient incremental approach
                auto result = executeSelect(*queryPtr);
                
                // Get only new rows (simplistic approach - assumes rows are appended)
                if (result.rows.size() > currentLastRowCount) {
                    std::vector<std::unordered_map<std::string, std::string>> newRows;
                    
                    for (size_t i = currentLastRowCount; i < result.rows.size(); i++) {
                        std::unordered_map<std::string, std::string> rowMap;
                        for (size_t j = 0; j < result.columns.size(); j++) {
                            rowMap[result.columns[j]] = result.rows[i][j];
                        }
                        newRows.push_back(rowMap);
                    }
                    
                    if (!newRows.empty()) {
                        plotter.streamUpdate(newRows);
                    }
                    
                    currentLastRowCount = result.rows.size();
                }
                
            } catch (const std::exception& e) {
                std::cerr << "Polling error: " << e.what() << std::endl;
            }
        }
    });
    
    // Wait for the plotter to finish (user closes window)
    while (plotter.isRunning()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    // Cleanup
    streaming = false;
    if (pollingThread.joinable()) {
        pollingThread.join();
    }
    
    // Return result
    ExecutionEngine::ResultSet result;
    result.columns = {"status", "statistics"};
    
    auto stats = plotter.getStatistics();
    std::string statsStr = "Points received: " + std::to_string(stats.totalPointsReceived) +
                          ", FPS: " + std::to_string(stats.currentFPS) +
                          ", Active series: " + std::to_string(stats.activeSeries);
    
    result.rows = {{"Real-time plot completed", statsStr}};
    return result;
}

ExecutionEngine::ResultSet ExecutionEngine::executeRealTimeCandlestick(AST::RealTimeCandlestickStatement& stmt) {
    // Validate configuration
    if (!stmt.isConfigured()) {
        throw std::runtime_error("Candlestick plot not properly configured. "
                                "Please specify open_column, close_column, high_column, low_column");
    }
    
    // Get column mapping
    auto mapping = stmt.getMapping();
    
    // Auto-detect columns if needed
    if (stmt.autoDetectColumns) {
        auto sampleResult = executeSelect(*stmt.query);
        
        // Look for standard column names
        for (const auto& col : sampleResult.columns) {
            std::string colLower = col;
            std::transform(colLower.begin(), colLower.end(), colLower.begin(), ::tolower);
            
            if (!mapping.useOpen && (colLower == "open" || colLower == "open_price")) {
                stmt.openColumn = col;
                mapping.useOpen = true;
            } else if (!mapping.useClose && (colLower == "close" || colLower == "close_price")) {
                stmt.closeColumn = col;
                mapping.useClose = true;
            } else if (!mapping.useHigh && (colLower == "high" || colLower == "high_price" || colLower == "max")) {
                stmt.highColumn = col;
                mapping.useHigh = true;
            } else if (!mapping.useLow && (colLower == "low" || colLower == "low_price" || colLower == "min")) {
                stmt.lowColumn = col;
                mapping.useLow = true;
            } else if (!mapping.useVolume && (colLower == "volume" || colLower == "vol")) {
                stmt.volumeColumn = col;
                mapping.useVolume = true;
            }
        }
        
        if (!mapping.hasAllRequired()) {
            throw std::runtime_error("Cannot auto-detect candlestick columns. "
                                    "Please specify open_column, close_column, high_column, low_column explicitly.");
        }
    }
    
    // Initialize the candlestick plotter
    Visualization::RealTimeCandlestickPlotter plotter;
    plotter.initialize(stmt);
    
    // Load initial data - FOLLOWING THE SAME PATTERN AS executeRealTimePlot
    std::cout << "[EXECUTION ENGINE] Exeuting SELECT query" << std::endl;
    auto initialResult = executeSelect(*stmt.query);
    std::cout << "[EXECUTION ENGINE] Done SELECT query execution" << std::endl;
    
    // Find column indices ONCE before using them
    auto mappingFinal = stmt.getMapping();
    int openIdx = -1, closeIdx = -1, highIdx = -1, lowIdx = -1, volumeIdx = -1;
    
    for (size_t i = 0; i < initialResult.columns.size(); i++) {
        if (initialResult.columns[i] == mappingFinal.open) openIdx = i;
        if (initialResult.columns[i] == mappingFinal.close) closeIdx = i;
        if (initialResult.columns[i] == mappingFinal.high) highIdx = i;
        if (initialResult.columns[i] == mappingFinal.low) lowIdx = i;
        if (mappingFinal.useVolume && initialResult.columns[i] == mappingFinal.volume) volumeIdx = i;
    }
    
    // Process initial data into candlesticks
    for (const auto& row : initialResult.rows) {
        double open = openIdx >= 0 ? std::stod(row[openIdx]) : 0;
        double close = closeIdx >= 0 ? std::stod(row[closeIdx]) : 0;
        double high = highIdx >= 0 ? std::stod(row[highIdx]) : 0;
        double low = lowIdx >= 0 ? std::stod(row[lowIdx]) : 0;
        double volume = volumeIdx >= 0 ? std::stod(row[volumeIdx]) : 0;
        
        Visualization::RealTimeCandlestick candle(
            open, high, low, close, volume,
            std::chrono::system_clock::now(),
            plotter.getCandleCount()
        );
        std::cout << "[EXECUTION] aDDING CANDLE STICK " << std::endl;
        plotter.addCandlestick(candle);
        std::cout << "[EXECUTION] Added candle stick" << std::endl;
    }
    
    // Start the plotter (this will create a window and start the plotting thread)
    plotter.start();
    
    // Set up streaming for new data - FOLLOWING THE SAME PATTERN AS executeRealTimePlot
    std::atomic<bool> streaming{true};
    
    // Store the last row count to only get new rows
    size_t lastRowCount = initialResult.rows.size();
    
    // Get pointer to query for polling
    auto* queryPtr = stmt.query.get();
    
    // Polling thread for incremental data - EXACT SAME PATTERN AS executeRealTimePlot
    std::thread pollingThread([&, lastRowCount]() mutable {
        size_t currentLastRowCount = lastRowCount;
        
        while (streaming && plotter.isRunning()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(stmt.intervalSeconds * 1000));
            
            if (plotter.isPaused()) continue;
            
            try {
                // Execute query to get all data
                auto result = executeSelect(*queryPtr);
                
                // Get only new rows (simplistic approach - assumes rows are appended)
                if (result.rows.size() > currentLastRowCount) {
                    // Process new rows
                    for (size_t i = currentLastRowCount; i < result.rows.size(); i++) {
                        const auto& row = result.rows[i];
                        
                        // Convert row to map for easy column access
                        std::unordered_map<std::string, std::string> rowMap;
                        for (size_t j = 0; j < result.columns.size(); j++) {
                            rowMap[result.columns[j]] = result.rows[i][j];
                        }
                        
                        // Extract values using column names
                        double close = 0.0, volume = 0.0;
                        
                        if (rowMap.count(stmt.closeColumn)) {
                            close = std::stod(rowMap[stmt.closeColumn]);
                        }
                        if (!stmt.volumeColumn.empty() && rowMap.count(stmt.volumeColumn)) {
                            volume = std::stod(rowMap[stmt.volumeColumn]);
                        }
                        
                        // Update the current candle with the close price
                        plotter.updateCurrentCandle(close, volume);
                    }
                    
                    currentLastRowCount = result.rows.size();
                }
                
            } catch (const std::exception& e) {
                std::cerr << "Polling error: " << e.what() << std::endl;
            }
        }
    });
    
    // Wait for the plotter to finish (user closes window) - EXACT SAME PATTERN
    while (plotter.isRunning()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    // Cleanup - EXACT SAME PATTERN
    streaming = false;
    if (pollingThread.joinable()) {
        pollingThread.join();
    }
    
    // Return result
    ExecutionEngine::ResultSet result;
    result.columns = {"status", "statistics"};
    
    auto stats = plotter.getStatistics();
    std::string statsStr = "Candles generated: " + std::to_string(stats.totalCandles) +
                          ", Bullish: " + std::to_string(stats.bullishCount) +
                          ", Bearish: " + std::to_string(stats.bearishCount);
    
    result.rows = {{"Real-time candlestick plot completed", statsStr}};
    return result;
}
