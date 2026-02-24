#include "plotter_includes/implotter.h"
#include "execution_engine_includes/executionengine_main.h"
#include "ai_execution_engine_final.h"
#include "database.h"
#include <thread>
#include <chrono>
#include <atomic>
#include <memory>

// Fixed executeSimulate function
ExecutionEngine::ResultSet AIExecutionEngineFinal::executeSimulate(
    AST::SimulateStatement& stmt) {
    
    std::cout << "[AIExecutionEngineFinal] Executing SIMULATE MARKET: "
              << stmt.model_name << std::endl;

    ExecutionEngine::ResultSet result;
    result.columns = {
        "simulation_id", "step", "path", "timestamp", "price",
        "return", "volatility", "volume", "regime", "sma_20",
        "rsi", "bid_price", "ask_price", "confidence"
    };

    try {
        // Get the model
        auto model = getOrLoadModel(stmt.model_name);
        if (!model) {
            throw std::runtime_error("Model not found: " + stmt.model_name);
        }

        // Create market simulator
        auto simulator = esql::ai::MarketSimulatorFactory::create_for_stocks(model, "SIM");

        // Apply simulation parameters
        simulator->set_noise_model("Gaussian", stmt.noise_level);
        simulator->set_volatility_model("GARCH", 0.02);
        simulator->set_mean_reversion_strength(/*stmt.include_mean_reversion,*/ stmt.mean_reversion_strength);
        simulator->set_include_volatility_clustering(stmt.include_volatility_clustering);
        
        if (stmt.simulate_microstructure) {
            simulator->set_microstructure_simulation(true, stmt.spread);
        }
        
        simulator->set_regime_detection(true);

        // Load initial conditions from input table if provided
        std::unordered_map<std::string, double> initial_conditions;
        if (!stmt.input_table.empty()) {
            auto data = data_extractor_->extract_table_data(
                db_.currentDatabase(),
                stmt.input_table,
                {},
                "",
                1,  // Just get one row
                0
            );

            if (!data.empty() && !data[0].empty()) {
                for (const auto& [col, datum] : data[0]) {
                    if (!datum.is_null()) {
                        try {
                            initial_conditions[col] = datum.as_double();
                        } catch (...) {
                            // Skip non-numeric columns
                        }
                    }
                }
            }
        }

        // Ensure we have at least a price
        if (initial_conditions.find("price") == initial_conditions.end()) {
            initial_conditions["price"] = 100.0;  // Default
        }

        // Calibrate simulator if we have historical data
        if (!stmt.input_table.empty()) {
            auto historical_data = data_extractor_->extract_table_data(
                db_.currentDatabase(),
                stmt.input_table,
                {},
                "",
                1000,  // Get up to 1000 rows for calibration
                0
            );

            if (historical_data.size() >= 100) {
                simulator->calibrate_from_historical_data(historical_data);
            }
        }

        // Apply scenario parameters
        std::unordered_map<std::string, double> scenario_params;
        for (const auto& [key, value] : stmt.scenario_params) {
            try {
                scenario_params[key] = std::stod(value);
            } catch (...) {
                std::cout << "[AIExecutionEngineFinal] Warning: Non-numeric scenario param: "
                          << key << " = " << value << std::endl;
            }
        }

        // Run simulation to get all paths
        std::cout << "[AIExecutionEngineFinal] Running simulation with "
                  << stmt.num_steps << " steps and " << stmt.num_paths << " paths..." << std::endl;
        
        auto paths = simulator->simulate(
            stmt.num_steps,
            stmt.num_paths,
            initial_conditions,
            scenario_params
        );

        std::cout << "[AIExecutionEngineFinal] Simulation returned " << paths.size() << " paths" << std::endl;
        if (!paths.empty()) {
            std::cout << "[AIExecutionEngineFinal] First path has " << paths[0].prices.size() << " prices" << std::endl;
        }

        // Generate simulation ID
        size_t simulation_id = std::hash<std::string>{}(stmt.model_name +
                               std::to_string(std::time(nullptr)));

        // FIRST, generate all result rows (this must happen regardless of plotting)
        for (size_t path_idx = 0; path_idx < paths.size(); ++path_idx) {
            const auto& path = paths[path_idx];
            
            // Generate timestamps
            auto base_time = std::chrono::system_clock::now();
            auto interval = parseTimeInterval(stmt.time_interval);
            
            for (size_t step = 0; step < path.prices.size(); ++step) {
                std::vector<std::string> row;
                
                // simulation_id
                row.push_back(std::to_string(simulation_id));
                
                // step
                row.push_back(std::to_string(step));
                
                // path
                row.push_back(std::to_string(path_idx));
                
                // timestamp
                auto time_point = base_time + interval * static_cast<int64_t>(step);
                auto tt = std::chrono::system_clock::to_time_t(time_point);
                std::stringstream ts;
                ts << std::put_time(std::localtime(&tt), "%Y-%m-%d %H:%M:%S");
                row.push_back(ts.str());
                
                // price
                row.push_back(std::to_string(path.prices[step]));
                
                // return
                if (step > 0 && step - 1 < path.returns.size()) {
                    row.push_back(std::to_string(path.returns[step-1]));
                } else {
                    row.push_back("0.0");
                }
                
                // volatility
                if (step < path.volatilities.size()) {
                    row.push_back(std::to_string(path.volatilities[step]));
                } else {
                    row.push_back("0.0");
                }
                
                // volume
                if (step < path.volumes.size()) {
                    row.push_back(std::to_string(path.volumes[step]));
                } else {
                    row.push_back("0.0");
                }
                
                // regime
                std::string regime_str;
                if (step < path.regimes.size()) {
                    switch (path.regimes[step]) {
                        case esql::ai::MarketRegime::BULL: regime_str = "BULL"; break;
                        case esql::ai::MarketRegime::BEAR: regime_str = "BEAR"; break;
                        case esql::ai::MarketRegime::SIDEWAYS: regime_str = "SIDEWAYS"; break;
                        case esql::ai::MarketRegime::HIGH_VOLATILITY: regime_str = "HIGH_VOL"; break;
                        case esql::ai::MarketRegime::LOW_VOLATILITY: regime_str = "LOW_VOL"; break;
                        case esql::ai::MarketRegime::MEAN_REVERTING: regime_str = "MEAN_REV"; break;
                        case esql::ai::MarketRegime::BREAKOUT: regime_str = "BREAKOUT"; break;
                        case esql::ai::MarketRegime::CRASH: regime_str = "CRASH"; break;
                        case esql::ai::MarketRegime::RALLY: regime_str = "RALLY"; break;
                        default: regime_str = "UNKNOWN";
                    }
                } else {
                    regime_str = "UNKNOWN";
                }
                row.push_back(regime_str);
                
                // indicators
                double sma_20_val = 0.0;
                if (!path.indicators.sma_20.empty()) {
                    size_t idx = std::min(step, path.indicators.sma_20.size() - 1);
                    sma_20_val = path.indicators.sma_20[idx];
                }
                row.push_back(std::to_string(sma_20_val));
                
                double rsi_val = 50.0;
                if (!path.indicators.rsi.empty()) {
                    size_t idx = std::min(step, path.indicators.rsi.size() - 1);
                    rsi_val = path.indicators.rsi[idx];
                }
                row.push_back(std::to_string(rsi_val));
                
                // bid/ask
                double bid_price = path.prices[step] * 0.9999;
                double ask_price = path.prices[step] * 1.0001;
                
                if (!path.events.empty() && step < path.events.size()) {
                    if (path.events[step].bid_price > 0) {
                        bid_price = path.events[step].bid_price;
                    }
                    if (path.events[step].ask_price > 0) {
                        ask_price = path.events[step].ask_price;
                    }
                }
                row.push_back(std::to_string(bid_price));
                row.push_back(std::to_string(ask_price));
                
                // confidence
                double confidence = 0.95;
                if (step < path.volatilities.size() && path.volatilities[step] > 0) {
                    confidence = 1.0 - std::min(0.5, path.volatilities[step]);
                    confidence = std::max(0.5, std::min(0.99, confidence));
                }
                row.push_back(std::to_string(confidence));
                
                result.rows.push_back(row);
            }
        }

        std::cout << "[AIExecutionEngineFinal] Generated " << result.rows.size() << " result rows" << std::endl;

        // NOW, handle plotting if requested (non-blocking)
        if (stmt.plot_config.has_value() && !paths.empty()) {
            const auto& plot_config = stmt.plot_config.value();
            
            std::cout << "[AIExecutionEngineFinal] Starting real-time plotting in separate thread..." << std::endl;
            
            // Create a shared pointer to the path data that will outlive this function
            auto path_data = std::make_shared<std::vector<esql::ai::SimulationPath>>(std::move(paths));
            
            // Start plotting in a detached thread so it doesn't block
            std::thread([this, path_data, plot_config]() {
                try {
                    // Initialize plotter if not already done
                    {
                        std::lock_guard<std::mutex> lock(plot_mutex_);
                        if (!plotter_) {
                            plotter_ = std::make_unique<Visualization::ImPlotSimulationPlotter>();
                            plotter_->initialize();
                        }
                    }
                    
                    // Setup plot window
                    plotter_->setupWindow(plot_config);
                    
                    // Get the first path for plotting
                    const auto& first_path = (*path_data)[0];
                    size_t total_steps = first_path.prices.size();
                    
                    std::cout << "[Plotting] Starting animation with " << total_steps << " steps..." << std::endl;
                    
                    // Animated playback
                    for (size_t step = 0; step < total_steps; ++step) {
                        auto start_time = std::chrono::steady_clock::now();
                        
                        // Plot current step
                        plotter_->plotSimulationCandlestick(
                            std::make_shared<esql::ai::SimulationPath>(first_path),
                            plot_config,
                            step
                        );
                        
                        // Check if window was closed
                        if (plotter_->isWindowClosed()) {
                            std::cout << "[Plotting] Window closed, stopping animation." << std::endl;
                            break;
                        }
                        
                        // Control update rate
                        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - start_time
                        );
                        
                        auto sleep_time = plot_config.update_interval - elapsed;
                        if (sleep_time > std::chrono::milliseconds(0)) {
                            std::this_thread::sleep_for(sleep_time);
                        }
                    }
                    
                    // Keep window open after animation
                    std::cout << "[Plotting] Animation complete. Window will stay open. Close it to continue." << std::endl;
                    
                    while (!plotter_->isWindowClosed()) {
                        plotter_->renderLoop();
                        std::this_thread::sleep_for(std::chrono::milliseconds(16));
                    }
                    
                    std::cout << "[Plotting] Plot window closed." << std::endl;
                    
                } catch (const std::exception& e) {
                    std::cerr << "[Plotting] Error in plotting thread: " << e.what() << std::endl;
                }
            }).detach();  // Detach so it runs independently
            
            std::cout << "[AIExecutionEngineFinal] Plotting thread started and detached." << std::endl;
        }

        // If output table specified, save results
        if (!stmt.output_table.empty()) {
            //saveSimulationResults(stmt.output_table, result);
        }

        logAIOperation("SIMULATE_MARKET", stmt.model_name, "SUCCESS",
                      "Generated " + std::to_string(stmt.num_paths) +
                      " paths with " + std::to_string(stmt.num_steps) + " steps");

    } catch (const std::exception& e) {
        logAIOperation("SIMULATE_MARKET", stmt.model_name, "FAILED", e.what());

        result.columns = {"error"};
        result.rows.push_back({std::string("ERROR: ") + e.what()});
    }

    return result;
}

std::chrono::seconds AIExecutionEngineFinal::parseTimeInterval(const std::string& interval) {
    char unit = interval.back();
    int value = std::stoi(interval.substr(0, interval.length() - 1));
    
    switch (unit) {
        case 's': return std::chrono::seconds(value);
        case 'm': return std::chrono::minutes(value);
        case 'h': return std::chrono::hours(value);
        case 'd': return std::chrono::hours(value * 24);
        default: return std::chrono::hours(1);  // Default to 1 hour
    }
}
