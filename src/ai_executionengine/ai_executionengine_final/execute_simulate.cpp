#include "market_simulator.h"
#include "execution_engine_includes/executionengine_main.h"
#include "ai_execution_engine_final.h"
#include "data_extractor.h"
#include "database.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <ctime>
#include <unordered_map>
#include <memory>

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

        // Run simulation
        auto paths = simulator->simulate(
            stmt.num_steps,
            stmt.num_paths,
            initial_conditions,
            scenario_params
        );

        // Generate output
        size_t simulation_id = std::hash<std::string>{}(stmt.model_name +
                               std::to_string(std::time(nullptr)));

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
                if (step > 0) {
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
                switch (path.regimes[std::min(step, path.regimes.size() - 1)]) {
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
                row.push_back(regime_str);

                // indicators (simplified)
                if (!path.indicators.sma_20.empty()) {
                    size_t idx = std::min(step, path.indicators.sma_20.size() - 1);
                    row.push_back(std::to_string(path.indicators.sma_20[idx]));
                } else {
                    row.push_back("0.0");
                }

                if (!path.indicators.rsi.empty()) {
                    size_t idx = std::min(step, path.indicators.rsi.size() - 1);
                    row.push_back(std::to_string(path.indicators.rsi[idx]));
                } else {
                    row.push_back("50.0");
                }

                // bid/ask
                if (!path.events.empty() && step < path.events.size()) {
                    row.push_back(std::to_string(path.events[step].bid_price));
                    row.push_back(std::to_string(path.events[step].ask_price));
                } else {
                    double price = path.prices[step];
                    row.push_back(std::to_string(price * 0.9999));  // Approx bid
                    row.push_back(std::to_string(price * 1.0001));  // Approx ask
                }

                // confidence
                double confidence = 1.0 - path.volatilities[std::min(step, path.volatilities.size() - 1)];
                row.push_back(std::to_string(std::max(0.5, std::min(0.99, confidence))));

                result.rows.push_back(row);
            }
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

// Helper function to parse time intervals
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

/*void AIExecutionEngineFinal::saveSimulationResults(
    const std::string& table_name,
    const ExecutionEngine::ResultSet& results) {

    // Create table if not exists
    AST::CreateTableStatement create_stmt;
    create_stmt.tablename = table_name;
    create_stmt.ifNotExists = true;

    // Define columns based on result set
    for (const auto& col : results.columns) {
        AST::ColumnDefination col_def;
        col_def.name = col;
        col_def.type = "TEXT";  // Simplification - in reality, map types properly
        create_stmt.columns.push_back(std::move(col_def));
    }

    base_engine_.executeCreateTable(create_stmt);

    // Insert results
    for (const auto& row : results.rows) {
        AST::InsertStatement insert_stmt;
        insert_stmt.tableName = table_name;

        // Create values
        for (const auto& value : row) {
            auto literal = std::make_unique<AST::Literal>();
            literal->value = value;
            insert_stmt.values.push_back(std::move(literal));
        }

        base_engine_.executeInsert(insert_stmt);
    }

    std::cout << "[AIExecutionEngineFinal] Saved " << results.rows.size()
              << " simulation results to table: " << table_name << std::endl;
}*/
