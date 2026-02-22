#pragma once
#ifndef MARKET_SIMULATOR_H
#define MARKET_SIMULATOR_H

#include "lightgbm_model.h"
#include "datum.h"
#include <vector>
#include <unordered_map>
#include <memory>
#include <random>
#include <chrono>
#include <functional>
#include <atomic>
#include <thread>
#include <mutex>
#include <queue>

namespace esql {
namespace ai {

// Market event types for real-time simulation
enum class MarketEventType {
    TICK,           // Individual price tick
    TRADE,          // Trade execution
    ORDER_BOOK_UPDATE, // Order book change
    SIGNAL,         // Technical signal
    ANOMALY,        // Anomaly detection
    REGIME_CHANGE,  // Market regime change
    NEWS_IMPACT     // News impact
};

struct MarketEvent {
    MarketEventType type;
    std::chrono::system_clock::time_point timestamp;
    std::string symbol;
    double price;
    double volume;
    std::unordered_map<std::string, double> data;
    
    // For order book
    double bid_price = 0.0;
    double ask_price = 0.0;
    size_t bid_size = 0;
    size_t ask_size = 0;
};

// Market microstructure simulation
struct OrderBook {
    struct Level {
        double price;
        size_t size;
        size_t order_count;
    };
    
    std::vector<Level> bids;
    std::vector<Level> asks;
    double spread = 0.0001;  // 1 basis point default
    
    void update(double mid_price, double volatility);
    double get_best_bid() const { return bids.empty() ? 0.0 : bids[0].price; }
    double get_best_ask() const { return asks.empty() ? 0.0 : asks[0].price; }
    double get_mid_price() const { return (get_best_bid() + get_best_ask()) / 2.0; }
};

// Market regime types
enum class MarketRegime {
    BULL,           // Trending up
    BEAR,           // Trending down
    SIDEWAYS,       // Range-bound
    HIGH_VOLATILITY,
    LOW_VOLATILITY,
    MEAN_REVERTING,
    BREAKOUT,
    CRASH,
    RALLY
};

struct TechnicalIndicators {
    std::vector<double> sma_20;
    std::vector<double> sma_50;
    std::vector<double> ema_12;
    std::vector<double> ema_26;
    std::vector<double> rsi;
    std::vector<double> macd;
    std::vector<double> macd_signal;
    std::vector<double> macd_histogram;
    std::vector<std::pair<double, double>> bollinger_bands;
    std::vector<double> atr;
    std::vector<double> obv;
    std::vector<double> volume_profile;
};

// Simulation path for Monte Carlo
struct SimulationPath {
    std::vector<double> prices;
    std::vector<double> returns;
    std::vector<double> volatilities;
    std::vector<double> volumes;
    std::vector<MarketEvent> events;
    std::vector<MarketRegime> regimes;
    TechnicalIndicators indicators;
    double final_price;
    double total_return;
    double sharpe_ratio;
    double max_drawdown;
    double var_95;  // Value at Risk
    double cvar_95; // Conditional VaR
};

class GARCHModel;
class RegimeDetector;
class MicrostructureEngine;

// Professional Market Simulator
class MarketSimulator {
public:
    MarketSimulator();
    ~MarketSimulator();
    
    // Initialize with model
    bool initialize(std::shared_ptr<AdaptiveLightGBMModel> model,
                   const std::string& time_column,
                   const std::vector<std::string>& value_columns);
    
    // Core simulation
    std::vector<SimulationPath> simulate(
        size_t num_steps,
        size_t num_paths,
        const std::unordered_map<std::string, double>& initial_conditions,
        const std::unordered_map<std::string, double>& scenario_params = {}
    );
    
    // Real-time simulation with event emission
    void start_real_time_simulation(
        size_t num_steps,
        std::chrono::milliseconds step_delay,
        std::function<void(const MarketEvent&)> event_callback
    );
    
    void stop_real_time_simulation();
    bool is_running() const { return simulation_running_; }
    
    // Advanced simulation features
    void set_noise_model(const std::string& model_type, double noise_level);
    void set_volatility_model(const std::string& model_type, double initial_vol);
    void set_regime_detection(bool enable);
    void set_microstructure_simulation(bool enable, double spread = 0.0001);
    
    // Technical indicator generation
    /*struct TechnicalIndicators {
        std::vector<double> sma_20;
        std::vector<double> sma_50;
        std::vector<double> ema_12;
        std::vector<double> ema_26;
        std::vector<double> rsi;
        std::vector<double> macd;
        std::vector<double> macd_signal;
        std::vector<double> macd_histogram;
        std::vector<std::pair<double, double>> bollinger_bands;
        std::vector<double> atr;
        std::vector<double> obv;
        std::vector<double> volume_profile;
    };*/
    
    TechnicalIndicators calculate_indicators(const SimulationPath& path);
    
    // Market analysis
    struct MarketAnalysis {
        MarketRegime current_regime;
        double trend_strength;
        double volatility_regime;
        double correlation_to_benchmark;
        std::vector<std::string> detected_patterns;
        std::unordered_map<std::string, double> risk_metrics;
    };
    
    MarketAnalysis analyze_path(const SimulationPath& path);
    
    // Calibration
    bool calibrate_from_historical_data(
        const std::vector<std::unordered_map<std::string, Datum>>& historical_data
    );
    
    // Performance metrics
    struct SimulationMetrics {
        double avg_return;
        double volatility;
        double sharpe_ratio;
        double sortino_ratio;
        double max_drawdown;
        double calmar_ratio;
        double win_rate;
        double profit_factor;
        double expectancy;
        std::vector<double> drawdown_series;
        std::unordered_map<std::string, double> custom_metrics;
    };
    
    SimulationMetrics calculate_metrics(const std::vector<SimulationPath>& paths);
    
    // Export/Import
    nlohmann::json to_json() const;
    static MarketSimulator from_json(const nlohmann::json& j);

private:
    std::shared_ptr<AdaptiveLightGBMModel> model_;
    std::string time_column_;
    std::vector<std::string> value_columns_;
    
    // Random number generation
    std::mt19937 rng_;
    std::normal_distribution<double> normal_dist_;
    std::uniform_real_distribution<double> uniform_dist_;
    
    // State
    std::atomic<bool> simulation_running_{false};
    std::thread simulation_thread_;
    std::mutex state_mutex_;
    
    // Models
    std::unique_ptr<class GARCHModel> volatility_model_;
    std::unique_ptr<class RegimeDetector> regime_detector_;
    std::unique_ptr<class MicrostructureEngine> microstructure_;
    
    // Parameters
    double noise_level_ = 0.01;
    double volatility_ = 0.02;
    bool detect_regimes_ = false;
    bool simulate_microstructure_ = false;
    bool include_mean_reversion_ = false;
    double mean_reversion_strength_ = 0.1;
    bool include_volatility_clustering_ = false;
    
    // Internal methods
    double predict_next_price(
        const std::vector<double>& history,
        const std::unordered_map<std::string, double>& features
    );
    
    double apply_noise(double price, double volatility);
    double apply_mean_reversion(double price, double mean, double strength);
    double apply_volatility_clustering(double volatility, double prev_volatility);
    double apply_market_impact(double price, double volume, double liquidity);
    
    MarketRegime detect_regime(const std::vector<double>& prices);
    void generate_technical_indicators(SimulationPath& path);
    void emit_event(MarketEventType type, double price, double volume);
    
    std::vector<double> compute_returns(const std::vector<double>& prices);
    std::vector<double> compute_volatilities(const std::vector<double>& returns);
    double compute_sharpe_ratio(const std::vector<double>& returns);
    double compute_max_drawdown(const std::vector<double>& prices);
};

// Factory for creating market simulators
class MarketSimulatorFactory {
public:
    static std::unique_ptr<MarketSimulator> create_for_forex(
        std::shared_ptr<AdaptiveLightGBMModel> model,
        const std::string& base_currency,
        const std::string& quote_currency
    );
    
    static std::unique_ptr<MarketSimulator> create_for_stocks(
        std::shared_ptr<AdaptiveLightGBMModel> model,
        const std::string& symbol,
        bool include_dividends = true
    );
    
    static std::unique_ptr<MarketSimulator> create_for_crypto(
        std::shared_ptr<AdaptiveLightGBMModel> model,
        const std::string& symbol,
        bool simulate_mining = false
    );
    
    static std::unique_ptr<MarketSimulator> create_for_commodities(
        std::shared_ptr<AdaptiveLightGBMModel> model,
        const std::string& commodity,
        bool include_futures_curve = true
    );
};

} // namespace ai
} // namespace esql

#endif // MARKET_SIMULATOR_H
