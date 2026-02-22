#include "market_simulator.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iomanip>

namespace esql {
namespace ai {

// ============================================
// GARCH Volatility Model
// ============================================
class GARCHModel {
private:
    double omega_;
    double alpha_;
    double beta_;
    double current_volatility_;

public:
    GARCHModel(double omega = 0.000001, double alpha = 0.1, double beta = 0.85)
        : omega_(omega), alpha_(alpha), beta_(beta), current_volatility_(0.02) {}

    double update(double return_value) {
        double var = omega_ +
                    alpha_ * return_value * return_value +
                    beta_ * current_volatility_ * current_volatility_;
        current_volatility_ = std::sqrt(var);
        return current_volatility_;
    }

    void calibrate(const std::vector<double>& returns) {
        // Simplified MLE calibration
        // In production, use full MLE or Bayesian methods
        double sum_sq = 0.0;
        for (double r : returns) {
            sum_sq += r * r;
        }
        double avg_var = sum_sq / returns.size();
        current_volatility_ = std::sqrt(avg_var);
    }

    double get_volatility() const { return current_volatility_; }
    void set_volatility(double vol) { current_volatility_ = vol; }
};

// ============================================
// Regime Detector
// ============================================
class RegimeDetector {
private:
    size_t lookback_ = 50;
    double bull_threshold_ = 0.05;
    double bear_threshold_ = -0.05;
    double high_vol_threshold_ = 0.03;

public:
    MarketRegime detect(const std::vector<double>& prices, double volatility) {
        if (prices.size() < lookback_) {
            return MarketRegime::SIDEWAYS;
        }

        // Calculate trend
        double start_price = prices[prices.size() - lookback_];
        double end_price = prices.back();
        double trend = (end_price - start_price) / start_price;

        // Detect regime
        if (volatility > high_vol_threshold_) {
            if (trend > bull_threshold_) {
                return MarketRegime::RALLY;
            } else if (trend < bear_threshold_) {
                return MarketRegime::CRASH;
            }
            return MarketRegime::HIGH_VOLATILITY;
        }

        if (trend > bull_threshold_) {
            return MarketRegime::BULL;
        } else if (trend < bear_threshold_) {
            return MarketRegime::BEAR;
        }

        // Check for mean reversion
        double mean_price = std::accumulate(prices.end() - 20, prices.end(), 0.0) / 20.0;
        double current_price = prices.back();
        double deviation = std::abs(current_price - mean_price) / mean_price;

        if (deviation < 0.01) {
            return MarketRegime::MEAN_REVERTING;
        }

        return MarketRegime::SIDEWAYS;
    }

    void set_lookback(size_t period) { lookback_ = period; }
    void set_thresholds(double bull, double bear, double high_vol) {
        bull_threshold_ = bull;
        bear_threshold_ = bear;
        high_vol_threshold_ = high_vol;
    }
};

// ============================================
// Microstructure Engine
// ============================================
class MicrostructureEngine {
private:
    double spread_;
    double liquidity_impact_;
    OrderBook order_book_;

public:
    MicrostructureEngine(double spread = 0.0001, double liquidity_impact = 0.01)
        : spread_(spread), liquidity_impact_(liquidity_impact) {
        order_book_.spread = spread;
    }

    double apply_microstructure(double fair_price, double volume, double volatility) {
        std::lock_guard<std::mutex> lock(mutex_);

        // Update order book
        order_book_.update(fair_price, volatility);

        // Calculate effective spread
        double effective_spread = spread_ * (1.0 + volatility * 10.0);

        // Apply liquidity impact
        double impact = volume * liquidity_impact_ * volatility;

        // Randomize execution price within spread
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-effective_spread/2, effective_spread/2);

        double execution_price = fair_price + dis(gen) + impact;

        // Update last trade
        last_trade_price_ = execution_price;
        last_trade_volume_ = volume;
        last_trade_time_ = std::chrono::system_clock::now();

        return execution_price;
    }

    OrderBook get_order_book() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return order_book_;
    }

    MarketEvent create_tick_event(double price, double volume) {
        MarketEvent event;
        event.type = MarketEventType::TICK;
        event.timestamp = std::chrono::system_clock::now();
        event.price = price;
        event.volume = volume;
        event.bid_price = order_book_.get_best_bid();
        event.ask_price = order_book_.get_best_ask();
        return event;
    }

private:
    mutable std::mutex mutex_;
    double last_trade_price_ = 0.0;
    double last_trade_volume_ = 0.0;
    std::chrono::system_clock::time_point last_trade_time_;
};

void OrderBook::update(double mid_price, double volatility) {
    // Simple order book model
    double half_spread = spread / 2.0;
    double vol_adjustment = volatility * spread;

    bids.clear();
    asks.clear();

    // Generate 5 levels on each side
    for (int i = 0; i < 5; ++i) {
        double level_spread = half_spread * (1.0 + i * 0.5) + vol_adjustment;

        Level bid;
        bid.price = mid_price - level_spread;
        bid.size = 1000 * (5 - i); // Decreasing size away from mid
        bid.order_count = 10 * (5 - i);
        bids.push_back(bid);

        Level ask;
        ask.price = mid_price + level_spread;
        ask.size = 1000 * (5 - i);
        ask.order_count = 10 * (5 - i);
        asks.push_back(ask);
    }
}

// ============================================
// MarketSimulator Implementation
// ============================================

MarketSimulator::MarketSimulator()
    : rng_(std::chrono::system_clock::now().time_since_epoch().count()),
      normal_dist_(0.0, 1.0),
      uniform_dist_(0.0, 1.0) {

    volatility_model_ = std::make_unique<GARCHModel>();
    regime_detector_ = std::make_unique<RegimeDetector>();
    microstructure_ = std::make_unique<MicrostructureEngine>();
}

MarketSimulator::~MarketSimulator() {
    stop_real_time_simulation();
}

bool MarketSimulator::initialize(std::shared_ptr<AdaptiveLightGBMModel> model,
                                const std::string& time_column,
                                const std::vector<std::string>& value_columns) {
    if (!model) {
        return false;
    }

    model_ = model;
    time_column_ = time_column;
    value_columns_ = value_columns;

    return true;
}

double MarketSimulator::predict_next_price(
    const std::vector<double>& history,
    const std::unordered_map<std::string, double>& features) {

    if (!model_) {
        return history.empty() ? 100.0 : history.back();
    }

    // Convert to tensor
    std::vector<float> input_features;
    input_features.reserve(features.size());

    for (const auto& [_, value] : features) {
        input_features.push_back(static_cast<float>(value));
    }

    Tensor input(std::move(input_features), {input_features.size()});

    try {
        Tensor prediction = model_->predict(input);
        return prediction.data.empty() ? history.back() : prediction.data[0];
    } catch (const std::exception& e) {
        std::cerr << "[MarketSimulator] Prediction failed: " << e.what() << std::endl;
        return history.empty() ? 100.0 : history.back();
    }
}

double MarketSimulator::apply_noise(double price, double volatility) {
    double noise = normal_dist_(rng_) * price * volatility * noise_level_;
    return price + noise;
}

double MarketSimulator::apply_mean_reversion(double price, double mean, double strength) {
    return price + strength * (mean - price);
}

double MarketSimulator::apply_volatility_clustering(double volatility, double prev_volatility) {
    if (!include_volatility_clustering_) {
        return volatility;
    }

    // GARCH-like volatility clustering
    double new_vol = volatility_model_->update(prev_volatility);
    return new_vol;
}

double MarketSimulator::apply_market_impact(double price, double volume, double liquidity) {
    if (!simulate_microstructure_) {
        return price;
    }

    return microstructure_->apply_microstructure(price, volume, volatility_);
}

MarketRegime MarketSimulator::detect_regime(const std::vector<double>& prices) {
    if (!detect_regimes_) {
        return MarketRegime::SIDEWAYS;
    }

    return regime_detector_->detect(prices, volatility_);
}

std::vector<SimulationPath> MarketSimulator::simulate(
    size_t num_steps,
    size_t num_paths,
    const std::unordered_map<std::string, double>& initial_conditions,
    const std::unordered_map<std::string, double>& scenario_params) {

    std::vector<SimulationPath> paths;
    paths.reserve(num_paths);

    // Initialize random number generator with seed for reproducibility
    std::mt19937 path_rng(rng_());

    for (size_t path_idx = 0; path_idx < num_paths; ++path_idx) {
        SimulationPath path;
        path.prices.reserve(num_steps + 1);
        path.returns.reserve(num_steps);
        path.volatilities.reserve(num_steps);
        path.volumes.reserve(num_steps);
        path.regimes.reserve(num_steps);

        // Set initial price
        double current_price = initial_conditions.at("price");
        path.prices.push_back(current_price);

        double current_volatility = volatility_;
        double prev_return = 0.0;

        // Track for mean reversion
        double historical_mean = current_price;
        std::vector<double> price_window;
        price_window.push_back(current_price);

        for (size_t step = 0; step < num_steps; ++step) {
            // Build features for prediction
            std::unordered_map<std::string, double> features = initial_conditions;
            features["price"] = current_price;
            features["volatility"] = current_volatility;
            features["step"] = static_cast<double>(step);
            features["path"] = static_cast<double>(path_idx);

            // Add scenario parameters
            for (const auto& [key, value] : scenario_params) {
                features[key] = value;
            }

            // Predict next price
            double predicted_price = predict_next_price(path.prices, features);

            // Apply mean reversion if enabled
            if (include_mean_reversion_) {
                // Calculate rolling mean
                if (price_window.size() > 20) {
                    price_window.erase(price_window.begin());
                }
                historical_mean = std::accumulate(price_window.begin(), price_window.end(), 0.0) / price_window.size();

                predicted_price = apply_mean_reversion(predicted_price, historical_mean, mean_reversion_strength_);
            }

            // Apply volatility clustering
            double return_val = (predicted_price - current_price) / current_price;
            current_volatility = apply_volatility_clustering(current_volatility, std::abs(return_val));

            // Apply noise
            predicted_price = apply_noise(predicted_price, current_volatility);

            // Apply market microstructure if enabled
            if (simulate_microstructure_) {
                double volume = 1000.0 * (1.0 + uniform_dist_(rng_));
                predicted_price = apply_market_impact(predicted_price, volume, current_volatility);
                path.volumes.push_back(volume);

                // Generate tick event
                auto event = microstructure_->create_tick_event(predicted_price, volume);
                path.events.push_back(event);
            }

            // Calculate return
            return_val = (predicted_price - current_price) / current_price;

            // Update path
            path.prices.push_back(predicted_price);
            path.returns.push_back(return_val);
            path.volatilities.push_back(current_volatility);

            // Detect regime
            MarketRegime regime = detect_regime(path.prices);
            path.regimes.push_back(regime);

            // Update state
            current_price = predicted_price;
            price_window.push_back(current_price);
            prev_return = return_val;
        }

        // Calculate path metrics
        path.final_price = path.prices.back();
        path.total_return = (path.final_price - path.prices[0]) / path.prices[0];

        // Calculate Sharpe ratio
        double mean_return = std::accumulate(path.returns.begin(), path.returns.end(), 0.0) / path.returns.size();
        double std_return = 0.0;
        for (double r : path.returns) {
            std_return += (r - mean_return) * (r - mean_return);
        }
        std_return = std::sqrt(std_return / path.returns.size());
        path.sharpe_ratio = (mean_return / std_return) * std::sqrt(252.0); // Annualized

        // Calculate max drawdown
        path.max_drawdown = compute_max_drawdown(path.prices);

        // Calculate VaR
        std::vector<double> sorted_returns = path.returns;
        std::sort(sorted_returns.begin(), sorted_returns.end());
        size_t var_idx = static_cast<size_t>(0.05 * sorted_returns.size());
        path.var_95 = -sorted_returns[var_idx];

        // Calculate CVaR
        double cvar_sum = 0.0;
        for (size_t i = 0; i < var_idx; ++i) {
            cvar_sum += -sorted_returns[i];
        }
        path.cvar_95 = cvar_sum / var_idx;

        // Generate technical indicators
        generate_technical_indicators(path);

        paths.push_back(std::move(path));
    }

    return paths;
}

void MarketSimulator::start_real_time_simulation(
    size_t num_steps,
    std::chrono::milliseconds step_delay,
    std::function<void(const MarketEvent&)> event_callback) {

    if (simulation_running_) {
        return;
    }

    simulation_running_ = true;

    simulation_thread_ = std::thread([this, num_steps, step_delay, event_callback]() {
        // Create initial conditions
        std::unordered_map<std::string, double> initial_conditions;
        initial_conditions["price"] = 100.0; // Should come from model

        double current_price = 100.0;
        double current_volatility = volatility_;
        std::vector<double> price_history;
        price_history.push_back(current_price);

        for (size_t step = 0; step < num_steps && simulation_running_; ++step) {
            // Simulate next price
            std::unordered_map<std::string, double> features = initial_conditions;
            features["price"] = current_price;
            features["volatility"] = current_volatility;
            features["step"] = static_cast<double>(step);

            double predicted_price = predict_next_price(price_history, features);
            predicted_price = apply_noise(predicted_price, current_volatility);

            // Create market event
            MarketEvent event;
            event.type = MarketEventType::TICK;
            event.timestamp = std::chrono::system_clock::now();
            event.price = predicted_price;
            event.volume = 1000.0 * (1.0 + uniform_dist_(rng_));

            // Add order book if microstructure enabled
            if (simulate_microstructure_) {
                auto order_book = microstructure_->get_order_book();
                event.bid_price = order_book.get_best_bid();
                event.ask_price = order_book.get_best_ask();
            }

            // Emit event
            if (event_callback) {
                event_callback(event);
            }

            // Update state
            current_price = predicted_price;
            price_history.push_back(current_price);

            // Apply delay
            std::this_thread::sleep_for(step_delay);
        }

        simulation_running_ = false;
    });
}

void MarketSimulator::stop_real_time_simulation() {
    simulation_running_ = false;
    if (simulation_thread_.joinable()) {
        simulation_thread_.join();
    }
}

void MarketSimulator::set_noise_model(const std::string& model_type, double noise_level) {
    noise_level_ = noise_level;
    // Could implement different noise models (Gaussian, t-distribution, etc.)
}

void MarketSimulator::set_volatility_model(const std::string& model_type, double initial_vol) {
    volatility_ = initial_vol;
    if (model_type == "GARCH") {
        volatility_model_ = std::make_unique<GARCHModel>();
    }
    // Add other volatility models as needed
}

void MarketSimulator::set_regime_detection(bool enable) {
    detect_regimes_ = enable;
}

void MarketSimulator::set_microstructure_simulation(bool enable, double spread) {
    simulate_microstructure_ = enable;
    microstructure_ = std::make_unique<MicrostructureEngine>(spread);
}

void MarketSimulator::generate_technical_indicators(SimulationPath& path) {
    const auto& prices = path.prices;
    size_t n = prices.size();

    if (n < 50) return;

    // Simple Moving Averages
    for (size_t i = 20; i <= n; ++i) {
        double sum_20 = 0.0;
        for (size_t j = i - 20; j < i; ++j) {
            sum_20 += prices[j];
        }
        path.indicators.sma_20.push_back(sum_20 / 20.0);
    }

    // RSI
    std::vector<double> gains, losses;
    for (size_t i = 1; i < n; ++i) {
        double change = prices[i] - prices[i-1];
        if (change >= 0) {
            gains.push_back(change);
            losses.push_back(0.0);
        } else {
            gains.push_back(0.0);
            losses.push_back(-change);
        }
    }

    for (size_t i = 14; i < gains.size(); ++i) {
        double avg_gain = std::accumulate(gains.begin() + i - 14, gains.begin() + i, 0.0) / 14.0;
        double avg_loss = std::accumulate(losses.begin() + i - 14, losses.begin() + i, 0.0) / 14.0;
        if (avg_loss > 0) {
            double rs = avg_gain / avg_loss;
            path.indicators.rsi.push_back(100.0 - (100.0 / (1.0 + rs)));
        } else {
            path.indicators.rsi.push_back(100.0);
        }
    }

    // Bollinger Bands
    for (size_t i = 20; i <= n; ++i) {
        double sum = 0.0;
        for (size_t j = i - 20; j < i; ++j) {
            sum += prices[j];
        }
        double mean = sum / 20.0;

        double sq_sum = 0.0;
        for (size_t j = i - 20; j < i; ++j) {
            sq_sum += (prices[j] - mean) * (prices[j] - mean);
        }
        double std = std::sqrt(sq_sum / 20.0);

        path.indicators.bollinger_bands.emplace_back(mean - 2 * std, mean + 2 * std);
    }
}

MarketSimulator::MarketAnalysis MarketSimulator::analyze_path(const SimulationPath& path) {
    MarketAnalysis analysis;

    // Detect current regime
    if (!path.prices.empty()) {
        analysis.current_regime = detect_regime(path.prices);
    }

    // Calculate trend strength using linear regression
    if (path.prices.size() > 20) {
        std::vector<double> x(path.prices.size());
        std::iota(x.begin(), x.end(), 0.0);

        double sum_x = std::accumulate(x.begin(), x.end(), 0.0);
        double sum_y = std::accumulate(path.prices.begin(), path.prices.end(), 0.0);
        double sum_xy = 0.0;
        double sum_xx = 0.0;

        for (size_t i = 0; i < path.prices.size(); ++i) {
            sum_xy += x[i] * path.prices[i];
            sum_xx += x[i] * x[i];
        }

        double slope = (path.prices.size() * sum_xy - sum_x * sum_y) /
                      (path.prices.size() * sum_xx - sum_x * sum_x);

        analysis.trend_strength = std::abs(slope);
    }

    // Calculate volatility regime
    if (!path.volatilities.empty()) {
        double avg_vol = std::accumulate(path.volatilities.begin(), path.volatilities.end(), 0.0) /
                        path.volatilities.size();

        if (avg_vol > 0.03) {
            analysis.volatility_regime = 2.0; // High
        } else if (avg_vol > 0.015) {
            analysis.volatility_regime = 1.0; // Normal
        } else {
            analysis.volatility_regime = 0.0; // Low
        }
    }

    // Detect patterns (simplified)
    if (path.prices.size() > 10) {
        // Check for head and shoulders pattern
        bool head_and_shoulders = false;
        // ... pattern detection logic

        // Check for double top/bottom
        bool double_top = false;
        // ... pattern detection logic

        if (head_and_shoulders) {
            analysis.detected_patterns.push_back("HEAD_AND_SHOULDERS");
        }
        if (double_top) {
            analysis.detected_patterns.push_back("DOUBLE_TOP");
        }
    }

    // Calculate risk metrics
    analysis.risk_metrics["var_95"] = path.var_95;
    analysis.risk_metrics["cvar_95"] = path.cvar_95;
    analysis.risk_metrics["max_drawdown"] = path.max_drawdown;
    analysis.risk_metrics["sharpe_ratio"] = path.sharpe_ratio;

    return analysis;
}

bool MarketSimulator::calibrate_from_historical_data(
    const std::vector<std::unordered_map<std::string, Datum>>& historical_data) {

    if (historical_data.size() < 100) {
        return false;
    }

    // Extract price series
    std::vector<double> prices;
    for (const auto& row : historical_data) {
        auto it = row.find("price");
        if (it != row.end() && !it->second.is_null()) {
            prices.push_back(it->second.as_double());
        }
    }

    if (prices.size() < 2) {
        return false;
    }

    // Calculate returns
    std::vector<double> returns;
    for (size_t i = 1; i < prices.size(); ++i) {
        returns.push_back((prices[i] - prices[i-1]) / prices[i-1]);
    }

    // Calibrate volatility model
    volatility_model_->calibrate(returns);

    // Set initial volatility
    volatility_ = volatility_model_->get_volatility();

    // Calibrate regime detector based on historical volatility
    double avg_vol = std::accumulate(returns.begin(), returns.end(), 0.0,
        [](double acc, double r) { return acc + r * r; }) / returns.size();
    avg_vol = std::sqrt(avg_vol);

    regime_detector_->set_thresholds(0.03, -0.03, avg_vol * 1.5);

    return true;
}

MarketSimulator::SimulationMetrics MarketSimulator::calculate_metrics(
    const std::vector<SimulationPath>& paths) {

    SimulationMetrics metrics;

    if (paths.empty()) {
        return metrics;
    }

    // Aggregate returns across paths
    std::vector<double> all_returns;
    for (const auto& path : paths) {
        all_returns.insert(all_returns.end(), path.returns.begin(), path.returns.end());
    }

    // Calculate basic statistics
    double sum_returns = std::accumulate(all_returns.begin(), all_returns.end(), 0.0);
    metrics.avg_return = sum_returns / all_returns.size();

    double sq_sum = 0.0;
    for (double r : all_returns) {
        sq_sum += (r - metrics.avg_return) * (r - metrics.avg_return);
    }
    metrics.volatility = std::sqrt(sq_sum / all_returns.size());

    // Sharpe ratio
    metrics.sharpe_ratio = (metrics.avg_return / metrics.volatility) * std::sqrt(252.0);

    // Sortino ratio (downside deviation)
    double downside_sum = 0.0;
    size_t downside_count = 0;
    for (double r : all_returns) {
        if (r < 0) {
            downside_sum += r * r;
            downside_count++;
        }
    }
    double downside_dev = downside_count > 0 ? std::sqrt(downside_sum / downside_count) : 0.0;
    metrics.sortino_ratio = (metrics.avg_return / downside_dev) * std::sqrt(252.0);

    // Max drawdown across all paths
    double max_dd = 0.0;
    for (const auto& path : paths) {
        max_dd = std::max(max_dd, path.max_drawdown);
    }
    metrics.max_drawdown = max_dd;

    // Calmar ratio
    if (metrics.max_drawdown > 0) {
        metrics.calmar_ratio = metrics.avg_return * 252.0 / metrics.max_drawdown;
    }

    // Win rate
    size_t winning_trades = 0;
    for (double r : all_returns) {
        if (r > 0) winning_trades++;
    }
    metrics.win_rate = static_cast<double>(winning_trades) / all_returns.size();

    // Profit factor (total gains / total losses)
    double total_gain = 0.0, total_loss = 0.0;
    for (double r : all_returns) {
        if (r > 0) {
            total_gain += r;
        } else {
            total_loss -= r;
        }
    }
    if (total_loss > 0) {
        metrics.profit_factor = total_gain / total_loss;
    }

    // Expectancy
    metrics.expectancy = metrics.avg_return;

    // Drawdown series
    for (const auto& path : paths) {
        double peak = path.prices[0];
        double max_dd_path = 0.0;
        for (double price : path.prices) {
            if (price > peak) {
                peak = price;
            }
            double dd = (peak - price) / peak;
            if (dd > max_dd_path) {
                max_dd_path = dd;
            }
        }
        metrics.drawdown_series.push_back(max_dd_path);
    }

    return metrics;
}

double MarketSimulator::compute_max_drawdown(const std::vector<double>& prices) {
    double max_dd = 0.0;
    double peak = prices[0];

    for (double price : prices) {
        if (price > peak) {
            peak = price;
        }
        double dd = (peak - price) / peak;
        if (dd > max_dd) {
            max_dd = dd;
        }
    }

    return max_dd;
}

// ============================================
// Factory Implementation
// ============================================

std::unique_ptr<MarketSimulator> MarketSimulatorFactory::create_for_forex(
    std::shared_ptr<AdaptiveLightGBMModel> model,
    const std::string& base_currency,
    const std::string& quote_currency) {

    auto simulator = std::make_unique<MarketSimulator>();
    simulator->set_noise_model("Gaussian", 0.005);  // Forex typically lower noise
    simulator->set_volatility_model("GARCH", 0.01);
    simulator->set_microstructure_simulation(true, 0.0001);  // 1 pip spread
    simulator->set_regime_detection(true);

    return simulator;
}

std::unique_ptr<MarketSimulator> MarketSimulatorFactory::create_for_stocks(
    std::shared_ptr<AdaptiveLightGBMModel> model,
    const std::string& symbol,
    bool include_dividends) {

    auto simulator = std::make_unique<MarketSimulator>();
    simulator->set_noise_model("Gaussian", 0.015);  // Higher noise for stocks
    simulator->set_volatility_model("GARCH", 0.02);
    simulator->set_microstructure_simulation(true, 0.0005);  // 5 basis points spread
    simulator->set_regime_detection(true);

    return simulator;
}

std::unique_ptr<MarketSimulator> MarketSimulatorFactory::create_for_crypto(
    std::shared_ptr<AdaptiveLightGBMModel> model,
    const std::string& symbol,
    bool simulate_mining) {

    auto simulator = std::make_unique<MarketSimulator>();
    simulator->set_noise_model("Gaussian", 0.03);  // High noise for crypto
    simulator->set_volatility_model("GARCH", 0.04);
    simulator->set_microstructure_simulation(true, 0.001);  // 10 basis points spread
    simulator->set_regime_detection(true);

    return simulator;
}

std::unique_ptr<MarketSimulator> MarketSimulatorFactory::create_for_commodities(
    std::shared_ptr<AdaptiveLightGBMModel> model,
    const std::string& commodity,
    bool include_futures_curve) {

    auto simulator = std::make_unique<MarketSimulator>();
    simulator->set_noise_model("Gaussian", 0.02);
    simulator->set_volatility_model("GARCH", 0.02);
    simulator->set_microstructure_simulation(true, 0.0002);
    simulator->set_regime_detection(true);

    return simulator;
}

} // namespace ai
} // namespace esql
