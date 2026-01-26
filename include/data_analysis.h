
// Enhanced data_analysis.h - Production grade analysis
// ============================================
#ifndef DATA_ANALYSIS_PRO_H
#define DATA_ANALYSIS_PRO_H

#include "datum.h"
#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <set>
#include <map>
#include <nlohmann/json.hpp>
#include <queue>
#include <random>
#include <chrono>
#include <mutex>
#include <future>
#include <thread>

namespace esql {
namespace analysis {

//class ProfessionalStatisticalCalculator;

// Enhanced statistical summary
struct ProfessionalColumnAnalysis {
    std::string name;
    std::string detected_type;
    size_t total_count = 0;
    size_t null_count = 0;
    size_t distinct_count = 0;
    size_t zero_count = 0;
    double min_value = 0.0;
    double max_value = 0.0;
    double range = 0.0;
    double mean = 0.0;
    double median = 0.0;
    double mode = 0.0;
    double trimmed_mean_5 = 0.0;
    double trimmed_mean_10 = 0.0;
    double winsorized_mean_5 = 0.0;
    double std_dev = 0.0;
    double variance = 0.0;
    double skewness = 0.0;
    double kurtosis = 0.0;
    double jarque_bera = 0.0;
    double shapiro_wilk = 0.0;
    double anderson_darling = 0.0;
    double q1 = 0.0;     // First quartile (25th percentile)
    double q2 = 0.0;     // Second quartile (50th percentile - median)
    double q3 = 0.0;     // Third quartile (75th percentile)
    double iqr = 0.0;    // Interquartile range
    double p10 = 0.0;    // 10th percentile
    double p90 = 0.0;    // 90th percentile
    double p95 = 0.0;    // 95th percentile
    double p99 = 0.0;    // 99th percentile
    double mad = 0.0;    // Median Absolute Deviation
    double coefficient_of_variation = 0.0;
    double entropy = 0.0;
    double gini_coefficient = 0.0;
    double theil_index = 0.0;
    double normality_p_value = 0.0;
    bool is_normal = false;
    bool is_categorical = false;
    bool has_outliers = false;
    bool is_stationary = false;
    bool is_constant = false;
    double missing_percentage = 0.0;
    double outlier_percentage = 0.0;
    std::vector<double> histogram_bins;
    std::vector<size_t> histogram_counts;
    std::vector<double> percentile_values;  // All percentiles from 1 to 99
    std::vector<double> outliers;
    std::vector<size_t> outlier_indices;
    std::vector<std::string> top_categories;
    std::map<std::string, size_t> value_frequencies;
    std::map<std::string, double> distribution_fit; // Best fitting distribution
    std::vector<double> autocorrelation; // For time series
    std::vector<double> partial_autocorrelation;
    double hurst_exponent = 0.0;

    // Data quality metrics
    struct DataQuality {
        double completeness_score = 0.0;
        double consistency_score = 0.0;
        double accuracy_score = 0.0;
        double uniqueness_score = 0.0;
        double timeliness_score = 0.0;
        double validity_score = 0.0;
        double overall_score = 0.0;
        std::vector<std::string> issues;
        std::vector<std::string> recommendations;
    } quality;

    // For categorical columns
    struct CategoricalStats {
        size_t max_category_length = 0;
        size_t min_category_length = 0;
        double average_category_length = 0.0;
        double cardinality_ratio = 0.0;
        std::map<std::string, double> category_probabilities;
        std::vector<std::string> rare_categories; // < 1% frequency
        std::vector<std::string> dominant_categories; // > 10% frequency
    } categorical_stats;

    // For numeric columns
    struct NumericStats {
        double geometric_mean = 0.0;
        double harmonic_mean = 0.0;
        double quadratic_mean = 0.0;
        double log_mean = 0.0;
        double power_mean_p1 = 0.0;
        double power_mean_p2 = 0.0;
        double power_mean_p3 = 0.0;
        double contraharmonic_mean = 0.0;
        double lehmer_mean_alpha2 = 0.0;
        double hodges_lehmann_estimator = 0.0;
        double midhinge = 0.0; // (Q1 + Q3)/2
        double trimean = 0.0;  // (Q1 + 2*Q2 + Q3)/4
        double midrange = 0.0; // (min + max)/2
        double interdecile_range = 0.0;
        double range_ratio = 0.0;
        std::vector<double> moments; // First 4 moments
        std::vector<double> l_moments; // First 4 L-moments
        std::vector<double> cumulants; // First 4 cumulants
    } numeric_stats;

    // Time series specific
    struct TimeSeriesStats {
        bool is_stationary_adf = false;
        bool is_stationary_kpss = false;
        double adf_pvalue = 0.0;
        double kpss_pvalue = 0.0;
        double hurst_exponent = 0.0;
        double lyapunov_exponent = 0.0;
        double autocorrelation_time = 0.0;
        std::vector<double> spectral_density;
        std::vector<double> periodogram;
    } time_series_stats;

    std::string to_detailed_string() const;
    nlohmann::json to_json(bool detailed = true) const;
    std::string to_markdown() const;
    std::string to_csv() const;
};

// Professional correlation analysis
struct ProfessionalCorrelationAnalysis {
    std::string column1;
    std::string column2;

    // Linear correlations
    double pearson_r = 0.0;
    double pearson_r_squared = 0.0;
    double pearson_p_value = 0.0;
    double pearson_confidence_lower = 0.0;
    double pearson_confidence_upper = 0.0;

    // Rank correlations
    double spearman_rho = 0.0;
    double spearman_p_value = 0.0;
    double kendall_tau = 0.0;
    double kendall_p_value = 0.0;

    // Distance correlation
    double distance_correlation = 0.0;
    double distance_p_value = 0.0;

    // Mutual information
    double mutual_information = 0.0;
    double normalized_mutual_information = 0.0;
    double adjusted_mutual_information = 0.0;

    // Maximal information coefficient
    double mic = 0.0;
    double mas = 0.0;
    double mev = 0.0;
    double mcn = 0.0;

    // Partial and semi-partial correlations
    double partial_correlation = 0.0;
    double semi_partial_correlation = 0.0;

    // Canonical correlation
    double canonical_correlation = 0.0;

    // Cross-correlation for time series
    double cross_correlation = 0.0;
    double cross_correlation_lag = 0.0;

    // Granger causality
    bool granger_causes = false;
    double granger_f_statistic = 0.0;
    double granger_p_value = 0.0;

    // Cointegration (for time series)
    bool cointegrated = false;
    double cointegration_p_value = 0.0;

    std::string relationship_strength; // "very_strong", "strong", "moderate", "weak", "very_weak"
    std::string relationship_direction; // "positive", "negative", "mixed"
    std::string relationship_type; // "linear", "monotonic", "nonlinear", "no_relationship"

    bool is_statistically_significant = false;
    bool is_practically_significant = false;
    double effect_size = 0.0; // Cohen's d

    // Visualization data
    std::vector<std::pair<double, double>> scatter_data;
    std::vector<double> regression_line;
    std::vector<double> confidence_band;
    std::vector<double> prediction_band;

    std::string to_detailed_string() const;
    nlohmann::json to_json() const;
};

// Professional feature importance
struct ProfessionalFeatureImportance {
    std::string feature_name;
    std::string feature_type;

    // Multiple importance measures
    struct ImportanceScores {
        double pearson = 0.0;
        double spearman = 0.0;
        double mutual_information = 0.0;
        double chi_square = 0.0;
        double anova_f = 0.0;
        double lasso_coefficient = 0.0;
        double random_forest = 0.0;
        double xgboost = 0.0;
        double lightgbm = 0.0;
        double shap = 0.0;
        double lime = 0.0;
        double permutation = 0.0;
        double boruta = 0.0;
        double relief = 0.0;
        double mrmr = 0.0;
        double fisher_score = 0.0;
        double laplacian_score = 0.0;
    } scores;

    // Statistical significance
    struct StatisticalSignificance {
        bool pearson_significant = false;
        bool spearman_significant = false;
        bool mi_significant = false;
        bool chi2_significant = false;
        bool anova_significant = false;
        double best_p_value = 0.0;
        std::string best_method;
    } significance;

    // Stability analysis
    struct Stability {
        double variance_across_folds = 0.0;
        double rank_stability = 0.0;
        double jaccard_similarity = 0.0;
        std::vector<double> importance_across_folds;
        std::vector<size_t> rank_across_folds;
    } stability;

    // Interaction effects
    std::vector<std::pair<std::string, double>> top_interactions;
    std::vector<std::pair<std::string, double>> synergy_effects;
    std::vector<std::pair<std::string, double>> redundancy_effects;

    // Non-linear effects
    double maximal_information_coefficient = 0.0;
    double distance_correlation = 0.0;
    double hoeffding_d = 0.0;

    // Partial dependence
    std::vector<std::pair<double, double>> partial_dependence;
    std::vector<std::pair<double, double>> individual_conditional_expectation;
    std::vector<std::pair<double, double>> accumulated_local_effects;

    // Shapley values distribution
    struct ShapleyAnalysis {
        double mean_abs_shap = 0.0;
        double std_shap = 0.0;
        double min_shap = 0.0;
        double max_shap = 0.0;
        std::vector<double> shap_values;
        std::vector<double> shap_interaction_values;
    } shapley;

    // Feature engineering suggestions
    std::vector<std::string> transformation_suggestions;
    std::vector<std::string> interaction_suggestions;
    std::vector<std::string> encoding_suggestions;

    // Business impact
    double business_value_score = 0.0;
    double data_collection_cost = 0.0;
    double feature_stability_score = 0.0;
    double feature_freshness_score = 0.0;

    // Recommendations
    enum Action {
        KEEP_AS_IS,
        TRANSFORM,
        CREATE_INTERACTION,
        BIN,
        ENCODE,
        DROP,
        MONITOR
    } recommended_action;

    std::string to_detailed_string() const;
    nlohmann::json to_json() const;
};

// Professional clustering analysis
struct ProfessionalClusterAnalysis {
    size_t cluster_id;
    std::string cluster_label;
    size_t size;
    double size_percentage;

    // Cluster characteristics
    std::vector<double> centroid;
    std::vector<double> centroid_std;
    std::vector<double> medoid;
    std::vector<std::string> defining_features;
    std::vector<std::pair<std::string, double>> feature_importances;

    // Cluster quality metrics
    double silhouette_score = 0.0;
    double davies_bouldin_index = 0.0;
    double calinski_harabasz_index = 0.0;
    double dun_index = 0.0;
    double c_index = 0.0;
    double gamma_index = 0.0;

    // Internal cohesion
    double within_cluster_sum_of_squares = 0.0;
    double average_intra_cluster_distance = 0.0;
    double maximum_intra_cluster_distance = 0.0;
    double diameter = 0.0;

    // Separation from other clusters
    std::vector<double> distances_to_other_clusters;
    double minimum_inter_cluster_distance = 0.0;
    double average_inter_cluster_distance = 0.0;

    // Density
    double density = 0.0;
    double sparsity = 0.0;

    // Stability
    double jaccard_stability = 0.0;
    double rand_stability = 0.0;

    // Outliers within cluster
    std::vector<size_t> outlier_indices;
    double outlier_percentage = 0.0;

    // Profile
    struct Profile {
        std::map<std::string, double> numeric_profile;
        std::map<std::string, std::string> categorical_profile;
        std::map<std::string, double> anomaly_scores;
    } profile;

    // Temporal stability (if time series)
    double temporal_stability = 0.0;
    std::vector<double> size_over_time;

    // Business interpretation
    std::string business_interpretation;
    std::vector<std::string> key_characteristics;
    std::vector<std::string> action_items;

    // Visualization
    std::vector<std::vector<double>> member_points;
    std::vector<double> convex_hull;

    std::string to_detailed_string() const;
    nlohmann::json to_json() const;
};

// Professional outlier analysis
struct ProfessionalOutlierAnalysis {
    // Detection results
    std::vector<size_t> outlier_indices;
    std::vector<double> outlier_scores;
    std::vector<std::string> outlier_types; // "global", "local", "collective", "contextual"
    std::vector<std::string> detection_methods; // "isolation_forest", "lof", "autoencoder", "mahalanobis"

    // Severity classification
    struct OutlierSeverity {
        size_t critical_count = 0;
        size_t high_count = 0;
        size_t medium_count = 0;
        size_t low_count = 0;
        std::vector<size_t> critical_indices;
        std::vector<size_t> high_indices;
        std::vector<size_t> medium_indices;
        std::vector<size_t> low_indices;
    } severity;

    // Root cause analysis
    struct RootCause {
        std::vector<std::string> contributing_features;
        std::vector<double> feature_contributions;
        std::string likely_cause; // "data_entry_error", "measurement_error", "natural_variation", "fraud", "system_error"
        double confidence = 0.0;
    } root_cause;

    // Impact analysis
    struct Impact {
        double on_mean = 0.0;
        double on_variance = 0.0;
        double on_correlation = 0.0;
        double on_model_performance = 0.0;
        double on_statistical_tests = 0.0;
    } impact;

    // Temporal patterns (if time series)
    struct TemporalPattern {
        bool is_clustered_in_time = false;
        double temporal_autocorrelation = 0.0;
        std::vector<size_t> temporal_clusters;
        std::string seasonality_pattern; // "daily", "weekly", "monthly", "yearly", "none"
    } temporal_pattern;

    // Recommended actions
    struct Recommendations {
        std::vector<size_t> should_investigate;
        std::vector<size_t> should_remove;
        std::vector<size_t> should_cap;
        std::vector<size_t> should_impute;
        std::vector<size_t> should_keep;
        std::string overall_strategy; // "remove", "winsorize", "transform", "keep"
    } recommendations;

    // Multivariate analysis
    struct MultivariateAnalysis {
        std::vector<std::string> affected_dimensions;
        std::vector<double> mahalanobis_distances;
        std::vector<double> cook_distances;
        std::vector<double> leverage_scores;
        std::vector<double> influence_scores;
    } multivariate;

    // Visualization data
    std::vector<std::vector<double>> outlier_points;
    std::vector<std::vector<double>> inlier_points;
    std::vector<double> decision_boundary;

    std::string to_detailed_string() const;
    nlohmann::json to_json() const;
};

// Professional distribution analysis
struct ProfessionalDistributionAnalysis {
    std::string column_name;

    // Distribution fitting
    struct FittedDistribution {
        std::string name; // "normal", "lognormal", "exponential", "gamma", "beta", "weibull", etc.
        std::map<std::string, double> parameters;
        double log_likelihood = 0.0;
        double aic = 0.0;
        double bic = 0.0;
        double ks_statistic = 0.0;
        double ks_p_value = 0.0;
        double ad_statistic = 0.0;
        double ad_p_value = 0.0;
        double cvm_statistic = 0.0;
        double cvm_p_value = 0.0;
        double chi2_statistic = 0.0;
        double chi2_p_value = 0.0;

        /*operator ProfessionalStatisticalCalculator::DistributionFit() const {
            ProfessionalStatisticalCalculator::DistributionFit df;
            df.name = name;
            df.parameters = parameters;
            df.log_likelihood = log_likelihood;
            df.aic = aic;
            df.bic = bic;
            df.ks_statistic = ks_statistic;
            df.ks_p_value = ks_p_value;
            return df;
        }*/
    };

    std::vector<FittedDistribution> fitted_distributions;
    FittedDistribution best_fit;

    // Goodness of fit tests
    struct GoodnessOfFit {
        double shapiro_wilk = 0.0;
        double shapiro_wilk_p = 0.0;
        double dagostino_k2 = 0.0;
        double dagostino_k2_p = 0.0;
        double jarque_bera = 0.0;
        double jarque_bera_p = 0.0;
        bool passes_normality = false;
    } goodness_of_fit;

    // Moments
    struct Moments {
        double mean = 0.0;
        double variance = 0.0;
        double skewness = 0.0;
        double kurtosis = 0.0;
        double excess_kurtosis = 0.0;
        std::vector<double> central_moments; // Up to 10th moment
        std::vector<double> standardized_moments;
        std::vector<double> cumulants;
    } moments;

    // L-moments (more robust)
    struct LMoments {
        double l1 = 0.0;
        double l2 = 0.0;
        double l3 = 0.0;
        double l4 = 0.0;
        double l_cv = 0.0;  // L-coefficient of variation
        double l_skew = 0.0; // L-skewness
        double l_kurt = 0.0; // L-kurtosis
    } l_moments;

    // Tail behavior
    struct TailAnalysis {
        double tail_index = 0.0;
        double hill_estimator = 0.0;
        double extreme_value_index = 0.0;
        bool heavy_tailed = false;
        bool light_tailed = false;
        double expected_shortfall_95 = 0.0;
        double expected_shortfall_99 = 0.0;
        double value_at_risk_95 = 0.0;
        double value_at_risk_99 = 0.0;
    } tail_analysis;

    // Multimodality
    struct Modality {
        size_t number_of_modes = 0;
        std::vector<double> mode_locations;
        std::vector<double> mode_heights;
        double dip_statistic = 0.0;
        double dip_p_value = 0.0;
        bool is_multimodal = false;
        bool is_bimodal = false;
        bool is_unimodal = false;
    } modality;

    // Transformation suggestions
    struct Transformations {
        bool log_recommended = false;
        bool box_cox_recommended = false;
        bool yeo_johnson_recommended = false;
        bool sqrt_recommended = false;
        bool reciprocal_recommended = false;
        bool power_recommended = false;
        double best_lambda = 0.0; // For Box-Cox
        std::string best_transformation;
    } transformations;

    // Distribution properties
    bool is_symmetric = false;
    bool is_uniform = false;
    bool is_exponential = false;
    bool is_power_law = false;
    bool is_poisson = false;
    bool is_binomial = false;

    // Visualization data
    std::vector<double> empirical_cdf;
    std::vector<double> theoretical_cdf;
    std::vector<double> qq_plot_data;
    std::vector<double> pp_plot_data;

    std::string to_detailed_string() const;
    nlohmann::json to_json() const;
};

// Professional time series analysis
struct ProfessionalTimeSeriesAnalysis {
    std::string timestamp_column;
    std::string value_column;

    // Stationarity tests
    struct Stationarity {
        double adf_statistic = 0.0;
        double adf_p_value = 0.0;
        double kpss_statistic = 0.0;
        double kpss_p_value = 0.0;
        double pp_statistic = 0.0;
        double pp_p_value = 0.0;
        bool is_stationary_adf = false;
        bool is_stationary_kpss = false;
        bool is_stationary_pp = false;
        int differencing_order = 0;
    } stationarity;

    // Decomposition
    struct Decomposition {
        std::vector<double> trend;
        std::vector<double> seasonal;
        std::vector<double> residual;
        std::vector<double> seasonal_indices;
        double trend_strength = 0.0;
        double seasonality_strength = 0.0;
        double residual_strength = 0.0;
        std::string seasonal_period; // "daily", "weekly", "monthly", "yearly"
        size_t seasonal_period_length = 0;
        bool has_seasonality;
    } decomposition;

    // Autocorrelation
    struct Autocorrelation {
        std::vector<double> acf_values;
        std::vector<double> acf_confidence;
        std::vector<double> pacf_values;
        std::vector<double> pacf_confidence;
        double autocorrelation_time = 0.0;
        double hurst_exponent = 0.0;
        double lyapunov_exponent = 0.0;
        bool is_long_memory = false;
        bool is_short_memory = false;
    } autocorrelation;

    // Spectral analysis
    struct Spectral {
        std::vector<double> periodogram;
        std::vector<double> spectral_density;
        std::vector<double> dominant_frequencies;
        std::vector<double> power_at_frequencies;
        double total_power = 0.0;
        double peak_frequency = 0.0;
        double peak_power = 0.0;
    } spectral;

    // Volatility
    struct Volatility {
        double unconditional_variance = 0.0;
        double conditional_variance = 0.0;
        double arch_effect = 0.0;
        double garch_effect = 0.0;
        bool has_volatility_clustering = false;
        std::vector<double> volatility_clusters;
    } volatility;

    // Forecasting properties
    struct Forecastability {
        double entropy_rate = 0.0;
        double predictability = 0.0;
        double sample_entropy = 0.0;
        double approximate_entropy = 0.0;
        bool is_chaotic = false;
        bool is_predictable = false;
        double forecast_horizon = 0.0;
    } forecastability;

    // Seasonality
    struct Seasonality {
        bool has_seasonality = false;
        double dominant_period;
        std::vector<size_t> seasonal_periods;
        std::vector<double> seasonal_strengths;
        std::string seasonal_type; // "additive", "multiplicative"
        std::vector<double> seasonal_indices;
    } seasonality;

    // Anomalies in time series
    struct TimeSeriesAnomalies {
        std::vector<size_t> point_anomaly_indices;
        std::vector<std::pair<size_t, size_t>> collective_anomaly_ranges;
        std::vector<size_t> trend_change_points;
        std::vector<size_t> seasonal_change_points;
    } anomalies;

    // Model suggestions
    struct ModelSuggestions {
        bool arima_recommended = false;
        bool ets_recommended = false;
        bool prophet_recommended = false;
        bool lstm_recommended = false;
        bool tbats_recommended = false;
        std::string best_model_type;
        std::map<std::string, std::string> model_parameters;
    } model_suggestions;

    std::string to_detailed_string() const;
    nlohmann::json to_json() const;
};

// Professional data quality report
struct ProfessionalDataQualityReport {
    // Scorecard
    struct Scores {
        double completeness = 0.0;
        double consistency = 0.0;
        double accuracy = 0.0;
        double timeliness = 0.0;
        double validity = 0.0;
        double uniqueness = 0.0;
        double integrity = 0.0;
        double freshness = 0.0;
        double lineage = 0.0;
        double overall = 0.0;

        struct Weighted {
            double completeness = 0.0;
            double consistency = 0.0;
            double accuracy = 0.0;
            double timeliness = 0.0;
            double validity = 0.0;
            double uniqueness = 0.0;
            double overall = 0.0;
        } weighted;
    } scores;

    // Dimensions
    struct Completeness {
        size_t total_cells = 0;
        size_t missing_cells = 0;
        double missing_percentage = 0.0;
        std::vector<std::string> columns_with_missing;
        std::map<std::string, double> column_missing_percentages;
        std::string missing_pattern; // "MCAR", "MAR", "MNAR"
        std::vector<size_t> rows_with_missing;
        double row_completeness = 0.0;
    } completeness;

    struct Consistency {
        std::vector<std::string> type_inconsistencies;
        std::vector<std::string> range_violations;
        std::vector<std::string> format_violations;
        std::vector<std::string> constraint_violations;
        std::vector<std::string> referential_integrity_violations;
        size_t total_violations = 0;
    } consistency;

    struct Accuracy {
        double ground_truth_accuracy = 0.0;
        double cross_source_agreement = 0.0;
        double self_consistency = 0.0;
        std::vector<std::string> accuracy_issues;
        std::map<std::string, double> column_accuracy_scores;
    } accuracy;

    struct Uniqueness {
        size_t total_rows = 0;
        size_t duplicate_rows = 0;
        double duplicate_percentage = 0.0;
        std::vector<std::vector<size_t>> duplicate_groups;
        std::vector<std::string> candidate_keys;
        bool has_primary_key = false;
    } uniqueness;

    struct Validity {
        std::vector<std::string> business_rule_violations;
        std::vector<std::string> domain_violations;
        std::vector<std::string> pattern_violations;
        size_t total_invalid_values = 0;
        double validity_rate = 0.0;
    } validity;

    struct Timeliness {
        double data_freshness = 0.0; // hours since last update
        double update_frequency = 0.0; // updates per day
        double latency = 0.0; // processing delay
        bool meets_sla = false;
    } timeliness;

    // Issues and recommendations
    struct Issues {
        std::vector<std::string> critical;
        std::vector<std::string> high;
        std::vector<std::string> medium;
        std::vector<std::string> low;
        size_t total_issues = 0;
    } issues;

    struct Recommendations {
        std::vector<std::string> immediate;
        std::vector<std::string> short_term;
        std::vector<std::string> long_term;
        std::vector<std::string> monitoring;
    } recommendations;

    // Metadata
    struct Metadata {
        std::string analysis_timestamp;
        size_t row_count = 0;
        size_t column_count = 0;
        size_t total_cells = 0;
        std::string data_source;
        std::string schema_version;
    } metadata;

    std::string to_detailed_string() const;
    nlohmann::json to_json() const;
    std::string to_markdown_report() const;
    std::string to_html_report() const;
};

// Professional comprehensive analysis report
struct ProfessionalComprehensiveAnalysisReport {
    // Metadata
    std::string analysis_id;
    std::string table_name;
    std::string analysis_timestamp;
    std::chrono::milliseconds analysis_duration;
    size_t row_count = 0;
    size_t column_count = 0;

    // Detailed analyses
    std::map<std::string, ProfessionalColumnAnalysis> column_analyses;
    std::vector<ProfessionalCorrelationAnalysis> correlations;
    std::vector<ProfessionalFeatureImportance> feature_importance;
    std::vector<ProfessionalClusterAnalysis> clusters;
    std::vector<ProfessionalOutlierAnalysis> outliers;
    std::vector<ProfessionalDistributionAnalysis> distributions;
    std::vector<ProfessionalTimeSeriesAnalysis> time_series_analyses;
    ProfessionalDataQualityReport data_quality;

    // Advanced analyses
    struct AdvancedAnalyses {
        // Multivariate analysis
        std::vector<double> principal_components;
        std::vector<double> explained_variance_ratio;
        double total_variance_explained = 0.0;

        // Manifold learning
        std::vector<std::vector<double>> tsne_embedding;
        std::vector<std::vector<double>> umap_embedding;

        // Association rules
        struct AssociationRule {
            std::vector<std::string> antecedent;
            std::vector<std::string> consequent;
            double support = 0.0;
            double confidence = 0.0;
            double lift = 0.0;
            double conviction = 0.0;
        };
        std::vector<AssociationRule> association_rules;

        // Causal inference
        struct CausalRelationship {
            std::string cause;
            std::string effect;
            double treatment_effect = 0.0;
            double confidence_interval_lower = 0.0;
            double confidence_interval_upper = 0.0;
            double p_value = 0.0;
        };
        std::vector<CausalRelationship> causal_relationships;

        // Network analysis
        struct Network {
            std::vector<std::string> nodes;
            std::vector<std::pair<std::string, std::string>> edges;
            std::vector<double> edge_weights;
            std::map<std::string, double> centrality_scores;
            std::vector<std::vector<std::string>> communities;
        } network;
    } advanced;

    // Insights (automatically generated)
    struct Insights {
        std::vector<std::string> data_quality;
        std::vector<std::string> statistical;
        std::vector<std::string> business;
        std::vector<std::string> predictive;
        std::vector<std::string> anomaly;
        std::vector<std::string> optimization;
    } insights;

    // Recommendations (prioritized)
    struct Recommendations {
        struct Recommendation {
            std::string title;
            std::string description;
            std::string category; // "data_quality", "feature_engineering", "modeling", "monitoring"
            std::string priority; // "critical", "high", "medium", "low"
            double expected_impact = 0.0;
            double implementation_effort = 0.0;
            std::vector<std::string> steps;
            std::vector<std::string> dependencies;
        };
        std::vector<Recommendation> all;
        std::vector<Recommendation> critical;
        std::vector<Recommendation> high;
        std::vector<Recommendation> medium;
        std::vector<Recommendation> low;
    } recommendations;

    // Performance metrics
    struct Performance {
        double memory_usage_mb = 0.0;
        double cpu_usage_percent = 0.0;
        std::chrono::milliseconds processing_time;
        std::map<std::string, std::chrono::milliseconds> component_times;
        bool within_sla = true;
    } performance;

    // Export methods
    nlohmann::json to_json(bool detailed = true) const;
    std::string to_markdown_report(bool detailed = true) const;
    std::string to_html_report(bool detailed = true) const;
    std::string to_pdf_report(bool detailed = true) const;
    std::string to_excel_report(bool detailed = true) const;

    // Serialization
    bool save_to_file(const std::string& filename, const std::string& format = "json") const;
    bool load_from_file(const std::string& filename);
};

// Statistical Calculator with advanced methods
class ProfessionalStatisticalCalculator {
public:
    // Basic statistics
    static double normal_quantile(double p);
    static double calculate_mean(const std::vector<double>& values);
    static double calculate_trimmed_mean(const std::vector<double>& values, double trim_proportion);
    static double calculate_winsorized_mean(const std::vector<double>& values, double winsorize_proportion);
    static double calculate_median(const std::vector<double>& values);
    static double calculate_quantile(const std::vector<double>& values, double q);
    static std::vector<double> calculate_quantiles(const std::vector<double>& values, const std::vector<double>& qs);
    static double calculate_mode(const std::vector<double>& values);
    static std::vector<double> calculate_modes(const std::vector<double>& values);

    // Dispersion
    static double calculate_variance(const std::vector<double>& values, bool population = false);
    static double calculate_std_dev(const std::vector<double>& values, bool population = false);
    static double calculate_mad(const std::vector<double>& values); // Median Absolute Deviation
    static double calculate_iqr(const std::vector<double>& values);
    static double calculate_range(const std::vector<double>& values);
    static double calculate_coefficient_of_variation(const std::vector<double>& values);

    // Shape
    static double calculate_skewness(const std::vector<double>& values, bool fisher = true);
    static double calculate_kurtosis(const std::vector<double>& values, bool fisher = true);
    static std::vector<double> calculate_moments(const std::vector<double>& values, size_t max_order = 4);
    static std::vector<double> calculate_central_moments(const std::vector<double>& values, size_t max_order = 4);
    static std::vector<double> calculate_standardized_moments(const std::vector<double>& values, size_t max_order = 4);
    static std::vector<double> calculate_l_moments(const std::vector<double>& values, size_t max_order = 4);

    // Normality tests
    static double shapiro_wilk(const std::vector<double>& values, double& p_value);
    static double jarque_bera(const std::vector<double>& values, double& p_value);
    static double anderson_darling(const std::vector<double>& values, const std::string& distribution = "normal");
    static double kolmogorov_smirnov(const std::vector<double>& values, const std::string& distribution = "normal");
    static double cramer_von_mises(const std::vector<double>& values, const std::string& distribution = "normal");

    // Correlation measures
    static double pearson_correlation(const std::vector<double>& x, const std::vector<double>& y,
                                     double& p_value, double& confidence_lower, double& confidence_upper);
    static double spearman_correlation(const std::vector<double>& x, const std::vector<double>& y, double& p_value);
    static double kendall_tau(const std::vector<double>& x, const std::vector<double>& y, double& p_value);
    static double distance_correlation(const std::vector<double>& x, const std::vector<double>& y);
    static double maximal_information_coefficient(const std::vector<double>& x, const std::vector<double>& y);
    static double hoeffding_d(const std::vector<double>& x, const std::vector<double>& y);

    // Information theory
    static double entropy(const std::vector<double>& values, size_t bins = 10);
    static double mutual_information(const std::vector<double>& x, const std::vector<double>& y, size_t bins = 10);
    static double conditional_entropy(const std::vector<double>& x, const std::vector<double>& y, size_t bins = 10);
    static double kl_divergence(const std::vector<double>& p, const std::vector<double>& q);
    static double js_divergence(const std::vector<double>& p, const std::vector<double>& q);

    // Distance measures
    static double euclidean_distance(const std::vector<double>& a, const std::vector<double>& b);
    static double manhattan_distance(const std::vector<double>& a, const std::vector<double>& b);
    static double chebyshev_distance(const std::vector<double>& a, const std::vector<double>& b);
    static double minkowski_distance(const std::vector<double>& a, const std::vector<double>& b, double p);
    static double mahalanobis_distance(const std::vector<double>& x, const std::vector<double>& mean,
                                      const std::vector<std::vector<double>>& covariance);
    static double cosine_similarity(const std::vector<double>& a, const std::vector<double>& b);
    static double jaccard_similarity(const std::vector<double>& a, const std::vector<double>& b);

    // Distribution fitting
    struct DistributionFit {
        std::string name;
        std::map<std::string, double> parameters;
        double log_likelihood;
        double aic;
        double bic;
        double ks_statistic;
        double ks_p_value;
        double ad_statistic;
        double ad_p_value;
        double cvm_statistic;
        double cvm_p_value;
        double chi2_statistic;
        double chi2_p_value;
    };

    static DistributionFit fit_distribution(const std::vector<double>& values, const std::string& distribution);
    static std::vector<DistributionFit> fit_multiple_distributions(const std::vector<double>& values);
    static DistributionFit find_best_fit(const std::vector<double>& values);

    // Time series analysis
    static std::vector<double> calculate_autocorrelation(const std::vector<double>& series, size_t max_lag);
    static std::vector<double> calculate_partial_autocorrelation(const std::vector<double>& series, size_t max_lag);
    static double calculate_hurst_exponent(const std::vector<double>& series);
    static double calculate_lyapunov_exponent(const std::vector<double>& series);
    static bool adf_test(const std::vector<double>& series, double& statistic, double& p_value);
    static bool kpss_test(const std::vector<double>& series, double& statistic, double& p_value);

    // Outlier detection
    static std::vector<size_t> detect_outliers_iqr(const std::vector<double>& values, double multiplier = 1.5);
    static std::vector<size_t> detect_outliers_zscore(const std::vector<double>& values, double threshold = 3.0);
    static std::vector<size_t> detect_outliers_mad(const std::vector<double>& values, double threshold = 3.5);
    static std::vector<size_t> detect_outliers_grubbs(const std::vector<double>& values, double alpha = 0.05);
    static std::vector<size_t> detect_outliers_dixon(const std::vector<double>& values, double alpha = 0.05);

    // Statistical tests
    static double t_test(const std::vector<double>& sample1, const std::vector<double>& sample2,
                        bool paired = false, bool equal_var = true);
    static double f_test(const std::vector<double>& sample1, const std::vector<double>& sample2);
    static double chi2_test(const std::vector<std::vector<double>>& contingency_table);
    static double anova(const std::vector<std::vector<double>>& groups);
    static double mann_whitney_u(const std::vector<double>& sample1, const std::vector<double>& sample2);
    static double kruskal_wallis(const std::vector<std::vector<double>>& groups);

private:
    static double erf_inv(double x);
    static double normal_cdf(double x);
    //static double normal_quantile(double p);
    static double gamma_function(double x);
    static double beta_function(double a, double b);
    static double incomplete_gamma(double a, double x);
    static double incomplete_beta(double a, double b, double x);
    static double incomplete_beta_cf(double a, double b, double x);
    static double digamma(double x);
    static double trigamma(double x);

    // Helper functions for distribution fitting
    static double normal_pdf(double x, double mu, double sigma);
    static double lognormal_pdf(double x, double mu, double sigma);
    static double exponential_pdf(double x, double lambda);
    static double gamma_pdf(double x, double alpha, double beta);
    static double beta_pdf(double x, double alpha, double beta);
    static double weibull_pdf(double x, double lambda, double k);

    // Numerical optimization for MLE
    static std::map<std::string, double> mle_normal(const std::vector<double>& values);
    static std::map<std::string, double> mle_lognormal(const std::vector<double>& values);
    static std::map<std::string, double> mle_exponential(const std::vector<double>& values);
    static std::map<std::string, double> mle_gamma(const std::vector<double>& values);
    static std::map<std::string, double> mle_beta(const std::vector<double>& values);
    static std::map<std::string, double> mle_weibull(const std::vector<double>& values);
};

// Professional Data Analyzer
class ProfessionalDataAnalyzer {
public:
    // Configuration
    struct Config {
        size_t max_rows_for_detailed_analysis;
        size_t sample_size_for_large_datasets;
        size_t max_correlation_pairs;
        size_t max_clusters;
        size_t max_features_for_importance;
        bool enable_parallel_processing;
        size_t num_threads;
        bool enable_gpu_acceleration;
        bool enable_caching;
        size_t cache_size_mb;
        std::string output_format; // "json", "markdown", "html", "csv"
        bool generate_visualizations;
        bool save_intermediate_results;
        std::string temp_directory;

        // Quality thresholds
        double data_quality_warning_threshold;
        double data_quality_critical_threshold;
        double correlation_significance_threshold;
        double outlier_contamination_threshold;

        // Performance tuning
        size_t chunk_size_for_processing;
        size_t max_memory_usage_mb;
        bool enable_progress_reporting;
        size_t progress_report_interval;

	// Constructor with default values
        Config() :
            max_rows_for_detailed_analysis(1000000),
            sample_size_for_large_datasets(100000),
            max_correlation_pairs(1000),
            max_clusters(20),
            max_features_for_importance(100),
            enable_parallel_processing(true),
            num_threads(std::thread::hardware_concurrency()),
            enable_gpu_acceleration(false),
            enable_caching(true),
            cache_size_mb(1024),
            output_format("json"),
            generate_visualizations(true),
            save_intermediate_results(false),
            temp_directory("/tmp/esql_analysis"),
            data_quality_warning_threshold(0.7),
            data_quality_critical_threshold(0.5),
            correlation_significance_threshold(0.05),
            outlier_contamination_threshold(0.05),
            chunk_size_for_processing(10000),
            max_memory_usage_mb(4096),
            enable_progress_reporting(true),
            progress_report_interval(1000)
        {}
    };

    ProfessionalDataAnalyzer(const Config& config = Config());
    ~ProfessionalDataAnalyzer();

    // Main analysis method
    ProfessionalComprehensiveAnalysisReport analyze_data(
        const std::vector<std::unordered_map<std::string, Datum>>& data,
        const std::string& target_column = "",
        const std::vector<std::string>& feature_columns = {},
        const std::string& analysis_type = "COMPREHENSIVE",
        const std::map<std::string, std::string>& options = {});

    // Streaming analysis for large datasets
    ProfessionalComprehensiveAnalysisReport analyze_data_streaming(
        const std::string& db_name,
        const std::string& table_name,
        const std::string& target_column = "",
        const std::vector<std::string>& feature_columns = {},
        const std::string& analysis_type = "COMPREHENSIVE",
        const std::map<std::string, std::string>& options = {});

    // Incremental analysis (update existing analysis with new data)
    ProfessionalComprehensiveAnalysisReport analyze_data_incremental(
        const ProfessionalComprehensiveAnalysisReport& previous_report,
        const std::vector<std::unordered_map<std::string, Datum>>& new_data,
        const std::map<std::string, std::string>& options = {});

    // Comparative analysis (compare two datasets)
    ProfessionalComprehensiveAnalysisReport analyze_data_comparative(
        const std::vector<std::unordered_map<std::string, Datum>>& data1,
        const std::vector<std::unordered_map<std::string, Datum>>& data2,
        const std::string& analysis_type = "COMPARATIVE",
        const std::map<std::string, std::string>& options = {});

    // Time series analysis
    ProfessionalComprehensiveAnalysisReport analyze_time_series(
        const std::vector<std::unordered_map<std::string, Datum>>& data,
        const std::string& timestamp_column,
        const std::string& value_column,
        const std::map<std::string, std::string>& options = {});

    // Get analysis status
    struct AnalysisStatus {
        std::string status; // "not_started", "running", "completed", "failed"
        double progress = 0.0; // 0 to 1
        std::chrono::milliseconds elapsed_time;
        std::chrono::milliseconds estimated_remaining_time;
	std::chrono::system_clock::time_point start_time;
        std::string current_operation;
        size_t rows_processed = 0;
        size_t total_rows = 0;
    };

    std::pair<std::vector<std::vector<double>>, std::vector<size_t>> perform_kmeans(const std::vector<std::vector<double>>& data,size_t k,size_t max_iterations);

    size_t find_elbow_point(const std::vector<double>& wcss_values);

    void generate_cluster_interpretation(ProfessionalClusterAnalysis& cluster,const std::vector<std::string>& feature_columns);

    std::vector<double> calculate_mahalanobis_distances(const std::vector<std::vector<double>>& data);

    std::vector<double> calculate_isolation_scores(const std::vector<std::vector<double>>& data);

    std::vector<double> calculate_lof_scores(const std::vector<std::vector<double>>& data);

    double normalize_score(double score,const std::vector<double>& all_scores);

    double calculate_mahalanobis_threshold(const std::vector<double>& distances);

    double calculate_lof_threshold(const std::vector<double>& scores);

    std::string determine_likely_cause(const ProfessionalOutlierAnalysis& analysis,const std::vector<std::vector<double>>& data,const std::vector<std::string>& feature_columns);

    double calculate_wcss(const std::vector<std::vector<double>>& data,const std::vector<std::vector<double>>& centroids,const std::vector<size_t>& labels);

    double calculate_variance_impact(const std::vector<std::vector<double>>& data_with_outliers,const std::vector<std::vector<double>>& data_without_outliers);

    void generate_outlier_recommendations(ProfessionalOutlierAnalysis& analysis,size_t total_points);

    std::vector<double> calculate_cooks_distances(const std::vector<std::vector<double>>& data);

    std::vector<double> calculate_leverage_scores(const std::vector<std::vector<double>>& data);

    void perform_tail_analysis(ProfessionalDistributionAnalysis::TailAnalysis& tail_analysis,const std::vector<double>& values);

    void perform_modality_analysis(ProfessionalDistributionAnalysis::Modality& modality,const std::vector<double>& values);

    void determine_distribution_properties(ProfessionalDistributionAnalysis& analysis,const std::vector<double>& values);

    void suggest_transformations(ProfessionalDistributionAnalysis::Transformations& transformations,const std::vector<double>& values);

    void generate_empirical_cdf(std::vector<double>& ecdf,const std::vector<double>& values);

    void generate_theoretical_cdf(std::vector<double>& tcdf,const std::vector<double>& values,const ProfessionalStatisticalCalculator::DistributionFit& distribution);

    void generate_qq_plot_data(std::vector<double>& qq_data,const std::vector<double>& values);

    void perform_time_series_decomposition(ProfessionalTimeSeriesAnalysis::Decomposition& decomposition,const std::vector<double>& series,const std::vector<time_t>& timestamps);

    void perform_volatility_analysis(ProfessionalTimeSeriesAnalysis::Volatility& volatility,const std::vector<double>& series);

    void perform_forecastability_analysis(ProfessionalTimeSeriesAnalysis::Forecastability& forecastability,const std::vector<double>& series);

    void detect_seasonality(ProfessionalTimeSeriesAnalysis::Seasonality& seasonality,const std::vector<double>& series,const std::vector<time_t>& timestamps);

    void detect_time_series_anomalies(ProfessionalTimeSeriesAnalysis::TimeSeriesAnomalies& anomalies,const std::vector<double>& series);

    void suggest_time_series_models(ProfessionalTimeSeriesAnalysis::ModelSuggestions& suggestions,const ProfessionalTimeSeriesAnalysis& analysis);

    double calculate_seasonal_strength(const std::vector<double>& series, size_t period);

    double calculate_autocorrelation_time(const std::vector<double>& acf_values);

    // Helper methods for data quality assessment
    void analyze_completeness(ProfessionalDataQualityReport::Completeness& completeness,const std::vector<std::unordered_map<std::string, Datum>>& data,const std::vector<std::string>& columns);

    void analyze_consistency(ProfessionalDataQualityReport::Consistency& consistency,const std::vector<std::unordered_map<std::string, Datum>>& data,const std::vector<std::string>& columns);

    void analyze_accuracy(ProfessionalDataQualityReport::Accuracy& accuracy,const std::vector<std::unordered_map<std::string, Datum>>& data,const std::vector<std::string>& columns);

    void analyze_uniqueness(ProfessionalDataQualityReport::Uniqueness& uniqueness,const std::vector<std::unordered_map<std::string, Datum>>& data);

    void analyze_validity(ProfessionalDataQualityReport::Validity& validity,const std::vector<std::unordered_map<std::string, Datum>>& data,const std::vector<std::string>& columns);

    void analyze_timeliness(ProfessionalDataQualityReport::Timeliness& timeliness,const std::vector<std::unordered_map<std::string, Datum>>& data);

    void calculate_data_quality_scores(ProfessionalDataQualityReport& report);

    void identify_data_quality_issues(ProfessionalDataQualityReport::Issues& issues,const ProfessionalDataQualityReport& report);

    void generate_data_quality_recommendations(ProfessionalDataQualityReport::Recommendations& recommendations,const ProfessionalDataQualityReport& report);

    void generate_comprehensive_recommendations(ProfessionalComprehensiveAnalysisReport::Recommendations& recommendations,const ProfessionalComprehensiveAnalysisReport& report);

    bool detect_seasonal_pattern(const std::vector<double>& series,const std::vector<time_t>& timestamps);

    size_t estimate_seasonal_period(const std::vector<double>& series);

    // Utility methods
    double calculate_variance(const std::vector<double>& values);
    double calculate_dip_statistic(const std::vector<size_t>& histogram);
    double estimate_box_cox_lambda(const std::vector<double>& values);
    double calculate_sample_entropy(const std::vector<double>& series, int m, double r);
    double calculate_approximate_entropy(const std::vector<double>& series, int m, double r);
    double calculate_distance(const std::vector<double>& a, const std::vector<double>& b);

    std::string generate_analysis_id() const;

    void calculate_cluster_quality(ProfessionalClusterAnalysis& cluster,const std::vector<std::vector<double>>& data,const std::vector<size_t>& cluster_indices,const std::vector<std::vector<double>>& centroids);

    void perform_spectral_analysis(ProfessionalTimeSeriesAnalysis::Spectral& spectral,const std::vector<double>& series);

    AnalysisStatus get_analysis_status(const std::string& analysis_id) const;

    // Cancel running analysis
    bool cancel_analysis(const std::string& analysis_id);

    // Export analysis results
    bool export_analysis(const std::string& analysis_id,
                        const std::string& format,
                        const std::string& output_path);

    // Get analysis history
    std::vector<std::string> get_analysis_history() const;

    // Clear analysis cache
    void clear_cache();

    // Performance metrics
    struct PerformanceMetrics {
        size_t total_analyses = 0;
        size_t successful_analyses = 0;
        size_t failed_analyses = 0;
        std::chrono::milliseconds total_processing_time;
        double average_processing_time_per_row = 0.0;
        size_t peak_memory_usage_mb = 0;
        double cache_hit_rate = 0.0;
    };

    PerformanceMetrics get_performance_metrics() const;

private:
    Config config_;

    // Core analysis components
    ProfessionalColumnAnalysis analyze_column_professional(
        const std::string& column_name,
        const std::vector<Datum>& values,
        const std::map<std::string, std::string>& options = {});

    ProfessionalCorrelationAnalysis analyze_correlation_professional(
        const std::string& col1,
        const std::string& col2,
        const std::vector<Datum>& values1,
        const std::vector<Datum>& values2,
        const std::map<std::string, std::string>& options = {});

    ProfessionalFeatureImportance analyze_feature_importance_professional(
        const std::string& feature_name,
        const std::vector<Datum>& feature_values,
        const std::vector<Datum>& target_values,
        const std::map<std::string, std::string>& options = {});

    std::vector<ProfessionalClusterAnalysis> perform_clustering_professional(
        const std::vector<std::unordered_map<std::string, Datum>>& data,
        const std::vector<std::string>& feature_columns,
        const std::map<std::string, std::string>& options = {});

    ProfessionalOutlierAnalysis detect_outliers_professional(
        const std::vector<std::unordered_map<std::string, Datum>>& data,
        const std::vector<std::string>& feature_columns,
        const std::map<std::string, std::string>& options = {});

    ProfessionalDistributionAnalysis analyze_distribution_professional(
        const std::string& column_name,
        const std::vector<Datum>& values,
        const std::map<std::string, std::string>& options = {});

    ProfessionalTimeSeriesAnalysis analyze_time_series_professional(
        const std::string& timestamp_column,
        const std::string& value_column,
        const std::vector<Datum>& timestamps,
        const std::vector<Datum>& values,
        const std::map<std::string, std::string>& options = {});

    ProfessionalDataQualityReport assess_data_quality_professional(
        const std::vector<std::unordered_map<std::string, Datum>>& data,
        const std::map<std::string, std::string>& options = {});

    // Helper methods
    std::string detect_data_type_professional(const std::vector<Datum>& values, size_t distinct_count) const;
    bool is_likely_categorical_professional(const std::vector<Datum>& values, size_t distinct_count) const;
    std::vector<double> extract_numeric_values_professional(const std::vector<Datum>& values) const;
    std::vector<std::string> extract_string_values_professional(const std::vector<Datum>& values) const;

    // Parallel processing helpers
    template<typename T, typename Func>
    std::vector<T> parallel_map(const std::vector<std::vector<Datum>>& data_chunks, Func func);

    size_t estimate_memory_usage(size_t rows, size_t cols) const;
    std::string generate_cache_key(
        const std::vector<std::unordered_map<std::string, Datum>>& data,
        const std::string& target_column,
        const std::vector<std::string>& feature_columns,
        const std::string& analysis_type,
        const std::map<std::string, std::string>& options) const;
    size_t estimate_report_size(const ProfessionalComprehensiveAnalysisReport& report) const;

    // Caching
    struct AnalysisCache {
        std::string key;
        ProfessionalComprehensiveAnalysisReport report;
        std::chrono::system_clock::time_point timestamp;
        size_t size_bytes;
    };

    std::map<std::string, AnalysisCache> cache_;
    size_t cache_size_bytes_ = 0;
    mutable std::mutex cache_mutex_;

    // Progress tracking
    struct AnalysisProgress {
        std::string analysis_id;
        AnalysisStatus status;
        std::future<ProfessionalComprehensiveAnalysisReport> future;
        std::chrono::system_clock::time_point start_time;

	AnalysisProgress() = default;
        AnalysisProgress(const AnalysisProgress&) = delete;
        AnalysisProgress& operator=(const AnalysisProgress&) = delete;
        AnalysisProgress(AnalysisProgress&&) = default;
        AnalysisProgress& operator=(AnalysisProgress&&) = default;

    };

    std::map<std::string, AnalysisProgress> active_analyses_;
    mutable std::mutex analysis_mutex_;

    // Performance tracking
    PerformanceMetrics performance_metrics_;
    mutable std::mutex metrics_mutex_;

    // Generate insights and recommendations
    void generate_insights_and_recommendations_professional(
        ProfessionalComprehensiveAnalysisReport& report,
        const std::string& target_column,
        const std::map<std::string, std::string>& options);

    // Validate input
    bool validate_input_data(const std::vector<std::unordered_map<std::string, Datum>>& data) const;

    // Resource management
    bool check_resource_availability(size_t estimated_memory_mb, size_t estimated_cpu_cores) const;
    void cleanup_resources();

    // Error handling
    void handle_analysis_error(const std::string& analysis_id, const std::exception& e);
    bool is_numeric_datum(const Datum& d) const;
    double get_double_from_datum(const Datum& d) const;
    std::string get_string_from_datum(const Datum& d) const;

    // Logging
    void log_analysis_progress(const std::string& analysis_id, const std::string& message, double progress);
    void log_performance_metric(const std::string& metric, double value) const;
};

} // namespace analysis
} // namespace esql

#endif // DATA_ANALYSIS_PRO_H
