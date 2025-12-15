// ============================================
// data_analysis.h - New header for analysis components
// ============================================
#ifndef DATA_ANALYSIS_H
#define DATA_ANALYSIS_H

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

namespace esql {
namespace analysis {

// Statistical summary for a column
struct ColumnAnalysis {
    std::string name;
    std::string detected_type;
    size_t total_count = 0;
    size_t null_count = 0;
    size_t distinct_count = 0;
    double min_value = 0.0;
    double max_value = 0.0;
    double mean = 0.0;
    double median = 0.0;
    double mode = 0.0;
    double std_dev = 0.0;
    double variance = 0.0;
    double skewness = 0.0;
    double kurtosis = 0.0;
    double q1 = 0.0;  // First quartile
    double q3 = 0.0;  // Third quartile
    double iqr = 0.0; // Interquartile range
    std::vector<double> histogram_bins;
    std::vector<size_t> histogram_counts;
    std::map<std::string, size_t> value_frequencies;
    std::vector<std::string> top_categories;
    double missing_percentage = 0.0;
    bool is_categorical = false;
    bool has_outliers = false;
    std::vector<double> outliers;
    double entropy = 0.0;  // For categorical columns

    std::string to_string() const;
    nlohmann::json to_json() const;
};

// Correlation analysis between columns
struct CorrelationAnalysis {
    std::string column1;
    std::string column2;
    double pearson_correlation = 0.0;
    double spearman_correlation = 0.0;
    double kendall_tau = 0.0;
    double covariance = 0.0;
    double p_value = 0.0;
    bool is_significant = false;
    std::string relationship_type; // "strong_positive", "weak_negative", etc.

    std::string to_string() const;
};

// Feature importance analysis
struct FeatureImportanceAnalysis {
    std::string feature_name;
    double importance_score = 0.0;
    double permutation_importance = 0.0;
    double shap_importance = 0.0;
    double information_gain = 0.0;
    double mutual_information = 0.0;
    double chi_square = 0.0;
    double anova_f_value = 0.0;
    std::vector<std::pair<double,double>> partial_dependence;

    std::string to_string() const;
};

// Clustering analysis
struct ClusterAnalysis {
    size_t cluster_id;
    size_t size;
    std::vector<double> centroid;
    std::vector<std::string> top_features;
    double silhouette_score = 0.0;
    double davies_bouldin_index = 0.0;
    double within_cluster_variance = 0.0;
    double between_cluster_variance = 0.0;
    std::vector<size_t> member_indices;

    std::string to_string() const;
};

// Anomaly/outlier detection
struct OutlierAnalysis {
    std::vector<size_t> outlier_indices;
    std::vector<double> outlier_scores;
    std::vector<std::string> outlier_reasons;
    std::string detection_method; // "isolation_forest", "lof", "elliptic_envelope"
    double contamination_rate = 0.0;
    std::vector<std::string> affected_columns;

    std::string to_string() const;
};

// Distribution analysis
struct DistributionAnalysis {
    std::string column_name;
    std::string distribution_type; // "normal", "uniform", "exponential", etc.
    std::vector<double> ks_test_results;
    std::vector<double> anderson_test_results;
    double shapiro_wilk_pvalue = 0.0;
    bool passes_normality_test = false;
    std::map<std::string, double> distribution_parameters;

    std::string to_string() const;
};

// Time series analysis (if applicable)
struct TimeSeriesAnalysis {
    std::string timestamp_column;
    std::string value_column;
    double autocorrelation = 0.0;
    double partial_autocorrelation = 0.0;
    double stationarity_pvalue = 0.0;
    bool is_stationary = false;
    std::vector<double> seasonal_decomposition;
    std::vector<double> trend_component;
    std::vector<double> seasonal_component;
    std::vector<double> residual_component;

    std::string to_string() const;
};

// Data quality assessment
struct DataQualityReport {
    std::map<std::string, double> quality_metrics = {
        {"completeness", 0.0},
        {"consistency", 0.0},
        {"accuracy", 0.0},
        {"timeliness", 0.0},
        {"validity", 0.0},
        {"uniqueness", 0.0}
    };
    std::vector<std::string> quality_issues;
    std::map<std::string, std::vector<std::string>> data_violations;
    double overall_quality_score = 0.0;

    std::string to_string() const;
};

// Complete analysis report
struct ComprehensiveAnalysisReport {
    std::map<std::string, ColumnAnalysis> column_analyses;
    std::vector<CorrelationAnalysis> correlations;
    std::vector<FeatureImportanceAnalysis> feature_importance;
    std::vector<ClusterAnalysis> clusters;
    std::vector<OutlierAnalysis> outliers;
    std::vector<DistributionAnalysis> distributions;
    std::vector<TimeSeriesAnalysis> time_series_analyses;
    DataQualityReport data_quality;
    std::vector<std::string> insights;
    std::vector<std::string> recommendations;

    nlohmann::json to_json() const;
    std::string to_markdown_report() const;
};

// Statistical calculator
class StatisticalCalculator {
public:
    static double calculate_mean(const std::vector<double>& values);
    static double calculate_median(const std::vector<double>& values);
    static double calculate_mode(const std::vector<double>& values);
    static double calculate_std_dev(const std::vector<double>& values);
    static double calculate_variance(const std::vector<double>& values);
    static double calculate_skewness(const std::vector<double>& values);
    static double calculate_kurtosis(const std::vector<double>& values);
    static std::pair<double, double> calculate_quartiles(const std::vector<double>& values);
    static double calculate_entropy(const std::vector<std::string>& values);
    static double calculate_pearson_correlation(const std::vector<double>& x, const std::vector<double>& y);
    static double calculate_spearman_correlation(const std::vector<double>& x, const std::vector<double>& y);
    static double calculate_kendall_tau(const std::vector<double>& x, const std::vector<double>& y);
    static double calculate_mutual_information(const std::vector<double>& x, const std::vector<double>& y);
    static double calculate_chi_square(const std::vector<std::string>& x, const std::vector<std::string>& y);
    static std::vector<double> detect_outliers_iqr(const std::vector<double>& values);
    static std::vector<double> detect_outliers_zscore(const std::vector<double>& values, double threshold = 3.0);

private:
    static double erf_inv(double x);
    static double normal_cdf(double x);
};

// Data analyzer
class DataAnalyzer {
public:
    ComprehensiveAnalysisReport analyze_data(
        const std::vector<std::unordered_map<std::string, Datum>>& data,
        const std::string& target_column = "",
        const std::vector<std::string>& feature_columns = {},
        const std::string& analysis_type = "COMPREHENSIVE");

private:
    ColumnAnalysis analyze_column(
        const std::string& column_name,
        const std::vector<Datum>& values,
        const std::vector<std::string>& categorical_thresholds = {"auto"});

    CorrelationAnalysis analyze_correlation(
        const std::string& col1,
        const std::string& col2,
        const std::vector<Datum>& values1,
        const std::vector<Datum>& values2);

    FeatureImportanceAnalysis analyze_feature_importance(
        const std::string& feature_name,
        const std::vector<Datum>& feature_values,
        const std::vector<Datum>& target_values);

    std::vector<ClusterAnalysis> perform_clustering(
        const std::vector<std::unordered_map<std::string, Datum>>& data,
        const std::vector<std::string>& feature_columns,
        size_t n_clusters = 3);

    OutlierAnalysis detect_outliers(
        const std::vector<std::unordered_map<std::string, Datum>>& data,
        const std::vector<std::string>& feature_columns);

    DistributionAnalysis analyze_distribution(
        const std::string& column_name,
        const std::vector<Datum>& values);

    DataQualityReport assess_data_quality(
        const std::vector<std::unordered_map<std::string, Datum>>& data);

    std::string detect_data_type(const Datum& value);
    std::string detect_data_type_series(const std::vector<Datum>& values);
    bool is_likely_categorical(const std::vector<Datum>& values, size_t distinct_count);

    std::vector<double> extract_numeric_values(const std::vector<Datum>& values);
    std::vector<std::string> extract_string_values(const std::vector<Datum>& values);

    std::vector<std::pair<double, double>> calculate_partial_dependence(
        const std::string& feature_name,
        const std::vector<Datum>& feature_values,
        const std::vector<Datum>& target_values);

    void generate_insights_and_recommendations(
        ComprehensiveAnalysisReport& report,
        const std::string& target_column);
};

} // namespace analysis
} // namespace esql

#endif // DATA_ANALYSIS_H
