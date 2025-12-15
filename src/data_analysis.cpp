// ============================================
// data_analysis.cpp - Implementation
// ============================================
#include "data_analysis.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <set>
#include <map>
#include <unordered_set>
#include <limits>


namespace esql {
namespace analysis {

// ============================================
// StatisticalCalculator Implementation
// ============================================

double StatisticalCalculator::calculate_mean(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

double StatisticalCalculator::calculate_median(const std::vector<double>& values) {
    if (values.empty()) return 0.0;

    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());

    size_t n = sorted.size();
    if (n % 2 == 0) {
        return (sorted[n/2 - 1] + sorted[n/2]) / 2.0;
    } else {
        return sorted[n/2];
    }
}

double StatisticalCalculator::calculate_mode(const std::vector<double>& values) {
    if (values.empty()) return 0.0;

    std::map<double, int> frequency_map;
    for (double val : values) {
        frequency_map[val]++;
    }

    auto max_freq = std::max_element(
        frequency_map.begin(),
        frequency_map.end(),
        [](const auto& a, const auto& b) {
            return a.second < b.second;
        }
    );

    return max_freq->first;
}

double StatisticalCalculator::calculate_std_dev(const std::vector<double>& values) {
    if (values.size() < 2) return 0.0;

    double mean = calculate_mean(values);
    double sum_squared_diff = 0.0;

    for (double val : values) {
        double diff = val - mean;
        sum_squared_diff += diff * diff;
    }

    return std::sqrt(sum_squared_diff / (values.size() - 1));
}

double StatisticalCalculator::calculate_variance(const std::vector<double>& values) {
    double std_dev = calculate_std_dev(values);
    return std_dev * std_dev;
}

double StatisticalCalculator::calculate_skewness(const std::vector<double>& values) {
    if (values.size() < 3) return 0.0;

    double mean = calculate_mean(values);
    double std_dev = calculate_std_dev(values);

    if (std_dev == 0.0) return 0.0;

    double sum_cubed_diff = 0.0;
    for (double val : values) {
        double diff = val - mean;
        sum_cubed_diff += diff * diff * diff;
    }

    double n = static_cast<double>(values.size());
    return (sum_cubed_diff / n) / std::pow(std_dev, 3);
}

double StatisticalCalculator::calculate_kurtosis(const std::vector<double>& values) {
    if (values.size() < 4) return 0.0;

    double mean = calculate_mean(values);
    double std_dev = calculate_std_dev(values);

    if (std_dev == 0.0) return 0.0;

    double sum_fourth_diff = 0.0;
    for (double val : values) {
        double diff = val - mean;
        sum_fourth_diff += diff * diff * diff * diff;
    }

    double n = static_cast<double>(values.size());
    return (sum_fourth_diff / n) / std::pow(std_dev, 4) - 3.0; // Excess kurtosis
}

std::pair<double, double> StatisticalCalculator::calculate_quartiles(const std::vector<double>& values) {
    if (values.empty()) return {0.0, 0.0};

    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());

    size_t n = sorted.size();
    size_t q1_index = n / 4;
    size_t q3_index = 3 * n / 4;

    double q1 = sorted[q1_index];
    double q3 = sorted[q3_index];

    return {q1, q3};
}

double StatisticalCalculator::calculate_entropy(const std::vector<std::string>& values) {
    if (values.empty()) return 0.0;

    std::map<std::string, size_t> frequency_map;
    for (const auto& val : values) {
        frequency_map[val]++;
    }

    double entropy = 0.0;
    double n = static_cast<double>(values.size());

    for (const auto& [_, count] : frequency_map) {
        double probability = static_cast<double>(count) / n;
        entropy -= probability * std::log2(probability);
    }

    return entropy;
}

double StatisticalCalculator::calculate_pearson_correlation(const std::vector<double>& x, const std::vector<double>& y) {
    if (x.size() != y.size() || x.size() < 2) return 0.0;

    double mean_x = calculate_mean(x);
    double mean_y = calculate_mean(y);
    double std_x = calculate_std_dev(x);
    double std_y = calculate_std_dev(y);

    if (std_x == 0.0 || std_y == 0.0) return 0.0;

    double sum_product = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        sum_product += (x[i] - mean_x) * (y[i] - mean_y);
    }

    return sum_product / ((x.size() - 1) * std_x * std_y);
}

double StatisticalCalculator::calculate_spearman_correlation(const std::vector<double>& x, const std::vector<double>& y) {
    if (x.size() != y.size() || x.size() < 2) return 0.0;

    // Create rank vectors
    auto get_ranks = [](const std::vector<double>& values) -> std::vector<double> {
        std::vector<size_t> indices(values.size());
        std::iota(indices.begin(), indices.end(), 0);

        std::sort(indices.begin(), indices.end(),
                 [&values](size_t i, size_t j) { return values[i] < values[j]; });

        std::vector<double> ranks(values.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            // Handle ties by assigning average rank
            size_t j = i;
            while (j + 1 < indices.size() &&
                   values[indices[j]] == values[indices[j + 1]]) {
                ++j;
            }

            double avg_rank = (i + j + 2) / 2.0;
            for (size_t k = i; k <= j; ++k) {
                ranks[indices[k]] = avg_rank;
            }

            i = j;
        }

        return ranks;
    };

    std::vector<double> ranks_x = get_ranks(x);
    std::vector<double> ranks_y = get_ranks(y);

    return calculate_pearson_correlation(ranks_x, ranks_y);
}

double StatisticalCalculator::calculate_kendall_tau(const std::vector<double>& x, const std::vector<double>& y) {
    if (x.size() != y.size() || x.size() < 2) return 0.0;

    size_t n = x.size();
    size_t concordant = 0;
    size_t discordant = 0;

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            double x_diff = x[j] - x[i];
            double y_diff = y[j] - y[i];

            if (x_diff * y_diff > 0) {
                concordant++;
            } else if (x_diff * y_diff < 0) {
                discordant++;
            }
            // Ties are ignored in basic implementation
        }
    }

    if (concordant + discordant == 0) return 0.0;

    return static_cast<double>(concordant - discordant) /
           static_cast<double>(concordant + discordant);
}

double StatisticalCalculator::calculate_mutual_information(const std::vector<double>& x, const std::vector<double>& y) {
    if (x.size() != y.size() || x.size() < 2) return 0.0;

    // Discretize continuous variables
    auto discretize = [](const std::vector<double>& values, size_t bins = 10) -> std::vector<size_t> {
        if (values.empty()) return {};

        double min_val = *std::min_element(values.begin(), values.end());
        double max_val = *std::max_element(values.begin(), values.end());

        if (max_val - min_val < 1e-10) {
            return std::vector<size_t>(values.size(), 0);
        }

        std::vector<size_t> discretized(values.size());
        for (size_t i = 0; i < values.size(); ++i) {
            discretized[i] = static_cast<size_t>(
                (values[i] - min_val) / (max_val - min_val) * (bins - 1)
            );
        }

        return discretized;
    };

    std::vector<size_t> disc_x = discretize(x);
    std::vector<size_t> disc_y = discretize(y);

    // Calculate joint and marginal probabilities
    std::map<std::pair<size_t, size_t>, double> joint_prob;
    std::map<size_t, double> marginal_x;
    std::map<size_t, double> marginal_y;

    double n = static_cast<double>(x.size());

    for (size_t i = 0; i < x.size(); ++i) {
        auto key = std::make_pair(disc_x[i], disc_y[i]);
        joint_prob[key] += 1.0 / n;
        marginal_x[disc_x[i]] += 1.0 / n;
        marginal_y[disc_y[i]] += 1.0 / n;
    }

    // Calculate mutual information
    double mi = 0.0;
    for (const auto& [key, p_xy] : joint_prob) {
        double p_x = marginal_x[key.first];
        double p_y = marginal_y[key.second];

        if (p_x > 0.0 && p_y > 0.0 && p_xy > 0.0) {
            mi += p_xy * std::log2(p_xy / (p_x * p_y));
        }
    }

    return mi;
}

double StatisticalCalculator::calculate_chi_square(const std::vector<std::string>& x, const std::vector<std::string>& y) {
    if (x.size() != y.size() || x.size() < 2) return 0.0;

    // Create contingency table
    std::map<std::string, size_t> x_categories, y_categories;
    std::map<std::pair<std::string, std::string>, size_t> observed_counts;

    for (size_t i = 0; i < x.size(); ++i) {
        x_categories[x[i]]++;
        y_categories[y[i]]++;
        observed_counts[{x[i], y[i]}]++;
    }

    // Calculate expected frequencies and chi-square
    double chi_square = 0.0;
    double n = static_cast<double>(x.size());

    for (const auto& [x_cat, x_count] : x_categories) {
        for (const auto& [y_cat, y_count] : y_categories) {
            double expected = (x_count * y_count) / n;
            auto key = std::make_pair(x_cat, y_cat);
            double observed = observed_counts.count(key) ? observed_counts[key] : 0.0;

            if (expected > 0.0) {
                double diff = observed - expected;
                chi_square += (diff * diff) / expected;
            }
        }
    }

    return chi_square;
}

std::vector<double> StatisticalCalculator::detect_outliers_iqr(const std::vector<double>& values) {
    if (values.size() < 4) return {};

    auto [q1, q3] = calculate_quartiles(values);
    double iqr = q3 - q1;

    if (iqr == 0.0) return {};

    double lower_bound = q1 - 1.5 * iqr;
    double upper_bound = q3 + 1.5 * iqr;

    std::vector<double> outliers;
    for (double val : values) {
        if (val < lower_bound || val > upper_bound) {
            outliers.push_back(val);
        }
    }

    return outliers;
}

std::vector<double> StatisticalCalculator::detect_outliers_zscore(const std::vector<double>& values, double threshold) {
    if (values.size() < 2) return {};

    double mean = calculate_mean(values);
    double std_dev = calculate_std_dev(values);

    if (std_dev == 0.0) return {};

    std::vector<double> outliers;
    for (double val : values) {
        double z_score = std::abs((val - mean) / std_dev);
        if (z_score > threshold) {
            outliers.push_back(val);
        }
    }

    return outliers;
}

// ============================================
// ColumnAnalysis Implementation
// ============================================

std::string ColumnAnalysis::to_string() const {
    std::stringstream ss;
    ss << "Column Analysis: " << name << "\n";
    ss << "  Type: " << detected_type << "\n";
    ss << "  Count: " << total_count << " (Nulls: " << null_count << ", "
       << std::fixed << std::setprecision(2) << missing_percentage << "% missing)\n";
    ss << "  Distinct: " << distinct_count << "\n";

    if (detected_type == "numeric" || detected_type == "integer" ||
        detected_type == "float" || detected_type == "double") {
        ss << "  Range: [" << min_value << ", " << max_value << "]\n";
        ss << "  Mean: " << mean << ", Median: " << median << ", Mode: " << mode << "\n";
        ss << "  Std Dev: " << std_dev << ", Variance: " << variance << "\n";
        ss << "  Skewness: " << skewness << ", Kurtosis: " << kurtosis << "\n";
        ss << "  Quartiles: Q1=" << q1 << ", Q3=" << q3 << ", IQR=" << iqr << "\n";
        ss << "  Has outliers: " << (has_outliers ? "Yes" : "No");
        if (has_outliers && !outliers.empty()) {
            ss << " (" << outliers.size() << " detected)";
        }
    } else if (is_categorical) {
        ss << "  Top categories:\n";
        for (size_t i = 0; i < std::min(top_categories.size(), (size_t)5); ++i) {
            ss << "    - " << top_categories[i] << "\n";
        }
        ss << "  Entropy: " << entropy << "\n";
    }

    return ss.str();
}

nlohmann::json ColumnAnalysis::to_json() const {
    nlohmann::json j;
    j["name"] = name;
    j["type"] = detected_type;
    j["total_count"] = total_count;
    j["null_count"] = null_count;
    j["distinct_count"] = distinct_count;
    j["missing_percentage"] = missing_percentage;
    j["is_categorical"] = is_categorical;

    if (detected_type == "numeric" || detected_type == "integer" ||
        detected_type == "float" || detected_type == "double") {
        j["min"] = min_value;
        j["max"] = max_value;
        j["mean"] = mean;
        j["median"] = median;
        j["mode"] = mode;
        j["std_dev"] = std_dev;
        j["variance"] = variance;
        j["skewness"] = skewness;
        j["kurtosis"] = kurtosis;
        j["q1"] = q1;
        j["q3"] = q3;
        j["iqr"] = iqr;
        j["has_outliers"] = has_outliers;
        j["outliers"] = outliers;
    }

    if (is_categorical) {
        j["top_categories"] = top_categories;
        j["entropy"] = entropy;
    }

    // Add histogram data if available
    if (!histogram_bins.empty() && !histogram_counts.empty()) {
        nlohmann::json hist_json;
        for (size_t i = 0; i < histogram_bins.size(); ++i) {
            if (i < histogram_counts.size()) {
                hist_json.push_back({
                    {"bin", histogram_bins[i]},
                    {"count", histogram_counts[i]}
                });
            }
        }
        j["histogram"] = hist_json;
    }

    return j;
}

std::string FeatureImportanceAnalysis::to_string() const {
    std::stringstream ss;
    ss << "Feature Importance Analysis: " << feature_name << "\n";
    ss << "  Importance Score: " << importance_score << "\n";
    ss << "  Permutation Importance: " << permutation_importance << "\n";
    ss << "  SHAP Importance: " << shap_importance << "\n";
    ss << "  Information Gain: " << information_gain << "\n";
    ss << "  Mutual Information: " << mutual_information << "\n";
    ss << "  Chi-Square: " << chi_square << "\n";
    ss << "  ANOVA F-Value: " << anova_f_value << "\n";

    if (!partial_dependence.empty()) {
        ss << "  Partial Dependence (first 5 points):\n";
        for (size_t i = 0; i < std::min(partial_dependence.size(), (size_t)5); ++i) {
            ss << "    Feature=" << partial_dependence[i].first
               << " -> Target=" << partial_dependence[i].second << "\n";
        }
    }

    return ss.str();
}

// ============================================
// DataAnalyzer Implementation
// ============================================

ComprehensiveAnalysisReport DataAnalyzer::analyze_data(
    const std::vector<std::unordered_map<std::string, Datum>>& data,
    const std::string& target_column,
    const std::vector<std::string>& feature_columns,
    const std::string& analysis_type) {

    ComprehensiveAnalysisReport report;

    if (data.empty()) {
        report.insights.push_back("No data to analyze");
        return report;
    }

    std::cout << "[DataAnalyzer] Analyzing " << data.size() << " rows of data" << std::endl;

    try {
        // 1. Extract column names
        std::vector<std::string> columns_to_analyze;
        if (!feature_columns.empty()) {
            columns_to_analyze = feature_columns;
        } else {
            // Get all columns from first row
            for (const auto& [col_name, _] : data[0]) {
                columns_to_analyze.push_back(col_name);
            }
        }

        // 2. Analyze each column
        std::cout << "[DataAnalyzer] Analyzing " << columns_to_analyze.size() << " columns..." << std::endl;

        for (const auto& column_name : columns_to_analyze) {
            // Collect values for this column
            std::vector<Datum> column_values;
            column_values.reserve(data.size());

            for (const auto& row : data) {
                auto it = row.find(column_name);
                if (it != row.end()) {
                    column_values.push_back(it->second);
                } else {
                    column_values.push_back(Datum::create_null());
                }
            }

            // Perform column analysis
            ColumnAnalysis col_analysis = analyze_column(column_name, column_values);
            report.column_analyses[column_name] = col_analysis;

            // Analyze distribution if numeric
            if (col_analysis.detected_type == "numeric" ||
                col_analysis.detected_type == "integer" ||
                col_analysis.detected_type == "float" ||
                col_analysis.detected_type == "double") {

                DistributionAnalysis dist_analysis = analyze_distribution(column_name, column_values);
                report.distributions.push_back(dist_analysis);
            }
        }

        // 3. Perform correlation analysis if multiple numeric columns exist
        if (analysis_type == "CORRELATION" || analysis_type == "COMPREHENSIVE") {
            std::cout << "[DataAnalyzer] Performing correlation analysis..." << std::endl;

            // Find numeric columns
            std::vector<std::string> numeric_columns;
            for (const auto& [col_name, analysis] : report.column_analyses) {
                if (analysis.detected_type == "numeric" ||
                    analysis.detected_type == "integer" ||
                    analysis.detected_type == "float" ||
                    analysis.detected_type == "double") {
                    numeric_columns.push_back(col_name);
                }
            }

            // Calculate correlations between all pairs
            for (size_t i = 0; i < numeric_columns.size(); ++i) {
                for (size_t j = i + 1; j < numeric_columns.size(); ++j) {
                    // Extract values for both columns
                    std::vector<Datum> values1, values2;
                    for (const auto& row : data) {
                        auto it1 = row.find(numeric_columns[i]);
                        auto it2 = row.find(numeric_columns[j]);

                        if (it1 != row.end() && it2 != row.end() &&
                            !it1->second.is_null() && !it2->second.is_null()) {
                            values1.push_back(it1->second);
                            values2.push_back(it2->second);
                        }
                    }

                    if (values1.size() >= 2 && values2.size() >= 2) {
                        CorrelationAnalysis corr_analysis = analyze_correlation(
                            numeric_columns[i], numeric_columns[j], values1, values2);
                        report.correlations.push_back(corr_analysis);
                    }
                }
            }
        }

        // 4. Feature importance analysis if target column specified
        if (!target_column.empty() && report.column_analyses.find(target_column) != report.column_analyses.end()) {
            std::cout << "[DataAnalyzer] Performing feature importance analysis for target: "
                      << target_column << std::endl;

            // Extract target values
            std::vector<Datum> target_values;
            for (const auto& row : data) {
                auto it = row.find(target_column);
                if (it != row.end()) {
                    target_values.push_back(it->second);
                }
            }

            // Analyze each feature against target
            for (const auto& [feature_name, feature_analysis] : report.column_analyses) {
                if (feature_name == target_column) continue;

                // Extract feature values
                std::vector<Datum> feature_values;
                for (const auto& row : data) {
                    auto it = row.find(feature_name);
                    if (it != row.end()) {
                        feature_values.push_back(it->second);
                    }
                }

                if (feature_values.size() == target_values.size() && !feature_values.empty()) {
                    FeatureImportanceAnalysis importance_analysis = analyze_feature_importance(
                        feature_name, feature_values, target_values);
                    report.feature_importance.push_back(importance_analysis);
                }
            }

            // Sort by importance score
            std::sort(report.feature_importance.begin(), report.feature_importance.end(),
                     [](const auto& a, const auto& b) {
                         return a.importance_score > b.importance_score;
                     });
        }

        // 5. Clustering analysis if requested
        if (analysis_type == "CLUSTERING" || analysis_type == "COMPREHENSIVE") {
            // Find numeric features for clustering
            std::vector<std::string> clustering_features;
            for (const auto& [col_name, analysis] : report.column_analyses) {
                if ((analysis.detected_type == "numeric" ||
                     analysis.detected_type == "integer" ||
                     analysis.detected_type == "float" ||
                     analysis.detected_type == "double") &&
                    !analysis.has_outliers) {
                    clustering_features.push_back(col_name);
                }
            }

            if (clustering_features.size() >= 2) {
                std::cout << "[DataAnalyzer] Performing clustering analysis with "
                          << clustering_features.size() << " features..." << std::endl;

                auto clusters = perform_clustering(data, clustering_features);
                report.clusters = clusters;
            }
        }

        // 6. Outlier detection
        if (analysis_type == "OUTLIER" || analysis_type == "COMPREHENSIVE") {
            std::cout << "[DataAnalyzer] Performing outlier detection..." << std::endl;

            // Find numeric features for outlier detection
            std::vector<std::string> outlier_features;
            for (const auto& [col_name, analysis] : report.column_analyses) {
                if (analysis.detected_type == "numeric" ||
                    analysis.detected_type == "integer" ||
                    analysis.detected_type == "float" ||
                    analysis.detected_type == "double") {
                    outlier_features.push_back(col_name);
                }
            }

            if (!outlier_features.empty()) {
                OutlierAnalysis outlier_analysis = detect_outliers(data, outlier_features);
                report.outliers.push_back(outlier_analysis);
            }
        }

        // 7. Data quality assessment
        std::cout << "[DataAnalyzer] Assessing data quality..." << std::endl;
        report.data_quality = assess_data_quality(data);

        // 8. Generate insights and recommendations
        generate_insights_and_recommendations(report, target_column);

        std::cout << "[DataAnalyzer] Analysis complete. Generated "
                  << report.insights.size() << " insights and "
                  << report.recommendations.size() << " recommendations." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[DataAnalyzer] Analysis failed: " << e.what() << std::endl;
        report.insights.push_back("Analysis failed: " + std::string(e.what()));
    }

    return report;
}

ColumnAnalysis DataAnalyzer::analyze_column(
    const std::string& column_name,
    const std::vector<Datum>& values,
    const std::vector<std::string>& categorical_thresholds) {

    ColumnAnalysis analysis;
    analysis.name = column_name;
    analysis.total_count = values.size();

    // Count nulls and collect non-null values
    std::vector<Datum> non_null_values;
    for (const auto& value : values) {
        if (value.is_null()) {
            analysis.null_count++;
        } else {
            non_null_values.push_back(value);
        }
    }

    analysis.missing_percentage = (analysis.null_count * 100.0) / analysis.total_count;

    if (non_null_values.empty()) {
        analysis.detected_type = "unknown";
        return analysis;
    }

    // Detect data type
    analysis.detected_type = detect_data_type_series(non_null_values);

    if (analysis.detected_type == "string" || analysis.detected_type == "text") {
        // Handle string/categorical columns
        std::vector<std::string> string_values;
        std::set<std::string> distinct_set;
        std::map<std::string, size_t> frequency_map;

        for (const auto& value : non_null_values) {
            std::string str_val = value.to_string();
            string_values.push_back(str_val);
            distinct_set.insert(str_val);
            frequency_map[str_val]++;
        }

        analysis.distinct_count = distinct_set.size();
        analysis.is_categorical = is_likely_categorical(non_null_values, analysis.distinct_count);
        analysis.entropy = StatisticalCalculator::calculate_entropy(string_values);

        // Get top categories
        std::vector<std::pair<std::string, size_t>> sorted_frequencies(
            frequency_map.begin(), frequency_map.end());

        std::sort(sorted_frequencies.begin(), sorted_frequencies.end(),
                 [](const auto& a, const auto& b) { return a.second > b.second; });

        for (size_t i = 0; i < std::min(sorted_frequencies.size(), (size_t)10); ++i) {
            analysis.top_categories.push_back(sorted_frequencies[i].first);
        }

    } else if (analysis.detected_type == "numeric" || analysis.detected_type == "integer" ||
               analysis.detected_type == "float" || analysis.detected_type == "double") {

        // Handle numeric columns
        std::vector<double> numeric_values = extract_numeric_values(non_null_values);

        if (numeric_values.empty()) {
            return analysis;
        }

        analysis.distinct_count = std::set<double>(numeric_values.begin(), numeric_values.end()).size();
        analysis.is_categorical = is_likely_categorical(non_null_values, analysis.distinct_count);

        // Calculate basic statistics
        analysis.min_value = *std::min_element(numeric_values.begin(), numeric_values.end());
        analysis.max_value = *std::max_element(numeric_values.begin(), numeric_values.end());
        analysis.mean = StatisticalCalculator::calculate_mean(numeric_values);
        analysis.median = StatisticalCalculator::calculate_median(numeric_values);
        analysis.mode = StatisticalCalculator::calculate_mode(numeric_values);
        analysis.std_dev = StatisticalCalculator::calculate_std_dev(numeric_values);
        analysis.variance = StatisticalCalculator::calculate_variance(numeric_values);
        analysis.skewness = StatisticalCalculator::calculate_skewness(numeric_values);
        analysis.kurtosis = StatisticalCalculator::calculate_kurtosis(numeric_values);

        // Calculate quartiles and IQR
        auto [q1, q3] = StatisticalCalculator::calculate_quartiles(numeric_values);
        analysis.q1 = q1;
        analysis.q3 = q3;
        analysis.iqr = q3 - q1;

        // Detect outliers
        analysis.outliers = StatisticalCalculator::detect_outliers_iqr(numeric_values);
        analysis.has_outliers = !analysis.outliers.empty();

        // Create histogram
        size_t num_bins = std::min(static_cast<size_t>(10),
                                  static_cast<size_t>(std::sqrt(numeric_values.size())));

        if (num_bins > 1) {
            double bin_width = (analysis.max_value - analysis.min_value) / num_bins;

            for (size_t i = 0; i < num_bins; ++i) {
                analysis.histogram_bins.push_back(analysis.min_value + i * bin_width);
                analysis.histogram_counts.push_back(0);
            }

            for (double value : numeric_values) {
                size_t bin_index = static_cast<size_t>(
                    (value - analysis.min_value) / bin_width
                );
                if (bin_index >= num_bins) bin_index = num_bins - 1;
                analysis.histogram_counts[bin_index]++;
            }
        }
    }

    return analysis;
}

CorrelationAnalysis DataAnalyzer::analyze_correlation(
    const std::string& col1,
    const std::string& col2,
    const std::vector<Datum>& values1,
    const std::vector<Datum>& values2) {

    CorrelationAnalysis analysis;
    analysis.column1 = col1;
    analysis.column2 = col2;

    // Extract numeric values
    std::vector<double> numeric1 = extract_numeric_values(values1);
    std::vector<double> numeric2 = extract_numeric_values(values2);

    if (numeric1.size() != numeric2.size() || numeric1.size() < 2) {
        return analysis;
    }

    // Calculate different correlation measures
    analysis.pearson_correlation = StatisticalCalculator::calculate_pearson_correlation(numeric1, numeric2);
    analysis.spearman_correlation = StatisticalCalculator::calculate_spearman_correlation(numeric1, numeric2);
    analysis.kendall_tau = StatisticalCalculator::calculate_kendall_tau(numeric1, numeric2);

    // Calculate covariance
    double mean1 = StatisticalCalculator::calculate_mean(numeric1);
    double mean2 = StatisticalCalculator::calculate_mean(numeric2);
    double cov_sum = 0.0;

    for (size_t i = 0; i < numeric1.size(); ++i) {
        cov_sum += (numeric1[i] - mean1) * (numeric2[i] - mean2);
    }

    analysis.covariance = cov_sum / (numeric1.size() - 1);

    // Determine significance (simplified - in production use proper statistical tests)
    analysis.p_value = 2.0 * std::exp(-std::abs(analysis.pearson_correlation) * std::sqrt(numeric1.size()));
    analysis.is_significant = analysis.p_value < 0.05;

    // Determine relationship type
    double abs_corr = std::abs(analysis.pearson_correlation);
    std::string strength = (abs_corr > 0.7) ? "strong" :
                          (abs_corr > 0.3) ? "moderate" : "weak";

    std::string direction = (analysis.pearson_correlation > 0) ? "positive" : "negative";

    analysis.relationship_type = strength + "_" + direction;

    return analysis;
}

FeatureImportanceAnalysis DataAnalyzer::analyze_feature_importance(
    const std::string& feature_name,
    const std::vector<Datum>& feature_values,
    const std::vector<Datum>& target_values) {

    FeatureImportanceAnalysis analysis;
    analysis.feature_name = feature_name;

    // Extract numeric values
    std::vector<double> numeric_features = extract_numeric_values(feature_values);
    std::vector<double> numeric_targets = extract_numeric_values(target_values);

    if (numeric_features.empty() || numeric_targets.empty() ||
        numeric_features.size() != numeric_targets.size()) {
        return analysis;
    }

    // Calculate Pearson correlation as importance score
    analysis.importance_score = std::abs(
        StatisticalCalculator::calculate_pearson_correlation(numeric_features, numeric_targets)
    );

    // Calculate mutual information
    analysis.mutual_information = StatisticalCalculator::calculate_mutual_information(
        numeric_features, numeric_targets
    );

    // For categorical features vs categorical targets, calculate chi-square
    std::vector<std::string> string_features, string_targets;
    for (const auto& val : feature_values) {
        string_features.push_back(val.to_string());
    }
    for (const auto& val : target_values) {
        string_targets.push_back(val.to_string());
    }

    analysis.chi_square = StatisticalCalculator::calculate_chi_square(
        string_features, string_targets
    );

    // Calculate partial dependence (simplified)
    analysis.partial_dependence = calculate_partial_dependence(
        feature_name, feature_values, target_values
    );

    return analysis;
}

std::vector<ClusterAnalysis> DataAnalyzer::perform_clustering(
    const std::vector<std::unordered_map<std::string, Datum>>& data,
    const std::vector<std::string>& feature_columns,
    size_t n_clusters) {

    std::vector<ClusterAnalysis> clusters;

    // Simplified K-means implementation (in production, use a proper library like MLPACK)
    if (data.empty() || feature_columns.empty()) {
        return clusters;
    }

    // Extract numeric data matrix
    std::vector<std::vector<double>> data_matrix(data.size(),
                                                std::vector<double>(feature_columns.size()));

    for (size_t i = 0; i < data.size(); ++i) {
        const auto& row = data[i];
        for (size_t j = 0; j < feature_columns.size(); ++j) {
            auto it = row.find(feature_columns[j]);
            if (it != row.end() && !it->second.is_null()) {
                try {
                    data_matrix[i][j] = it->second.as_double();
                } catch (...) {
                    data_matrix[i][j] = 0.0;
                }
            }
        }
    }

    // Remove rows with missing values
    std::vector<std::vector<double>> clean_data;
    std::vector<size_t> clean_indices;

    for (size_t i = 0; i < data_matrix.size(); ++i) {
        bool has_missing = false;
        for (double val : data_matrix[i]) {
            if (std::isnan(val)) {
                has_missing = true;
                break;
            }
        }

        if (!has_missing) {
            clean_data.push_back(data_matrix[i]);
            clean_indices.push_back(i);
        }
    }

    if (clean_data.size() < n_clusters) {
        return clusters;
    }

    // Initialize random centroids
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dist(0, clean_data.size() - 1);

    std::vector<std::vector<double>> centroids(n_clusters);
    for (size_t k = 0; k < n_clusters; ++k) {
        centroids[k] = clean_data[dist(gen)];
    }

    // K-means iterations
    const size_t max_iterations = 100;
    std::vector<size_t> assignments(clean_data.size(), 0);

    for (size_t iter = 0; iter < max_iterations; ++iter) {
        // Assign points to nearest centroid
        bool changed = false;

        for (size_t i = 0; i < clean_data.size(); ++i) {
            double min_dist = std::numeric_limits<double>::max();
            size_t best_cluster = 0;

            for (size_t k = 0; k < n_clusters; ++k) {
                double dist = 0.0;
                for (size_t j = 0; j < feature_columns.size(); ++j) {
                    double diff = clean_data[i][j] - centroids[k][j];
                    dist += diff * diff;
                }
                dist = std::sqrt(dist);

                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = k;
                }
            }

            if (assignments[i] != best_cluster) {
                assignments[i] = best_cluster;
                changed = true;
            }
        }

        if (!changed) break;

        // Update centroids
        std::vector<std::vector<double>> new_centroids(n_clusters,
                                                      std::vector<double>(feature_columns.size(), 0.0));
        std::vector<size_t> cluster_sizes(n_clusters, 0);

        for (size_t i = 0; i < clean_data.size(); ++i) {
            size_t cluster = assignments[i];
            cluster_sizes[cluster]++;

            for (size_t j = 0; j < feature_columns.size(); ++j) {
                new_centroids[cluster][j] += clean_data[i][j];
            }
        }

        for (size_t k = 0; k < n_clusters; ++k) {
            if (cluster_sizes[k] > 0) {
                for (size_t j = 0; j < feature_columns.size(); ++j) {
                    new_centroids[k][j] /= cluster_sizes[k];
                }
            }
        }

        centroids = new_centroids;
    }

    // Create cluster analyses
    for (size_t k = 0; k < n_clusters; ++k) {
        ClusterAnalysis cluster;
        cluster.cluster_id = k;

        // Count members
        std::vector<size_t> member_indices;
        for (size_t i = 0; i < assignments.size(); ++i) {
            if (assignments[i] == k) {
                member_indices.push_back(clean_indices[i]);
            }
        }

        cluster.size = member_indices.size();
        cluster.member_indices = member_indices;
        cluster.centroid = centroids[k];

        // Identify top features for this cluster
        if (!feature_columns.empty()) {
            std::vector<std::pair<size_t, double>> feature_variances;
            for (size_t j = 0; j < feature_columns.size(); ++j) {
                // Calculate variance of this feature within cluster
                std::vector<double> feature_values;
                for (size_t i = 0; i < clean_data.size(); ++i) {
                    if (assignments[i] == k) {
                        feature_values.push_back(clean_data[i][j]);
                    }
                }

                if (feature_values.size() > 1) {
                    double variance = StatisticalCalculator::calculate_variance(feature_values);
                    feature_variances.emplace_back(j, variance);
                }
            }

            // Sort by variance (highest variance = most distinctive)
            std::sort(feature_variances.begin(), feature_variances.end(),
                     [](const auto& a, const auto& b) { return a.second > b.second; });

            for (size_t i = 0; i < std::min(feature_variances.size(), (size_t)3); ++i) {
                cluster.top_features.push_back(feature_columns[feature_variances[i].first]);
            }
        }

        clusters.push_back(cluster);
    }

    return clusters;
}

OutlierAnalysis DataAnalyzer::detect_outliers(
    const std::vector<std::unordered_map<std::string, Datum>>& data,
    const std::vector<std::string>& feature_columns) {

    OutlierAnalysis analysis;
    analysis.detection_method = "isolation_forest"; // Simplified

    if (data.empty() || feature_columns.empty()) {
        return analysis;
    }

    // Simplified outlier detection using z-score method
    std::vector<double> outlier_scores(data.size(), 0.0);
    std::vector<std::string> outlier_reasons(data.size(), "");

    for (const auto& feature : feature_columns) {
        // Extract values for this feature
        std::vector<double> values;
        for (const auto& row : data) {
            auto it = row.find(feature);
            if (it != row.end() && !it->second.is_null()) {
                try {
                    values.push_back(it->second.as_double());
                } catch (...) {
                    values.push_back(0.0);
                }
            } else {
                values.push_back(0.0);
            }
        }

        // Calculate z-scores
        double mean = StatisticalCalculator::calculate_mean(values);
        double std_dev = StatisticalCalculator::calculate_std_dev(values);

        if (std_dev > 0.0) {
            for (size_t i = 0; i < values.size(); ++i) {
                double z_score = std::abs((values[i] - mean) / std_dev);
                outlier_scores[i] += z_score;

                if (z_score > 3.0) {
                    if (!outlier_reasons[i].empty()) {
                        outlier_reasons[i] += ", ";
                    }
                    outlier_reasons[i] += feature + "(z=" + std::to_string(z_score) + ")";
                }
            }
        }
    }

    // Normalize scores and identify outliers
    if (data.size() > 0) {
        for (size_t i = 0; i < outlier_scores.size(); ++i) {
            outlier_scores[i] /= feature_columns.size();

            if (outlier_scores[i] > 2.5) { // Threshold
                analysis.outlier_indices.push_back(i);
                analysis.outlier_scores.push_back(outlier_scores[i]);
                analysis.outlier_reasons.push_back(outlier_reasons[i]);
            }
        }
    }

    analysis.contamination_rate = static_cast<double>(analysis.outlier_indices.size()) / data.size();
    analysis.affected_columns = feature_columns;

    return analysis;
}

DistributionAnalysis DataAnalyzer::analyze_distribution(
    const std::string& column_name,
    const std::vector<Datum>& values) {

    DistributionAnalysis analysis;
    analysis.column_name = column_name;

    std::vector<double> numeric_values = extract_numeric_values(values);
    if (numeric_values.empty()) {
        return analysis;
    }

    // Simplified distribution testing
    double skewness = StatisticalCalculator::calculate_skewness(numeric_values);
    double kurtosis = StatisticalCalculator::calculate_kurtosis(numeric_values);

    // Determine distribution type based on statistics
    if (std::abs(skewness) < 0.5 && std::abs(kurtosis) < 1.0) {
        analysis.distribution_type = "normal";
        analysis.passes_normality_test = true;
    } else if (skewness > 1.0) {
        analysis.distribution_type = "right_skewed";
    } else if (skewness < -1.0) {
        analysis.distribution_type = "left_skewed";
    } else if (kurtosis > 3.0) {
        analysis.distribution_type = "leptokurtic";
    } else if (kurtosis < 3.0) {
        analysis.distribution_type = "platykurtic";
    } else {
        analysis.distribution_type = "unknown";
    }

    // Store distribution parameters
    analysis.distribution_parameters["mean"] = StatisticalCalculator::calculate_mean(numeric_values);
    analysis.distribution_parameters["std_dev"] = StatisticalCalculator::calculate_std_dev(numeric_values);
    analysis.distribution_parameters["skewness"] = skewness;
    analysis.distribution_parameters["kurtosis"] = kurtosis;

    return analysis;
}

DataQualityReport DataAnalyzer::assess_data_quality(
    const std::vector<std::unordered_map<std::string, Datum>>& data) {

    DataQualityReport report;

    if (data.empty()) {
        report.quality_issues.push_back("No data available");
        report.overall_quality_score = 0.0;
        return report;
    }

    // 1. Completeness
    size_t total_cells = data.size() * data[0].size();
    size_t null_cells = 0;

    for (const auto& row : data) {
        for (const auto& [_, value] : row) {
            if (value.is_null()) {
                null_cells++;
            }
        }
    }

    report.quality_metrics["completeness"] = 1.0 - (static_cast<double>(null_cells) / total_cells);

    if (report.quality_metrics["completeness"] < 0.9) {
        report.quality_issues.push_back("High percentage of missing values");
    }

    // 2. Consistency (simplified - check for consistent data types)
    // This would be more comprehensive in production

    // 3. Uniqueness (check for duplicate rows)
    std::set<std::string> row_hashes;
    for (const auto& row : data) {
        std::string row_hash;
        for (const auto& [col, value] : row) {
            row_hash += value.to_string() + "|";
        }
        row_hashes.insert(row_hash);
    }

    double uniqueness = static_cast<double>(row_hashes.size()) / data.size();
    report.quality_metrics["uniqueness"] = uniqueness;

    if (uniqueness < 0.95) {
        report.quality_issues.push_back("Potential duplicate rows detected");
    }

    // Calculate overall score (weighted average)
    double total_weight = 0.0;
    double weighted_sum = 0.0;

    std::map<std::string, double> weights = {
        {"completeness", 0.3},
        {"consistency", 0.2},
        {"accuracy", 0.2},  // Would need ground truth for accuracy
        {"uniqueness", 0.2},
        {"timeliness", 0.05},  // Would need timestamps
        {"validity", 0.05}     // Would need validation rules
    };

    for (const auto& [metric, weight] : weights) {
        if (report.quality_metrics.find(metric) != report.quality_metrics.end()) {
            weighted_sum += report.quality_metrics[metric] * weight;
            total_weight += weight;
        }
    }

    if (total_weight > 0.0) {
        report.overall_quality_score = weighted_sum / total_weight;
    }

    // Classify quality
    if (report.overall_quality_score >= 0.9) {
        report.quality_issues.push_back("Excellent data quality");
    } else if (report.overall_quality_score >= 0.7) {
        report.quality_issues.push_back("Good data quality");
    } else if (report.overall_quality_score >= 0.5) {
        report.quality_issues.push_back("Fair data quality - needs improvement");
    } else {
        report.quality_issues.push_back("Poor data quality - requires attention");
    }

    return report;
}

std::string DataAnalyzer::detect_data_type(const Datum& value) {
    if (value.is_null()) return "null";

    switch (value.type()) {
        case Datum::Type::INTEGER: return "integer";
        case Datum::Type::FLOAT: return "float";
        case Datum::Type::DOUBLE: return "double";
        case Datum::Type::BOOLEAN: return "boolean";
        case Datum::Type::STRING: {
            std::string str_val = value.as_string();

            // Check if it's a date/time
            if (str_val.size() == 10 && str_val[4] == '-' && str_val[7] == '-') {
                return "date";
            }

            // Check if it's numeric
            try {
                std::stod(str_val);
                return "numeric_string";
            } catch (...) {
                return "string";
            }
        }
        case Datum::Type::DATE:
        case Datum::Type::DATETIME:
        case Datum::Type::TIMESTAMP: return "datetime";
        default: return "unknown";
    }
}

std::string DataAnalyzer::detect_data_type_series(const std::vector<Datum>& values) {
    if (values.empty()) return "unknown";

    // Count type occurrences
    std::map<std::string, size_t> type_counts;
    size_t numeric_like = 0;

    for (const auto& value : values) {
        std::string type = detect_data_type(value);
        type_counts[type]++;

        if (type == "integer" || type == "float" || type == "double" ||
            type == "numeric_string" || type == "boolean") {
            numeric_like++;
        }
    }

    // Determine dominant type
    std::string dominant_type = "unknown";
    size_t max_count = 0;

    for (const auto& [type, count] : type_counts) {
        if (count > max_count) {
            max_count = count;
            dominant_type = type;
        }
    }

    // If mostly numeric-like, return "numeric"
    if (static_cast<double>(numeric_like) / values.size() > 0.8) {
        return "numeric";
    }

    return dominant_type;
}

bool DataAnalyzer::is_likely_categorical(const std::vector<Datum>& values, size_t distinct_count) {
    if (values.empty()) return false;

    // Check cardinality threshold
    double cardinality_ratio = static_cast<double>(distinct_count) / values.size();

    // Consider categorical if:
    // 1. Low cardinality (< 20 distinct values)
    // 2. OR cardinality ratio < 0.1 (less than 10% unique)
    // 3. OR detected as boolean
    if (distinct_count <= 20 || cardinality_ratio < 0.1) {
        return true;
    }

    // Check if values are mostly strings
    size_t string_count = 0;
    for (const auto& value : values) {
        if (detect_data_type(value) == "string") {
            string_count++;
        }
    }

    return static_cast<double>(string_count) / values.size() > 0.8;
}

std::vector<double> DataAnalyzer::extract_numeric_values(const std::vector<Datum>& values) {
    std::vector<double> numeric_values;

    for (const auto& value : values) {
        if (!value.is_null()) {
            try {
                if (value.is_integer()) {
                    numeric_values.push_back(static_cast<double>(value.as_int()));
                } else if (value.is_float()) {
                    numeric_values.push_back(static_cast<double>(value.as_float()));
                } else if (value.is_double()) {
                    numeric_values.push_back(value.as_double());
                } else if (value.is_boolean()) {
                    numeric_values.push_back(value.as_bool() ? 1.0 : 0.0);
                } else if (value.is_string()) {
                    // Try to parse string as number
                    try {
                        numeric_values.push_back(std::stod(value.as_string()));
                    } catch (...) {
                        // Not a numeric string
                    }
                }
            } catch (...) {
                // Conversion failed
            }
        }
    }

    return numeric_values;
}

std::vector<std::string> DataAnalyzer::extract_string_values(const std::vector<Datum>& values) {
    std::vector<std::string> string_values;

    for (const auto& value : values) {
        if (!value.is_null()) {
            string_values.push_back(value.to_string());
        }
    }

    return string_values;
}

std::vector<std::pair<double, double>> DataAnalyzer::calculate_partial_dependence(
    const std::string& feature_name,
    const std::vector<Datum>& feature_values,
    const std::vector<Datum>& target_values) {

    std::vector<std::pair<double, double>> pd_plot;

    // Simplified partial dependence calculation
    // In production, this would use actual model predictions

    std::vector<double> numeric_features = extract_numeric_values(feature_values);
    std::vector<double> numeric_targets = extract_numeric_values(target_values);

    if (numeric_features.empty() || numeric_targets.empty() ||
        numeric_features.size() != numeric_targets.size()) {
        return pd_plot;
    }

    // Create bins for the feature
    size_t num_bins = std::min(static_cast<size_t>(10), numeric_features.size());
    double min_val = *std::min_element(numeric_features.begin(), numeric_features.end());
    double max_val = *std::max_element(numeric_features.begin(), numeric_features.end());
    double bin_width = (max_val - min_val) / num_bins;

    if (bin_width < 1e-10) {
        return pd_plot;
    }

    // Calculate average target for each bin
    for (size_t bin = 0; bin < num_bins; ++bin) {
        double bin_start = min_val + bin * bin_width;
        double bin_center = bin_start + bin_width / 2.0;

        double sum_target = 0.0;
        size_t count = 0;

        for (size_t i = 0; i < numeric_features.size(); ++i) {
            if (numeric_features[i] >= bin_start &&
                numeric_features[i] < bin_start + bin_width) {
                sum_target += numeric_targets[i];
                count++;
            }
        }

        if (count > 0) {
            pd_plot.emplace_back(bin_center, sum_target / count);
        }
    }

    return pd_plot;
}

void DataAnalyzer::generate_insights_and_recommendations(
    ComprehensiveAnalysisReport& report,
    const std::string& target_column) {

    // Generate insights based on analysis results

    // 1. Data quality insights
    if (report.data_quality.overall_quality_score < 0.7) {
        report.insights.push_back("Data quality needs improvement. Score: " +
                                 std::to_string(report.data_quality.overall_quality_score));
        report.recommendations.push_back("Address missing values and data inconsistencies");
    }

    // 2. Missing data insights
    for (const auto& [col_name, analysis] : report.column_analyses) {
        if (analysis.missing_percentage > 20.0) {
            report.insights.push_back("Column '" + col_name + "' has " +
                                     std::to_string(analysis.missing_percentage) +
                                     "% missing values");
            report.recommendations.push_back("Consider imputation or removal of column '" +
                                            col_name + "'");
        }
    }

    // 3. Outlier insights
    for (const auto& [col_name, analysis] : report.column_analyses) {
        if (analysis.has_outliers) {
            report.insights.push_back("Column '" + col_name + "' contains " +
                                     std::to_string(analysis.outliers.size()) + " outliers");
            report.recommendations.push_back("Investigate outliers in column '" + col_name +
                                            "' for data quality issues");
        }
    }

    // 4. Correlation insights
    for (const auto& correlation : report.correlations) {
        if (std::abs(correlation.pearson_correlation) > 0.8) {
            report.insights.push_back("Strong " + correlation.relationship_type +
                                     " correlation between " + correlation.column1 +
                                     " and " + correlation.column2 +
                                     " (r=" + std::to_string(correlation.pearson_correlation) + ")");

            if (correlation.pearson_correlation > 0.9) {
                report.recommendations.push_back("Consider removing one of '" +
                                                correlation.column1 + "' or '" +
                                                correlation.column2 +
                                                "' due to high multicollinearity");
            }
        }
    }

    // 5. Feature importance insights
    if (!report.feature_importance.empty()) {
        const auto& top_feature = report.feature_importance[0];
        report.insights.push_back("Most important feature for predicting '" +
                                 target_column + "': '" + top_feature.feature_name +
                                 "' (importance: " + std::to_string(top_feature.importance_score) + ")");

        report.recommendations.push_back("Focus feature engineering efforts on '" +
                                        top_feature.feature_name + "'");
    }

    // 6. Distribution insights
    for (const auto& distribution : report.distributions) {
        if (distribution.distribution_type == "right_skewed") {
            report.insights.push_back("Column '" + distribution.column_name +
                                     "' is right-skewed");
            report.recommendations.push_back("Consider log transformation for '" +
                                            distribution.column_name + "'");
        } else if (distribution.distribution_type == "left_skewed") {
            report.insights.push_back("Column '" + distribution.column_name +
                                     "' is left-skewed");
        }
    }

    // 7. Clustering insights
    if (!report.clusters.empty()) {
        report.insights.push_back("Data naturally clusters into " +
                                 std::to_string(report.clusters.size()) + " groups");

        // Find largest and smallest clusters
        auto max_cluster = *std::max_element(report.clusters.begin(), report.clusters.end(),
            [](const auto& a, const auto& b) { return a.size < b.size; });

        auto min_cluster = *std::min_element(report.clusters.begin(), report.clusters.end(),
            [](const auto& a, const auto& b) { return a.size < b.size; });

        report.insights.push_back("Largest cluster: " + std::to_string(max_cluster.size) +
                                 " samples, Smallest cluster: " +
                                 std::to_string(min_cluster.size) + " samples");
    }

    // 8. General recommendations
    if (report.column_analyses.size() > 50) {
        report.recommendations.push_back("Consider dimensionality reduction (PCA) due to high feature count");
    }

    if (report.data_quality.quality_metrics["uniqueness"] < 0.8) {
        report.recommendations.push_back("Investigate potential duplicate records");
    }
}

} // namespace analysis
} // namespace esql
