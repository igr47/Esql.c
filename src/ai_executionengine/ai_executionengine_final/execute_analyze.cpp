// ============================================
// ExecuteAnalyze
// ============================================
#include "ai_execution_engine_final.h"
#include "database.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <nlohmann/json.hpp>

// ============================================
// Enhanced executeAnalyzeData
// ============================================
ExecutionEngine::ResultSet AIExecutionEngineFinal::executeAnalyzeData(
    AST::AnalyzeDataStatement& stmt) {

    std::cout << "[AIExecutionEngineFinal] Executing comprehensive ANALYZE DATA: " << stmt.table_name << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();
    ExecutionEngine::ResultSet result;

    try {
        // 1. Extract data from table
        std::cout << "[AIExecutionEngineFinal] Extracting data from table: " << stmt.table_name << std::endl;

        auto table_data = data_extractor_->extract_table_data(
            db_.currentDatabase(),
            stmt.table_name,
            {} // All columns
        );

        if (table_data.empty()) {
            throw std::runtime_error("Table is empty or doesn't exist");
        }

        std::cout << "[AIExecutionEngineFinal] Extracted " << table_data.size()  << " rows for analysis" << std::endl;

        // 2. Perform comprehensive analysis
        esql::analysis::DataAnalyzer analyzer;
        auto analysis_report = analyzer.analyze_data(
            table_data,
            stmt.target_column,
            stmt.feature_columns,
            stmt.analysis_type
        );

        // 3. Format results based on analysis type and output format
        if (stmt.output_format == "JSON") {
            // Return as JSON
            result.columns = {"analysis_report"};

            nlohmann::json report_json;
            report_json["table_name"] = stmt.table_name;
            report_json["analysis_type"] = stmt.analysis_type;
            report_json["timestamp"] = std::chrono::system_clock::to_time_t(
                std::chrono::system_clock::now()
            );
            report_json["row_count"] = table_data.size();

            // Add column analyses
            nlohmann::json columns_json;
            for (const auto& [col_name, col_analysis] : analysis_report.column_analyses) {
                columns_json[col_name] = col_analysis.to_json();
            }
            report_json["column_analyses"] = columns_json;

            // Add correlations
            nlohmann::json correlations_json = nlohmann::json::array();
            for (const auto& corr : analysis_report.correlations) {
                nlohmann::json corr_json;
                corr_json["column1"] = corr.column1;
                corr_json["column2"] = corr.column2;
                corr_json["pearson"] = corr.pearson_correlation;
                corr_json["spearman"] = corr.spearman_correlation;
                corr_json["relationship"] = corr.relationship_type;
                corr_json["significant"] = corr.is_significant;
                correlations_json.push_back(corr_json);
            }
            report_json["correlations"] = correlations_json;

            // Add feature importance
            if (!analysis_report.feature_importance.empty()) {
                nlohmann::json importance_json = nlohmann::json::array();
                for (const auto& importance : analysis_report.feature_importance) {
                    nlohmann::json imp_json;
                    imp_json["feature"] = importance.feature_name;
                    imp_json["importance_score"] = importance.importance_score;
                    imp_json["mutual_information"] = importance.mutual_information;
                    importance_json.push_back(imp_json);
                }
                report_json["feature_importance"] = importance_json;
            }

            // Add data quality
            nlohmann::json quality_json;
            quality_json["overall_score"] = analysis_report.data_quality.overall_quality_score;
            quality_json["metrics"] = analysis_report.data_quality.quality_metrics;
            quality_json["issues"] = analysis_report.data_quality.quality_issues;
            report_json["data_quality"] = quality_json;

            // Add insights and recommendations
            report_json["insights"] = analysis_report.insights;
            report_json["recommendations"] = analysis_report.recommendations;

            result.rows.push_back({report_json.dump(2)});

        } else if (stmt.output_format == "MARKDOWN") {
            // Return as markdown report
            result.columns = {"analysis_report"};

            std::stringstream markdown;
            markdown << "# Data Analysis Report\n\n";
            markdown << "**Table:** " << stmt.table_name << "\n";
            markdown << "**Analysis Type:** " << stmt.analysis_type << "\n";
            markdown << "**Rows Analyzed:** " << table_data.size() << "\n";
            markdown << "**Timestamp:** "
                    << std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())
                    << "\n\n";

            markdown << "## Data Quality Assessment\n";
            markdown << "- **Overall Quality Score:** " << std::fixed << std::setprecision(2)
                    << analysis_report.data_quality.overall_quality_score * 100 << "%\n";

            for (const auto& [metric, score] : analysis_report.data_quality.quality_metrics) {
                markdown << "- **" << metric << ":** " << std::setprecision(2) << score * 100 << "%\n";
            }

            if (!analysis_report.data_quality.quality_issues.empty()) {
                markdown << "\n### Quality Issues\n";
                for (const auto& issue : analysis_report.data_quality.quality_issues) {
                    markdown << "- " << issue << "\n";
                }
            }

            markdown << "\n## Column Analysis\n";
            for (const auto& [col_name, col_analysis] : analysis_report.column_analyses) {
                markdown << "### " << col_name << "\n";
                markdown << "- **Type:** " << col_analysis.detected_type << "\n";
                markdown << "- **Missing:** " << std::setprecision(1)
                        << col_analysis.missing_percentage << "%\n";
                markdown << "- **Distinct:** " << col_analysis.distinct_count << "\n";

                if (col_analysis.detected_type == "numeric") {
                    markdown << "- **Range:** [" << col_analysis.min_value << ", "
                            << col_analysis.max_value << "]\n";
                    markdown << "- **Mean:** " << col_analysis.mean << "\n";
                    markdown << "- **Std Dev:** " << col_analysis.std_dev << "\n";

                    if (col_analysis.has_outliers) {
                        markdown << "- **Outliers:** " << col_analysis.outliers.size() << " detected\n";
                    }
                }

                markdown << "\n";
            }

            if (!analysis_report.correlations.empty()) {
                markdown << "\n## Correlation Analysis\n";
                markdown << "| Feature 1 | Feature 2 | Pearson | Spearman | Relationship |\n";
                markdown << "|-----------|-----------|---------|----------|--------------|\n";

                for (const auto& corr : analysis_report.correlations) {
                    if (std::abs(corr.pearson_correlation) > 0.3) { // Show moderate+ correlations
                        markdown << "| " << corr.column1
                                << " | " << corr.column2
                                << " | " << std::fixed << std::setprecision(3) << corr.pearson_correlation
                                << " | " << corr.spearman_correlation
                                << " | " << corr.relationship_type << " |\n";
                    }
                }
            }

            if (!analysis_report.feature_importance.empty() && !stmt.target_column.empty()) {
                markdown << "\n## Feature Importance for Target: " << stmt.target_column << "\n";
                markdown << "| Feature | Importance Score | Mutual Information |\n";
                markdown << "|---------|-----------------|-------------------|\n";

                for (size_t i = 0; i < std::min(analysis_report.feature_importance.size(), (size_t)10); ++i) {
                    const auto& importance = analysis_report.feature_importance[i];
                    markdown << "| " << importance.feature_name
                            << " | " << std::fixed << std::setprecision(4) << importance.importance_score
                            << " | " << importance.mutual_information << " |\n";
                }
            }

            if (!analysis_report.clusters.empty()) {
                markdown << "\n## Clustering Analysis\n";
                markdown << "Found " << analysis_report.clusters.size() << " natural clusters\n\n";

                for (const auto& cluster : analysis_report.clusters) {
                    markdown << "### Cluster " << cluster.cluster_id << "\n";
                    markdown << "- **Size:** " << cluster.size << " samples ("
                            << std::setprecision(1)
                            << (static_cast<double>(cluster.size) / table_data.size() * 100)
                            << "% of data)\n";

                    if (!cluster.top_features.empty()) {
                        markdown << "- **Top Features:** ";
                        for (size_t i = 0; i < cluster.top_features.size(); ++i) {
                            if (i > 0) markdown << ", ";
                            markdown << cluster.top_features[i];
                        }
                        markdown << "\n";
                    }
                    markdown << "\n";
                }
            }

            if (!analysis_report.outliers.empty() && !analysis_report.outliers[0].outlier_indices.empty()) {
                markdown << "\n## Outlier Detection\n";
                markdown << "- **Method:** " << analysis_report.outliers[0].detection_method << "\n";
                markdown << "- **Contamination Rate:** "
                        << std::setprecision(2)
                        << analysis_report.outliers[0].contamination_rate * 100 << "%\n";
                markdown << "- **Outliers Detected:** "
                        << analysis_report.outliers[0].outlier_indices.size() << "\n";

                if (!analysis_report.outliers[0].affected_columns.empty()) {
                    markdown << "- **Affected Columns:** ";
                    for (size_t i = 0; i < analysis_report.outliers[0].affected_columns.size(); ++i) {
                        if (i > 0) markdown << ", ";
                        markdown << analysis_report.outliers[0].affected_columns[i];
                    }
                    markdown << "\n";
                }
            }

            if (!analysis_report.insights.empty()) {
                markdown << "\n## Key Insights\n";
                for (const auto& insight : analysis_report.insights) {
                    markdown << "- " << insight << "\n";
                }
            }

            if (!analysis_report.recommendations.empty()) {
                markdown << "\n## Recommendations\n";
                for (const auto& recommendation : analysis_report.recommendations) {
                    markdown << "- " << recommendation << "\n";
                }
            }

            result.rows.push_back({markdown.str()});

        } else {
            // Default: TABLE format with summary
            if (stmt.analysis_type == "CORRELATION") {
                result.columns = {"feature1", "feature2", "pearson_correlation",
                                 "spearman_correlation", "relationship", "significant"};

                for (const auto& corr : analysis_report.correlations) {
                    if (std::abs(corr.pearson_correlation) > 0.3) {
                        std::vector<std::string> row;
                        row.push_back(corr.column1);
                        row.push_back(corr.column2);
                        row.push_back(std::to_string(corr.pearson_correlation));
                        row.push_back(std::to_string(corr.spearman_correlation));
                        row.push_back(corr.relationship_type);
                        row.push_back(corr.is_significant ? "Yes" : "No");
                        result.rows.push_back(row);
                    }
                }

            } else if (stmt.analysis_type == "IMPORTANCE") {
                result.columns = {"feature", "importance_score", "mutual_information",
                                 "data_type", "missing_pct"};

                for (const auto& importance : analysis_report.feature_importance) {
                    // Find corresponding column analysis
                    auto col_it = analysis_report.column_analyses.find(importance.feature_name);
                    if (col_it != analysis_report.column_analyses.end()) {
                        std::vector<std::string> row;
                        row.push_back(importance.feature_name);
                        row.push_back(std::to_string(importance.importance_score));
                        row.push_back(std::to_string(importance.mutual_information));
                        row.push_back(col_it->second.detected_type);
                        row.push_back(std::to_string(col_it->second.missing_percentage) + "%");
                        result.rows.push_back(row);
                    }
                }

            } else if (stmt.analysis_type == "CLUSTERING") {
                result.columns = {"cluster_id", "size", "percentage", "top_features"};

                for (const auto& cluster : analysis_report.clusters) {
                    std::vector<std::string> row;
                    row.push_back(std::to_string(cluster.cluster_id));
                    row.push_back(std::to_string(cluster.size));

                    double percentage = (static_cast<double>(cluster.size) / table_data.size()) * 100;
                    row.push_back(std::to_string(percentage) + "%");

                    std::stringstream features_ss;
                    for (size_t i = 0; i < std::min(cluster.top_features.size(), (size_t)3); ++i) {
                        if (i > 0) features_ss << ", ";
                        features_ss << cluster.top_features[i];
                    }
                    row.push_back(features_ss.str());

                    result.rows.push_back(row);
                }

            } else if (stmt.analysis_type == "SUMMARY") {
                result.columns = {"column", "type", "total", "nulls", "missing_pct",
                                 "distinct", "mean", "std_dev", "min", "max", "has_outliers"};

                for (const auto& [col_name, col_analysis] : analysis_report.column_analyses) {
                    std::vector<std::string> row;
                    row.push_back(col_name);
                    row.push_back(col_analysis.detected_type);
                    row.push_back(std::to_string(col_analysis.total_count));
                    row.push_back(std::to_string(col_analysis.null_count));
                    row.push_back(std::to_string(col_analysis.missing_percentage) + "%");
                    row.push_back(std::to_string(col_analysis.distinct_count));

                    if (col_analysis.detected_type == "numeric") {
                        row.push_back(std::to_string(col_analysis.mean));
                        row.push_back(std::to_string(col_analysis.std_dev));
                        row.push_back(std::to_string(col_analysis.min_value));
                        row.push_back(std::to_string(col_analysis.max_value));
                        row.push_back(col_analysis.has_outliers ? "Yes" : "No");
                    } else {
                        row.push_back("N/A");
                        row.push_back("N/A");
                        row.push_back("N/A");
                        row.push_back("N/A");
                        row.push_back("N/A");
                    }

                    result.rows.push_back(row);
                }

            } else if (stmt.analysis_type == "QUALITY") {
                result.columns = {"metric", "score", "status"};

                for (const auto& [metric, score] : analysis_report.data_quality.quality_metrics) {
                    std::vector<std::string> row;
                    row.push_back(metric);
                    row.push_back(std::to_string(score * 100) + "%");

                    if (score >= 0.9) row.push_back("Excellent");
                    else if (score >= 0.7) row.push_back("Good");
                    else if (score >= 0.5) row.push_back("Fair");
                    else row.push_back("Poor");

                    result.rows.push_back(row);
                }

                // Add overall score
                std::vector<std::string> overall_row;
                overall_row.push_back("OVERALL");
                overall_row.push_back(std::to_string(analysis_report.data_quality.overall_quality_score * 100) + "%");

                if (analysis_report.data_quality.overall_quality_score >= 0.9)
                    overall_row.push_back("Excellent");
                else if (analysis_report.data_quality.overall_quality_score >= 0.7)
                    overall_row.push_back("Good");
                else if (analysis_report.data_quality.overall_quality_score >= 0.5)
                    overall_row.push_back("Fair");
                else
                    overall_row.push_back("Poor");

                result.rows.push_back(overall_row);

            } else {
                // COMPREHENSIVE analysis - show insights
                result.columns = {"type", "message", "severity"};

                // Add data quality insights
                if (analysis_report.data_quality.overall_quality_score < 0.7) {
                    result.rows.push_back({"QUALITY", "Data quality needs improvement (score: " + std::to_string(analysis_report.data_quality.overall_quality_score * 100) + "%)","WARNING"});
                }

                // Add missing data insights
                for (const auto& [col_name, col_analysis] : analysis_report.column_analyses) {
                    if (col_analysis.missing_percentage > 20.0) {
                        result.rows.push_back({"MISSING_DATA","Column '" + col_name + "' has " + std::to_string(col_analysis.missing_percentage) + "% missing values","WARNING"});
                    }
                }

                // Add correlation insights
                for (const auto& corr : analysis_report.correlations) {
                    if (std::abs(corr.pearson_correlation) > 0.8) {
                        result.rows.push_back({"CORRELATION","Strong correlation between " + corr.column1 + " and " + corr.column2 + " (r=" + std::to_string(corr.pearson_correlation) + ")","INFO"});
                    }
                }

                // Add feature importance insights
                if (!analysis_report.feature_importance.empty() && !stmt.target_column.empty()) {
                    const auto& top_feature = analysis_report.feature_importance[0];
                    result.rows.push_back({"FEATURE_IMPORTANCE","Top feature for predicting '" + stmt.target_column + "': '" + top_feature.feature_name + "'","INFO"});
                }

                // Add outlier insights
                for (const auto& [col_name, col_analysis] : analysis_report.column_analyses) {
                    if (col_analysis.has_outliers) {
                        result.rows.push_back({"OUTLIERS","Column '" + col_name + "' has " + std::to_string(col_analysis.outliers.size()) + " outliers","WARNING"});
                    }
                }
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time
        );

        std::cout << "[AIExecutionEngineFinal] Analysis completed in " << duration.count() << "ms" << std::endl;

        // Log successful operation
        logAIOperation("ANALYZE_DATA", stmt.table_name, "SUCCESS", "Type: " + stmt.analysis_type + ", Format: " + stmt.output_format +  ", Duration: " + std::to_string(duration.count()) + "ms");

    } catch (const std::exception& e) {
        logAIOperation("ANALYZE_DATA", stmt.table_name, "FAILED", e.what());

        result.columns = {"error"};
        result.rows.push_back({std::string("ERROR: ") + e.what()});
    }

    return result;
}

