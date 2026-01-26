// ============================================
// Enhanced executeAnalyzeData - Production grade
// ============================================
#include "ai_execution_engine_final.h"
#include "data_analysis.h"
#include "database.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <thread>
#include <future>

// Add namespace alias for clarity
//using esql::Datum;

ExecutionEngine::ResultSet AIExecutionEngineFinal::executeAnalyzeData(
    AST::AnalyzeDataStatement& stmt) {

    auto start_time = std::chrono::high_resolution_clock::now();
    ExecutionEngine::ResultSet result;
    
    try {
        // Validate input
        if (stmt.table_name.empty()) {
            throw std::runtime_error("Table name is required for analysis");
        }
        
        std::cout << "[AIExecutionEngineFinal] Starting professional analysis of table: " 
                  << stmt.table_name << std::endl;
        
        // Configuration
        esql::analysis::ProfessionalDataAnalyzer::Config config;
        config.max_rows_for_detailed_analysis = 1000000;
        config.enable_parallel_processing = true;
        config.num_threads = std::thread::hardware_concurrency();
        config.output_format = stmt.output_format;
        config.generate_visualizations = true;
        
        // Set analysis type
        std::string analysis_type = stmt.analysis_type;
        if (analysis_type.empty()) {
            analysis_type = "COMPREHENSIVE";
        }
        
        // Convert options
        std::map<std::string, std::string> options;
        for (const auto& [key, value] : stmt.options) {
            options[key] = value;
        }
        
        // Check for output format specific options
        if (stmt.output_format == "JSON" || stmt.output_format == "MARKDOWN") {
            options["detailed"] = "true";
        }
        
        // Get data extractor
        if (!data_extractor_) {
            throw std::runtime_error("Data extractor not initialized");
        }
        
        // Extract data with proper chunking for large tables
        std::cout << "[AIExecutionEngineFinal] Extracting data from table: " 
                  << stmt.table_name << std::endl;
        
        size_t total_rows = 0;
        std::vector<std::unordered_map<std::string, esql::Datum>> table_data;
        
        // Use cursor-based extraction for large tables
        auto cursor = data_extractor_->create_cursor(
            db_.currentDatabase(),
            stmt.table_name,
            stmt.feature_columns.empty() ? std::vector<std::string>{} : stmt.feature_columns
        );
        
        if (cursor) {
            size_t chunk_size = 10000;
            std::cout << "[AIExecutionEngineFinal] Using cursor-based extraction with chunk size: " 
                      << chunk_size << std::endl;
            
            while (cursor->has_next()) {
                auto chunk = extract_chunk(*cursor, chunk_size);
                table_data.insert(table_data.end(), 
                                 std::make_move_iterator(chunk.begin()),
                                 std::make_move_iterator(chunk.end()));
                total_rows += chunk.size();
                
                if (total_rows % 100000 == 0) {
                    std::cout << "[AIExecutionEngineFinal] Extracted " << total_rows 
                              << " rows so far..." << std::endl;
                }
                
                // Early exit for very large tables (sampling)
                if (total_rows > 1000000 && stmt.options.find("sample_large_tables") != stmt.options.end()) {
                    std::cout << "[AIExecutionEngineFinal] Large table detected (" << total_rows 
                              << " rows). Switching to sampling mode." << std::endl;
                    break;
                }
            }
        } else {
            // Fallback to regular extraction
            auto fallback_data = data_extractor_->extract_table_data(
                db_.currentDatabase(),
                stmt.table_name,
                stmt.feature_columns.empty() ? std::vector<std::string>{} : stmt.feature_columns
            );
            table_data = std::move(fallback_data);
            total_rows = table_data.size();
        }
        
        if (table_data.empty()) {
            throw std::runtime_error("Table is empty or doesn't exist");
        }
        
        std::cout << "[AIExecutionEngineFinal] Extracted " << total_rows 
                  << " rows for analysis" << std::endl;
        
        // Create professional analyzer
        esql::analysis::ProfessionalDataAnalyzer analyzer(config);
        
        // Perform analysis
        std::cout << "[AIExecutionEngineFinal] Starting professional analysis..." << std::endl;
        
        auto analysis_report = analyzer.analyze_data(
            table_data,
            stmt.target_column,
            stmt.feature_columns,
            analysis_type,
            options
        );
        
        // Format results
        if (stmt.output_format == "JSON") {
            result.columns = {"analysis_report"};
            
            nlohmann::json report_json = analysis_report.to_json(true);
            
            // Add metadata
            report_json["metadata"]["table_name"] = stmt.table_name;
            report_json["metadata"]["analysis_type"] = analysis_type;
            report_json["metadata"]["target_column"] = stmt.target_column;
            report_json["metadata"]["row_count"] = total_rows;
            report_json["metadata"]["column_count"] = table_data.empty() ? 0 : table_data[0].size();
            report_json["metadata"]["analysis_timestamp"] = 
                std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            report_json["metadata"]["processing_time_ms"] = 
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - start_time
                ).count();
            
            result.rows.push_back({report_json.dump(2)});
            
        } else if (stmt.output_format == "MARKDOWN") {
            result.columns = {"analysis_report"};
            
            std::string markdown_report = analysis_report.to_markdown_report(true);
            result.rows.push_back({markdown_report});
            
        } else if (stmt.output_format == "HTML") {
            result.columns = {"analysis_report"};
            
            std::string html_report = analysis_report.to_html_report(true);
            result.rows.push_back({html_report});
            
        } else {
            // TABLE format - Show detailed results based on analysis type
            format_table_output(result, analysis_report, analysis_type, stmt);
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "[AIExecutionEngineFinal] Professional analysis completed in " 
                  << duration.count() << "ms" << std::endl;
        
        // Log successful operation
        logAIOperation("ANALYZE_DATA", stmt.table_name, "SUCCESS", 
                      "Type: " + analysis_type + 
                      ", Format: " + stmt.output_format + 
                      ", Rows: " + std::to_string(total_rows) + 
                      ", Duration: " + std::to_string(duration.count()) + "ms");
        
        // Save report to file if requested
        if (stmt.options.find("save_report") != stmt.options.end()) {
            std::string filename = stmt.options.at("save_report");
            if (filename.empty()) {
                filename = "analysis_report_" + stmt.table_name + "_" + 
                          std::to_string(std::chrono::system_clock::to_time_t(
                              std::chrono::system_clock::now())) + ".json";
            }
            analysis_report.save_to_file(filename, stmt.output_format);
            std::cout << "[AIExecutionEngineFinal] Report saved to: " << filename << std::endl;
        }
        
    } catch (const std::exception& e) {
        logAIOperation("ANALYZE_DATA", stmt.table_name, "FAILED", e.what());
        
        result.columns = {"error"};
        result.rows.push_back({std::string("ERROR: ") + e.what()});
    }
    
    return result;
}

// Helper function to extract chunks using cursor
std::vector<std::unordered_map<std::string, esql::Datum>> 
AIExecutionEngineFinal::extract_chunk(esql::DataExtractor::DataCursor& cursor, size_t chunk_size) {
    std::vector<std::unordered_map<std::string, esql::Datum>> chunk;
    chunk.reserve(chunk_size);
    
    for (size_t i = 0; i < chunk_size && cursor.has_next(); ++i) {
        chunk.push_back(cursor.next());
    }
    
    return chunk;
}

// Format table output based on analysis type
void AIExecutionEngineFinal::format_table_output(
    ExecutionEngine::ResultSet& result,
    const esql::analysis::ProfessionalComprehensiveAnalysisReport& report,
    const std::string& analysis_type,
    const AST::AnalyzeDataStatement& stmt) {
    
    if (analysis_type == "SUMMARY" || analysis_type.empty()) {
        format_summary_output(result, report);
        
    } else if (analysis_type == "CORRELATION") {
        format_correlation_output(result, report);
        
    } else if (analysis_type == "IMPORTANCE") {
        format_importance_output(result, report, stmt.target_column);
        
    } else if (analysis_type == "CLUSTERING") {
        format_clustering_output(result, report);
        
    } else if (analysis_type == "OUTLIER") {
        format_outlier_output(result, report);
        
    } else if (analysis_type == "DISTRIBUTION") {
        format_distribution_output(result, report);
        
    } else if (analysis_type == "QUALITY") {
        format_quality_output(result, report);
        
    } else if (analysis_type == "TIMESERIES") {
        format_timeseries_output(result, report);
        
    } else if (analysis_type == "COMPREHENSIVE") {
        format_comprehensive_output(result, report);
        
    } else {
        // Default to insights
        format_insights_output(result, report);
    }
}

// Format summary output
void AIExecutionEngineFinal::format_summary_output(
    ExecutionEngine::ResultSet& result,
    const esql::analysis::ProfessionalComprehensiveAnalysisReport& report) {
    
    result.columns = {
        "column", "type", "total", "nulls", "null_pct", "distinct", 
        "mean", "std_dev", "min", "q1", "median", "q3", "max", 
        "skewness", "kurtosis", "has_outliers", "data_quality"
    };
    
    for (const auto& entry : report.column_analyses) {
        const auto& col_name = entry.first;
        const auto& col_analysis = entry.second;
        
        std::vector<std::string> row;
        row.push_back(col_name);
        row.push_back(col_analysis.detected_type);
        row.push_back(std::to_string(col_analysis.total_count));
        row.push_back(std::to_string(col_analysis.null_count));
        row.push_back(std::to_string(col_analysis.missing_percentage) + "%");
        row.push_back(std::to_string(col_analysis.distinct_count));
        
        if (col_analysis.detected_type == "numeric" ||
            col_analysis.detected_type == "integer" ||
            col_analysis.detected_type == "float" ||
            col_analysis.detected_type == "double") {
            
            row.push_back(std::to_string(col_analysis.mean));
            row.push_back(std::to_string(col_analysis.std_dev));
            row.push_back(std::to_string(col_analysis.min_value));
            row.push_back(std::to_string(col_analysis.q1));
            row.push_back(std::to_string(col_analysis.median));
            row.push_back(std::to_string(col_analysis.q3));
            row.push_back(std::to_string(col_analysis.max_value));
            row.push_back(std::to_string(col_analysis.skewness));
            row.push_back(std::to_string(col_analysis.kurtosis));
            row.push_back(col_analysis.has_outliers ? "Yes" : "No");
            row.push_back(std::to_string(col_analysis.quality.overall_score * 100) + "%");
        } else {
            // For non-numeric columns
            row.push_back("N/A");
            row.push_back("N/A");
            row.push_back("N/A");
            row.push_back("N/A");
            row.push_back("N/A");
            row.push_back("N/A");
            row.push_back("N/A");
            row.push_back("N/A");
            row.push_back("N/A");
            row.push_back("N/A");
            row.push_back(std::to_string(col_analysis.quality.overall_score * 100) + "%");
        }
        
        result.rows.push_back(row);
    }
}

// Format correlation output
void AIExecutionEngineFinal::format_correlation_output(
    ExecutionEngine::ResultSet& result,
    const esql::analysis::ProfessionalComprehensiveAnalysisReport& report) {
    
    result.columns = {
        "feature1", "feature2", "pearson_r", "pearson_p", 
        "spearman_rho", "spearman_p", "relationship", "strength", 
        "significant", "effect_size"
    };
    
    for (const auto& corr : report.correlations) {
        // Only show moderate to strong correlations
        if (std::abs(corr.pearson_r) > 0.3 || corr.is_statistically_significant) {
            std::vector<std::string> row;
            row.push_back(corr.column1);
            row.push_back(corr.column2);
            row.push_back(std::to_string(corr.pearson_r));
            row.push_back(std::to_string(corr.pearson_p_value));
            row.push_back(std::to_string(corr.spearman_rho));
            row.push_back(std::to_string(corr.spearman_p_value));
            row.push_back(corr.relationship_direction);
            row.push_back(corr.relationship_strength);
            row.push_back(corr.is_statistically_significant ? "Yes" : "No");
            row.push_back(std::to_string(corr.effect_size));
            
            result.rows.push_back(row);
        }
    }
    
    // If no correlations found, add a message
    if (result.rows.empty()) {
        result.columns = {"message"};
        result.rows.push_back({"No significant correlations found (|r| > 0.3)"});
    }
}

// Format feature importance output
void AIExecutionEngineFinal::format_importance_output(
    ExecutionEngine::ResultSet& result,
    const esql::analysis::ProfessionalComprehensiveAnalysisReport& report,
    const std::string& target_column) {
    
    result.columns = {
        "rank", "feature", "type", "importance_score", 
        "shap_value", "mutual_info", "p_value", 
        "stability", "recommendation"
    };
    
    // Sort features by importance
    std::vector<std::pair<std::string, double>> sorted_features;
    for (const auto& importance : report.feature_importance) {
        sorted_features.emplace_back(importance.feature_name, 
                                    importance.scores.random_forest);
    }
    
    std::sort(sorted_features.begin(), sorted_features.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    size_t rank = 1;
    for (const auto& feature_pair : sorted_features) {
        const std::string& feature_name = feature_pair.first;
        double importance_score = feature_pair.second;
        
        // Find the full importance analysis
        auto it = std::find_if(report.feature_importance.begin(),
                              report.feature_importance.end(),
                              [&](const auto& imp) { return imp.feature_name == feature_name; });
        
        if (it != report.feature_importance.end()) {
            std::vector<std::string> row;
            row.push_back(std::to_string(rank++));
            row.push_back(feature_name);
            
            // Find column type
            std::string feature_type = "unknown";
            auto col_it = report.column_analyses.find(feature_name);
            if (col_it != report.column_analyses.end()) {
                feature_type = col_it->second.detected_type;
            }
            row.push_back(feature_type);
            
            row.push_back(std::to_string(importance_score));
            row.push_back(std::to_string(it->shapley.mean_abs_shap));
            row.push_back(std::to_string(it->scores.mutual_information));
            row.push_back(std::to_string(it->significance.best_p_value));
            row.push_back(std::to_string(it->stability.rank_stability));
            
            // Recommendation
            std::string recommendation;
            switch (it->recommended_action) {
                case esql::analysis::ProfessionalFeatureImportance::KEEP_AS_IS:
                    recommendation = "Keep as is";
                    break;
                case esql::analysis::ProfessionalFeatureImportance::TRANSFORM:
                    recommendation = "Transform";
                    break;
                case esql::analysis::ProfessionalFeatureImportance::CREATE_INTERACTION:
                    recommendation = "Create interaction";
                    break;
                case esql::analysis::ProfessionalFeatureImportance::BIN:
                    recommendation = "Bin";
                    break;
                case esql::analysis::ProfessionalFeatureImportance::ENCODE:
                    recommendation = "Encode";
                    break;
                case esql::analysis::ProfessionalFeatureImportance::DROP:
                    recommendation = "Drop";
                    break;
                case esql::analysis::ProfessionalFeatureImportance::MONITOR:
                    recommendation = "Monitor";
                    break;
                default:
                    recommendation = "No recommendation";
            }
            row.push_back(recommendation);
            
            result.rows.push_back(row);
        }
    }
}

// Format clustering output
void AIExecutionEngineFinal::format_clustering_output(
    ExecutionEngine::ResultSet& result,
    const esql::analysis::ProfessionalComprehensiveAnalysisReport& report) {
    
    result.columns = {
        "cluster_id", "size", "percentage", "silhouette", 
        "davies_bouldin", "key_features", "characteristics"
    };
    
    for (const auto& cluster : report.clusters) {
        std::vector<std::string> row;
        row.push_back(std::to_string(cluster.cluster_id));
        row.push_back(std::to_string(cluster.size));
        row.push_back(std::to_string(cluster.size_percentage) + "%");
        row.push_back(std::to_string(cluster.silhouette_score));
        row.push_back(std::to_string(cluster.davies_bouldin_index));
        
        // Key features
        std::stringstream features_ss;
        for (size_t i = 0; i < std::min(cluster.defining_features.size(), (size_t)3); ++i) {
            if (i > 0) features_ss << ", ";
            features_ss << cluster.defining_features[i];
        }
        row.push_back(features_ss.str());
        
        // Characteristics
        std::stringstream chars_ss;
        for (size_t i = 0; i < std::min(cluster.key_characteristics.size(), (size_t)2); ++i) {
            if (i > 0) chars_ss << "; ";
            chars_ss << cluster.key_characteristics[i];
        }
        row.push_back(chars_ss.str());
        
        result.rows.push_back(row);
    }
}

// Format outlier output
void AIExecutionEngineFinal::format_outlier_output(
    ExecutionEngine::ResultSet& result,
    const esql::analysis::ProfessionalComprehensiveAnalysisReport& report) {
    
    if (!report.outliers.empty()) {
        const auto& outlier_analysis = report.outliers[0];
        
        result.columns = {
            "severity", "count", "percentage", "main_features", 
            "likely_cause", "recommendation"
        };
        
        // Add rows for each severity level
        std::vector<std::pair<std::string, size_t>> severity_counts = {
            {"Critical", outlier_analysis.severity.critical_count},
            {"High", outlier_analysis.severity.high_count},
            {"Medium", outlier_analysis.severity.medium_count},
            {"Low", outlier_analysis.severity.low_count}
        };
        
        size_t total_outliers = outlier_analysis.outlier_indices.size();
        
        for (const auto& severity_pair : severity_counts) {
            const std::string& severity = severity_pair.first;
            size_t count = severity_pair.second;
            
            if (count > 0) {
                std::vector<std::string> row;
                row.push_back(severity);
                row.push_back(std::to_string(count));
                row.push_back(std::to_string((count * 100.0) / total_outliers) + "%");
                
                // Main features contributing to outliers
                std::stringstream features_ss;
                for (size_t i = 0; i < std::min(outlier_analysis.root_cause.contributing_features.size(), 
                                               (size_t)3); ++i) {
                    if (i > 0) features_ss << ", ";
                    features_ss << outlier_analysis.root_cause.contributing_features[i];
                }
                row.push_back(features_ss.str());
                
                row.push_back(outlier_analysis.root_cause.likely_cause);
                row.push_back(outlier_analysis.recommendations.overall_strategy);
                
                result.rows.push_back(row);
            }
        }
        
        // Add summary row
        if (total_outliers > 0) {
            std::vector<std::string> summary_row;
            summary_row.push_back("TOTAL");
            summary_row.push_back(std::to_string(total_outliers));
            summary_row.push_back(std::to_string((total_outliers * 100.0) / 
                (report.column_analyses.empty() ? 1 : 
                 report.column_analyses.begin()->second.total_count)) + "%");
            summary_row.push_back("");
            summary_row.push_back("");
            summary_row.push_back("");
            
            result.rows.push_back(summary_row);
        }
    } else {
        result.columns = {"message"};
        result.rows.push_back({"No outliers detected"});
    }
}

// Format data quality output
void AIExecutionEngineFinal::format_quality_output(
    ExecutionEngine::ResultSet& result,
    const esql::analysis::ProfessionalComprehensiveAnalysisReport& report) {
    
    result.columns = {
        "metric", "score", "status", "issues", "recommendations"
    };
    
    const auto& quality = report.data_quality.scores;
    
    std::vector<std::pair<std::string, double>> metrics = {
        {"Completeness", quality.completeness},
        {"Consistency", quality.consistency},
        {"Accuracy", quality.accuracy},
        {"Timeliness", quality.timeliness},
        {"Validity", quality.validity},
        {"Uniqueness", quality.uniqueness},
        {"Overall", quality.overall}
    };
    
    for (const auto& metric_pair : metrics) {
        const std::string& metric_name = metric_pair.first;
        double score = metric_pair.second;
        
        std::vector<std::string> row;
        row.push_back(metric_name);
        row.push_back(std::to_string(score * 100) + "%");
        
        // Status
        std::string status;
        if (score >= 0.9) status = "Excellent";
        else if (score >= 0.7) status = "Good";
        else if (score >= 0.5) status = "Fair";
        else status = "Poor";
        row.push_back(status);
        
        // Issues (simplified)
        std::string issues;
        if (metric_name == "Completeness" && report.data_quality.completeness.missing_percentage > 0) {
            issues = std::to_string(report.data_quality.completeness.missing_percentage) + "% missing";
        } else if (metric_name == "Uniqueness" && report.data_quality.uniqueness.duplicate_percentage > 0) {
            issues = std::to_string(report.data_quality.uniqueness.duplicate_percentage) + "% duplicates";
        }
        row.push_back(issues);
        
        // Recommendations (simplified)
        std::string recommendations;
        if (score < 0.7) {
            if (metric_name == "Completeness") recommendations = "Impute missing values";
            else if (metric_name == "Uniqueness") recommendations = "Remove duplicates";
            else if (metric_name == "Consistency") recommendations = "Validate data types";
        }
        row.push_back(recommendations);
        
        result.rows.push_back(row);
    }
}

// Format time series output
void AIExecutionEngineFinal::format_timeseries_output(
    ExecutionEngine::ResultSet& result,
    const esql::analysis::ProfessionalComprehensiveAnalysisReport& report) {
    
    if (!report.time_series_analyses.empty()) {
        const auto& ts_analysis = report.time_series_analyses[0];
        
        result.columns = {
            "property", "value", "interpretation", "recommendation"
        };
        
        // Fixed: Use 3-element tuples instead of 4
        std::vector<std::tuple<std::string, std::string, std::string>> properties = {
            {"Stationarity (ADF)", 
             std::to_string(ts_analysis.stationarity.adf_p_value),
             ts_analysis.stationarity.is_stationary_adf ? "Stationary" : "Non-stationary"},
            
            {"Seasonality", 
             ts_analysis.seasonality.has_seasonality ? "Yes" : "No",
             ts_analysis.seasonality.has_seasonality ? 
                 "Has seasonality" : "No seasonality"},
            
            {"Hurst Exponent", 
             std::to_string(ts_analysis.autocorrelation.hurst_exponent),
             ts_analysis.autocorrelation.hurst_exponent > 0.5 ? "Long memory" : "Short memory"},
            
            {"Forecastability", 
             std::to_string(ts_analysis.forecastability.predictability),
             ts_analysis.forecastability.predictability > 0.7 ? "Highly predictable" : 
             ts_analysis.forecastability.predictability > 0.3 ? "Moderately predictable" : "Hard to predict"}
        };
        
        for (const auto& property_tuple : properties) {
            std::vector<std::string> row;
            row.push_back(std::get<0>(property_tuple));  // property
            row.push_back(std::get<1>(property_tuple));  // value
            
            // Add interpretation and recommendations based on property
            std::string interpretation = std::get<2>(property_tuple);
            std::string recommendation = "";
            
            // Add specific recommendations
            if (std::get<0>(property_tuple) == "Stationarity (ADF)" && 
                !ts_analysis.stationarity.is_stationary_adf) {
                recommendation = "Consider differencing";
            } else if (std::get<0>(property_tuple) == "Seasonality" && 
                      ts_analysis.seasonality.has_seasonality) {
                recommendation = "Use seasonal decomposition";
            } else if (std::get<0>(property_tuple) == "Hurst Exponent") {
                recommendation = ts_analysis.autocorrelation.hurst_exponent > 0.5 ? 
                               "Consider ARFIMA" : "Consider ARIMA";
            } else if (std::get<0>(property_tuple) == "Forecastability") {
                recommendation = ts_analysis.forecastability.predictability > 0.7 ? 
                               "Good for forecasting" : "Consider more features";
            }
            
            row.push_back(interpretation);
            row.push_back(recommendation);
            
            result.rows.push_back(row);
        }
    }
}

// Format comprehensive output (insights)
void AIExecutionEngineFinal::format_comprehensive_output(
    ExecutionEngine::ResultSet& result,
    const esql::analysis::ProfessionalComprehensiveAnalysisReport& report) {
    
    result.columns = {"category", "insight", "severity", "action"};
    
    // Add data quality insights
    if (report.data_quality.scores.overall < 0.7) {
        std::vector<std::string> row;
        row.push_back("DATA_QUALITY");
        row.push_back("Data quality needs improvement (score: " + 
                     std::to_string(report.data_quality.scores.overall * 100) + "%)");
        row.push_back(report.data_quality.scores.overall < 0.5 ? "CRITICAL" : "WARNING");
        row.push_back("Address data quality issues before analysis");
        result.rows.push_back(row);
    }
    
    // Add missing data insights
    if (report.data_quality.completeness.missing_percentage > 5.0) {
        std::vector<std::string> row;
        row.push_back("MISSING_DATA");
        row.push_back("Significant missing data (" + 
                     std::to_string(report.data_quality.completeness.missing_percentage) + "%)");
        row.push_back("WARNING");
        row.push_back("Consider imputation or data collection");
        result.rows.push_back(row);
    }
    
    // Add correlation insights
    for (const auto& corr : report.correlations) {
        if (std::abs(corr.pearson_r) > 0.8) {
            std::vector<std::string> row;
            row.push_back("CORRELATION");
            row.push_back("Very strong correlation between " + corr.column1 + 
                         " and " + corr.column2 + " (r=" + std::to_string(corr.pearson_r) + ")");
            row.push_back("INFO");
            row.push_back("Check for multicollinearity in models");
            result.rows.push_back(row);
        }
    }
    
    // Add outlier insights
    if (!report.outliers.empty() && !report.outliers[0].outlier_indices.empty()) {
        size_t outlier_count = report.outliers[0].outlier_indices.size();
        double outlier_percentage = (outlier_count * 100.0) / 
            (report.column_analyses.empty() ? 1 : report.column_analyses.begin()->second.total_count);
        
        if (outlier_percentage > 1.0) {
            std::vector<std::string> row;
            row.push_back("OUTLIERS");
            row.push_back(std::to_string(outlier_count) + " outliers detected (" + 
                         std::to_string(outlier_percentage) + "%)");
            row.push_back(outlier_percentage > 5.0 ? "WARNING" : "INFO");
            row.push_back("Investigate outliers for data quality");
            result.rows.push_back(row);
        }
    }
    
    // Add feature importance insights
    if (!report.feature_importance.empty()) {
        // Find top feature
        auto top_feature = std::max_element(
            report.feature_importance.begin(),
            report.feature_importance.end(),
            [](const auto& a, const auto& b) {
                return a.scores.random_forest < b.scores.random_forest;
            }
        );
        
        if (top_feature != report.feature_importance.end() && 
            top_feature->scores.random_forest > 0.1) {
            std::vector<std::string> row;
            row.push_back("FEATURE_IMPORTANCE");
            row.push_back("Most important feature: '" + top_feature->feature_name + 
                         "' (importance: " + std::to_string(top_feature->scores.random_forest) + ")");
            row.push_back("INFO");
            row.push_back("Focus feature engineering on this feature");
            result.rows.push_back(row);
        }
    }
    
    // Add clustering insights
    if (!report.clusters.empty()) {
        // Check for imbalanced clusters
        auto max_cluster = *std::max_element(report.clusters.begin(), report.clusters.end(),
            [](const auto& a, const auto& b) { return a.size < b.size; });
        
        auto min_cluster = *std::min_element(report.clusters.begin(), report.clusters.end(),
            [](const auto& a, const auto& b) { return a.size < b.size; });
        
        double imbalance_ratio = static_cast<double>(max_cluster.size) / min_cluster.size;
        
        if (imbalance_ratio > 5.0) {
            std::vector<std::string> row;
            row.push_back("CLUSTERING");
            row.push_back("Highly imbalanced clusters (ratio: " + std::to_string(imbalance_ratio) + ":1)");
            row.push_back("WARNING");
            row.push_back("Consider stratified sampling or cluster balancing");
            result.rows.push_back(row);
        }
    }
    
    // If no insights, add a message
    if (result.rows.empty()) {
        result.columns = {"message"};
        result.rows.push_back({"No significant insights found. Data appears to be clean and well-structured."});
    }
}

// Format insights output
void AIExecutionEngineFinal::format_insights_output(
    ExecutionEngine::ResultSet& result,
    const esql::analysis::ProfessionalComprehensiveAnalysisReport& report) {
    
    result.columns = {"type", "message", "severity"};
    
    // Use insights from the report
    for (const auto& insight : report.insights.data_quality) {
        std::vector<std::string> row;
        row.push_back("DATA_QUALITY");
        row.push_back(insight);
        row.push_back("INFO");
        result.rows.push_back(row);
    }
    
    for (const auto& insight : report.insights.statistical) {
        std::vector<std::string> row;
        row.push_back("STATISTICAL");
        row.push_back(insight);
        row.push_back("INFO");
        result.rows.push_back(row);
    }
    
    for (const auto& insight : report.insights.business) {
        std::vector<std::string> row;
        row.push_back("BUSINESS");
        row.push_back(insight);
        row.push_back("INFO");
        result.rows.push_back(row);
    }
    
    // If no insights, generate some basic ones
    if (result.rows.empty()) {
        // Check data quality
        if (report.data_quality.scores.overall < 0.9) {
            std::vector<std::string> row;
            row.push_back("DATA_QUALITY");
            row.push_back("Data quality score: " + 
                         std::to_string(report.data_quality.scores.overall * 100) + "%");
            row.push_back(report.data_quality.scores.overall < 0.7 ? "WARNING" : "INFO");
            result.rows.push_back(row);
        }
        
        // Check for missing data
        if (report.data_quality.completeness.missing_percentage > 0) {
            std::vector<std::string> row;
            row.push_back("MISSING_DATA");
            row.push_back(std::to_string(report.data_quality.completeness.missing_percentage) + 
                         "% missing values");
            row.push_back(report.data_quality.completeness.missing_percentage > 5.0 ? 
                         "WARNING" : "INFO");
            result.rows.push_back(row);
        }
    }
}

void AIExecutionEngineFinal::format_distribution_output(
    ExecutionEngine::ResultSet& result,
    const esql::analysis::ProfessionalComprehensiveAnalysisReport& report) {

    result.columns = {"column", "distribution_type", "kurtosis", "skewness", "normality_test", "is_normal"};

    for (const auto& [col_name, col_analysis] : report.column_analyses) {
        if (col_analysis.detected_type == "numeric" ||
            col_analysis.detected_type == "integer" ||
            col_analysis.detected_type == "float" ||
            col_analysis.detected_type == "double") {

            std::vector<std::string> row;
            row.push_back(col_name);

            // Determine distribution type based on skewness and kurtosis
            std::string dist_type = "unknown";
            if (std::abs(col_analysis.skewness) < 0.5 && std::abs(col_analysis.kurtosis) < 0.5) {
                dist_type = "normal";
            } else if (col_analysis.skewness > 0) {
		               dist_type = "right_skewed";
            } else if (col_analysis.skewness < 0) {
                dist_type = "left_skewed";
            } else if (col_analysis.kurtosis > 0) {
                dist_type = "leptokurtic";
            } else if (col_analysis.kurtosis < 0) {
                dist_type = "platykurtic";
            }

            row.push_back(dist_type);
            row.push_back(std::to_string(col_analysis.kurtosis));
            row.push_back(std::to_string(col_analysis.skewness));
            row.push_back("N/A"); // Placeholder for normality test
            row.push_back(std::abs(col_analysis.skewness) < 0.5 && std::abs(col_analysis.kurtosis) < 0.5 ? "Yes" : "No");

            result.rows.push_back(row);
        }
    }

    if (result.rows.empty()) {
        result.columns = {"message"};
	      result.rows.push_back({"No numeric columns found for distribution analysis"});
    }
}
