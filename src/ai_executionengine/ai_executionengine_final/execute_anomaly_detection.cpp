#include "ai/anomaly_detection.h"
#include "ai_execution_engine_final.h"
#include "datum.h"
#include "database.h"
#include <iostream>
#include <chrono>
#include <string>
#include <mutex>
#include <algorithm>
#include <vector>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <filesystem>

// Add missing declaration for AST::Statement
namespace AST {
    class Statement;
}

ExecutionEngine::ResultSet AIExecutionEngineFinal::executeDetectAnomaly(
    AST::DetectAnomalyStatement& stmt) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        std::cout << "[AnomalyDetection] Starting anomaly detection on table: " 
                  << stmt.input_table << std::endl;
        
        // ============================================
        // 1. Extract Data Using Existing DataExtractor
        // ============================================
        
        std::cout << "[AnomalyDetection] Extracting data from table..." << std::endl;
        
        // Use the already initialized data_extractor_
        auto rows = data_extractor_->extract_table_data(
            "default",  // Assuming default database
            stmt.input_table, 
            {},  // All columns
            stmt.where_clause, 
            0,   // No limit
            0    // No offset
        );
        
        if (rows.empty()) {
            throw std::runtime_error("No data found in table: " + stmt.input_table);
        }
        
        std::cout << "[AnomalyDetection] Extracted " << rows.size() 
                  << " rows from table" << std::endl;
        
        // ============================================
        // 2. Analyze Columns and Get Column Statistics
        // ============================================
        
        std::cout << "[AnomalyDetection] Analyzing column statistics..." << std::endl;
        
        // Get all column names from first row
        std::vector<std::string> all_columns;
        if (!rows.empty()) {
            for (const auto& [col_name, _] : rows[0]) {
                all_columns.push_back(col_name);
            }
        }
        
        // Get column statistics using DataExtractor
        auto column_stats = data_extractor_->analyze_columns(
            "default", stmt.input_table, all_columns);
        
        std::cout << "[AnomalyDetection] Analyzed " << column_stats.size() 
                  << " columns" << std::endl;
        
        // ============================================
        // 3. Select Feature Columns
        // ============================================
        
        std::vector<std::string> feature_columns;
        
        // If specific features are provided in parameters, use them
        auto features_it = stmt.parameters.find("features");
        if (features_it != stmt.parameters.end()) {
            // Parse comma-separated feature list
            std::stringstream ss(features_it->second);
            std::string feature;
            while (std::getline(ss, feature, ',')) {
                // Trim whitespace
                feature.erase(0, feature.find_first_not_of(" \t"));
                feature.erase(feature.find_last_not_of(" \t") + 1);
                if (!feature.empty()) {
                    feature_columns.push_back(feature);
                }
            }
        } else {
            // Auto-select features: exclude ID, timestamp, and low-variance columns
            for (const auto& [col_name, stats] : column_stats) {
                // Skip likely ID and timestamp columns
                std::string col_lower = col_name;
                std::transform(col_lower.begin(), col_lower.end(), 
                              col_lower.begin(), ::tolower);
                
                if (col_lower.find("id") != std::string::npos ||
                    col_lower.find("timestamp") != std::string::npos ||
                    col_lower.find("date") != std::string::npos ||
                    col_lower.find("time") != std::string::npos) {
                    continue;
                }
                
                // Skip constant or near-constant columns
                if (stats.std_value < 0.001) {
                    continue;
                }
                
                // Skip columns with too many nulls (>50%)
                if (static_cast<float>(stats.null_count) / stats.total_count > 0.5) {
                    continue;
                }
                
                feature_columns.push_back(col_name);
            }
        }
        
        if (feature_columns.empty()) {
            throw std::runtime_error("No suitable feature columns found for anomaly detection");
        }
        
        std::cout << "[AnomalyDetection] Selected " << feature_columns.size() 
                  << " feature columns" << std::endl;
        
        // ============================================
        // 4. Convert Rows to Tensors
        // ============================================
        
        std::cout << "[AnomalyDetection] Converting data to tensors..." << std::endl;
        
        std::vector<esql::ai::Tensor> tensors;
        tensors.reserve(rows.size());
        
        size_t skipped_rows = 0;
        for (size_t row_idx = 0; row_idx < rows.size(); ++row_idx) {
            const auto& row = rows[row_idx];
            std::vector<float> features;
            features.reserve(feature_columns.size());
            
            bool row_valid = true;
            
            for (const auto& col_name : feature_columns) {
                auto col_it = row.find(col_name);
                if (col_it == row.end() || col_it->second.is_null()) {
                    // Missing value - impute with mean if available
                    auto stats_it = column_stats.find(col_name);
                    if (stats_it != column_stats.end()) {
                        features.push_back(static_cast<float>(stats_it->second.mean_value));
                    } else {
                        features.push_back(0.0f);
                    }
                    continue;
                }
                
                const esql::Datum& datum = col_it->second;
                float feature_value = 0.0f;
                
                try {
                    // Convert Datum to float based on type
                    if (datum.is_integer()) {
                        feature_value = static_cast<float>(datum.as_int());
                    } else if (datum.is_float()) {
                        feature_value = datum.as_float();
                    } else if (datum.is_double()) {
                        feature_value = static_cast<float>(datum.as_double());
                    } else if (datum.is_boolean()) {
                        feature_value = datum.as_bool() ? 1.0f : 0.0f;
                    } else if (datum.is_string()) {
                        // String encoding: use frequency encoding from DataExtractor
                        std::string str_val = datum.as_string();
                        feature_value = data_extractor_->encode_string_feature(col_name, str_val);
                    } else if (datum.is_datetime()) {
                        // Convert datetime to Unix timestamp
                        auto tp = datum.as_datetime();
                        auto duration = tp.time_since_epoch();
                        feature_value = static_cast<float>(
                            std::chrono::duration_cast<std::chrono::seconds>(duration).count());
                    } else {
                        // Unsupported type
                        feature_value = 0.0f;
                    }
                    
                    // Apply normalization using column statistics
                    auto stats_it = column_stats.find(col_name);
                    if (stats_it != column_stats.end() && stats_it->second.std_value > 0.001) {
                        // Z-score normalization
                        feature_value = (feature_value - stats_it->second.mean_value) / 
                                      stats_it->second.std_value;
                        // Clip extreme values
                        feature_value = std::clamp(feature_value, -3.0f, 3.0f);
                    }
                    
                    features.push_back(feature_value);
                    
                } catch (const std::exception& e) {
                    std::cerr << "[TensorConversion] Error converting column '" << col_name 
                              << "' in row " << row_idx << ": " << e.what() << std::endl;
                    features.push_back(0.0f);
                }
            }
            
            if (row_valid) {
                // Create tensor with 1D shape (feature vector)
                tensors.emplace_back(std::move(features), 
                                    std::vector<size_t>{feature_columns.size()});
            } else {
                skipped_rows++;
            }
        }
        
        if (tensors.empty()) {
            throw std::runtime_error("No valid data could be converted to tensors");
        }
        
        std::cout << "[AnomalyDetection] Converted " << tensors.size() 
                  << " rows to tensors (skipped " << skipped_rows << " invalid rows)" << std::endl;
        
        // ============================================
        // 5. Create or Load Anomaly Detector
        // ============================================
        
        std::unique_ptr<esql::ai::IAnomalyDetector> detector;
        bool is_new_model = false;
        
        if (!stmt.model_name.empty()) {
            // Try to load existing model
            std::cout << "[AnomalyDetection] Loading existing model: " 
                      << stmt.model_name << std::endl;
            
            auto& registry = esql::ai::ModelRegistry::instance();
            
            // First check if model exists in registry
            if (registry.model_exists(stmt.model_name)) {
                auto* model = registry.get_model(stmt.model_name);
                if (model) {
                    // Try dynamic cast to IAnomalyDetector
                    detector.reset(dynamic_cast<esql::ai::IAnomalyDetector*>(model));
                    
                    if (!detector) {
                        std::cerr << "[AnomalyDetection] Model exists but is not an anomaly detector" 
                                  << std::endl;
                        // Continue to create new detector
                    } else {
                        std::cout << "[AnomalyDetection] Loaded model from registry: " 
                                  << stmt.model_name << std::endl;
                    }
                }
            }
            
            // If not in registry, try to load from disk
            if (!detector) {
                std::string model_path = "models/" + stmt.model_name + ".anomaly";
                std::ifstream test_file(model_path);
                if (test_file.good()) {
                    test_file.close();
                    
                    // Create detector and load from file
                    esql::ai::AnomalyDetectionConfig config;
                    detector = esql::ai::AnomalyDetectorFactory::create_detector(
                        stmt.algorithm.empty() ? "isolation_forest" : stmt.algorithm, 
                        config);
                    
                    if (detector->load_detector(model_path)) {
                        std::cout << "[AnomalyDetection] Loaded model from disk: " 
                                  << stmt.model_name << std::endl;
                    }
                }
            }
        }
        
        // Create new detector if needed
        if (!detector) {
            is_new_model = true;
            
            // Parse configuration from SQL parameters
            esql::ai::AnomalyDetectionConfig config;
            config.algorithm = stmt.algorithm.empty() ? "isolation_forest" : stmt.algorithm;
            config.detection_mode = "unsupervised";
            
            // Set contamination rate
            auto cont_it = stmt.parameters.find("contamination");
            if (cont_it != stmt.parameters.end()) {
                try {
                    config.contamination = std::stof(cont_it->second);
                    config.contamination = std::clamp(config.contamination, 0.01f, 0.5f);
                } catch (...) {
                    std::cerr << "[AnomalyDetection] Invalid contamination, using default 0.1" 
                              << std::endl;
                    config.contamination = 0.1f;
                }
            } else {
                config.contamination = 0.1f;  // Default 10% expected anomalies
            }
            
            // Set scaling method
            auto scale_it = stmt.parameters.find("scaling");
            if (scale_it != stmt.parameters.end()) {
                config.scaling_method = scale_it->second;
            } else {
                config.scaling_method = "robust";  // Robust scaling for anomaly detection
            }
            
            // Set threshold method
            auto thresh_it = stmt.parameters.find("threshold_method");
            if (thresh_it != stmt.parameters.end()) {
                config.threshold_method = thresh_it->second;
            }
            
            // Check for manual threshold
            auto manual_thresh_it = stmt.parameters.find("threshold");
            if (manual_thresh_it != stmt.parameters.end()) {
                try {
                    config.manual_threshold = std::stof(manual_thresh_it->second);
                    config.threshold_auto_tune = false;
                } catch (...) {
                    std::cerr << "[AnomalyDetection] Invalid manual threshold, using auto-tune" 
                              << std::endl;
                }
            }
            
            // Create detector
            std::cout << "[AnomalyDetection] Creating new " << config.algorithm 
                      << " detector with contamination=" << config.contamination << std::endl;
            
            detector = esql::ai::AnomalyDetectorFactory::create_detector(
                config.algorithm, config);
            
            if (!detector) {
                throw std::runtime_error("Failed to create anomaly detector");
            }
            
            // Train the detector
            std::cout << "[AnomalyDetection] Training detector on " 
                      << tensors.size() << " samples..." << std::endl;
            
            auto train_start = std::chrono::high_resolution_clock::now();
            
            if (!detector->train_unsupervised(tensors)) {
                throw std::runtime_error("Failed to train anomaly detector");
            }
            
            auto train_end = std::chrono::high_resolution_clock::now();
            auto train_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                train_end - train_start);
            
            std::cout << "[AnomalyDetection] Training completed in " 
                      << train_duration.count() << "ms" << std::endl;
            
            // Save the model if a name was provided
            if (!stmt.model_name.empty()) {
                std::string model_path = "models/" + stmt.model_name + ".anomaly";
                
                // Ensure models directory exists
                std::filesystem::create_directories("models");
                
                if (detector->save_detector(model_path)) {
                    std::cout << "[AnomalyDetection] Model saved to: " << model_path << std::endl;
                } else {
                    std::cerr << "[AnomalyDetection] Warning: Failed to save model to disk" << std::endl;
                }
            }
        }
        
        // ============================================
        // 6. Perform Anomaly Detection
        // ============================================
        
        std::cout << "[AnomalyDetection] Running anomaly detection..." << std::endl;
        
        auto detect_start = std::chrono::high_resolution_clock::now();
        
        // Perform batch detection
        auto results = detector->detect_anomalies_batch(tensors);
        
        auto detect_end = std::chrono::high_resolution_clock::now();
        auto detect_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            detect_end - detect_start);
        
        // Count anomalies
        size_t anomaly_count = 0;
        size_t high_confidence_anomalies = 0;
        float avg_anomaly_score = 0.0f;
        
        for (const auto& result : results) {
            if (result.is_anomaly) {
                anomaly_count++;
                avg_anomaly_score += result.anomaly_score;
                
                if (result.confidence > 0.8f) {
                    high_confidence_anomalies++;
                }
            }
        }
        
        if (anomaly_count > 0) {
            avg_anomaly_score /= anomaly_count;
        }
        
        std::cout << "[AnomalyDetection] Detection completed in " 
                  << detect_duration.count() << "ms" << std::endl;
        std::cout << "[AnomalyDetection] Results: " << anomaly_count << " anomalies found (" 
                  << std::fixed << std::setprecision(1) 
                  << (static_cast<float>(anomaly_count) / results.size() * 100) 
                  << "%), Avg score: " << avg_anomaly_score << std::endl;
        
        // ============================================
        // 7. Generate Alerts if Requested
        // ============================================
        
        if (stmt.generate_alerts && anomaly_count > 0) {
            std::cout << "\n[AnomalyDetection] GENERATING ALERTS:" << std::endl;
            std::cout << "========================================" << std::endl;
            std::cout << "Table: " << stmt.input_table << std::endl;
            std::cout << "Total anomalies: " << anomaly_count << std::endl;
            std::cout << "High confidence anomalies: " << high_confidence_anomalies << std::endl;
            std::cout << "Average anomaly score: " << avg_anomaly_score << std::endl;
            std::cout << "========================================\n" << std::endl;
        }
        
        // ============================================
        // 8. Save Results to Output Table if Specified
        // ============================================
        
        if (!stmt.output_table.empty()) {
            std::cout << "[AnomalyDetection] Saving results to table: " 
                      << stmt.output_table << std::endl;
            
            try {
                // Create output table SQL
                std::string create_sql = 
                    "CREATE TABLE IF NOT EXISTS " + stmt.output_table + " (\n"
                    "  row_id INTEGER PRIMARY KEY,\n"
                    "  is_anomaly BOOLEAN,\n"
                    "  anomaly_score FLOAT,\n"
                    "  confidence FLOAT,\n"
                    "  threshold FLOAT,\n"
                    "  reasons TEXT,\n"
                    "  detection_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP\n"
                    ")";
                
                // Execute via database
                auto create_result = db_.executeQuery(create_sql);
                
                // Insert results
                for (size_t i = 0; i < results.size(); ++i) {
                    const auto& result = results[i];
                    
                    // Combine reasons
                    std::string reasons_str;
                    for (const auto& reason : result.reasons) {
                        if (!reasons_str.empty()) reasons_str += "; ";
                        reasons_str += reason;
                    }
                    
                    // Escape quotes for SQL
                    std::replace(reasons_str.begin(), reasons_str.end(), '\'', '`');
                    
                    std::string insert_sql = 
                        "INSERT INTO " + stmt.output_table + 
                        " (row_id, is_anomaly, anomaly_score, confidence, threshold, reasons) " +
                        "VALUES (" + std::to_string(i) + ", " +
                        (result.is_anomaly ? "true" : "false") + ", " +
                        std::to_string(result.anomaly_score) + ", " +
                        std::to_string(result.confidence) + ", " +
                        std::to_string(result.threshold) + ", '" +
                        reasons_str + "')";
                    
                    db_.executeQuery(insert_sql);
                }
                
                std::cout << "[AnomalyDetection] Saved " << results.size() 
                          << " results to table: " << stmt.output_table << std::endl;
                
            } catch (const std::exception& e) {
                std::cerr << "[AnomalyDetection] Failed to save results: " << e.what() << std::endl;
            }
        }
        
        // ============================================
        // 9. Prepare and Return Result Set
        // ============================================
        
        ExecutionEngine::ResultSet result_set;
        
        // Set columns for result display
        result_set.columns = {
            "row_id", 
            "is_anomaly", 
            "anomaly_score", 
            "confidence", 
            "threshold", 
            "reasons"
        };
        
        // Add sample results (first 100 or all if less)
        size_t sample_size = std::min(static_cast<size_t>(100), results.size());
        for (size_t i = 0; i < sample_size; ++i) {
            const auto& result = results[i];
            
            std::vector<std::string> row;
            row.push_back(std::to_string(i));
            row.push_back(result.is_anomaly ? "true" : "false");
            row.push_back(std::to_string(result.anomaly_score));
            row.push_back(std::to_string(result.confidence));
            row.push_back(std::to_string(result.threshold));
            
            // Combine reasons
            std::string reasons_str;
            for (const auto& reason : result.reasons) {
                if (!reasons_str.empty()) reasons_str += "; ";
                reasons_str += reason;
            }
            row.push_back(reasons_str);
            
            result_set.rows.push_back(row);
        }
        
        // Add summary information
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);
        
        std::stringstream message;
        message << "Anomaly detection completed successfully.\n";
        message << "• Table: " << stmt.input_table << "\n";
        message << "• Rows processed: " << results.size() << "\n";
        message << "• Anomalies detected: " << anomaly_count << " (" 
                << std::fixed << std::setprecision(1) 
                << (static_cast<float>(anomaly_count) / results.size() * 100) 
                << "%)\n";
        message << "• Model: " << (is_new_model ? "Created new " : "Used existing ") 
                << detector->get_config().algorithm << "\n";
        message << "• Time: " << total_duration.count() << "ms total (" 
                << detect_duration.count() << "ms detection)";
        
        // Log final summary
        std::cout << "\n[AnomalyDetection] " << message.str() << std::endl;
        
        return result_set;
        
    } catch (const std::exception& e) {
        std::cerr << "[AnomalyDetection] ERROR: " << e.what() << std::endl;
        
        ExecutionEngine::ResultSet result_set;
        result_set.columns = {"error"};
        result_set.rows.push_back({std::string("Anomaly detection failed: ") + e.what()});
        
        return result_set;
    }
}
