#include "ai/lightgbm_model.h"
#include <iostream>
#include <fstream>
#include <algorithm>

namespace esql {
namespace ai {

// ============================================
// ModelSchema Implementation
// ============================================
nlohmann::json ModelSchema::to_json() const {
    nlohmann::json j;
    j["model_id"] = model_id;
    j["description"] = description;
    j["target_column"] = target_column;
    j["problem_type"] = problem_type;
    j["created_at"] = std::chrono::system_clock::to_time_t(created_at);
    j["last_updated"] = std::chrono::system_clock::to_time_t(last_updated);
    j["training_samples"] = training_samples;
    j["accuracy"] = accuracy;
    j["drift_score"] = drift_score;

    // Comprehensive metrics section
    nlohmann::json metrics_json;

    if (problem_type == "binary_classification") {
        metrics_json["accuracy"] = get_metadata_float("accuracy", 0.0f);
        metrics_json["precision"] = get_metadata_float("precision", 0.0f);
        metrics_json["recall"] = get_metadata_float("recall", 0.0f);
        metrics_json["f1_score"] = get_metadata_float("f1_score", 0.0f);
        metrics_json["auc_score"] = get_metadata_float("auc_score", 0.0f);
        metrics_json["specificity"] = get_metadata_float("specificity", 0.0f);

    } else if (problem_type == "multiclass") {
        metrics_json["accuracy"] = get_metadata_float("accuracy", 0.0f);
        metrics_json["macro_precision"] = get_metadata_float("macro_precision", 0.0f);
        metrics_json["macro_recall"] = get_metadata_float("macro_recall", 0.0f);
        metrics_json["macro_f1"] = get_metadata_float("macro_f1", 0.0f);
        metrics_json["micro_precision"] = get_metadata_float("micro_precision", 0.0f);

    } else {
        // Regression metrics
        metrics_json["r2_score"] = get_metadata_float("r2_score", 0.0f);
        metrics_json["rmse"] = get_metadata_float("rmse", 0.0f);
        metrics_json["mae"] = get_metadata_float("mae", 0.0f);
        metrics_json["mape"] = get_metadata_float("mape", 0.0f);
        metrics_json["medae"] = get_metadata_float("medae", 0.0f);

        // "Precision-like" metrics for regression
        metrics_json["within_5_percent"] = get_metadata_float("within_5_percent", 0.0f);
        metrics_json["within_10_percent"] = get_metadata_float("within_10_percent", 0.0f);
        metrics_json["within_20_percent"] = get_metadata_float("within_20_percent", 0.0f);
        metrics_json["within_1_std"] = get_metadata_float("within_1_std", 0.0f);
        metrics_json["within_2_std"] = get_metadata_float("within_2_std", 0.0f);
        metrics_json["coverage_90"] = get_metadata_float("coverage_90", 0.0f);
        metrics_json["coverage_95"] = get_metadata_float("coverage_95", 0.0f);
        metrics_json["coverage_99"] = get_metadata_float("coverage_99", 0.0f);
    }

    j["metrics"] = metrics_json;

    // Feature information
    j["features"] = nlohmann::json::array();
    for (const auto& feature : features) {
        j["features"].push_back(feature.to_json());
    }

    // All metadata (including raw metrics)
    j["metadata"] = metadata;

    // Statistics
    j["stats"] = nlohmann::json::object({
        {"total_predictions", stats.total_predictions},
        {"failed_predictions", stats.failed_predictions},
        {"avg_confidence", stats.avg_confidence},
        {"avg_inference_time_us", stats.avg_inference_time.count()}
    });

    return j;
}

float ModelSchema::get_metadata_float(const std::string& key, float default_value) const {
    auto it = metadata.find(key);
    if (it != metadata.end()) {
        try {
            return std::stof(it->second);
        } catch (...) {
            return default_value;
        }
    }
    return default_value;
}

ModelSchema ModelSchema::from_json(const nlohmann::json& j) {
    ModelSchema schema;
    schema.model_id = j["model_id"];
    schema.description = j["description"];
    schema.target_column = j["target_column"];
    schema.problem_type = j["problem_type"];
    schema.created_at = std::chrono::system_clock::from_time_t(j["created_at"]);
    schema.last_updated = std::chrono::system_clock::from_time_t(j["last_updated"]);
    schema.training_samples = j["training_samples"];
    schema.accuracy = j["accuracy"];
    schema.drift_score = j["drift_score"];

    for (const auto& f : j["features"]) {
        schema.features.push_back(FeatureDescriptor::from_json(f));
    }

    if (j.contains("metadata")) {
        schema.metadata = j["metadata"].get<std::unordered_map<std::string, std::string>>();
    }

    if (j.contains("stats")) {
        schema.stats.total_predictions = j["stats"]["total_predictions"];
        schema.stats.failed_predictions = j["stats"]["failed_predictions"];
        schema.stats.avg_confidence = j["stats"]["avg_confidence"];
        schema.stats.avg_inference_time = std::chrono::microseconds(j["stats"]["avg_inference_time_us"]);
    }

    return schema;
}

bool ModelSchema::matches_row(const std::unordered_map<std::string, Datum>& row) const {
    for (const auto& feature : features) {
        if (feature.required && row.find(feature.db_column) == row.end()) {
            return false;
        }
    }
    return true;
}

std::vector<float> ModelSchema::extract_features(const std::unordered_map<std::string, Datum>& row) const {
    std::vector<float> result;
    result.reserve(features.size());

    for (const auto& feature : features) {
        auto it = row.find(feature.db_column);
        if (it != row.end()) {
            try {
                result.push_back(feature.transform(it->second));
            } catch (const std::exception& e) {
                // Use default value if transformation fails
                result.push_back(feature.default_value);
            }
        } else if (feature.required) {
            throw std::runtime_error("Missing required feature: " + feature.db_column);
        } else {
            result.push_back(feature.default_value);
        }
    }

    return result;
}

std::vector<std::string> ModelSchema::get_missing_features(const std::unordered_map<std::string, Datum>& row) const {
    std::vector<std::string> missing;
    for (const auto& feature : features) {
        if (feature.required && row.find(feature.db_column) == row.end()) {
            missing.push_back(feature.db_column);
        }
    }
    return missing;
}

} // namespace ai
} // namespace esql
