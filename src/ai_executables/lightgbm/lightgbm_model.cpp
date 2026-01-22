#include "ai/lightgbm_model.h"
#include "ai/algorithm_registry.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <unordered_map>

namespace esql {
namespace ai {

// ============================================
// AdaptiveLightGBMModel Core Implementation
// ============================================

AdaptiveLightGBMModel::AdaptiveLightGBMModel()
    : schema_() {
    schema_.created_at = std::chrono::system_clock::now();
    schema_.last_updated = schema_.created_at;
}

AdaptiveLightGBMModel::AdaptiveLightGBMModel(const ModelSchema& schema)
    : schema_(schema) {
    schema_.created_at = std::chrono::system_clock::now();
    schema_.last_updated = schema_.created_at;
}

AdaptiveLightGBMModel::~AdaptiveLightGBMModel() {
    std::lock_guard<std::mutex> lock(model_mutex_);
    if (booster_) {
        LGBM_BoosterFree(booster_);
        booster_ = nullptr;
    }
}

bool AdaptiveLightGBMModel::load(const std::string& path) {
    std::lock_guard<std::mutex> lock(model_mutex_);

    // Try to load model file
    int num_iterations = 0;
    int result = LGBM_BoosterCreateFromModelfile(path.c_str(), &num_iterations, &booster_);
    if (result != 0) {
        std::cerr << "[LightGBM] Failed to load model: " << path
                  << ", error: " << LGBM_GetLastError() << std::endl;
        return false;
    }

    // Try to load schema
    std::string schema_path = path + ".schema.json";
    if (std::filesystem::exists(schema_path)) {
        try {
            std::ifstream schema_file(schema_path);
            nlohmann::json j;
            schema_file >> j;
            schema_ = ModelSchema::from_json(j);
            std::cout << "[LightGBM] Loaded schema for model: " << schema_.model_id << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[LightGBM] Failed to load schema: " << e.what() << std::endl;
            create_minimal_schema(path);
        }
    } else {
        std::cout << "[LightGBM] No schema found, creating minimal schema" << std::endl;
        create_minimal_schema(path);
    }

    // Validate feature count matches model
    int num_features = 0;
    LGBM_BoosterGetNumFeature(booster_, &num_features);

    if (static_cast<size_t>(num_features) != schema_.features.size()) {
        std::cout << "[LightGBM] Feature count mismatch: model expects " << num_features
                  << ", schema has " << schema_.features.size()
                  << ". Adjusting schema..." << std::endl;
        adjust_schema_to_model(static_cast<size_t>(num_features));
    }

    // Initialize buffers
    input_buffer_.resize(schema_.features.size() * batch_size_);
    output_buffer_.resize(get_output_size() * batch_size_);

    is_loaded_ = true;
    drift_detector_.last_drift_check = std::chrono::system_clock::now();
    std::cout << "[LightGBM] Model loaded successfully: " << schema_.model_id
              << " (" << schema_.features.size() << " features)" << std::endl;

    return true;
}

bool AdaptiveLightGBMModel::save(const std::string& path) {
    std::lock_guard<std::mutex> lock(model_mutex_);

    if (!booster_) {
        std::cerr << "[LightGBM] Cannot save - no model loaded" << std::endl;
        return false;
    }

    // Save LightGBM model
    int result = LGBM_BoosterSaveModel(booster_, 0, -1, 0, path.c_str());
    if (result != 0) {
        std::cerr << "[LightGBM] Failed to save model: " << LGBM_GetLastError() << std::endl;
        return false;
    }

    // Save schema
    std::string schema_path = path + ".schema.json";
    try {
        std::ofstream schema_file(schema_path);
        schema_file << schema_.to_json().dump(2);
        std::cout << "[LightGBM] Model saved to: " << path << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[LightGBM] Failed to save schema: " << e.what() << std::endl;
        return false;
    }
}

Tensor AdaptiveLightGBMModel::predict(const Tensor& input) {
    if (!is_loaded_) {
        throw std::runtime_error("[LightGBM] Model not loaded");
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    std::lock_guard<std::mutex> lock(model_mutex_);

    // Validate input size
    if (input.total_size() != schema_.features.size()) {
        std::stringstream ss;
        ss << "[LightGBM] Input size mismatch. Expected " << schema_.features.size()
           << " features, got " << input.total_size();
        throw std::runtime_error(ss.str());
    }

    // Copy input to buffer
    std::copy(input.data.begin(), input.data.end(), input_buffer_.begin());

    // Run prediction
    int64_t out_len = 0;
    double out_result = 0.0;

    int result = LGBM_BoosterPredictForMatSingleRow(
        booster_,
        input_buffer_.data(),
        C_API_DTYPE_FLOAT32,
        static_cast<int>(schema_.features.size()),
        1, // row major
        0, // normal prediction
        0, // start iteration
        -1, // all iterations
        "",
        &out_len,
        &out_result
    );

    if (result != 0 || !out_result) {
        schema_.stats.failed_predictions++;
        throw std::runtime_error("[LightGBM] Prediction failed: " +
                                std::string(LGBM_GetLastError()));
    }

    // Convert output
    std::vector<float> output;
    output.reserve(out_len);
    output.push_back(static_cast<float>(out_result));

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time
    );

    // Update statistics
    prediction_count_++;
    schema_.stats.total_predictions++;

    // Update average inference time using exponential moving average
    if (schema_.stats.total_predictions == 1) {
        schema_.stats.avg_inference_time = duration;
    } else {
        // EMA with alpha = 0.1
        auto new_avg = schema_.stats.avg_inference_time * 0.9 + duration * 0.1;
        schema_.stats.avg_inference_time = std::chrono::duration_cast<std::chrono::microseconds>(new_avg);
    }

    // Update drift detection
    float prediction_value = (out_len == 1) ? output[0] : 0.0f;
    drift_detector_.add_sample(input.data, prediction_value);

    // Periodically check for drift (every 100 predictions or 1 hour)
    auto now = std::chrono::system_clock::now();
    if (prediction_count_ % 100 == 0 ||
        now - drift_detector_.last_drift_check > std::chrono::hours(1)) {
        schema_.drift_score = drift_detector_.calculate_drift_score();
        drift_detector_.last_drift_check = now;

        if (schema_.drift_score > 0.3f) {
            std::cout << "[LightGBM] WARNING: Drift detected for model " << schema_.model_id << " (score: " << schema_.drift_score << ")" << std::endl;
        }
    }

    return Tensor(std::move(output), {static_cast<size_t>(out_len)});
}

Tensor AdaptiveLightGBMModel::predict_row(const std::unordered_map<std::string, Datum>& row) {
    if (!is_loaded_) {
        throw std::runtime_error("[LightGBM] Model not loaded");
    }

    // Extract features from row using schema
    std::vector<float> features = schema_.extract_features(row);

    // Create tensor and predict
    Tensor input_tensor(std::move(features), {schema_.features.size()});
    return predict(input_tensor);
}

std::vector<Tensor> AdaptiveLightGBMModel::predict_batch(const std::vector<Tensor>& inputs) {
    if (!is_loaded_ || inputs.empty()) {
        return {};
    }

    std::lock_guard<std::mutex> lock(model_mutex_);

    size_t batch_size = inputs.size();

    // Resize buffers if needed
    if (batch_size_ != batch_size) {
        batch_size_ = batch_size;
        input_buffer_.resize(schema_.features.size() * batch_size_);
        output_buffer_.resize(get_output_size() * batch_size_);
    }

    // Validate all inputs have correct size
    for (size_t i = 0; i < batch_size; ++i) {
        if (inputs[i].total_size() != schema_.features.size()) {
            throw std::runtime_error("[LightGBM] Batch input size mismatch at index " +
                                    std::to_string(i));
        }
    }

    // Flatten batch
    for (size_t i = 0; i < batch_size; ++i) {
        std::copy(inputs[i].data.begin(), inputs[i].data.end(),
                 input_buffer_.begin() + i * schema_.features.size());
    }

    // Batch prediction
    int64_t out_len = 0;
    double out_result = 0.0;

    int result = LGBM_BoosterPredictForMat(
        booster_,
        input_buffer_.data(),
        C_API_DTYPE_FLOAT32,
        static_cast<int32_t>(batch_size),
        static_cast<int32_t>(schema_.features.size()),
        1, // row major
        0, // normal prediction
        0, // start iteration
        -1, // all iterations
        "",
        &out_len,
        &out_result
    );

    if (result != 0 || !out_result) {
        schema_.stats.failed_predictions++;
        throw std::runtime_error("[LightGBM] Batch prediction failed: " +
                                std::string(LGBM_GetLastError()));
    }

    // Split results
    std::vector<Tensor> results;
    results.reserve(batch_size);

    size_t per_sample_output = out_len / batch_size;
    
    // For single output models (binary classification, regression)
    if (per_sample_output == 1) {
        double* out_result_ptr = reinterpret_cast<double*>(&out_result);
        for (size_t i = 0; i < batch_size; ++i) {
            std::vector<float> sample_output;
            sample_output.push_back(static_cast<float>(out_result_ptr[i]));
            results.push_back(Tensor(std::move(sample_output), {1}));
        }
    }

    // Update statistics
    prediction_count_ += batch_size;
    schema_.stats.total_predictions += batch_size;

    return results;
}

ModelMetadata AdaptiveLightGBMModel::get_metadata() const {
    std::lock_guard<std::mutex> lock(model_mutex_);

    ModelMetadata meta;
    meta.name = schema_.model_id;
    meta.type = ModelType::LIGHTGBM;
    meta.input_size = schema_.features.size();
    meta.output_size = get_output_size();
    meta.accuracy = schema_.accuracy;
    meta.model_size = get_model_size();
    meta.avg_inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        schema_.stats.avg_inference_time
    );

    // Extract metrics based on problem type
    if (schema_.problem_type == "binary_classification") {
        meta.precision = get_metric_from_metadata("precision", 0.0f);
        meta.recall = get_metric_from_metadata("recall", 0.0f);
        meta.f1_score = get_metric_from_metadata("f1_score", 0.0f);
        meta.auc_score = get_metric_from_metadata("auc_score", 0.0f);
    } else if (schema_.problem_type == "multiclass") {
        meta.precision = get_metric_from_metadata("macro_precision", 0.0f);
        meta.recall = get_metric_from_metadata("macro_recall", 0.0f);
        meta.f1_score = get_metric_from_metadata("macro_f1", 0.0f);
    } else {
        meta.r2_score = get_metric_from_metadata("r2_score", 0.0f);
        meta.rmse = get_metric_from_metadata("rmse", 0.0f);
        meta.mae = get_metric_from_metadata("mae", 0.0f);
        meta.within_10_percent = get_metric_from_metadata("within_10_percent", 0.0f);
        meta.within_1_std = get_metric_from_metadata("within_1_std", 0.0f);
        meta.coverage_95 = get_metric_from_metadata("coverage_95", 0.0f);
    }

    // Add algorithm info
    auto& algo_registry = esql::ai::AlgorithmRegistry::instance();
    const auto* algo_info = algo_registry.get_algorithm(schema_.algorithm);

    meta.parameters["algorithm"] = schema_.algorithm;
    if (algo_info) {
        meta.parameters["algorithm_description"] = algo_info->description;
        meta.parameters["lightgbm_objective"] = algo_info->lightgbm_objective;
    }

    // Add comprehensive metrics based on problem type
    if (schema_.problem_type == "binary_classification") {
        add_binary_classification_metrics(meta.parameters);
    } else if (schema_.problem_type == "multiclass") {
        add_multiclass_metrics(meta.parameters);
    } else {
        add_regression_metrics(meta.parameters);
    }

    // Add common schema info
    meta.parameters["problem_type"] = schema_.problem_type;
    meta.parameters["target_column"] = schema_.target_column;
    meta.parameters["features"] = std::to_string(schema_.features.size());
    meta.parameters["drift_score"] = std::to_string(schema_.drift_score);
    meta.parameters["created_at"] = std::to_string(
        std::chrono::system_clock::to_time_t(schema_.created_at)
    );
    meta.parameters["training_samples"] = std::to_string(schema_.training_samples);
    meta.parameters["total_predictions"] = std::to_string(schema_.stats.total_predictions);

    return meta;
}

void AdaptiveLightGBMModel::add_all_metrics_to_parameters(std::unordered_map<std::string, std::string>& params) const {
    // Add all available metrics from metadata
    for (const auto& [key, value] : schema_.metadata) {
        // Skip non-metric entries
        if (key == "created_via" || key == "source_table" ||
            key == "target_column" || key == "training_samples") {
            continue;
        }

        params[key] = value;
    }
}

float AdaptiveLightGBMModel::get_metric_from_metadata(const std::string& key, float default_value) const {
    auto it = schema_.metadata.find(key);
    if (it != schema_.metadata.end()) {
        try {
            return std::stof(it->second);
        } catch (...) {
            return default_value;
        }
    }
    return default_value;
}

void AdaptiveLightGBMModel::set_batch_size(size_t batch_size) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    batch_size_ = batch_size;
    input_buffer_.resize(schema_.features.size() * batch_size_);
    output_buffer_.resize(get_output_size() * batch_size_);
}

void AdaptiveLightGBMModel::warmup(size_t iterations) {
    if (!is_loaded_) return;

    std::cout << "[LightGBM] Warming up model: " << schema_.model_id << std::endl;

    // Create dummy input with valid range
    std::vector<float> dummy_features(schema_.features.size(), 0.5f);
    Tensor dummy_input(std::move(dummy_features), {schema_.features.size()});

    for (size_t i = 0; i < iterations; ++i) {
        try {
            predict(dummy_input);
        } catch (const std::exception& e) {
            std::cerr << "[LightGBM] Warmup iteration failed: " << e.what() << std::endl;
        }
    }

    std::cout << "[LightGBM] Warmup complete: " << iterations << " iterations" << std::endl;
}

size_t AdaptiveLightGBMModel::get_memory_usage() const {
    std::lock_guard<std::mutex> lock(model_mutex_);

    size_t total = 0;
    total += input_buffer_.capacity() * sizeof(float);
    total += output_buffer_.capacity() * sizeof(double);
    total += schema_.features.size() * sizeof(FeatureDescriptor);

    // Estimate LightGBM model size (very rough)
    total += schema_.features.size() * 100; // ~100 bytes per feature

    return total;
}

void AdaptiveLightGBMModel::release_unused_memory() {
    std::lock_guard<std::mutex> lock(model_mutex_);

    // Shrink buffers to current size
    std::vector<float>().swap(input_buffer_);
    std::vector<double>().swap(output_buffer_);

    // Reallocate with current size
    input_buffer_.resize(schema_.features.size() * batch_size_);
    output_buffer_.resize(get_output_size() * batch_size_);
}

bool AdaptiveLightGBMModel::can_handle_row(const std::unordered_map<std::string, Datum>& row) const {
    return schema_.matches_row(row);
}

void AdaptiveLightGBMModel::update_schema(const ModelSchema& new_schema) {
    std::lock_guard<std::mutex> lock(model_mutex_);

    // Validate feature count matches model
    if (booster_) {
        int num_features = 0;
        LGBM_BoosterGetNumFeature(booster_, &num_features);

        if (static_cast<size_t>(num_features) != new_schema.features.size()) {
            throw std::runtime_error("[LightGBM] New schema feature count doesn't match model");
        }
    }

    schema_ = new_schema;
    schema_.last_updated = std::chrono::system_clock::now();

    // Update buffers
    input_buffer_.resize(schema_.features.size() * batch_size_);
    output_buffer_.resize(get_output_size() * batch_size_);
}

void AdaptiveLightGBMModel::update_feature(const FeatureDescriptor& feature) {
    std::lock_guard<std::mutex> lock(model_mutex_);

    // Check if feature exists
    for (auto& existing : schema_.features) {
        if (existing.db_column == feature.db_column) {
            existing = feature;
            schema_.last_updated = std::chrono::system_clock::now();
            return;
        }
    }

    // Add new feature
    schema_.features.push_back(feature);
    schema_.last_updated = std::chrono::system_clock::now();

    std::cout << "[LightGBM] WARNING: New feature added. Model needs retraining." << std::endl;
}

bool AdaptiveLightGBMModel::needs_retraining() const {
    return schema_.drift_score > 0.3f || // High drift
           (prediction_count_ > 10000 && schema_.accuracy < 0.8f); // Low accuracy after many predictions
}

void AdaptiveLightGBMModel::reset_drift_detector() {
    drift_detector_.recent_features.clear();
    drift_detector_.recent_predictions.clear();
    drift_detector_.current_drift_score = 0.0f;
    schema_.drift_score = 0.0f;
    drift_detector_.last_drift_check = std::chrono::system_clock::now();
}

float AdaptiveLightGBMModel::get_avg_inference_time_ms() const {
    return schema_.stats.avg_inference_time.count() / 1000.0f;
}

} // namespace ai
} // namespace esql
