#include "ai/lightgbm_model.h"
#include "ai/algorithm_registry.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <fstream>

namespace esql {
namespace ai {

// ============================================
// FeatureDescriptor Implementation
// ============================================

FeatureDescriptor::FeatureDescriptor()
    : default_value(0.0f), required(true), is_categorical(false),
      min_value(0.0f), max_value(1.0f), mean_value(0.0f), std_value(1.0f) {}

float FeatureDescriptor::transform(const Datum& datum) const {
    if (datum.is_null()) {
        return default_value;
    }

    switch (datum.type()) {
        case Datum::Type::INTEGER:
            return transform_value(static_cast<float>(datum.as_int()));
        case Datum::Type::FLOAT:
            return transform_value(datum.as_float());
        case Datum::Type::DOUBLE:
            return transform_value(static_cast<float>(datum.as_double()));
        case Datum::Type::BOOLEAN:
            return transform_value(datum.as_bool() ? 1.0f : 0.0f);
        case Datum::Type::STRING:
            return transform_string(datum.as_string());
        default:
            return default_value;
    }
}

float FeatureDescriptor::transform_value(float value) const {
    if (transformation == "normalize") {
        float range = max_value - min_value;
        if (range < 1e-6f) return 0.5f;
        return (value - min_value) / range;
    } else if (transformation == "standardize") {
        if (std_value < 1e-6f) return 0.0f;
        return (value - mean_value) / std_value;
    } else if (transformation == "log") {
        return std::log(value + 1.0f);
    } else if (transformation == "binary") {
        return value > 0.5f ? 1.0f : 0.0f;
    } else if (transformation == "sigmoid") {
        return 1.0f / (1.0f + std::exp(-value));
    }
    return value; // direct transformation
}

float FeatureDescriptor::transform_string(const std::string& str) const {
    if (is_categorical) {
        // Find category index
        auto it = std::find(categories.begin(), categories.end(), str);
        if (it != categories.end()) {
            return static_cast<float>(std::distance(categories.begin(), it));
        }
        return 0.0f; // Unknown category (use first category as default)
    } else {
        // Hash string to float
        size_t hash_val = 0;
        for (char c : str) {
            hash_val = hash_val * 31 + static_cast<size_t>(c);
        }
        return static_cast<float>(hash_val % 1000) / 1000.0f;
    }
}

nlohmann::json FeatureDescriptor::to_json() const {
    nlohmann::json j;
    j["name"] = name;
    j["db_column"] = db_column;
    j["data_type"] = data_type;
    j["transformation"] = transformation;
    j["default_value"] = default_value;
    j["required"] = required;
    j["is_categorical"] = is_categorical;
    j["min_value"] = min_value;
    j["max_value"] = max_value;
    j["mean_value"] = mean_value;
    j["std_value"] = std_value;

    if (is_categorical) {
        j["categories"] = categories;
    }

    return j;
}

FeatureDescriptor FeatureDescriptor::from_json(const nlohmann::json& j) {
    FeatureDescriptor fd;
    fd.name = j["name"];
    fd.db_column = j["db_column"];
    fd.data_type = j["data_type"];
    fd.transformation = j["transformation"];
    fd.default_value = j["default_value"];
    fd.required = j["required"];
    fd.is_categorical = j["is_categorical"];
    fd.min_value = j["min_value"];
    fd.max_value = j["max_value"];
    fd.mean_value = j["mean_value"];
    fd.std_value = j["std_value"];

    if (fd.is_categorical && j.contains("categories")) {
        fd.categories = j["categories"].get<std::vector<std::string>>();
    }

    return fd;
}

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

    j["features"] = nlohmann::json::array();
    for (const auto& feature : features) {
        j["features"].push_back(feature.to_json());
    }

    j["metadata"] = metadata;
    j["stats"] = nlohmann::json::object({
        {"total_predictions", stats.total_predictions},
        {"failed_predictions", stats.failed_predictions},
        {"avg_confidence", stats.avg_confidence},
        {"avg_inference_time_us", stats.avg_inference_time.count()}
    });

    return j;
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

// ============================================
// DriftDetector Implementation
// ============================================

void AdaptiveLightGBMModel::DriftDetector::add_sample(const std::vector<float>& features, float prediction) {
    recent_features.push_back(features);
    recent_predictions.push_back(prediction);

    // Keep only last 1000 samples
    if (recent_features.size() > 1000) {
        recent_features.erase(recent_features.begin());
        recent_predictions.erase(recent_predictions.begin());
    }
}

float AdaptiveLightGBMModel::DriftDetector::calculate_drift_score() {
    if (recent_features.size() < 100) return 0.0f;

    // Calculate KL divergence between recent and historical distributions
    // Simplified version: measure change in prediction confidence
    float avg_confidence = 0.0f;
    for (float pred : recent_predictions) {
        // For binary classification, confidence is distance from 0.5
        avg_confidence += std::abs(pred - 0.5f) * 2.0f;
    }
    avg_confidence /= recent_predictions.size();

    // Lower average confidence indicates potential drift
    current_drift_score = std::max(0.0f, 1.0f - avg_confidence);
    return current_drift_score;
}

// ============================================
// AdaptiveLightGBMModel Implementation
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
    /*for (int64_t i = 0; i < out_len; ++i) {
        output.push_back(static_cast<float>(out_result[i]));
    }*/
    output.push_back(static_cast<float>(out_result));

    //LGBM_FreeStringMemory(out_result);

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

    // Periodically check for drift (every 100 predictions or 1 hour. Will specify later)
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
        double* out_result = reinterpret_cast<double*>(&out_result);
        for (size_t i = 0; i < batch_size; ++i) {
            std::vector<float> sample_output;
            sample_output.push_back(static_cast<float>(out_result[i]));
            results.push_back(Tensor(std::move(sample_output), {1}));
        }
    }
    /*for (size_t i = 0; i < batch_size; ++i) {
        std::vector<float> sample_output;
        sample_output.reserve(per_sample_output);

        for (size_t j = 0; j < per_sample_output; ++j) {
            sample_output.push_back(static_cast<float>(
                out_result[i * per_sample_output + j]
            ));
        }

        results.emplace_back(std::move(sample_output), {per_sample_output});
    }*/

    //LGBM_FreeStringMemory(out_result);

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
    meta.avg_inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(schema_.stats.avg_inference_time);

    // Add schema info to parameters
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


bool AdaptiveLightGBMModel::train(const std::vector<std::vector<float>>& features,
                                 const std::vector<float>& labels,
                                 const std::unordered_map<std::string, std::string>& params) {
    if (features.empty() || features.size() != labels.size()) {
        std::cerr << "[LightGBM] Invalid training data" << std::endl;
        return false;
    }

    size_t num_samples = features.size();
    size_t num_features = features[0].size();

    // Validate all samples have same number of features
    for (const auto& sample : features) {
        if (sample.size() != num_features) {
            std::cerr << "[LightGBM] Inconsistent feature sizes in training data" << std::endl;
            return false;
        }
    }

    std::cout << "[LightGBM] Training model with " << num_samples << " samples, "
              << num_features << " features..." << std::endl;

    // Debug: Print first few samples
    std::cout << "[LightGBM] DEBUG: First 3 training samples:" << std::endl;
    for (size_t i = 0; i < std::min((size_t)3, num_samples); ++i) {
        std::cout << "  Sample " << i << ": Features [";
        for (size_t j = 0; j < std::min((size_t)5, num_features); ++j) {
            std::cout << features[i][j] << " ";
        }
        std::cout << "], Label: " << labels[i] << std::endl;
    }

    // Flatten features for LightGBM
    std::vector<float> flat_features;
    flat_features.reserve(num_samples * num_features);
    for (const auto& sample : features) {
        flat_features.insert(flat_features.end(), sample.begin(), sample.end());
    }

    // Create dataset from matrix
    DatasetHandle dataset = nullptr;

    // Generate parameters string
    std::string param_str = generate_parameters(params);
    const char* param_cstr = param_str.c_str();

    std::cout << "[LightGBM] Creating dataset with " << num_samples
              << " rows and " << num_features << " columns" << std::endl;

    // First create an empty dataset
    int result = LGBM_DatasetCreateFromMat(
        flat_features.data(),
        C_API_DTYPE_FLOAT32,
        static_cast<int32_t>(num_samples),
        static_cast<int32_t>(num_features),
        1,  // row major
        "", // Empty parameters for dataset creation
        nullptr, // No reference dataset
        &dataset
    );

    if (result != 0 || !dataset) {
        std::cerr << "[LightGBM] Failed to create dataset: " << LGBM_GetLastError() << std::endl;
        return false;
    }

    // Prepare labels in correct format (float for Ligh)
    std::vector<float> labels_float(labels.begin(), labels.end());

    std::cout << "[LightGBM] Setting labels. First few labels: ";
    for (int i = 0; i < std::min(5, (int)labels_float.size()); ++i) {
        std::cout << labels_float[i] << " ";
    }
    std::cout << std::endl;

    // Set labels with correct parameters
    result = LGBM_DatasetSetField(
        dataset,
        "label",
        labels_float.data(),
        static_cast<int>(labels_float.size()),
        C_API_DTYPE_FLOAT32  // Use float32 for labels
    );

    if (result != 0) {
        std::cerr << "[LightGBM] Failed to set labels: " << LGBM_GetLastError() << std::endl;
        std::cerr << "[LightGBM] Labels count: " << labels_float.size()
                  << ", Dataset samples: " << num_samples << std::endl;
        LGBM_DatasetFree(dataset);
        return false;
    }

    std::cout << "[LightGBM] Dataset created successfully. Creating booster..." << std::endl;

    // Create booster with full parameters
    BoosterHandle new_booster = nullptr;
    result = LGBM_BoosterCreate(dataset, param_cstr, &new_booster);

    if (result != 0 || !new_booster) {
        std::cerr << "[LightGBM] Failed to create booster: " << LGBM_GetLastError() << std::endl;
        LGBM_DatasetFree(dataset);
        return false;
    }

    // Train model with proper iterations
    int num_iterations = 100;
    if (params.find("num_iterations") != params.end()) {
        try {
            num_iterations = std::stoi(params.at("num_iterations"));
        } catch (...) {
            num_iterations = 100;
        }
    }

    std::cout << "[LightGBM] Training for " << num_iterations << " iterations..." << std::endl;

    bool training_success = true;
    for (int i = 0; i < num_iterations; ++i) {
        int is_finished = 0;
        result = LGBM_BoosterUpdateOneIter(new_booster, &is_finished);

        if (result != 0) {
            std::cerr << "[LightGBM] Training iteration " << i + 1 << " failed: "
                      << LGBM_GetLastError() << std::endl;
            training_success = false;
            break;
        }

        if (is_finished) {
            std::cout << "[LightGBM] Early stopping at iteration " << i + 1 << std::endl;
            break;
        }

        if ((i + 1) % 10 == 0) {
            std::cout << "[LightGBM] Completed iteration " << (i + 1) << std::endl;
        }
    }

    // Now we can safely free the dataset
    if (dataset) {
        LGBM_DatasetFree(dataset);
    }

    if (!training_success) {
        if (new_booster) {
            LGBM_BoosterFree(new_booster);
        }
        return false;
    }

    // Swap boosters
    std::lock_guard<std::mutex> lock(model_mutex_);
    if (booster_) {
        LGBM_BoosterFree(booster_);
    }
    booster_ = new_booster;

    // Update schema statistics
    schema_.training_samples = num_samples;
    schema_.last_updated = std::chrono::system_clock::now();

    // Set algorithm if not already set
    if (schema_.algorithm.empty()) {
        schema_.algorithm = "LIGHTGBM";
    }

    // Calculate accuracy/performance metrics
    calculate_training_metrics(features, labels);

    // Reset drift detector
    reset_drift_detector();

    std::cout << "[LightGBM] Training completed successfully!" << std::endl;
    return true;
}

void AdaptiveLightGBMModel::calculate_training_metrics(
    const std::vector<std::vector<float>>& features,
    const std::vector<float>& labels) {

    if (features.empty() || labels.empty()) {
        schema_.accuracy = 0.0f;
        return;
    }

    size_t num_samples = std::min(features.size(), static_cast<size_t>(1000));
    double total_error = 0.0;
    double total_squared_error = 0.0;
    double mean_label = 0.0;

    // Calculate mean label
    for (size_t i = 0; i < num_samples; ++i) {
        mean_label += labels[i];
    }
    mean_label /= num_samples;

    size_t valid_predictions = 0;

    for (size_t i = 0; i < num_samples; ++i) {
        try {
            // Make prediction
            std::vector<float> features_copy = features[i];
            std::vector<size_t> shape = {features_copy.size()};
            esql::ai::Tensor input_tensor(std::move(features_copy), std::move(shape));
            auto prediction = predict(input_tensor);

            float pred_value = prediction.data[0];
            float true_value = labels[i];

            // Accumulate errors
            double error = pred_value - true_value;
            total_error += std::abs(error);
            total_squared_error += error * error;

            valid_predictions++;

        } catch (const std::exception& e) {
            // Skip failed predictions
            continue;
        }
    }

    if (valid_predictions > 0) {
        if (schema_.problem_type == "binary_classification") {
            // For classification, we'd need different metrics
            // For now, use a placeholder
            schema_.accuracy = 0.85f; // Placeholder
        } else {
            // For regression: calculate R² and RMSE
            double mae = total_error / valid_predictions;
            double mse = total_squared_error / valid_predictions;
            double rmse = std::sqrt(mse);

            // Calculate R² (coefficient of determination)
            double total_variance = 0.0;
            for (size_t i = 0; i < num_samples; ++i) {
                double diff = labels[i] - mean_label;
                total_variance += diff * diff;
            }

            if (total_variance > 0) {
                double r2 = 1.0 - (total_squared_error / total_variance);
                schema_.accuracy = static_cast<float>(r2);
            } else {
                schema_.accuracy = 0.0f;
            }

            // Store additional metrics in metadata
            schema_.metadata["rmse"] = std::to_string(rmse);
            schema_.metadata["mae"] = std::to_string(mae);
            schema_.metadata["r2"] = std::to_string(schema_.accuracy);
        }
    }
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

// ============================================
// Private Helper Methods
// ============================================

void AdaptiveLightGBMModel::create_minimal_schema(const std::string& model_path) {
    schema_.model_id = std::filesystem::path(model_path).stem().string();
    schema_.description = "Auto-generated schema for LightGBM model";
    schema_.problem_type = "binary_classification"; // Default

    // Try to get feature count from model
    if (booster_) {
        int num_features = 0;
        LGBM_BoosterGetNumFeature(booster_, &num_features);

        std::cout << "[LightGBM] Model has " << num_features << " features" << std::endl;

        // Create generic features
        for (int64_t i = 0; i < num_features; ++i) {
            FeatureDescriptor feature;
            feature.name = "feature_" + std::to_string(i);
            feature.db_column = feature.name;
            feature.data_type = "float";
            feature.transformation = "direct";
            feature.default_value = 0.0f;
            feature.required = true;
            feature.is_categorical = false;

            schema_.features.push_back(feature);
        }
    } else {
        std::cerr << "[LightGBM] Cannot create schema - no model loaded" << std::endl;
    }
}

void AdaptiveLightGBMModel::adjust_schema_to_model(size_t expected_features) {
    if (schema_.features.size() < expected_features) {
        // Add missing features
        std::cout << "[LightGBM] Adding " << (expected_features - schema_.features.size())
                  << " missing features" << std::endl;

        for (size_t i = schema_.features.size(); i < expected_features; ++i) {
            FeatureDescriptor feature;
            feature.name = "auto_feature_" + std::to_string(i);
            feature.db_column = feature.name;
            feature.data_type = "float";
            feature.transformation = "direct";
            feature.default_value = 0.0f;
            feature.required = false; // Not required since auto-added
            feature.is_categorical = false;

            schema_.features.push_back(feature);
        }
    } else if (schema_.features.size() > expected_features) {
        // Truncate features
        std::cout << "[LightGBM] Truncating schema from " << schema_.features.size()
                  << " to " << expected_features << " features" << std::endl;
        schema_.features.resize(expected_features);
    }
}


std::string AdaptiveLightGBMModel::generate_parameters(
    const std::unordered_map<std::string, std::string>& params) {

    // Default parameters for LightGBM
    std::unordered_map<std::string, std::string> default_params;

    // Set objective based on problem type
    if (schema_.problem_type == "binary_classification") {
        default_params["objective"] = "binary";
        default_params["metric"] = "binary_logloss";
    } else if (schema_.problem_type == "multiclass") {
        default_params["objective"] = "multiclass";
        default_params["metric"] = "multi_logloss";
        auto it = schema_.metadata.find("num_classes");
        if (it != schema_.metadata.end()) {
            default_params["num_class"] = it->second;
        } else {
            default_params["num_class"] = "3";
        }
    } else {
        // For regression
        default_params["objective"] = "regression";
        default_params["metric"] = "rmse";
        default_params["boosting"] = "gbdt";
    }

    default_params["num_leaves"] = "31";
    default_params["learning_rate"] = "0.05";
    default_params["feature_fraction"] = "0.9";
    default_params["bagging_fraction"] = "0.8";
    default_params["bagging_freq"] = "5";
    default_params["min_data_in_leaf"] = "20";
    default_params["min_sum_hessian_in_leaf"] = "0.001";
    default_params["lambda_l1"] = "0.0";
    default_params["lambda_l2"] = "0.0";
    default_params["min_gain_to_split"] = "0.0";
    default_params["max_depth"] = "-1";
    default_params["verbose"] = "1";
    default_params["num_threads"] = "4";

    // Override with user parameters
    for (const auto& [key, value] : params) {
        default_params[key] = value;
    }

    // Build parameter string
    std::string param_str;
    for (const auto& [key, value] : default_params) {
        param_str += key + "=" + value + " ";
    }

    std::cout << "[LightGBM] Generated parameters: " << param_str << std::endl;
    return param_str;
}


size_t AdaptiveLightGBMModel::get_output_size() const {
    if (schema_.problem_type == "binary_classification") return 1;
    if (schema_.problem_type == "multiclass") {
        auto it = schema_.metadata.find("num_classes");
        if (it != schema_.metadata.end()) {
            try {
                return std::stoi(it->second);
            } catch (...) {
                return 1;
            }
        }
        return 1;
    }
    return 1; // regression
}

size_t AdaptiveLightGBMModel::get_model_size() const {
    if (!is_loaded_) return 0;

    size_t size = 0;

    // LightGBM model size
    size += schema_.features.size() * 100;

    // Buffer sizes
    size += input_buffer_.capacity() * sizeof(float);
    size += output_buffer_.capacity() * sizeof(double);

    // Schema size
    size += sizeof(ModelSchema);
    size += schema_.features.size() * sizeof(FeatureDescriptor);

    // Feature categories
    for (const auto& feature : schema_.features) {
        size += feature.categories.capacity() * sizeof(std::string);
    }

    return size;
}

} // namespace ai
} // namespace esql
