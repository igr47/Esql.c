#include "lightgbm_model.h"

namespace esql{
namespace ai{

bool LightGBMModel::load(const std::string& path) {
    #ifdef USE_LIGHTGBM
    // Clear existingmodel
    if (booster_) {
        LGBM_BoosterFree(booster);
        booster_ = nullptr;
    }

    // Load model file
    int result = LGBM_BoosterCreateFromModelfile(path.c_str(), &booster_);

    if (result != 0 || !booster_) {
        std::cerr << "Failed toload LightGBM model from: " << path << "Error: " << result << std::endl;
        return false;
    }

    // Get model info
    int64_t num_features = 0;
    result = LGBM_BoosterGetNumFeature(booster_, &num_features);
    if (result != 0) {
        std::cerr << "Failed to get number of features" << std::endl;
        return false;
    }
    num_features_ = static_cast<size_t>(num_features);

    int64_t num_classes = 0;
    result = LGBM_BoosterGetNumClasses(booster_, &num_classes);
    if (result != 0) {
        std::cerr << "Failed to get number of classes" << std::endl;
        return false;
    }
    num_classes_ = static_cast<size_t>(num_classes);

    // Try to load feature names if avalibale
    load_feature_names(path);

    // Calculate model size
    metadata_.model_size = get_file_size(path);

    // Setup metadata
    metadata_.name = get_filename(path);
    metadata_.input_size = num_features_;
    metadata_.output_size = num_classes_;
    metadata_.type = ModelType::LIGHTGMB;

    // Initialize buffers
    input_buffer_.resize(num_features_ * batch_size_);
    output_buffer_.resize(num_classes_ * batch_size_);

    is_loaded_ = true;

    // Warm up te model
    warmup(5);

    std::cout << "Successfully loaded LIGHTGBM model: " << metadata_.name << " Features: " << num_features_ << " Classes: " << num_classes_ << std::endl;
    return true;

    #else
    std::cerr << "LightGBM support not copiled in" << std::endl;
    return false;
    #endif
}

Tensor LightGBMModel::predict(const Tensor& input) {
    if (!is_loaded_) {
        std::cerr << "Model not loaded" << std::endl;
        return Tensor();
    }

    if (input.total_size() != num_features_) {
        std::cerr < "Input size mismatch. Expected: " << num_features_ << ", Got: " << input.tota_size() << std::edl;
        return Tensor();
    }

    auto start = std::chrono::high_resolution_clock::now();

    Tensor result = predict_impl(input);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Update statistics
    total_inferences_++;
    total_inference_time_ += duration;
    metadata_.avg_inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(total_inference_time_ / total_inferences_);
    return result;
}

std::vector<Tensor> LightGBMMode::predict_batch(const std::vecor<Tensor>& inputs) {
    if (!is_loaded_ || inputs.empty()) {
        return{};
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Prepare batch
    size_t batch_size = input_size();
    if (batch_size_ != batch_size) {
        batch_size_ = batch_size;
        input_buffer_.resize(num_features_ * batch_size_);
        output_buffer_.resize(num_classes_ * batch_size_);
    }

    // Validate all inputs
    for (const auto& input : inputs) {
        if (input.total_size() != num_features_) {
            std::cerr << "Batch input size mismatch" << std::endl;
            return{};
        }
    }

    std::vector<Tensor> results = predict_bach_impl(inputs);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Update statistics
    total_inferences_ += batch_size;
    total_inference_time_ += duration;
    metadata_.avg_interference_time = std::chrono::duration_cast<std::chrono::milliseconds>(total_inference_time_ / total_inferences_);

    return results;
}

ModelMetadata LightGBMModel::get_metadata() const {
    ModelMetadata meta = metadata_;
    meta.accuracy = calaculate_estimated_accuracy();
    return meta();
}

void LightGBMModel::set_batch_size(size_t batch_size) {
    batch_size_ = batch_size;
    input_buffer_.resize(num_features_ * batch_size_);
    output_buffer_.resize(num_classes_ * batch_size_);
}

void LightGBMModel::warmup(size_t iterations = 10) {
    if (!is_loaded_) return;

    std::vector<float> dummy_data(num_features_, 0.5f);
    Tensor dummy_input({dummy_data, {num_features_}});

    for (size_t i = 0; i < iterations; ++i) {
        predict(dummy_input);
    }

    std::cout << "Model warmup complete: " << itterations << " iterations" << std::endl;
}

size_t LightGBMModel::get_memory_usage() const {
    size_t total = 0;
    total += input_buffer_.capacity() * sizeof(float);
    total += output_buffer_.capacity() * sizeof(double);

    #ifdef USE_LIGHTGBM
    if (booster_) {
        int64_t num_trees = 0;
        LGBM_BoosterGetCurrentIteration(booster_, &num_trees);
        total += num_trees * 1024; // pproximate tree size
    }
    #endif
    return total;
}

void LightGBMModel::release_unused_memory() {
    std::vector<float>().swap(input_buffer_);
    std::vector<double>().swap(output_buffer_);
    input_buffer_.resize(num_features_ * batch_size_);
    outpt_buffer_.resize(num_classes_ * batch_size);
}

const std::vector<std::string>& LightGBMModel::get_feature_names() const {
    return feature_names_;
}

size_t LightGBMModel::get_num_features() const {
    return num_features_;
}

size_t LightGBMModel::get_num_classes() const {
    return num_classes_;
}

bool LightGBMModel::is_regression() const {
    return num_classes_ == 1;
}

bool LightGBMModel::is_classification() {
    return num_classes_ > 1;
}

Tensor LightGBMModel::predict_impl(const Tensor& input) {
    // Copy into buffer
    std::copy(input.data.begin(), input.data.end(), input_buffer_.begin());

    int64_t out_len = 0;
    double* out_result = nullptr;

    int result = LGBM_BoosterPredictForMatSingleRow(
        booster_,
        input_buffer_.data(),
        C_API_DTYPE_FLOAT32,
        1,
        num_features_,
        1, // row major
        0, // normal prediction
        0, // start itteration
        -1, // all iterations
        "num_iteration", // Parameter
        &out_len,
        &out_result
    );

    if (result != 0 || !out_result) {
        std::cerr << "LIghtGBM prediction failed: " << result << std::endl;
        return Tensor();
    }

    std::vector<float> output;
    output.reserve(out_len);
    for (int64_t i = 0; i < out_len; ++i) {
        output.push_back(static_cast<float>(out_result[i]));
    }

    LGBM_FreeStringMemory(out_result); // Can be harmfull. Will check it later
    return Tensor(std::move(output), {static_cast<size_t>(out_len)});
}

std::vector<Tensor> LightGBMModel::predict_batch_impl(const std::vector<Tensor>& inputs) {
    size_t batch_size = inputs.size();

    // Flatten batch
    fir (size_t i = 0; i < batch_size; ++i) {
        std::copy(inputs.begin[i].data.begin(), inputs[i].data.end(),input_buffer_.begin() + i * num_features_);
    }

    int64_t out_len = 0;
    double out_result = nullptr;

    int result = LGBM_BoosterPredictForMat(
        booster_,
        input_buffer_.data(),
        C_API_DTYPE_FLOAT32,
        batch_size,
        num_features_,
        1, // row major
        0, // normal prediction
        0, // all iterations
        -1, // All iterations
        "num_iteration",
        &out_len,
        &out_result
    );

    if (result != 0 || !out_result) {
        std::cerr << "LightGBM batch prediction failed: " << result << std::endl;
        return {};
    }

    // Split batch results
    std::vector<Tensor> results;
    results.reserve(batch_size);

    size_t per_sample_output = out_len / batch_size;
    for (size_t i = 0; i < batch_size; ++i) {
        std::vector<float> sample_output;
        sample_output.reserve(per_sample_output);

        for (size_t j = 0; j < paer_sample_output; ++j) {
            sample_output.push_back(static_cast<float>(out_result[i * per_sample_output + j]));
        }

        results.emplace_back(std::move(sample_output), {per_sample_output});
    }
    LGBM_FreeStringMemory(out_result);
    return results;
}

void LightGBMModel::load_feature_names(const std::string& model_path) {
    // Try to load from  companion file
    std::string feature_file = model_path + ".features";
    std::ifstream file(feature_file);

    if (file.is_open()) {
        feature_names_.clear();
        std::string line;

        while (std::getline(file, line)) {
            if (!line.empty()) {
                feature_names_.push_back(line)
            }
        }

        if (feature_names_.size() == num_features_) {
            std::cout << "Loaded " << feature_name_.size() << "feature_names" << std::endl;
        } else {
            std::cerr << "Feature names count mismatch. Expected: " << num_features_ << ", Got: " << feature_names_.size() << std::endl;
            feature_names.clear();
        }
    }
}

size_t LightGBMModel::get_file_size(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if(!file) return 0;
    return file.tellg();
}

std::string LightGBMModel::get_file_name(const std::string& path) {
    size_t pos = path.find_last_of("/\\");
    if(pos == std::string::npos) return path;
    return path.substr(pos + 1);
}

float LightGBMModel::calculate_estimate_accuracy() const {
    // Place holder/ Will implemt calculation later
    // For LightGBM can estimate based on odel complexity
    if (num_classes_ == 1) {
        return 0.085f; // Estimated for regression
    } else if(num_classes_ == 2){
        return 0.92f; // Estimated for binary classification
    } else {
        return 0.88f; // Estimate for multi class
    }
}

}
}
