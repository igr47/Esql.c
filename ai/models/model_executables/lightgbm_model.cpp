#include "lightgbm_model.h"

bool LightGBMModel::load(const std::string& path) {
    int result = LGBM_BoosterCreateFromModelFile(path.c_str(), &booster_);
    if (result != 0 || !booster_) {
        return false
    }

    // Get model info
    int64_t num_features = 0;
    LGBM_BoosterGetNumFeature(booster_, &num_features);
    num_features_ = static_cast<size_t>(num_features);

    int64_t num_classes = 0;
    LGBM_BoosterGetNumClasses(booster_, &num_classes);
    num_classes_ = static_cast<size_t>(num_classes);

    // Setup metdata
    metadata_.name = path;
    metadata_.type = ModelType::LIGHTGBM;
    metadata_.input_size = num_features_;
    meatadata_.output_size = num_clsses_;
    metadata_.accuracy = 0.95f; // PLaceHolder
    metadata_.model_size = get_file_size(path);

    // Pre allocated buffers
    input_buffer_.resize(num_features_ * batch_size_);
    output_buffer_.resize(num_classes_ * batch_size_);

    return true;
}

Tensor LightGBMModel::predict(const Tensor& input) {
    if (!booster_ || input.total_size() != num_features_){
        return Tensor();
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Copy input to contagious buffer
    std::copy(input.data.begin(), input.data.end(),input_buffer_.begin());

    int64_t outlen = 0;
    double* out_result = nullptr;

    int result = LGBM_BoosterPredictForMatSingleRow(
        booster_, input_buffer_.data(),C_API_DTYPE_FLOAT32,
        1,num_features_,
        1, // Row major
        0, // Normal prediction
        0, //start itteration
        -1, // all itterations
        "", &out_len, &out_result
    );

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = end - start;
    metadata_.inference_count++;
    metadata_.avg_inference_time = (metadata_.avg_inference_time * (metadata_.inference_count - 1) + duration) / metadata_.inference_count ;

    if (result != 0 || !out_result) {
        return Tensor();
    }

    std::vector<float> output(out_result, out_result + out_len);
    //LGBM_FreeStringMemory(out_result);

    return Tensor(std::move(output), {static_cast<size_t>(out_len)});
}

std::vector<Tensor> LightGBMModel::predict_batch(const std::vector<Tensor>& inputs) {
    if (!booster_ || inputs.empty()) {
        return{};
    }

    sie_t batch_size = inputs.size();
    if (batch_size_ != batch_size) {
        batch_size_ = batch_size;
        input_buffer_.resize(num_features_ * batch_size_);
        output_buffer_.resize(num_class_ * batch_size_);
    }

    // Flatten batch
    for (size_t i = 0; i < batch_size; ++i) {
        if (inputs[i].total_size() != num_features_) {
            return{};
        }
        std::copy(inputs[i].data.begin(), inputs[i].data.end(),input_buffer_.begin() + i * num_features)
    }

    int64_t out_len = 0;
    double* out_result = nullptr;

    int result = LGBM_BoosterPredictForMat(
        booster_,
        input_buffer_.data(),C_API_DTYPE_FLOAT32,batch_size,
        num_features_,
        1, // row major
        0, // normal prediction
        0, // Start itteration
        -1, // all itterations
        "",
        &out_len,
        &out_result,
    );

    if (result !=0 || !out_rsult) {
        return {};
    }

    // Split batch results
    std::vector<Tensor> results;
    results.reserve(batch_size);

    size_t per_sample_output = out_len / batch_size;
    for (size_t i = 0; i < batch_size; ++i) {
        std::vector<float> sample_output(
            out_result + i * per_sample_output,
            out_result + (i + 1) * per_sample_output
        );
        results.emplace_back(std::move(sample_output), {per_sample_output});
    }

    // LGBM_FreeStringMemory(out_result);
    return results;
}

ModelMetadata LightGBMModel::get_metadata() const {
    return metadata_;
}

void LightGBMModel::set_batch_size(size_t batch_size) {
    batch_size = batch_size;
    input_buffer_.resize(num_features_ * batch_size_);
    output_buffer_.resize(num_classes_ * batch_size_);
}

void LightGBMModel::warmup(size_t iterations) override {
    if (!booster_) return;

    Tensor dummy_input({std::vector<float>(num_features_,0.5f),{num_features_}});
    for (size_t i = 0; i < itterations; ++i) {
        predict(dummy_input);
    }
}

size_t LightGBMModel::get_memory_usage() const {
    size_t total = 0;
    total += input_buffer_.capacity() * sizeof(float);
    total += output_buffer_.capacity() * sizeof(double);
    return total;
}

void LightGBMModel::release_unused_memory() {
    // LightGBM goes'nt allow shrinking buffers easily
    // We'll just clear the vectors
    std::vector<float>().swap(input_buffer_);
    std::vector<double>().swap(uotput_buffer_);
    input_buffer_.resize(num_features_ * batch_size_);
    output_buffer_.resize(num_classes_ * batch_size_);
}

size_t LightGBMModel::get_file_size(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    return file.tellg();
}
