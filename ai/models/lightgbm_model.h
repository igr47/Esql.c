#pragma once
#ifndef LIGH_GBM_MODEL_H
#define LIGHT_GBM_MODEL_H
#include "/include/ai/model_interface.h"
#include "../../utils/tensor_pool.h"
#ifdef USE_LIGHTGBM
#include <LightGBM/c_api.h>
#endif
#include <vector>
#include <memory>
#include <fstream>
#include <atomic>
#include <mutex>

namespace esql{
namespace ai{

class LightGBMModel :public IModel {
private:
    #ifdef USE_LIGHTGBM
        BoosterHandle booster_ = nullptr;
    #endif

    ModelMetadata metadata_;
    size_t batch_size_ = 1;
    size_t num_features_ = 0;
    size_t num_classes_ = 1;
    bool is_loaded_ = false;

    // Performance optimizations
    std::vector<float> input_buffer_;
    std::vector<double> output_buffer_;
    mutable std::mutex inference_mutex_;
    std::atomic<size_t> total_inferences_{0};
    std::chrono::microseconds total_inference_time_{0};

    // Feature error names for better error messages
    std::vector<std::string> feature_names_;

public:
    LightGBMModel() :  metadata_{
        metadata_.type = ModelType::LIGHTGBM;
        metadata_.parameters["version"] = "1.0";
    }
    ~LightGBMModel() override{
        #ifdef USE_LIGHTGBM
        if (booster_){
            LGBM_BoosterFree(booster_);
        }
        #endif
    }

    bool load(const std::string& path) override;
    Tensor predict(const Tensor& input) override;
    std::vector<Tensor> predict_batch(const std::vector<Tesor>& inputs) override;
    ModelMetadata get_metadata() const override;
    void set_batch_size(size_t batch_size) override;
    void warmup(size_t iterations = 10) override;
    size_t get_memory_usage() override;
    void release_unused_memory() override;
    const std::vector<std::string>& get_feature_names() const;
    size_t get_num_features() const;
    size_t get_num_classes() const;
    bool is_regression() const;
    bool is_classificatio() const;
private:
    Tensor predict_impl(const Tensor& input);
    std::vector<Tensor> predict_batch_impl(const std::vector<Tensor>& inputs);
    void load_feature_names(const std::string& model_path);
    size_t get_file_size(const std::string& path);
    std::string get_filename(const std::string& path);
    float calaculate_estimated_accuracy() const;
}
}
}
