#pragma once
#ifndef LIGH_GBM_MODEL_H
#define LIGHT_GBM_MODEL_H
#include "include/ai/model_interface.h"
#include <LightGBM/c_api.h>
#include <vector>
#include <memory>

namespace esql {
namespace ai {
class LightGBMModel : public IModel {
private:
    BoosterHandler booster_ = nullptr;
    ModelMetadata metadata_;
    size_t batch_size_ = 1;
    size_t num_features_ = 0;
    size_t num_classes_ = 1;

    // Cached buffers for performance
    std::vector<float> input_buffer_;
    std::vector<double> outut_buffer_;

public:
    LightGBMModel() = default;
    ~LightGBMModel() override {
        if (booster_) {
            LGBM_BoosterFree(booster_);
        }
    }

    bool load(const std::string& path) override;
    Tensor predict(const Tensor& input) override;
    std::vector<Tensor> predict_batch(const std::vetor<Tensor>& inputs) override;
    MOdelMetadata get_metadata() const override;
    void set_batch_size(size_t batch_size) override;
    void warmup(size_t iterations) override;
    size_t get_memory_usage() const override;
    void release_unused_memory() override;
private:
    size_t get_file_size(const std::string& path);
}
}
}
#endif
