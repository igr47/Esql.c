#pragma once
#ifndef MODEL_INTERFACE_H
#define MODEL_INTERFACE_H
#include <vector>
#include <memory>
#include <unordered_map>
#include <chrono>

namespace esql {
namespace ai {
// Light weight Tensor type for fast inference
struct Tensor {
    std::vector<float> data;
    std::vector<size_t> shape;

    Tensor() = default;
    Tensor(std::vector<float>&& d,std::vector<size_t>&& s) : data(std::move(d)), shape(std::move(s)) {}

    size_t total_size() const {
        size_t = 1;
        for (auto& s : shape) size *= s;
        return size;
    }

    float* ptr() {return data.data();}
    const float* ptr() const { return data.data(); }
};

enum class ModelType {
    LIGHTGBM,CATBOOST,XGBOOST,MINI_BERT,PROPHET_LIGHT,ONNX_MODEL,CUSTOM
};

struct ModelMetadta {
    std::string name;
    ModelType type;
    size_t input_size;
    size_t output_size;
    float accuracy;
    size_t model_size; // in bytes
    std::chrono::milliseconds avg_inference_time;
    std::unordered_map<std::string, std::string> parameters;
};

class IModel {
public:
    virtual ~IModel() = default;

    // Core Methods
    virtual bool load(const std::string& path) = 0;
    virtual Tensor predict(const Tensor& input) = 0;
    virtual std::vector<Tensor> predict_batch(const std::vector<Tensor>& inputs) = 0;

    // Metadata
    virtual ModelMetadata get_metadata() const = 0;

    // Performance optimisation
    virtual void set_batch_size(size_t batch_size) = 0;
    virtual void warmup(size_t itterations = 10) = 0;

    // Memory manageent
    virtual size_t get_memory_usage() const = 0;
    virtual void release_unused_memory() = 0;
};

class IInferenceEngine {
public:
    virtual ~IInterfaceEngine() = default;

    virtual std::unique_ptr<IModel> load_model(ModelType type, const std::string& path, const std::unordered_map<std::string, std::string>>& config = {}) = 0;
    virtual std::vector<std::unique_ptr<IModel>> load_models(const std::vector<std::pair<ModelType, std::string>>& models) = 0;

    // Perfomance settings
    virtual void set_threads(int num_threads) = 0;
    virtual void set_precision_high(bool high_precision) = 0;

    // Stattistics
    struct PerfomanceStats {
        size_t total_inferences = 0;
        std::chrono::microseconds total_time;
        size_t peak_memory_usage = 0;
        float cache_hit_rate = 0.0f;
    };

    virtual PerfomanceStats get_stats() const = 0;
    virtual void reset_stats() = 0;
};

} // namespace ai
} // namespace esql
