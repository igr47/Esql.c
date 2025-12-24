#pragma once
#ifndef MODEL_REGISTRY_H
#define MODEL_REGISTRY_H

#include "lightgbm_model.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <shared_mutex>
#include <vector>

namespace esql {
namespace ai {

class ModelRegistry {
private:
    ModelRegistry() = default;
    ~ModelRegistry() = default;

    std::unordered_map<std::string, std::unique_ptr<AdaptiveLightGBMModel>> models_;
    std::unordered_map<std::string, ModelMetadata> metadata_;
    mutable std::shared_mutex mutex_;

    // Model persistence
    std::string models_directory_ = "models";

public:
    static ModelRegistry& instance() {
        static ModelRegistry instance;
        return instance;
    }

    // Disable copying
    ModelRegistry(const ModelRegistry&) = delete;
    ModelRegistry& operator=(const ModelRegistry&) = delete;

    // Core methods
    bool register_model(const std::string& name, std::unique_ptr<AdaptiveLightGBMModel> model);
    AdaptiveLightGBMModel* get_model(const std::string& name);
    const ModelMetadata* get_metadata(const std::string& name) const;
    bool unregister_model(const std::string& name);
    std::vector<std::string> list_models() const;
    size_t count() const;
    void clear();

    // Model persistence
    bool save_model(const std::string& name, const std::string& path = "");
    std::unique_ptr<AdaptiveLightGBMModel> load_model(const std::string& name, const std::string& path = "");
    bool model_exists(const std::string& name) const;

    // Batch operations
    std::vector<std::pair<std::string, Tensor>> predict_batch(
        const std::string& model_name,
        const std::vector<std::unordered_map<std::string, Datum>>& rows);

    // Model monitoring
    struct ModelStatus {
        std::string name;
        bool is_loaded;
        size_t features;
        float accuracy;
        float drift_score;
        size_t predictions;
        std::chrono::milliseconds avg_inference_time;
    };

    std::vector<ModelStatus> get_all_model_status() const;
    bool save_model_schema(const std::string& name, const std::string& schema_path);

    // Auto-reloading
    void auto_reload_models();
    void set_models_directory(const std::string& dir);

private:
    std::string get_model_path(const std::string& name) const;
    std::string get_schema_path(const std::string& name) const;
    void scan_models_directory();
};

} // namespace ai
} // namespace esql

#endif // MODEL_REGISTRY_H
