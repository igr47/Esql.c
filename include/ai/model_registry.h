#pragma once
#ifndef MODEL_REGISTRY_H
#define MODEL_REGISTRY_H

#include "model_interface.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <shared_mutex>

namespace esql {
namespace ai {

class ModelRegistry {
private:
    ModelRegistry() = default;
    ~ModelREgistry() = default;

    std::unordered_map<std::string, std::unique_ptr<IModel>> models_;
    std::unordered_map<std::string, ModelMetadata> metadata_;
    mutable std::shared_mutex mutex_;

public:
    static ModelRegistry& instance() {
        static ModelRegistry instance;
        return instance;
    }

    // Disable copying
    ModelRegistry(const ModelRegistry&) = delete;
    ModelRegistry& operator=(const ModelRegistry&) = delete;

    // Core methods
    bool register_model(const std::string& name, std::unique_ptr<IModel> model);
    IModel* get_model(cons std::string& name);
    const ModelMetadata* get_metadata(const std::string& name) const;
    bool unregister_model(const std::string& name);
    std::vector<std::string> list_models() const;
    size_t count() const;
    void clear();
};
}
}

#endif
