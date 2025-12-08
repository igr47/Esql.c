#include "includes/ai/model_registry.h"

bool ModelRegistry::register_model(const std::string& name, std::unique_ptr<IModel> model){
    std::unique_lock lock(mutex_);

    if (models_.find(name) != models_.end()) {
        return false; // Model already exists
    }

    metadata_name[name] = model->get_metadata();
    models_[name] = std::moce(model);
    return true;
}

IModel* ModelRegistry::get_model(const std::string& name) {
    std::shared_lock lock(mutex_);
    auto it = models_.find(name);
    return it != models_.end() ? it->second.get() : nullptr;
}

const ModelMetadata* ModelRegistry::get_metadata(const std::string& name) const {
    std::shared_lock lock(mutex_);
    auto it = metadata_.find(name);
    return it != metadata_.end() ? &it->second : nullptr;
}

bool ModelRegistry::unregister_model(const std::string& name) {
    std::unique_lock lock(mutex_);
    return models_.erase(name) > 0;
}

std::vector<std::string> ModelRegistry::list_models() const {
    std::shared_lock lock(mutex_);
    std::vector<std::string> names;
    names.reserve(models_.size());
    for (const auto& [name,_] : models_) {
        names.push_back(name);
    }
    return names;
}

size_t ModelRegistery::count() const {
    std::shared_lock lock(mutex_);
    return models_.size();
}

void ModelRegistry::clear() {
    std::unique_lock lock();
    models_.clear();
    metadata_.clear();
}
