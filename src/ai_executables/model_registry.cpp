#include "ai/model_registry.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <string>

namespace esql {
namespace ai {

bool ModelRegistry::register_model(const std::string& name, std::unique_ptr<AdaptiveLightGBMModel> model) {
    std::unique_lock lock(mutex_);

    if (models_.find(name) != models_.end()) {
        std::cerr << "[ModelRegistry] Model already exists: " << name << std::endl;
        return false;
    }

    metadata_[name] = model->get_metadata();
    models_[name] = std::move(model);

    std::cout << "[ModelRegistry] Registered model: " << name << std::endl;
    return true;
}

AdaptiveLightGBMModel* ModelRegistry::get_model(const std::string& name) {
    std::shared_lock lock(mutex_);

    auto it = models_.find(name);
    if (it != models_.end()) {
        return it->second.get();
    }

    // Try to load from disk
    lock.unlock();
    auto model = load_model(name);

    if (model) {
        std::unique_lock write_lock(mutex_);
        models_[name] = std::move(model);
        metadata_[name] = models_[name]->get_metadata();
        return models_[name].get();
    }

    return nullptr;
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

    for (const auto& [name, _] : models_) {
        names.push_back(name);
    }

    // Also check disk for models not in memory
    try {
        if (std::filesystem::exists(models_directory_)) {
            for (const auto& entry : std::filesystem::directory_iterator(models_directory_)) {
                if (entry.path().extension() == ".txt") {
                    std::string model_name = entry.path().stem().string();
                    if (std::find(names.begin(), names.end(), model_name) == names.end()) {
                        names.push_back(model_name);
                    }
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[ModelRegistry] Error scanning models directory: " << e.what() << std::endl;
    }

    return names;
}

size_t ModelRegistry::count() const {
    std::shared_lock lock(mutex_);
    return models_.size();
}

void ModelRegistry::clear() {
    std::unique_lock lock(mutex_);
    models_.clear();
    metadata_.clear();
}

bool ModelRegistry::save_model(const std::string& name, const std::string& path) {
    std::shared_lock lock(mutex_);

    auto it = models_.find(name);
    if (it == models_.end()) {
        std::cerr << "[ModelRegistry] Cannot save - model not found: " << name << std::endl;

        // Try to check if model exists on disk but not in memory
        std::string disk_path = get_model_path(name);
        if (std::filesystem::exists(disk_path)) {
            std::cout << "[ModelRegistry] Model exists on disk but not in memory: " << name << std::endl;
            return true; // Model already saved on disk
        }

        return false;
    }

    std::string save_path = path.empty() ? get_model_path(name) : path;

        // Ensure directory exists
    std::filesystem::create_directories(std::filesystem::path(save_path).parent_path());

    bool success = it->second->save(save_path);

    //return it->second->save(save_path);
    if (success) {
        std::cout << "[ModelRegistry] Model saved to: " << save_path << std::endl;
    } else {
        std::cerr << "[ModelRegistry] Failed to save model: " << name << std::endl;
        // Save at least the schema even if model can't be saved
        save_model_schema(name, save_path + ".schema.json");
    }

    return success;
}

bool ModelRegistry::save_model_schema(const std::string& name, const std::string& schema_path) {
    std::shared_lock lock(mutex_);

    auto it = models_.find(name);
    if (it == models_.end()) {
        return false;
    }

    const auto& schema = it->second->get_schema();

    try {
        std::ofstream schema_file(schema_path);
        nlohmann::json j = schema.to_json();
        schema_file << j.dump(2);
        std::cout << "[ModelRegistry] Schema saved to: " << schema_path << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[ModelRegistry] Failed to save schema: " << e.what() << std::endl;
        return false;
    }
}

std::unique_ptr<AdaptiveLightGBMModel> ModelRegistry::load_model(const std::string& name, const std::string& path) {
    std::string load_path = path.empty() ? get_model_path(name) : path;

    if (!std::filesystem::exists(load_path)) {
        std::cerr << "[ModelRegistry] Model file not found: " << load_path << std::endl;
        return nullptr;
    }

    auto model = std::make_unique<AdaptiveLightGBMModel>();
    if (!model->load(load_path)) {
        std::cerr << "[ModelRegistry] Failed to load model: " << name << std::endl;
        return nullptr;
    }

    model->warmup(5);
    std::cout << "[ModelRegistry] Loaded model from disk: " << name << std::endl;

    return model;
}

bool ModelRegistry::model_exists(const std::string& name) const {
    std::shared_lock lock(mutex_);

    if (models_.find(name) != models_.end()) {
        return true;
    }

    // Check disk
    return std::filesystem::exists(get_model_path(name));
}

std::vector<std::pair<std::string, Tensor>> ModelRegistry::predict_batch(
    const std::string& model_name,
    const std::vector<std::unordered_map<std::string, Datum>>& rows) {

    auto* model = get_model(model_name);
    if (!model) {
        throw std::runtime_error("Model not found: " + model_name);
    }

    std::vector<std::pair<std::string, Tensor>> results;
    results.reserve(rows.size());

    for (const auto& row : rows) {
        try {
            Tensor prediction = model->predict_row(row);
            results.emplace_back("success", std::move(prediction));
        } catch (const std::exception& e) {
            // Create error tensor
            Tensor error_tensor({0.0f}, {1});
            results.emplace_back(std::string("error: ") + e.what(), std::move(error_tensor));
        }
    }

    return results;
}

std::vector<ModelRegistry::ModelStatus> ModelRegistry::get_all_model_status() const {
    std::shared_lock lock(mutex_);

    std::vector<ModelStatus> status_list;
    status_list.reserve(models_.size());

    for (const auto& [name, model] : models_) {
        auto metadata = model->get_metadata();
        auto schema = model->get_schema();

        ModelStatus status;
        status.name = name;
        status.is_loaded = true;
        status.features = schema.features.size();
        status.accuracy = schema.accuracy;
        status.drift_score = schema.drift_score;
        status.predictions = schema.stats.total_predictions;
        status.avg_inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            schema.stats.avg_inference_time
        );

        status_list.push_back(status);
    }

    return status_list;
}

void ModelRegistry::auto_reload_models() {
    std::unique_lock lock(mutex_);

    std::cout << "[ModelRegistry] Auto-reloading models from: " << models_directory_ << std::endl;

    try {
        if (!std::filesystem::exists(models_directory_)) {
            std::filesystem::create_directories(models_directory_);
            return;
        }

        size_t loaded_count = 0;
        for (const auto& entry : std::filesystem::directory_iterator(models_directory_)) {
            if (entry.path().extension() == ".txt") {
                std::string model_name = entry.path().stem().string();

                // Check if already loaded
                if (models_.find(model_name) != models_.end()) {
                    continue;
                }

                // Try to load
                auto model = std::make_unique<AdaptiveLightGBMModel>();
                if (model->load(entry.path().string())) {
                    model->warmup(3);
                    models_[model_name] = std::move(model);
                    metadata_[model_name] = models_[model_name]->get_metadata();
                    loaded_count++;

                    std::cout << "[ModelRegistry] Auto-loaded: " << model_name << std::endl;
                }
            }
        }

        std::cout << "[ModelRegistry] Auto-reload complete. Loaded " << loaded_count << " new models." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[ModelRegistry] Auto-reload error: " << e.what() << std::endl;
    }
}

void ModelRegistry::set_models_directory(const std::string& dir) {
    std::unique_lock lock(mutex_);
    models_directory_ = dir;
}

std::string ModelRegistry::get_model_path(const std::string& name) const {
    return models_directory_ + "/" + name + ".txt";
}

std::string ModelRegistry::get_schema_path(const std::string& name) const {
    return models_directory_ + "/" + name + ".schema.json";
}

void ModelRegistry::scan_models_directory() {
    // Already implemented in list_models and auto_reload_models
}

} // namespace ai
} // namespace esql
