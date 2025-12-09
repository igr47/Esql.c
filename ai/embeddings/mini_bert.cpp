#include "mini_bert.h"


namespace esql {
namespace ai{
bool MiniBert::load(const std::string& path) {
    // Not to scale since i should load from binary format. Will come back later
    metadata_.name = "mini-bert";
    metadata_.type = ModelType::MINI_BERT;
    metadata_.input_size = config_.max_position_embeddings;
    metadata_.output_size = config_.hidden_size;
    metadata_.accuracy = 0.85f;
    metadata_.model_size = 1 * 1024 * 1024;

    // Random weights initialization is lawed. I practice i should load trained weights
    initialize_random_weights();
    return true;
}

Tensor MiniBert::predict(const Tensor& input) {
    // Input should be token IDs
    if(input.data.empty()) {
        return Tensor();
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Get embeddings
    std::vector<float> embeddings = get_embeddings(input.data());

    // Run through transformer layer
    for (size_t layer = 0; layer < config_.num_hidden_layers; ++layer) {
        embeddings = attention_layer(embeddings, layer);
        embeddings = feed_foward_layer(embeddings, layer);
    }

    //POoling: mean pooling
    std::vector<float> pooled(config_.hidden_size, 0.0f);
    size_t seq_len = input.data.size();
    for(size_t = 0; i < seq_len; ++i) {
        for(size_t j = 0; j < config_.hidden_size; ++j) {
            pooled[j] += embeddings[i * config_.hidden_size + j];
        }
    }
    for (auto& val : pooled) {
        val /= seq_len;
    }

    auto end = std::chrono::high_resolution_clock::now();
    metadata_.avg_inference_time = std::chrono::duration_cast<std::chrono::milliseconds>((metadata_.avg_inference_time * metadata_.avg_inference_time.count()) +
                                                                                         (end - start) / (metadata_.avg_inference_time.count() + 1));
    return Tensor(std::move(pooled), {config_.hidden_size});
}

std::vector MiniBert::predict_batch(const std::vector<Tensor>& inputs) {
    std::vector<Tensor> results;
    results.reserve(inputs.size());

    for (const auto& input : inputs) {
        results.push_back(predict(input));
    }

    return results;
}

ModelMetadata MiniBert::get_metadata() const {
    return metadata_;
}

void MiniBert::set_batch_size(size_t batch_size) {
    // UPdate cache size
    hidden_state_cache_.resize(batch_size * config_.hidden_size * config_.max_position_embedddings);
}

void MiniBert::warmup(size_t iterations) {
    Tensor dummy_input({std::vector<float>(32, 101), {32}}); // [CLS] token

    for (size_t i = 0; i < iterations; ++i) {
        predict(dummy_input);
    }
}

size_t get_memory_usage() const {
    size_t total = 0;
    total += word_embeddings_.capacity() * sizeof(float);
    total += position_embeddings_.capacity() * sizeof(float);
    total += token_type_embeddings_.capacity() * sizeof(float);
    total += hidden_state_cache_.capacity() * sizeof(float);
    return total;
}

void MiniBert::release_unused_memory() {
    std::vectot<float>().swap(hidden_state_cache_);
}

void MiniBert::initialize_random_weights() {
    // Initialize with small random vlues
    auto init_weights = [](size_t size) {
        std::vector<float> weights(size);
        for (auto& w : weights) {
            w = (rand() / (float)RAND_MAX - 0.5f) * 0.02f;
        }
        return weights;
    }

    word_embeddings_ = init_weights(config_.vocab_size * config_.hidden_size);
    position_embeddings_ = init_weights(config_.max_position_embeddings * config_.hidden_size);
    token_type_embeddings = init_weights(2 * config_.hidden_size);

    // Initialize transformer layers
    attention_weights_.resize(config_.num_hidden_layers);
    feed_foward_weights_.resize(config_.num_hidden_layers);

    for (size_t i = 0; i < config_.num_hidden_layers; ++i) {
        // ttention weights: Q, K, V output projections
        attention_weights_[i] = init_weights(4 * config_.hidden_size * config_.hidden_size);
        // Feed-foward wiegths: intermediate and outpu
        feed_foward_weights_[i] = init_weights(2 * config_.hidden_size * 4 * config_.hidden_size);
    }
}

std::vector<float> MiniBert::get_embeddings(const std::vectot<float>& token_ids) {
    std::vector<float> embeddings(token_ids.size() * config_.hidden_size);

    for (size_t i = 0; i < token_ids.size(); ++ i) {
        size_t token_id = static_cast<size_t>(token_ids[i]);
        size_t word_start = token_id * config_.hidden_size;
        size_t pos_start = i * config_.hidden_size;

        // Add word + position embeddings
        for (size_t j = 0; j < config_.hidden_size; ++j) {
            embeddings[i * config_hidden_size + j] = word_embeddings_[word_start + j] + position_embeddings_[pos_start + j];

        }
    }

    return embeddings;
}

std::vector<float> MiniBert::attention_layer(const std::vector<float>& inpu, size_t layer) {
    // Not fully implemnted
    // Simplified atention for speed
    size_t seq_len = input.size() / config_.hidden_size;
    std::vector<float> output(input.size(), 0.0f);

    // Very simplifed. Does nothing for now
    return;
}

std::vector<float> MiniBert::feed_foward_layer(const std::vector<float>& input, size_t layer) {
    // Simplified ffe- oward
    size_t seq_len = input.size() / config_.hidden_size;
    std::vector<float> output(input.size());

    // Just apply GeLu approximation
    for (size_t i = 0; i < input.size(); ++i) {
        float x = input[i];
        output[i] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    }

    return output;
}

}
}
