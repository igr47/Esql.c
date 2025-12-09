#pragma once
#ifndef MINI_BERT
#define MINI_BERT
#include "includes/ai/mode_interface.h"
#include <vector>
#include <string>
#include <cmath>

namespace esql {
namespace ai {

class MiniBert : public IModel {
private:
    // Tiny BERT-Like architechtutre for fast embedings
    struct TransaformerConfig {
    size_t vocab_size = 30522;
    size_t hidden_size = 128; // Reduced from 768 for speed
    size_t num_attention_heads = 4; // Reduced from 12
    size_t num_hidden_layers = 2; // Reduced from 12
    size_t max_position_embeddings = 512;
    };

    TransformerConfig config_;
    ModelMetadata metadata_;

    // Embedding tble
    std::vector<float> word_embeddings_;
    std::vector<float> position_embeddings_;
    std::vector<float> token_type_embeddings_;

    // Transformer weights
    std::vector<std::vector<float>> attention_weights_;
    std::vector<std::vector<float>> feed_foward_weights_;

    // Cache for performance
    std::vector<float> hidden_state_cache_;

public:
    bool load(const std::string& path) override;
    Tensor predict(const Tensor& input) override;
    std::vector<Tensor> predict_batch(const std::vector<Tensor>& inputs) override;
    ModelMetadata get_metadata() const override;
    void set_batch_size(size_t batch_size) override;
    size_t warmup(size_t iterations) override;
    size_t get_memory_usage() const override;
    void release_unused_memory() override;

private:
    void initialize_random_weights();
    std::vector<float> get_embeddings(const std::vector<float>& token_ids);
    std::vector<float> attention_layer(const std::vector<float>& input,size_t layer);
    std::vector<float> feed_foward_layer(const std::vector<float>& input, size_t layer);


}
}
}
