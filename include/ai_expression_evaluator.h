// ai_expression_evaluator.h
#ifndef AI_EXPRESSION_EVALUATOR_H
#define AI_EXPRESSION_EVALUATOR_H

#include "ai_grammer.h"
#include "ai/model_registry.h"
#include "ai/lightgbm_model.h"
#include "database.h"
#include <memory>
#include <unordered_map>
#include <string>

class ExecutionEngine;

class AIExpressionEvaluator {
private:
    esql::ai::ModelRegistry& registry_;
    Database& db_;
    ExecutionEngine& engine_;

    // Cache for frequently used predictions
    struct PredictionCache {
        std::string model_name;
        std::string row_hash;
        std::vector<std::string> results;
        std::chrono::steady_clock::time_point timestamp;
    };

    std::vector<PredictionCache> cache_;
    size_t max_cache_size_ = 1000;
    std::chrono::seconds cache_ttl_{60}; // 1 minute cache

public:
    AIExpressionEvaluator(Database& db, ExecutionEngine& engine_);

    // Main evaluation methods
    std::string evaluateAIFunction(const AST::AIFunctionCall* func,
                                   const std::unordered_map<std::string, std::string>& row);

    std::string evaluateAIScalar(const AST::AIScalarExpression* expr,
                                 const std::unordered_map<std::string, std::string>& row);

    std::string evaluateModelFunction(const AST::ModelFunctionCall* func,
                                      const std::unordered_map<std::string, std::string>& row);

    // Batch prediction for optimization
    std::vector<std::string> evaluateBatchPredictions(
        const std::vector<const AST::AIFunctionCall*>& functions,
        const std::vector<std::unordered_map<std::string, std::string>>& rows);

private:
    // Core prediction logic
    std::vector<float> extractFeaturesFromRow(
        const std::string& model_name,
        const std::unordered_map<std::string, std::string>& row);

    std::string formatPredictionResult(
        const esql::ai::Tensor& prediction,
        const esql::ai::ModelSchema& schema,
        AST::AIFunctionType func_type,
        const std::unordered_map<std::string, std::string>& options);

    // Cache management
    std::string generateCacheKey(const std::string& model_name,
                                 const std::unordered_map<std::string, std::string>& row);
    std::optional<std::vector<std::string>> getCachedResult(const std::string& key);
    void cacheResult(const std::string& key, const std::vector<std::string>& result);
    void cleanupCache();

    // Model loading with caching
    std::shared_ptr<esql::ai::AdaptiveLightGBMModel> getModel(const std::string& model_name);
    std::unordered_map<std::string, std::shared_ptr<esql::ai::AdaptiveLightGBMModel>> model_cache_;
    std::shared_mutex model_cache_mutex_;
};

#endif // AI_EXPRESSION_EVALUATOR_H
