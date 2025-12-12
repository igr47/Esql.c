
#ifndef AI_EXECUTION_ENGINE_H
#define AI_EXECUTION_ENGINE_H

#include "execution_engine_includes/executionengine_main.h"
#include "ai_grammer.h"
#include "data_extractor.h"
#include "ai/model_registry.h"
#include <memory>
#include <unordered_map>
#include <chrono>

class AIExecutionEngine {
private:
    Database& db_;
    fractal::DiskStorage& storage_;
    esql::DataExtractor data_extractor_;

public:
    AIExecutionEngine(Database& db, fractal::DiskStorage& storage);

    // Main execution method
    ExecutionEngine::ResultSet executeAIStatement(std::unique_ptr<AST::Statement> stmt);

    // Individual execution methods
    ExecutionEngine::ResultSet executeTrainModel(AST::TrainModelStatement& stmt);
    ExecutionEngine::ResultSet executePredict(AST::PredictStatement& stmt);
    ExecutionEngine::ResultSet executeShowModels(AST::ShowModelsStatement& stmt);
    ExecutionEngine::ResultSet executeDropModel(AST::DropModelStatement& stmt);
    ExecutionEngine::ResultSet executeModelMetrics(AST::ModelMetricsStatement& stmt);
    ExecutionEngine::ResultSet executeExplain(AST::ExplainStatement& stmt);
    ExecutionEngine::ResultSet executeFeatureImportance(AST::FeatureImportanceStatement& stmt);

    // Batch operations
    ExecutionEngine::ResultSet executeBatchPredict(const std::vector<AST::PredictStatement>& stmts);
    ExecutionEngine::ResultSet executeBatchTrain(const std::vector<AST::TrainModelStatement>& stmts);

    // Model management
    bool loadModel(const std::string& model_name);
    bool unloadModel(const std::string& model_name);
    bool saveModelToDisk(const std::string& model_name);
    bool deleteModelFromDisk(const std::string& model_name);

    // Performance monitoring
    struct PerformanceStats {
        size_t total_training_operations = 0;
        size_t total_prediction_operations = 0;
        std::chrono::milliseconds total_training_time;
        std::chrono::milliseconds total_prediction_time;
        size_t peak_memory_usage = 0;

        void reset();
        std::string toString() const;
    };

    PerformanceStats getPerformanceStats() const;
    void resetPerformanceStats();

    // Resource management
    void setMaxConcurrentModels(size_t max);
    void setModelCacheSize(size_t cache_size_mb);
    void cleanupUnusedModels();

private:
    // Internal helper methods
    std::unique_ptr<esql::ai::AdaptiveLightGBMModel> createModelFromConfig(
        AST::TrainModelStatement& stmt,
        const esql::ai::ModelSchema& schema);

    std::vector<std::vector<float>> extractFeaturesForTraining(
        const std::string& table,
        const std::vector<std::string>& feature_columns,
        const std::string& where_clause);

    std::vector<float> extractTargetForTraining(
        const std::string& table,
        const std::string& target_column,
        const std::string& where_clause);

    // Model training helpers
    std::unordered_map<std::string, std::string> prepareTrainingParameters(
        const std::unordered_map<std::string, std::string>& user_params,
        const std::string& problem_type);

    esql::ai::ModelSchema createModelSchema(
        AST::TrainModelStatement& stmt,
        const std::vector<std::vector<float>>& features,
        const std::vector<float>& labels);

    // Prediction helpers
    std::vector<std::unordered_map<std::string, std::string>> preparePredictionInput(
        const std::string& table,
        const std::string& where_clause,
        size_t limit);

    void savePredictionsToTable(AST::PredictStatement& stmt,const ExecutionEngine::ResultSet& predictions);

    // Output table creation
    void createModelOutputTable(AST::TrainModelStatement& stmt);
    void createPredictionOutputTable(AST::PredictStatement& stmt);

    // Metrics calculation
    void calculateTestMetrics(
        AST::ModelMetricsStatement& stmt,
        esql::ai::AdaptiveLightGBMModel& model,
        ExecutionEngine::ResultSet& result);

    std::unordered_map<std::string, float> calculateModelMetrics(
        esql::ai::AdaptiveLightGBMModel& model,
        const std::vector<std::vector<float>>& test_features,
        const std::vector<float>& test_labels);

    // Feature importance calculation
    std::vector<std::pair<std::string, float>> calculateFeatureImportance(
        esql::ai::AdaptiveLightGBMModel& model);

    // Explanation generation
    std::unordered_map<std::string, std::string> generateModelExplanation(
        esql::ai::AdaptiveLightGBMModel& model,
        const std::vector<float>& input_features);

    // Performance tracking
    PerformanceStats stats_;
    mutable std::mutex stats_mutex_;

    // Model cache
    std::unordered_map<std::string, std::unique_ptr<esql::ai::AdaptiveLightGBMModel>> model_cache_;
    mutable std::shared_mutex cache_mutex_;
    size_t max_cached_models_ = 10;

    // Resource limits
    size_t max_concurrent_models_ = 5;
    size_t max_training_memory_mb_ = 4096; // 4GB

    // Utility methods
    std::string generateUniqueTableName(const std::string& prefix);
    std::string formatFloat(float value, int precision = 4);
    std::string formatDateTime(const std::chrono::system_clock::time_point& tp);

    // Error handling
    void handleTrainingError(const std::string& model_name, const std::exception& e);
    void handlePredictionError(const std::string& model_name, const std::exception& e);

    // Cleanup
    void cleanupTemporaryTables();
    void cleanupTemporaryFiles();
};

#endif // AI_EXECUTION_ENGINE_H
