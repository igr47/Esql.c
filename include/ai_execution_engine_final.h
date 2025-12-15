// ============================================
// ai_execution_engine_final.h
// ============================================
#ifndef AI_EXECUTION_ENGINE_FINAL_H
#define AI_EXECUTION_ENGINE_FINAL_H

#include "ai_execution_engine.h"
#include "execution_engine_includes/executionengine_main.h"
#include "ai_parser.h"
#include "data_analysis.h"
#include "datum.h"
#include "schema_discovery.h"
#include <mutex>
#include <queue>
#include <future>

class Datum;

class AIExecutionEngineFinal {
private:

    // Reference to the base execution engine
    ExecutionEngine& base_engine_;
    Database& db_;
    fractal::DiskStorage& storage_;

    std::unique_ptr<AIExecutionEngine> ai_engine_;
    std::unique_ptr<esql::DataExtractor> data_extractor_;
    std::unique_ptr<esql::ai::SchemaDiscoverer> schema_discoverer_;

    // Thread pool for parallel execution
    std::vector<std::thread> worker_threads_;
    std::queue<std::function<void()>> task_queue_;
    std::mutex queue_mutex_;
    std::condition_variable cv_;
    bool stop_workers_ = false;

    // Cache for frequently used models
    std::unordered_map<std::string,
    std::shared_ptr<esql::ai::AdaptiveLightGBMModel>> model_cache_;
    std::shared_mutex cache_mutex_;
    size_t max_cache_size_ = 20;

    // Statistics
    struct AIStats {
        size_t total_ai_queries = 0;
        size_t total_predictions = 0;
        size_t total_training_operations = 0;
        std::chrono::milliseconds total_execution_time;
        std::unordered_map<std::string, size_t> query_type_counts;
    };
    AIStats ai_stats_;
    mutable std::mutex stats_mutex_;

public:
    AIExecutionEngineFinal(ExecutionEngine& base_engine,Database& db, fractal::DiskStorage& storage);
    ~AIExecutionEngineFinal();

    // Main execution override
    ExecutionEngine::ResultSet execute(std::unique_ptr<AST::Statement> stmt);

    // Getter for the base engine (for access to its methods)
    ExecutionEngine& getBaseEngine() { return base_engine_; }
    Database& getDatabase() { return db_; }
    fractal::DiskStorage& getStorage() { return storage_; }

    // Enhanced AI methods
    ExecutionEngine::ResultSet executeCreateModel(AST::CreateModelStatement& stmt);
    ExecutionEngine::ResultSet executeCreateOrReplaceModel(AST::CreateOrReplaceModelStatement& stmt);
    ExecutionEngine::ResultSet executeDescribeModel(AST::DescribeModelStatement& stmt);
    ExecutionEngine::ResultSet executeAnalyzeData(AST::AnalyzeDataStatement& stmt);
    ExecutionEngine::ResultSet executeCreatePipeline(AST::CreatePipelineStatement& stmt);
    ExecutionEngine::ResultSet executeBatchAI(AST::BatchAIStatement& stmt);

    // SQL-integrated AI functions
    ExecutionEngine::ResultSet executeSelectWithAIFunctions(AST::SelectStatement& stmt);
    ExecutionEngine::ResultSet executeCreateTableAsAIPrediction(AST::CreateTableStatement& stmt);

    // Model lifecycle management
    bool deployModelToProduction(const std::string& model_name);
    bool rollbackModel(const std::string& model_name, const std::string& version);
    std::vector<std::string> getModelVersions(const std::string& model_name);

    // Performance optimization
    void preloadPopularModels(const std::vector<std::string>& model_names);
    void warmupModelCache();
    void clearModelCache();

    // Monitoring and observability
    AIStats getAIStats() const;
    std::vector<esql::ai::ModelRegistry::ModelStatus> getModelHealthStatus();
    std::unordered_map<std::string, float> getModelPerformanceMetrics();

    // Data quality and drift detection
    ExecutionEngine::ResultSet detectDataDrift(const std::string& model_name, const std::string& reference_table,const std::string& current_table);
    ExecutionEngine::ResultSet validateModelInputs(const std::string& model_name,const std::string& input_table);

    // A/B testing for models
    ExecutionEngine::ResultSet runABTest(const std::string& model_a,const std::string& model_b,const std::string& test_data_table,const std::string& metric = "accuracy");

    // Feature store integration
    ExecutionEngine::ResultSet registerFeatures(const std::string& feature_set_name,const std::vector<std::string>& feature_definitions);
    ExecutionEngine::ResultSet getFeatureStatistics(const std::string& feature_set_name);

private:
    void initializeWorkerThreads(size_t num_threads = 4);
    void stopWorkerThreads();
    void addTask(std::function<void()> task);

    // Enhanced helper methods
    std::shared_ptr<esql::ai::AdaptiveLightGBMModel> getOrLoadModel(const std::string& model_name);
    bool ensureModelLoaded(const std::string& model_name);

    ExecutionEngine::ResultSet executeAIFunctionInSelect(const AST::AIFunctionCall& func_call,const std::vector<std::unordered_map<std::string, std::string>>& data);
    ExecutionEngine::ResultSet executeAIScalarFunction(const AST::AIScalarExpression& expr,const std::vector<std::unordered_map<std::string, std::string>>& data);
    ExecutionEngine::ResultSet executeModelFunction(const AST::ModelFunctionCall& func_call,const std::vector<std::unordered_map<std::string, std::string>>& data);

    // Feature engineering
    std::vector<std::vector<float>> engineerFeatures(const std::vector<std::unordered_map<std::string, std::string>>& raw_data,const std::vector<esql::ai::FeatureDescriptor>& feature_descriptors);

    // Model validation and testing
    std::unordered_map<std::string, float> validateModel(const std::string& model_name,const std::string& validation_table);

    // Data preprocessing
    std::vector<std::unordered_map<std::string, Datum>> preprocessData(const std::vector<std::unordered_map<std::string, std::string>>& raw_data,const esql::ai::ModelSchema& schema);

    // Result formatting
    std::string formatAIPredictionResult(const esql::ai::Tensor& prediction,const esql::ai::ModelSchema& schema);

    // Error handling and recovery
    void handleModelFailure(const std::string& model_name,const std::exception& e);
    bool retryModelOperation(const std::function<bool()>& operation,int max_retries = 3);

    // Security and access control
    bool validateModelAccess(const std::string& model_name,const std::string& user_role = "default");
    bool validateTrainingDataAccess(const std::string& table_name,const std::string& user_role = "default");

    // Resource management
    size_t estimateMemoryRequirements(const std::string& operation,const std::string& model_name = "",size_t data_size = 0);
    bool checkResourceAvailability(size_t required_memory_mb, size_t required_cpu_cores = 1);

    // Logging and auditing
    void logAIOperation(const std::string& operation,const std::string& model_name,const std::string& status,const std::string& details = "");

    // Backup and recovery
    bool backupModel(const std::string& model_name,const std::string& backup_path = "");
    bool restoreModel(const std::string& model_name,const std::string& backup_path = "");

    // Model registry helpers
    bool registerModelWithMetadata(const std::string& name,std::unique_ptr<esql::ai::AdaptiveLightGBMModel> model,const std::unordered_map<std::string, std::string>& metadata);

    // Query parsing helpers
    bool isAIFunctionExpression(const AST::Expression* expr);
    std::vector<std::string> extractAIFunctionsFromSelect(AST::SelectStatement& stmt);

    // Batch operation helpers
    ExecutionEngine::ResultSet executeParallelTraining(const std::vector<AST::TrainModelStatement>& training_tasks);
    ExecutionEngine::ResultSet executeParallelPrediction(const std::vector<AST::PredictStatement>& prediction_tasks);

    // Schema evolution
    ExecutionEngine::ResultSet migrateModelSchema(const std::string& model_name,const esql::ai::ModelSchema& new_schema);

    // Experiment tracking
    ExecutionEngine::ResultSet createExperiment(const std::string& experiment_name,const std::unordered_map<std::string, std::string>& params);
    ExecutionEngine::ResultSet logExperimentResult(const std::string& experiment_name,const std::unordered_map<std::string, float>& results);
    bool isValidModelName(const std::string& name) const;
};

#endif // AI_EXECUTION_ENGINE_FINAL_H
