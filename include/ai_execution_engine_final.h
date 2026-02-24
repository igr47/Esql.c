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

    class HyperparameterValidator {
    private:
        static const std::set<std::string> valid_parameters;
        static const std::unordered_map<std::string, std::pair<float, float>> param_ranges;
        static const std::unordered_map<std::string, std::set<std::string>> param_dependencies;

    public:
        static void validate(const std::unordered_map<std::string, std::string>& params,const std::string& algorithm,const std::string& problem_type,size_t num_classes = 0);
        static std::unordered_map<std::string, std::string> getDefaultParameters(const std::string& algorithm,const std::string& problem_type,size_t sample_count,size_t feature_count,size_t num_classes = 0);

    private:
        static void validateParameterRange(const std::string& param,const std::string& value_str,const std::pair<float, float>& range,std::vector<std::string>& errors);
        static void checkRequiredParameters(const std::unordered_map<std::string, std::string>& params,const std::string& algorithm,const std::string& problem_type,size_t num_classes,std::vector<std::string>& errors);
        static void checkParameterDependencies(const std::unordered_map<std::string, std::string>& params,const std::string& algorithm,std::vector<std::string>& errors,std::vector<std::string>& warnings);
        static void checkAlgorithmConstraints(const std::unordered_map<std::string, std::string>& params,const std::string& algorithm,std::vector<std::string>& errors);
    };

    class DataPreprocessor {
    public:
        static esql::DataExtractor::TrainingData preprocess(const esql::DataExtractor::TrainingData& original_data,const std::string& sampling_method,float sampling_ratio,
            bool feature_scaling,const std::string& scaling_method,const std::vector<esql::ai::FeatureDescriptor>& feature_descriptors,int seed);
        static esql::DataExtractor::TrainingData applySampling(const esql::DataExtractor::TrainingData& data,const std::string& method,float ratio,int seed);
        static esql::DataExtractor::TrainingData applyScaling(const esql::DataExtractor::TrainingData& data,const std::string& method,const std::vector<esql::ai::FeatureDescriptor>& feature_descriptors);
    };

    class FeatureSelector {
    public:
        static std::pair<esql::DataExtractor::TrainingData, std::vector<size_t>> selectFeatures(const esql::DataExtractor::TrainingData& data,
            const std::vector<esql::ai::FeatureDescriptor>& original_features,const std::string& method,int max_features,const std::vector<float>& labels);
    private:
        static std::vector<float> calculateFeatureImportance(const esql::DataExtractor::TrainingData& data,const std::vector<float>& labels,const std::string& method);
        static float calculateCorrelation(const esql::DataExtractor::TrainingData& data,const std::vector<float>& labels,size_t feature_idx);
        static std::vector<size_t> selectTopFeatures(const std::vector<float>& importance_scores,int max_features);
    };

       // Hyperparameter tuner
    class HyperparameterTuner {
    private:
        struct TrialResult {
            std::unordered_map<std::string, std::string> parameters;
            float score;
            std::chrono::milliseconds duration;
            std::unordered_map<std::string, float> metrics;

            bool operator<(const TrialResult& other) const {
                return score < other.score;  // For max-heap
            }
        };

    public:
        static std::unordered_map<std::string, std::string> tune(const esql::DataExtractor::TrainingData& data,const std::string& algorithm,
            const std::string& problem_type,const AST::TuningOptions& tuning_options,const std::vector<esql::ai::FeatureDescriptor>& feature_descriptors,int seed);

    private:
        struct SearchSpace {
            std::unordered_map<std::string, std::vector<std::string>> categorical;
            std::unordered_map<std::string, std::pair<float, float>> continuous;
            std::unordered_map<std::string, std::pair<int, int>> integer;
        };

        static SearchSpace generateSearchSpace(const std::string& algorithm,const std::string& problem_type,const AST::TuningOptions& tuning_options);
        static std::unordered_map<std::string, std::string> generateParameterCombination(const SearchSpace& space,const std::string& method,std::mt19937& rng);
        static float evaluateSingleFold(const esql::DataExtractor::TrainingData& data,const std::unordered_map<std::string, std::string>& params,const std::string& algorithm,const std::string& problem_type,const std::vector<esql::ai::FeatureDescriptor>& feature_descriptors,int seed);
        static float evaluateParameters(const esql::DataExtractor::TrainingData& data,const std::unordered_map<std::string, std::string>& params,const std::string& algorithm,const std::string& problem_type,const std::vector<esql::ai::FeatureDescriptor>& feature_descriptors,int folds,int seed);
        static float evaluateModel(const esql::ai::AdaptiveLightGBMModel& model,const esql::DataExtractor::TrainingData& test_data,const std::string& problem_type);
        static float calculateAccuracy(const std::vector<float>& predictions,const std::vector<float>& labels);
        static float calculateRSquared(const std::vector<float>& predictions,const std::vector<float>& labels);
        static float calculateQuantileScore(const std::vector<float>& predictions,const std::vector<float>& labels);
    };

    // Reference to the base execution engine
    ExecutionEngine& base_engine_;
    Database& db_;
    fractal::DiskStorage& storage_;
    std::unique_ptr<Visualization::ImPlotSimulationPlotter> plotter_;
    std::atomic<bool> plotting_started_{false};
    std::thread plotting_thread_;
    std::mutex plot_mutex_;

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

    //Hyper parameter preparation function
    std::unordered_map<std::string, std::string> prepareHyperparameters(AST::CreateModelStatement& stmt,const esql::DataExtractor::TrainingData& training_data,const std::string& detected_problem_type,size_t num_classes);
    void applyTrainingOptions(std::unordered_map<std::string, std::string>& params,const AST::TrainingOptions& options);
    void logTrainingParameters(const std::string& model_name,const std::unordered_map<std::string, std::string>& params,const std::string& problem_type);

    // Enhanced AI methods
    ExecutionEngine::ResultSet executeCreateModel(AST::CreateModelStatement& stmt);
    ExecutionEngine::ResultSet executeCreateOrReplaceModel(AST::CreateOrReplaceModelStatement& stmt);
    ExecutionEngine::ResultSet executeDescribeModel(AST::DescribeModelStatement& stmt);
    ExecutionEngine::ResultSet executeAnalyzeData(AST::AnalyzeDataStatement& stmt);
    ExecutionEngine::ResultSet executeCreatePipeline(AST::CreatePipelineStatement& stmt);
    ExecutionEngine::ResultSet executeBatchAI(AST::BatchAIStatement& stmt);
    ExecutionEngine::ResultSet executeForecast(AST::ForecastStatement& stmt);
    ExecutionEngine::ResultSet executeDetectAnomaly(AST::DetectAnomalyStatement& stmt);
    ExecutionEngine::ResultSet executeSimulate(AST::SimulateStatement& stmt);
    std::chrono::seconds parseTimeInterval(const std::string& interval);
    //void saveSimulationResults(const std::string& table_name,const ExecutionEngine::ResultSet& results);

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

       // Helper methods for executeAnalyzeData
    std::vector<std::unordered_map<std::string, esql::Datum>> extract_chunk(esql::DataExtractor::DataCursor& cursor, size_t chunk_size);

    void format_table_output(ExecutionEngine::ResultSet& result,const esql::analysis::ProfessionalComprehensiveAnalysisReport& report,const std::string& analysis_type,const AST::AnalyzeDataStatement& stmt);

    void format_summary_output(ExecutionEngine::ResultSet& result,const esql::analysis::ProfessionalComprehensiveAnalysisReport& report);

    void format_correlation_output(ExecutionEngine::ResultSet& result,const esql::analysis::ProfessionalComprehensiveAnalysisReport& report);

    void format_importance_output(ExecutionEngine::ResultSet& result,const esql::analysis::ProfessionalComprehensiveAnalysisReport& report,const std::string& target_column);

    void format_clustering_output(ExecutionEngine::ResultSet& result,const esql::analysis::ProfessionalComprehensiveAnalysisReport& report);

    void format_outlier_output(ExecutionEngine::ResultSet& result,const esql::analysis::ProfessionalComprehensiveAnalysisReport& report);

    void format_distribution_output(ExecutionEngine::ResultSet& result,const esql::analysis::ProfessionalComprehensiveAnalysisReport& report);

    void format_quality_output(ExecutionEngine::ResultSet& result,const esql::analysis::ProfessionalComprehensiveAnalysisReport& report);

    void format_timeseries_output(ExecutionEngine::ResultSet& result,const esql::analysis::ProfessionalComprehensiveAnalysisReport& report);

    void format_comprehensive_output(ExecutionEngine::ResultSet& result,const esql::analysis::ProfessionalComprehensiveAnalysisReport& report);

    void format_insights_output(ExecutionEngine::ResultSet& result,const esql::analysis::ProfessionalComprehensiveAnalysisReport& report);

    // Enhanced helper methods
    std::shared_ptr<esql::ai::AdaptiveLightGBMModel> getOrLoadModel(const std::string& model_name);
    bool ensureModelLoaded(const std::string& model_name);
    bool saveModelToDisk(const std::string& model_name);

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
