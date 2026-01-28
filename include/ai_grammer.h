#ifndef AI_GRAMMAR_H
#define AI_GRAMMAR_H

#include "scanner.h"
#include "parser.h"
#include <vector>
#include <string>
#include <memory>
#include <nlohmann/json.hpp>
#include <unordered_map>

// AI-specific token types to add to Token::Type enum
/*
    // AI/ML Operations
    TRAIN, PREDICT, INFERENCE, MODEL, MODELS, USING, WITH_MODEL,
    FEATURES, TARGET, TEST_SPLIT, ITERATIONS, ACCURACY, PRECISION,
    RECALL, F1_SCORE, CONFUSION_MATRIX, SAVE_MODEL, LOAD_MODEL,
    DROP_MODEL, SHOW_MODELS, MODEL_METRICS, FEATURE_IMPORTANCE,
    HYPERPARAMETERS, EPOCHS, BATCH_SIZE, LEARNING_RATE, REGULARIZATION,
    CROSS_VALIDATION, INFER, FORECAST, ANOMALY_DETECTION, CLUSTER,
    CLASSIFY, REGRESS, EXPLAIN, SHAP_VALUES,

    // Model types
    LIGHTGBM, XGBOOST, CATBOOST, RANDOM_FOREST, LINEAR_REGRESSION,
    LOGISTIC_REGRESSION, NEURAL_NETWORK, KMEANS, SVM,

    // Evaluation metrics
    MAE, MSE, RMSE, R2, AUC_ROC, AUC_PR,

    // Feature engineering
    SCALER, NORMALIZER, ONE_HOT_ENCODER, LABEL_ENCODER, PCA,
    FEATURE_SELECTION, FEATURE_EXTRACTION,

    // Time series
    TIMESERIES, LAG, ROLLING_MEAN, EXPONENTIAL_SMOOTHING,
*/

// AI SQL statement types
namespace AST {
    class AIStatement : public Statement {
    public:
        virtual ~AIStatement() = default;
        virtual std::string toEsql() const = 0;
    };

    enum class AIFunctionType {
        PREDICT,            // Generic prediction
        PREDICT_CLASS,      // Class prediction with probabilities
        PREDICT_VALUE,      // Regression value prediction
        PREDICT_PROBA,      // Probability for classification
        PREDICT_CLUSTER,    // Cluster assignment
        PREDICT_ANOMALY,    // Anomaly score
        EXPLAIN,           // Model explanation/SHAP values
        TRAIN_MODEL,       // Model training function
        MODEL_METRICS,     // Get model metrics
        FEATURE_IMPORTANCE // Get feature importance
    };

    // Training options structure for CREATE MODEL
    struct TrainingOptions {
        bool cross_validation = false;
        int cv_folds = 5;
        bool early_stopping = true;
        int early_stopping_rounds = 10;
        std::string validation_table;  // Separate validation table
        float validation_split = 0.2f;
        bool use_gpu = false;
        std::string device_type = "cpu";  // cpu or gpu
        int num_threads = -1;  // auto-detect if -1
        std::string metric = "auto";  // auto or specific metric
        std::string boosting_type = "gbdt";  // gbdt, dart, goss, rf
        int seed = 42;
        bool deterministic = true;

        nlohmann::json to_json() const;
        static TrainingOptions from_json(const nlohmann::json& j);
    };

    // Hyperparameter tuning options structure
    struct TuningOptions {
        bool tune_hyperparameters = false;
        std::string tuning_method = "grid";  // grid, random, bayesian
        int tuning_iterations = 10;
        int tuning_folds = 3;
        std::string scoring_metric = "auto";
        std::unordered_map<std::string, std::vector<std::string>> param_grid;
        std::unordered_map<std::string, std::pair<float, float>> param_ranges; // for random/bayesian
        bool parallel_tuning = true;
        int tuning_jobs = -1;  // -1 for all cores

        nlohmann::json to_json() const;
        static TuningOptions from_json(const nlohmann::json& j);
    };

    class TrainModelStatement : public AIStatement {
    public:
        std::string model_name;
        std::string algorithm = "LIGHTGBM";
        std::string source_table;
        std::string target_column;
        std::vector<std::string> feature_columns;
        std::unordered_map<std::string, std::string> hyperparameters;
        float test_split = 0.2f;
        int iterations = 100;
        bool save_model = true;
        bool if_not_exists = false;
        bool replace = false;
        std::string output_table;
        std::string where_clause;

        TrainModelStatement() = default;
        std::string toEsql() const override;
    };

    class CreateModelStatement : public AIStatement {
    public:
        std::string model_name;
        std::string algorithm;
        std::vector<std::pair<std::string, std::string>> features; // name, type
        std::string target_type; // "CLASSIFICATION", "REGRESSION", "CLUSTERING"
        std::unordered_map<std::string, std::string> parameters;

        TrainingOptions training_options;
        TuningOptions tuning_options;

        std::string data_sampling = "none";  // none, oversample, undersample, smote
        float sampling_ratio = 1.0f;
        bool feature_selection = false;
        std::string feature_selection_method = "auto";
        int max_features_to_select = -1;
        bool feature_scaling = true;
        std::string scaling_method = "standard";  // standard, minmax, robust

        std::string toEsql() const override;
        nlohmann::json to_json() const;
        static CreateModelStatement from_json(const nlohmann::json& j);
    };

    class ForecastStatement : public AIStatement {
    public:
	std::string model_name;
        std::string input_table;
        std::string time_column;
        std::vector<std::string> value_columns;
        size_t horizon = 1;  // Number of steps to forecast
        bool include_confidence = false;
        bool include_scenarios = false;
        size_t num_scenarios = 100;
        std::string output_table;
        std::string scenario_type = "monte_carlo"; // monte_carlo, bootstrap, parametric

        std::string toEsql() const override;
    };

    class SimulateStatement : public AIStatement {
    public:
        std::string model_name;
        std::string base_table;
        std::string intervention_table;  // Table with what-if scenarios
        std::vector<std::string> scenario_columns;
        size_t simulation_steps = 1;
        std::string output_table;
        bool compare_scenarios = false;
        std::string comparison_metric = "absolute_difference";

        std::string toEsql() const override;
    };

    class MultiPredictStatement : public AIStatement {
    public:
        std::string model_name;
        std::string input_table;
        std::string output_table;
        std::vector<std::string> output_columns;
        size_t num_predictions = 1;  // For sequence prediction
        bool parallel_predictions = false;
        size_t batch_size = 100;

        std::string toEsql() const override;
    };

    class AnalyzeUncertaintyStatement : public AIStatement {
    public:
        std::string model_name;
        std::string input_table;
        std::string output_table;
        std::string uncertainty_method = "bootstrapping"; // bootstrapping, dropout, ensemble
        size_t num_samples = 100;
        float confidence_level = 0.95;

        std::string toEsql() const override;
   };

    class AIFunctionCall : public Expression {
    public:
        AIFunctionType function_type;
        std::string model_name;
        std::vector<std::unique_ptr<Expression>> arguments;
        std::unique_ptr<Expression> alias;
        std::unordered_map<std::string, std::string> options; // Additional options

        std::unique_ptr<Expression> clone() const override;
        AIFunctionCall(AIFunctionType type, const std::string& model,
                      std::vector<std::unique_ptr<Expression>> args,
                      std::unique_ptr<Expression> al = nullptr,
                      const std::unordered_map<std::string, std::string>& opts = {});
        std::string toString() const override;
    };

    class CreateOrReplaceModelStatement : public AIStatement {
    public:
        std::string model_name;
        std::string algorithm;
        std::string source_table;
        std::string target_column;
        std::vector<std::string> feature_columns;
        std::unordered_map<std::string, std::string> parameters;
        bool replace = false;

        std::string toEsql() const override;
    };

    class PredictStatement : public AIStatement {
    public:
        std::string model_name;
        std::string input_table;
        std::string output_table;
        std::vector<std::string> output_columns;
        bool include_probabilities = false;
        bool include_confidence = false;
        std::string where_clause;
        size_t limit = 0;

        std::string toEsql() const override;
    };

    class InferenceStatement : public AIStatement {
    public:
        std::string model_name;
        std::unique_ptr<Expression> input_data;
        bool batch_mode = false;

        std::string toEsql() const override;
    };

    class ShowModelsStatement : public AIStatement {
    public:
        std::string pattern;
        bool detailed = false;
        std::string model_type; // Filter by type

        std::string toEsql() const override;
    };

    class DropModelStatement : public AIStatement {
    public:
        std::string model_name;
        bool if_exists = false;

        std::string toEsql() const override;
    };

    class ModelMetricsStatement : public AIStatement {
    public:
        std::string model_name;
        std::string test_data_table;
        std::unordered_map<std::string, std::string> metrics_options;

        std::string toEsql() const override;
    };

    class DescribeModelStatement : public AIStatement {
    public:
        std::string model_name;
        bool extended = false;
        std::vector<std::string> sections; // e.g., "PARAMETERS", "FEATURES", "METRICS"

        std::string toEsql() const override;
    };

    class AnalyzeDataStatement : public AIStatement {
    public:
        std::string table_name;
        std::string target_column;
        std::vector<std::string> feature_columns;
        std::string analysis_type; // "CORRELATION", "IMPORTANCE", "CLUSTERING"
        std::unordered_map<std::string, std::string> options;
        std::string output_format = "TABLE"; // TABLE, JSON, CHART

        std::string toEsql() const override;
    };

    class CreatePipelineStatement : public AIStatement {
    public:
        std::string pipeline_name;
        std::vector<std::pair<std::string, std::string>> steps; // step_type, config
        std::unordered_map<std::string, std::string> parameters;
        std::string describtion;
        bool replace = false;

        std::string toEsql() const override;
    };

    class BatchAIStatement : public AIStatement {
    public:
        std::string operation; // "TRAIN", "PREDICT", "EVALUATE"
        std::vector<std::unique_ptr<AIStatement>> statements;
        bool parallel = false;
        int max_concurrent = 1;
        std::string on_error = "CONTINUE"; // STOP, CONTINUE, ROLLBACK

        std::string toEsql() const override;
    };

    class ExplainStatement : public AIStatement {
    public:
        std::string model_name;
        std::unique_ptr<Expression> input_row;
        bool shap_values = false;

        std::string toEsql() const override;
    };

    class FeatureImportanceStatement : public AIStatement {
    public:
        std::string model_name;
        int top_n = 10;

        std::string toEsql() const override;
    };

    // AI-specific expressions
    class ModelFunctionCall : public Expression {
    public:
        std::string model_name;
        std::vector<std::unique_ptr<Expression>> arguments;
        std::unique_ptr<Expression> alias;

        std::unique_ptr<Expression> clone() const override;
        ModelFunctionCall(const std::string& name,
                         std::vector<std::unique_ptr<Expression>> args,
                         std::unique_ptr<Expression> al = nullptr);
        std::string toString() const override;
    };

    class AIScalarExpression : public Expression {
    public:
        enum class AIType {
            PREDICT, PROBABILITY, CONFIDENCE, ANOMALY_SCORE,
            CLUSTER_ID, FORECAST_VALUE, RESIDUAL, INFLUENCE,
            SIMILARITY, RECOMMENDATION_SCORE
        };

        AIType ai_type;
        std::string model_name;
        std::vector<std::unique_ptr<Expression>> inputs;
        std::unordered_map<std::string, std::string> options;
        std::unique_ptr<Expression> alias;

        std::unique_ptr<Expression> clone() const override;
        AIScalarExpression(AIType type, const std::string& model,
                          std::vector<std::unique_ptr<Expression>> ins,
                          std::unique_ptr<Expression> al = nullptr,
                          const std::unordered_map<std::string, std::string>& opts = {});
        std::string toString() const override;
    };
}

#endif // AI_GRAMMAR_H
