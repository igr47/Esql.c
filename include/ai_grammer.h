
#ifndef AI_GRAMMAR_H
#define AI_GRAMMAR_H

#include "scanner.h"
#include "parser.h"
#include <vector>
#include <string>
#include <memory>

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
    };

    class TrainModelStatement : public AIStatement {
    public:
        std::string model_name;
        std::string algorithm;
        std::string source_table;
        std::string target_column;
        std::vector<std::string> feature_columns;
        std::unordered_map<std::string, std::string> hyperparameters;
        float test_split = 0.2f;
        int iterations = 100;
        bool save_model = true;
        std::string output_table;
        std::string where_clause;

        TrainModelStatement() = default;
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

        PredictStatement() = default;
    };

    class InferenceStatement : public AIStatement {
    public:
        std::string model_name;
        std::unique_ptr<Expression> input_data;
        bool batch_mode = false;

        InferenceStatement() = default;
    };

    class ShowModelsStatement : public AIStatement {
    public:
        std::string pattern;
        bool detailed = false;

        ShowModelsStatement() = default;
    };

    class DropModelStatement : public AIStatement {
    public:
        std::string model_name;
        bool if_exists = false;

        DropModelStatement() = default;
    };

    class ModelMetricsStatement : public AIStatement {
    public:
        std::string model_name;
        std::string test_data_table;

        ModelMetricsStatement() = default;
    };

    class ExplainStatement : public AIStatement {
    public:
        std::string model_name;
        std::unique_ptr<Expression> input_row;
        bool shap_values = false;

        ExplainStatement() = default;
    };

    class FeatureImportanceStatement : public AIStatement {
    public:
        std::string model_name;
        int top_n = 10;

        FeatureImportanceStatement() = default;
    };

    // AI-specific expressions
    class ModelFunctionCall : public Expression {
    public:
        std::string model_name;
        std::vector<std::unique_ptr<Expression>> arguments;
        std::unique_ptr<Expression> alias;

        std::unique_ptr<Expression> clone() const override {
            std::vector<std::unique_ptr<Expression>> cloned_args;
            for (const auto& arg : arguments) {
                cloned_args.push_back(arg->clone());
            }
            return std::make_unique<ModelFunctionCall>(
                model_name,
                std::move(cloned_args),
                alias ? alias->clone() : nullptr
            );
        }

        ModelFunctionCall(const std::string& name,
                         std::vector<std::unique_ptr<Expression>> args,
                         std::unique_ptr<Expression> al = nullptr)
            : model_name(name), arguments(std::move(args)), alias(std::move(al)) {}

        std::string toString() const override {
            std::string result = model_name + "(";
            for (size_t i = 0; i < arguments.size(); ++i) {
                result += arguments[i]->toString();
                if (i < arguments.size() - 1) result += ", ";
            }
            result += ")";
            if (alias) {
                result += " AS " + alias->toString();
            }
            return result;
        }
    };

    class AIScalarExpression : public Expression {
    public:
        enum class AIType {
            PREDICT, PROBABILITY, CONFIDENCE, ANOMALY_SCORE,
            CLUSTER_ID, FORECAST_VALUE, RESIDUAL, INFLUENCE
        };

        AIType ai_type;
        std::string model_name;
        std::vector<std::unique_ptr<Expression>> inputs;

        std::unique_ptr<Expression> clone() const override {
            std::vector<std::unique_ptr<Expression>> cloned_inputs;
            for (const auto& input : inputs) {
                cloned_inputs.push_back(input->clone());
            }
            return std::make_unique<AIScalarExpression>(ai_type, model_name, std::move(cloned_inputs));
        }

        AIScalarExpression(AIType type, const std::string& model,
                          std::vector<std::unique_ptr<Expression>> ins)
            : ai_type(type), model_name(model), inputs(std::move(ins)) {}

        std::string toString() const override {
            std::string type_str;
            switch(ai_type) {
                case AIType::PREDICT: type_str = "PREDICT"; break;
                case AIType::PROBABILITY: type_str = "PROBABILITY"; break;
                case AIType::CONFIDENCE: type_str = "CONFIDENCE"; break;
                case AIType::ANOMALY_SCORE: type_str = "ANOMALY_SCORE"; break;
                case AIType::CLUSTER_ID: type_str = "CLUSTER_ID"; break;
                case AIType::FORECAST_VALUE: type_str = "FORECAST"; break;
                case AIType::RESIDUAL: type_str = "RESIDUAL"; break;
                case AIType::INFLUENCE: type_str = "INFLUENCE"; break;
            }

            std::string result = type_str + "_USING_" + model_name + "(";
            for (size_t i = 0; i < inputs.size(); ++i) {
                result += inputs[i]->toString();
                if (i < inputs.size() - 1) result += ", ";
            }
            result += ")";
            return result;
        }
    };
}

#endif // AI_GRAMMAR_H
