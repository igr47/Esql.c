
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
        std::string toEsql() const override {
            std::string result = "TRAIN MODEL " + model_name;

            if (!algorithm.empty()) {
                result += " USING " + algorithm;
            }

            if (!source_table.empty()) {
                result += " ON " + source_table;
            }

            if (!target_column.empty()) {
                result += " TARGET " + target_column;
            }

            if (!feature_columns.empty()) {
                result += " FEATURES (";
                for (size_t i = 0; i < feature_columns.size(); ++i) {
                    if (i > 0) result += ", ";
                    result += feature_columns[i];
                }
                result += ")";
            }

            if (!hyperparameters.empty()) {
                result += " WITH HYPERPARAMETERS (";
                bool first = true;
                for (const auto& [key, value] : hyperparameters) {
                    if (!first) result += ", ";
                    result += key + " = " + value;
                    first = false;
                }
            result += ")";
            }

            if (test_split > 0) {
                result += " TEST_SPLIT = " + std::to_string(test_split);
            }

            if (iterations > 0) {
                result += " ITERATIONS = " + std::to_string(iterations);
            }

            if (!where_clause.empty()) {
                result += " WHERE " + where_clause;
            }

            if (!output_table.empty()) {
                result += " INTO " + output_table;
            }

             return result;
        }
    };

    class CreateModelStatement : public AIStatement {
    public:
        std::string model_name;
        std::string algorithm;
        std::vector<std::pair<std::string, std::string>> features; // name, type
        std::string target_type; // "CLASSIFICATION", "REGRESSION", "CLUSTERING"
        std::unordered_map<std::string, std::string> parameters;

        std::string toEsql() const override {
            std::string result = "CREATE MODEL " + model_name;

            if (!algorithm.empty()) {
                result += " USING " + algorithm;
            }

            if (!features.empty()) {
                result += " FEATURES (";
                for (size_t i = 0; i < features.size(); ++i) {
                    if (i > 0) result += ", ";
                    result += features[i].first;
                    if (!features[i].second.empty() && features[i].second != "AUTO") {
                        result += " AS " + features[i].second;
                    }
                }
                result += ")";
            }

            if (!target_type.empty()) {
                result += " TARGET " + target_type;
            }

            if (!parameters.empty()) {
                result += " WITH (";
                bool first = true;
                for (const auto& [key, value] : parameters) {
                    if (!first) result += ", ";
                    result += key + " = " + value;
                    first = false;
                }
                result += ")";
            }

             return result;
        }
    };

    class AIFunctionCall : public Expression {
    public:
        AIFunctionType function_type;
        std::string model_name;
        std::vector<std::unique_ptr<Expression>> arguments;
        std::unique_ptr<Expression> alias;
        std::unordered_map<std::string, std::string> options; // Additional options

        std::unique_ptr<Expression> clone() const override {
            std::vector<std::unique_ptr<Expression>> cloned_args;
            for (const auto& arg : arguments) {
                cloned_args.push_back(arg->clone());
            }

            std::unique_ptr<Expression> cloned_alias = nullptr;
            if (alias) cloned_alias = alias->clone();

            auto result = std::make_unique<AIFunctionCall>(function_type, model_name, std::move(cloned_args), std::move(cloned_alias), options);
            return result;
        }

        AIFunctionCall(AIFunctionType type, const std::string& model,std::vector<std::unique_ptr<Expression>> args,std::unique_ptr<Expression> al = nullptr,
                   const std::unordered_map<std::string, std::string>& opts = {}) : function_type(type), model_name(model), arguments(std::move(args)),alias(std::move(al)), options(opts) {}

        std::string toString() const override {
            std::string func_name;
            switch(function_type) {
                case AIFunctionType::PREDICT: func_name = "AI_PREDICT"; break;
                case AIFunctionType::PREDICT_CLASS: func_name = "AI_PREDICT_CLASS"; break;
                case AIFunctionType::PREDICT_VALUE: func_name = "AI_PREDICT_VALUE"; break;
                case AIFunctionType::PREDICT_PROBA: func_name = "AI_PREDICT_PROBA"; break;
                case AIFunctionType::PREDICT_CLUSTER: func_name = "AI_PREDICT_CLUSTER"; break;
                case AIFunctionType::PREDICT_ANOMALY: func_name = "AI_PREDICT_ANOMALY"; break;
                case AIFunctionType::EXPLAIN: func_name = "AI_EXPLAIN"; break;
                case AIFunctionType::TRAIN_MODEL: func_name = "AI_TRAIN"; break;
                case AIFunctionType::MODEL_METRICS: func_name = "AI_MODEL_METRICS"; break;
                case AIFunctionType::FEATURE_IMPORTANCE: func_name = "AI_FEATURE_IMPORTANCE"; break;
            }

            std::string result = func_name + "('" + model_name + "'";

            if (!arguments.empty()) {
                result += ", ";
                for (size_t i = 0; i < arguments.size(); ++i) {
                    result += arguments[i]->toString();
                    if (i < arguments.size() - 1) result += ", ";
                }
            }

            result += ")";

            if (!options.empty()) {
                result += " WITH (";
                bool first = true;
                for (const auto& [key, value] : options) {
                    if (!first) result += ", ";
                    result += key + " = " + value;
                    first = false;
                }
                result += ")";
            }

            if (alias) {
                result += " AS " + alias->toString();
            }

            return result;
        }
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

        std::string toEsql() const override {
            std::string result = replace ? "CREATE OR REPLACE MODEL " : "CREATE MODEL ";
            result += model_name;

            if (!algorithm.empty()) {
                result += " USING " + algorithm;
            }

            if (!source_table.empty()) {
                result += " ON " + source_table;
            }

            if (!target_column.empty()) {
                result += " TARGET " + target_column;
            }

            if (!feature_columns.empty()) {
                result += " FEATURES (";
                for (size_t i = 0; i < feature_columns.size(); ++i) {
                    if (i > 0) result += ", ";
                    result += feature_columns[i];
                }
                result += ")";
            }

            if (!parameters.empty()) {
                result += " WITH (";
                bool first = true;
                for (const auto& [key, value] : parameters) {
                    if (!first) result += ", ";
                    result += key + " = " + value;
                    first = false;
                }
                result += ")";
            }

            return result;
        }
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

        //PredictStatement() = default;
        std::string toEsql() const override {
            std::string result = "PREDICT USING " + model_name;

            if (!input_table.empty()) {
                result += " ON " + input_table;
            }

            if (!where_clause.empty()) {
                result += " WHERE " + where_clause;
            }

            if (!output_columns.empty()) {
                result += " OUTPUT (";
                for (size_t i = 0; i < output_columns.size(); ++i) {
                    if (i > 0) result += ", ";
                    result += output_columns[i];
                }
                result += ")";
            }

            if (include_probabilities) {
                result += " WITH PROBABILITIES";
            } else if (include_confidence) {
                result += " WITH CONFIDENCE";
            }

            if (!output_table.empty()) {
                result += " INTO " + output_table;
            }

            if (limit > 0) {
                result += " LIMIT " + std::to_string(limit);
            }

            return result;
        }
    };

    class InferenceStatement : public AIStatement {
    public:
        std::string model_name;
        std::unique_ptr<Expression> input_data;
        bool batch_mode = false;

        //InferenceStatement() = default;
        std::string toEsql() const override {
            std::string result = "INFERECE USING" + model_name;

            if (input_data) {
                result += " FOR " + input_data->toString();
            }
            if (batch_mode) {
                result += "IN BATCH";
            }
            return result;
        }
    };

    class ShowModelsStatement : public AIStatement {
    public:
        std::string pattern;
        bool detailed = false;
        std::string model_type; // Filter by type

        //ShowModelsStatement() = default;
        std::string toEsql() const override {
            std::string result = "SHOW MODELS";

            if (!pattern.empty()) {
                result += " LIKE '" + pattern + "'";
            }

            if (detailed) {
                result += " DETAILED";
            }

            if (!model_type.empty()) {
                result += " TYPE " + model_type;
            }

            return result;
        }
    };

    class DropModelStatement : public AIStatement {
    public:
        std::string model_name;
        bool if_exists = false;

        //DropModelStatement() = default;
        std::string toEsql() const override {
           std::string result = "DROP MODEL";

           if (if_exists) {
              result += " IF EXISTS";
           }

           result += " " + model_name;

           return result;
        }
    };

    class ModelMetricsStatement : public AIStatement {
    public:
        std::string model_name;
        std::string test_data_table;

        //ModelMetricsStatement() = default;
        std::string toEsql() const override {
            std::string result = "MODEL METRICS FOR " + model_name;

            if (!test_data_table.empty()) {
                result += " ON " + test_data_table;
            }

            return result;
        }
    };

    class DescribeModelStatement : public AIStatement {
    public:
        std::string model_name;
        bool extended = false;

        std::string toEsql() const override {
            std::string result = "DESCRIBE MODEL " + model_name;

            if (extended) {
                result += " EXTENDED";
            }

            return result;
        }
    };

    class AnalyzeDataStatement : public AIStatement {
    public:
        std::string table_name;
        std::string target_column;
        std::vector<std::string> feature_columns;
        std::string analysis_type; // "CORRELATION", "IMPORTANCE", "CLUSTERING"
        std::unordered_map<std::string, std::string> options;

        std::string toEsql() const override {
           std::string result = "ANALYZE DATA " + table_name;

           if (!target_column.empty()) {
                result += " TARGET " + target_column;
           }

           if (!feature_columns.empty()) {
                result += " FEATURES (";
                for (size_t i = 0; i < feature_columns.size(); ++i) {
                      if (i > 0) result += ", ";
                      result += feature_columns[i];
                }
                result += ")";
           }

           if (!analysis_type.empty()) {
                result += " TYPE " + analysis_type;
           }

           if (!options.empty()) {
                result += " WITH (";
                bool first = true;
                for (const auto& [key, value] : options) {
                    if (!first) result += ", ";
                    result += key + " = " + value;
                    first = false;
                }
                result += ")";
           }

           return result;
        }
    };

    class CreatePipelineStatement : public AIStatement {
    public:
        std::string pipeline_name;
        std::vector<std::pair<std::string, std::string>> steps; // step_type, config
        std::unordered_map<std::string, std::string> parameters;

        std::string toEsql() const override {
           std::string result = "CREATE PIPELINE " + pipeline_name;

           if (!steps.empty()) {
                result += " STEPS (";
                for (size_t i = 0; i < steps.size(); ++i) {
                    if (i > 0) result += ", ";
                    result += steps[i].first + "(" + steps[i].second + ")";
                }
                result += ")";
           }

           if (!parameters.empty()) {
                result += " WITH (";
                bool first = true;
                for (const auto& [key, value] : parameters) {
                    if (!first) result += ", ";
                    result += key + " = " + value;
                    first = false;
                }
                result += ")";
           }
           return result;
        }
    };


    class ExplainStatement : public AIStatement {
    public:
        std::string model_name;
        std::unique_ptr<Expression> input_row;
        bool shap_values = false;

        //ExplainStatement() = default;
        std::string toEsql() const override {
           std::string result = "EXPLAIN MODEL " + model_name;

            if (input_row) {
                result += " FOR ";
                // Simplified - you might want to format the input_row properly
                result += input_row->toString();
            }

            if (shap_values) {
                result += " WITH SHAP_VALUES";
            }

            return result;
        }
    };

    class FeatureImportanceStatement : public AIStatement {
    public:
        std::string model_name;
        int top_n = 10;

        //FeatureImportanceStatement() = default;
        std::string toEsql() const override {
           std::string result = "FEATURE IMPORTANCE FOR " + model_name;

           if (top_n > 0) {
                result += " TOP " + std::to_string(top_n);
           }

           return result;
        }
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
