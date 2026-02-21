#include "ai_expression_evaluator.h"
#include "ai/model_registry.h"
#include "execution_engine_includes/executionengine_main.h"
#include "datum.h"
#include <sstream>
#include <iomanip>
#include <functional>

AIExpressionEvaluator::AIExpressionEvaluator(Database& db,ExecutionEngine& engine)
    : registry_(esql::ai::ModelRegistry::instance()), db_(db), engine_(engine) {}

std::string AIExpressionEvaluator::evaluateAIFunction(
    const AST::AIFunctionCall* func,
    const std::unordered_map<std::string, std::string>& row) {

    try {
        // Get model
        auto model = getModel(func->model_name);
        if (!model) {
            return "NULL";
        }

        const auto& schema = model->get_schema();

        // Extract features from arguments or row
        std::vector<float> features;

        if (!func->arguments.empty()) {
            // Use explicit arguments: PREDICT_PROBA(model_name, feature1, feature2, ...)
            for (const auto& arg : func->arguments) {
                // Evaluate the argument expression
                std::string arg_value = engine_.evaluateExpressionWrapper(arg.get(), row);
                try {
                    features.push_back(std::stof(arg_value));
                } catch (...) {
                    features.push_back(0.0f);
                }
            }
        } else {
            // Use schema-based extraction: PREDICT(model_name)
            // Extract all features according to schema
            //  First convert the string map to Datum map
            std::unordered_map<std::string, esql::Datum> datum_row;
            for (const auto& [key, value] : row) {
                // Try to convert string map to appropriate Daum type
                if (value == "NULL" || value.empty()) {
                    datum_row[key] = esql::Datum();
                } else {
                    try {
                        size_t pos;
                        int int_val = std::stoi(value, &pos);
                        if (pos == value.length()) {
                            datum_row[key] = esql::Datum(int_val);
                        } else {
                            double double_val = std::stod(value);
                            datum_row[key] = esql::Datum(double_val);
                        }
                    } catch (...) {
                        // Default to string
                        datum_row[key] = esql::Datum(value);
                    }
                }
            }

            // Extract features using the schema
            features = schema.extract_features(datum_row);
            /*features = schema.extract_features([&](const std::string& col) -> float {
                auto it = row.find(col);
                if (it != row.end()) {
                    try {
                        return std::stof(it->second);
                    } catch (...) {
                        return 0.0f;
                    }
                }
                // Find feature by name
                for (const auto& feature : schema.features) {
                    if (feature.name == col) {
                        return feature.default_value;
                    }
                }
                return 0.0f;
            });*/
        }

        // Create tensor and predict
        esql::ai::Tensor input_tensor(std::move(features), {features.size()});
        auto prediction = model->predict(input_tensor);

        // Format result based on function type
        return formatPredictionResult(prediction, schema, func->function_type, func->options);

    } catch (const std::exception& e) {
        std::cerr << "[AI Expression] Error: " << e.what() << std::endl;
        return "NULL";
    }
}

std::string AIExpressionEvaluator::formatPredictionResult(
    const esql::ai::Tensor& prediction,
    const esql::ai::ModelSchema& schema,
    AST::AIFunctionType func_type,
    const std::unordered_map<std::string, std::string>& options) {

    std::stringstream result;

    switch (func_type) {
        case AST::AIFunctionType::PREDICT_CLASS:
            if (schema.problem_type == "binary_classification") {
                result << (prediction.data[0] > 0.5f ? "1" : "0");
            } else {
                // Multiclass - get class with highest probability
                int max_idx = 0;
                for (size_t i = 1; i < prediction.data.size(); ++i) {
                    if (prediction.data[i] > prediction.data[max_idx]) {
                        max_idx = i;
                    }
                }
                result << max_idx;
            }
            break;

        case AST::AIFunctionType::PREDICT_PROBA:
            if (schema.problem_type == "binary_classification") {
                result << std::fixed << std::setprecision(6) << prediction.data[0];
            } else {
                // Return JSON array for multiclass
                result << "[";
                for (size_t i = 0; i < prediction.data.size(); ++i) {
                    if (i > 0) result << ",";
                    result << std::fixed << std::setprecision(6) << prediction.data[i];
                }
                result << "]";
            }
            break;

        case AST::AIFunctionType::PREDICT_VALUE:
        default:
            result << std::fixed << std::setprecision(6) << prediction.data[0];
            break;
    }

    return result.str();
}

std::shared_ptr<esql::ai::AdaptiveLightGBMModel> AIExpressionEvaluator::getModel(
    const std::string& model_name) {

    // Check cache first
    {
        std::shared_lock lock(model_cache_mutex_);
        auto it = model_cache_.find(model_name);
        if (it != model_cache_.end()) {
            return it->second;
        }
    }

    // Load model
    auto* model = registry_.get_model(model_name);
    if (!model) {
        return nullptr;
    }

    // Wrap in shared_ptr and cache
    auto shared_model = std::shared_ptr<esql::ai::AdaptiveLightGBMModel>(
        model, [](esql::ai::AdaptiveLightGBMModel*) {});

    {
        std::unique_lock lock(model_cache_mutex_);
        model_cache_[model_name] = shared_model;
    }

    return shared_model;
}

std::string AIExpressionEvaluator::evaluateModelFunction(
    const AST::ModelFunctionCall* func,
    const std::unordered_map<std::string, std::string>& row) {

    // Direct model call: model_name(feature1, feature2, ...)
    try {
        auto model = getModel(func->model_name);
        if (!model) {
            return "NULL";
        }

        std::vector<float> features;
        for (const auto& arg : func->arguments) {
            std::string arg_value = engine_.evaluateExpressionWrapper(arg.get(), row);
            try {
                features.push_back(std::stof(arg_value));
            } catch (...) {
                features.push_back(0.0f);
            }
        }

        esql::ai::Tensor input_tensor(std::move(features), {features.size()});
        auto prediction = model->predict(input_tensor);

        return std::to_string(prediction.data[0]);

    } catch (const std::exception& e) {
        return "NULL";
    }
}

std::string AIExpressionEvaluator::evaluateAIScalar(
    const AST::AIScalarExpression* expr,
    const std::unordered_map<std::string, std::string>& row) {

    // Handle scalar expressions like PREDICT_USING_model_name(features...)
    try {
        auto model = getModel(expr->model_name);
        if (!model) {
            return "NULL";
        }

        std::vector<float> features;
        for (const auto& input : expr->inputs) {
            std::string val = engine_.evaluateExpressionWrapper(input.get(), row);
            try {
                features.push_back(std::stof(val));
            } catch (...) {
                features.push_back(0.0f);
            }
        }

        esql::ai::Tensor input_tensor(std::move(features), {features.size()});
        auto prediction = model->predict(input_tensor);

        switch (expr->ai_type) {
            case AST::AIScalarExpression::AIType::PREDICT:
                return std::to_string(prediction.data[0]);
            case AST::AIScalarExpression::AIType::PROBABILITY:
                return std::to_string(prediction.data[0]);
            case AST::AIScalarExpression::AIType::CONFIDENCE:
                return std::to_string(std::abs(prediction.data[0] - 0.5f) * 2.0f);
            default:
                return std::to_string(prediction.data[0]);
        }

    } catch (const std::exception& e) {
        return "NULL";
    }
}
