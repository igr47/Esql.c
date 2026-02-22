#include "ai_grammer.h"
#include <sstream>

namespace AST {

    // TrainingOptions implementations
    nlohmann::json TrainingOptions::to_json() const {
        nlohmann::json j;
        j["cross_validation"] = cross_validation;
        j["cv_folds"] = cv_folds;
        j["early_stopping"] = early_stopping;
        j["early_stopping_rounds"] = early_stopping_rounds;
        j["validation_table"] = validation_table;
        j["validation_split"] = validation_split;
        j["use_gpu"] = use_gpu;
        j["device_type"] = device_type;
        j["num_threads"] = num_threads;
        j["metric"] = metric;
        j["boosting_type"] = boosting_type;
        j["seed"] = seed;
        j["deterministic"] = deterministic;
        return j;
    }

    TrainingOptions TrainingOptions::from_json(const nlohmann::json& j) {
        TrainingOptions opts;
        opts.cross_validation = j.value("cross_validation", false);
        opts.cv_folds = j.value("cv_folds", 5);
        opts.early_stopping = j.value("early_stopping", true);
        opts.early_stopping_rounds = j.value("early_stopping_rounds", 10);
        opts.validation_table = j.value("validation_table", "");
        opts.validation_split = j.value("validation_split", 0.2f);
        opts.use_gpu = j.value("use_gpu", false);
        opts.device_type = j.value("device_type", "cpu");
        opts.num_threads = j.value("num_threads", -1);
        opts.metric = j.value("metric", "auto");
        opts.boosting_type = j.value("boosting_type", "gbdt");
        opts.seed = j.value("seed", 42);
        opts.deterministic = j.value("deterministic", true);
        return opts;
    }

    // TuningOptions implementations
    nlohmann::json TuningOptions::to_json() const {
        nlohmann::json j;
        j["tune_hyperparameters"] = tune_hyperparameters;
        j["tuning_method"] = tuning_method;
        j["tuning_iterations"] = tuning_iterations;
        j["tuning_folds"] = tuning_folds;
        j["scoring_metric"] = scoring_metric;
        j["parallel_tuning"] = parallel_tuning;
        j["tuning_jobs"] = tuning_jobs;

        // Serialize param_grid
        nlohmann::json grid_json;
        for (const auto& [param, values] : param_grid) {
            grid_json[param] = values;
        }
        j["param_grid"] = grid_json;

        // Serialize param_ranges
        nlohmann::json ranges_json;
        for (const auto& [param, range] : param_ranges) {
            nlohmann::json range_json;
            range_json["min"] = range.first;
            range_json["max"] = range.second;
            ranges_json[param] = range_json;
        }
        j["param_ranges"] = ranges_json;

        return j;
    }

    TuningOptions TuningOptions::from_json(const nlohmann::json& j) {
        TuningOptions opts;
        opts.tune_hyperparameters = j.value("tune_hyperparameters", false);
        opts.tuning_method = j.value("tuning_method", "grid");
        opts.tuning_iterations = j.value("tuning_iterations", 10);
        opts.tuning_folds = j.value("tuning_folds", 3);
        opts.scoring_metric = j.value("scoring_metric", "auto");
        opts.parallel_tuning = j.value("parallel_tuning", true);
        opts.tuning_jobs = j.value("tuning_jobs", -1);

        // Deserialize param_grid
        if (j.contains("param_grid")) {
            for (auto& [key, value] : j["param_grid"].items()) {
                std::vector<std::string> values;
                for (const auto& v : value) {
                    values.push_back(v.get<std::string>());
                }
                opts.param_grid[key] = values;
            }
        }

        // Deserialize param_ranges
        if (j.contains("param_ranges")) {
            for (auto& [key, value] : j["param_ranges"].items()) {
                float min_val = value["min"];
                float max_val = value["max"];
                opts.param_ranges[key] = {min_val, max_val};
            }
        }

        return opts;
    }

    // TrainModelStatement implementations
    std::string TrainModelStatement::toEsql() const {
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

    // CreateModelStatement implementations
    std::string CreateModelStatement::toEsql() const {
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

        if (training_options.cross_validation) {
            result += " CROSS_VALIDATION FOLDS " + std::to_string(training_options.cv_folds);
        }

        if (training_options.early_stopping) {
            result += " EARLY_STOPPING ROUNDS " + std::to_string(training_options.early_stopping_rounds);
            if (!training_options.validation_table.empty()) {
                result += " VALIDATION_TABLE " + training_options.validation_table;
            } else {
                result += " VALIDATION_SPLIT " + std::to_string(training_options.validation_split);
            }
        }

        if (training_options.use_gpu) {
            result += " DEVICE GPU";
        }

        if (training_options.num_threads > 0) {
            result += " NUM_THREADS " + std::to_string(training_options.num_threads);
        }

        if (training_options.metric != "auto") {
            result += " METRIC " + training_options.metric;
        }

        if (training_options.boosting_type != "gbdt") {
            result += " BOOSTING " + training_options.boosting_type;
        }

        if (training_options.seed != 42) {
            result += " SEED " + std::to_string(training_options.seed);
        }

        // Add tuning options
        if (tuning_options.tune_hyperparameters) {
            result += " TUNE_HYPERPARAMETERS USING " + tuning_options.tuning_method + " ITERATIONS " + std::to_string(tuning_options.tuning_iterations);

            if (!tuning_options.parallel_tuning) {
                result += " SEQUENTIAL";
            }

            if (tuning_options.tuning_jobs > 0) {
                result += " JOBS " + std::to_string(tuning_options.tuning_jobs);
            }
        }

        // Add advanced options
        if (data_sampling != "none") {
            result += " DATA_SAMPLING " + data_sampling;
            if (sampling_ratio != 1.0f) {
                result += " RATIO " + std::to_string(sampling_ratio);
            }
        }

        if (feature_selection) {
            result += " FEATURE_SELECTION USING " + feature_selection_method;
            if (max_features_to_select > 0) {
                result += " MAX_FEATURES " + std::to_string(max_features_to_select);
            }
        }

        if (!feature_scaling) {
            result += " NO_FEATURE_SCALING";
        } else if (scaling_method != "standard") {
            result += " SCALING " + scaling_method;
        }

        // Add hyperparameters
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

    nlohmann::json CreateModelStatement::to_json() const {
        nlohmann::json j;
        j["model_name"] = model_name;
        j["algorithm"] = algorithm;
        j["target_type"] = target_type;
        j["parameters"] = parameters;
        j["training_options"] = training_options.to_json();
        j["tuning_options"] = tuning_options.to_json();

        // Serialize features
        nlohmann::json features_json = nlohmann::json::array();
        for (const auto& [name, type] : features) {
            nlohmann::json feature;
            feature["name"] = name;
            feature["type"] = type;
            features_json.push_back(feature);
        }
        j["features"] = features_json;
        j["data_sampling"] = data_sampling;
        j["sampling_ratio"] = sampling_ratio;
        j["feature_selection"] = feature_selection;
        j["feature_selection_method"] = feature_selection_method;
        j["max_features_to_select"] = max_features_to_select;
        j["feature_scaling"] = feature_scaling;
        j["scaling_method"] = scaling_method;

        return j;
    }

    CreateModelStatement CreateModelStatement::from_json(const nlohmann::json& j) {
        CreateModelStatement stmt;
        stmt.model_name = j["model_name"];
        stmt.algorithm = j["algorithm"];
        stmt.target_type = j["target_type"];
        stmt.parameters = j["parameters"].get<std::unordered_map<std::string, std::string>>();
        stmt.training_options = TrainingOptions::from_json(j["training_options"]);
        stmt.tuning_options = TuningOptions::from_json(j["tuning_options"]);

        // Deserialize features
        for (const auto& feature_json : j["features"]) {
            stmt.features.emplace_back(
                feature_json["name"],
                feature_json["type"]
            );
        }

        // Advanced options
        stmt.data_sampling = j["data_sampling"];
        stmt.sampling_ratio = j["sampling_ratio"];
        stmt.feature_selection = j["feature_selection"];
        stmt.feature_selection_method = j["feature_selection_method"];
        stmt.max_features_to_select = j["max_features_to_select"];
        stmt.feature_scaling = j["feature_scaling"];
        stmt.scaling_method = j["scaling_method"];

        return stmt;
    }

    // AIFunctionCall implementations
    std::unique_ptr<Expression> AIFunctionCall::clone() const {
        std::vector<std::unique_ptr<Expression>> cloned_args;
        for (const auto& arg : arguments) {
            cloned_args.push_back(arg->clone());
        }

        std::unique_ptr<Expression> cloned_alias = nullptr;
        if (alias) cloned_alias = alias->clone();

        auto result = std::make_unique<AIFunctionCall>(function_type, model_name, std::move(cloned_args), std::move(cloned_alias), options);
        return result;
    }

    AIFunctionCall::AIFunctionCall(AIFunctionType type, const std::string& model,
                                   std::vector<std::unique_ptr<Expression>> args,
                                   std::unique_ptr<Expression> al,
                                   const std::unordered_map<std::string, std::string>& opts) 
        : function_type(type), model_name(model), arguments(std::move(args)), alias(std::move(al)), options(opts) {}

    std::string AIFunctionCall::toString() const {
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
            case AIFunctionType::FORECAST: func_name = "AI_FORECAST"; break;
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

    // CreateOrReplaceModelStatement implementations
    std::string CreateOrReplaceModelStatement::toEsql() const {
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

    std::string ForecastStatement::toEsql() const {
	std::string result = "FORECAST USING " + model_name;

	if (!input_table.empty()) {
	    result += " ON " + input_table;
	}

	if (!time_column.empty()) {
	   result += " WITH TIME " + time_column;
	}

	if (!value_columns.empty()) {
	   result += " OUTPUT (";
	   for (size_t i = 0; i < value_columns.size(); ++i) {
	       if (i > 0) result += ", ";
	       result += value_columns[i];
	   }
	   result += ")";
	}

	result += " STEPS " + std::to_string(horizon);

	if (include_confidence) {
	   result += " WITH CONFIDENCE ";
	}
        
        if (include_scenarios) {
	   result += " WITH SCENARIOS ";
	}
        result += " NUMBER SCENARIOS " + std::to_string(num_scenarios);

	if (!output_table.empty()) {
	   result += " ON TABLE " + output_table;
	}

	if (!scenario_type.empty()) {
	   result += " SCENARIO TYPE " +scenario_type;
	}

	return result;
    }

    std::string SimulateStatement::toEsql() const {
        std::string result = " SIMULATING USING " + model_name;

        if (!input_table.empty()) {
            result += " ON TABLE " + input_table;
        }

        if (!output_table.empty()) {
            result += " OUTPUT TABLE " + output_table;
        } 

        if (!time_column.empty()) {
            result += " TIME " + time_column;
        }

        if (!time_interval.empty()) {
            result += " INTERVAL " + time_interval;
        }

        return result;
    }

    /*std::string SimulateStatement::toEsql() const {
	 std::string result = " SIMULATING USING " + model_name;

	 if (!base_table.empty()) {
	    result += " ON TABLE " + base_table;
	 }

	 if (!intervention_table.empty()) {
	    result += " WITH INTERVENTION " + intervention_table;
	 }

	 if (!scenario_columns.empty()) {
	    result += " SCENARIO COLUMNS ( ";
	    for (size_t i = 0; i < scenario_columns.size(); ++i) {
		if (i > 0) result += ", ";
		result += scenario_columns[i];
	    }
	    result += ")";
	 }

	 result += " SIMULATION STEPS " + std::to_string(simulation_steps);

	 if (!output_table.empty()) {
	    result += " OUTPUT TABLE " + output_table;
	 }

	 if (compare_scenarios) {
	    result += " WITH SCENARIO COMPARISON ";
	 }

	 if (!comparison_metric.empty()) {
	    result += " COMPARISON METRIC " + comparison_metric;
	 }

	 return result;
    }*/

    std::string DetectAnomalyStatement::toEsql() const {
        std::stringstream ss;
        ss << "DETECT ANOMALY USING ";
        if (!model_name.empty()) {
            ss << "MODEL " << model_name;
        } else {
            ss << algorithm;
        }
        ss << " ON " << input_table;
        if (!parameters.empty()) {
            ss << " WITH (";
            bool first = true;
            for (const auto& [key, value] : parameters) {
                if (!first) ss << ", ";
                ss << key << " = " << value;
                first = false;
            }
            ss << ")";
        }
        if (!where_clause.empty()) {
            ss << " WHERE " << where_clause;
        }
        if (!output_table.empty()) {
            ss << " INTO " << output_table;
	}
        if (generate_alerts) {
            ss << " WITH ALERTS";
        }
        return ss.str();
    }

    std::string MultiPredictStatement::toEsql() const {
	 std::string result = " MULTI PREDICTING USING " + model_name;

	 if (!input_table.empty()) {
	    result += " ON TABLE " + input_table;
	 }

	 if (!output_table.empty()) {
	    result += " OUTPUT TABLE " + output_table;
	 }

	 return result;
    }

    std::string AnalyzeUncertaintyStatement::toEsql() const {
	 std::string result = " ANALYZING UNCERTAINITY USING " + model_name;

	 if (!input_table.empty()) {
	    result += " ON TABLE " + input_table;
	 }

	 if (!output_table.empty()) {
	    result += " OUTPUT TABLE " + output_table;
	 }

	 return result;
    }

    // PredictStatement implementations
    std::string PredictStatement::toEsql() const {
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

    // InferenceStatement implementations
    std::string InferenceStatement::toEsql() const {
        std::string result = "INFERECE USING" + model_name;

        if (input_data) {
            result += " FOR " + input_data->toString();
        }
        if (batch_mode) {
            result += "IN BATCH";
        }
        return result;
    }

    // ShowModelsStatement implementations
    std::string ShowModelsStatement::toEsql() const {
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

    // DropModelStatement implementations
    std::string DropModelStatement::toEsql() const {
        std::string result = "DROP MODEL";

        if (if_exists) {
            result += " IF EXISTS";
        }

        result += " " + model_name;

        return result;
    }

    // ModelMetricsStatement implementations
    std::string ModelMetricsStatement::toEsql() const {
        std::string result = "MODEL METRICS FOR " + model_name;

        if (!test_data_table.empty()) {
            result += " ON " + test_data_table;
        }

        if (!metrics_options.empty()) {
            result += " WITH (";
            bool first = true;
            for (const auto& [key, value] : metrics_options) {
                if (!first) result += ", ";
                result += key + " = " + value;
                first = false;
            }
            result += ")";
        }

        return result;
    }

    // DescribeModelStatement implementations
    std::string DescribeModelStatement::toEsql() const {
        std::string result = "DESCRIBE MODEL " + model_name;

        if (extended) {
            result += " EXTENDED";
        }

        if (!sections.empty()) {
            result += " SECTIONS (";
            for (size_t i = 0; i < sections.size(); ++i) {
                if (i > 0) result += ", ";
                result += sections[i];
            }
            result += ")";
        }

        return result;
    }

    // AnalyzeDataStatement implementations
    std::string AnalyzeDataStatement::toEsql() const {
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

    // CreatePipelineStatement implementations
    std::string CreatePipelineStatement::toEsql() const {
        std::string result = replace ? "CREATE OR REPLACE PIPELINE " : "CREATE PIPELINE ";
        result += pipeline_name;

        if (!describtion.empty()) {
            result += " DESCRIPTION '" + describtion + "'";
        }

        if (!steps.empty()) {
            result += " STEPS (";
            for (size_t i = 0; i < steps.size(); ++i) {
                if (i > 0) result += ", ";
                result += steps[i].first;
                if (!steps[i].second.empty()) {
                    result += "(" + steps[i].second + ")";
                }
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

    // BatchAIStatement implementations
    std::string BatchAIStatement::toEsql() const {
        std::string result = "BEGIN BATCH ";

        if (parallel) {
            result += "PARALLEL " + std::to_string(max_concurrent) + " ";
        }

        result += "ON ERROR " + on_error + "\n";

        for (const auto& stmt : statements) {
            result += "  " + stmt->toEsql() + ";\n";
        }

        result += "END BATCH";
        return result;
    }

    // ExplainStatement implementations
    std::string ExplainStatement::toEsql() const {
        std::string result = "EXPLAIN MODEL " + model_name;

        if (input_row) {
            result += " FOR ";
            result += input_row->toString();
        }

        if (shap_values) {
            result += " WITH SHAP_VALUES";
        }

        return result;
    }

    // FeatureImportanceStatement implementations
    std::string FeatureImportanceStatement::toEsql() const {
        std::string result = "FEATURE IMPORTANCE FOR " + model_name;

        if (top_n > 0) {
            result += " TOP " + std::to_string(top_n);
        }

        return result;
    }

    // ModelFunctionCall implementations
    std::unique_ptr<Expression> ModelFunctionCall::clone() const {
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

    ModelFunctionCall::ModelFunctionCall(const std::string& name,
                                        std::vector<std::unique_ptr<Expression>> args,
                                        std::unique_ptr<Expression> al)
        : model_name(name), arguments(std::move(args)), alias(std::move(al)) {}

    std::string ModelFunctionCall::toString() const {
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

    // AIScalarExpression implementations
    std::unique_ptr<Expression> AIScalarExpression::clone() const {
        std::vector<std::unique_ptr<Expression>> cloned_inputs;
        for (const auto& input : inputs) {
            cloned_inputs.push_back(input->clone());
        }

        auto cloned_alias = alias ? alias->clone() : nullptr;

        return std::make_unique<AIScalarExpression>(ai_type, model_name, std::move(cloned_inputs), std::move(cloned_alias), options);
    }

    AIScalarExpression::AIScalarExpression(AIType type, const std::string& model,
                                          std::vector<std::unique_ptr<Expression>> ins,
                                          std::unique_ptr<Expression> al,
                                          const std::unordered_map<std::string, std::string>& opts)
        : ai_type(type), model_name(model), inputs(std::move(ins)), alias(std::move(al)), options(opts) {}

    std::string AIScalarExpression::toString() const {
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
            case AIType::SIMILARITY: type_str = "SIMILARITY"; break;
            case AIType::RECOMMENDATION_SCORE: type_str = "RECOMMENDATION_SCORE"; break;
        }

        std::string result = type_str + "_USING_" + model_name + "(";
        for (size_t i = 0; i < inputs.size(); ++i) {
            result += inputs[i]->toString();
            if (i < inputs.size() - 1) result += ", ";
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

} // namespace AST
