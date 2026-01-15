#include "ai_grammer.h"
#include "ai_parser.h"
#include "parser.h"
#include "scanner.h"
#include "ai/algorithm_registry.h"
#include <sstream>
#include <algorithm>

AIParser::AIParser(Lexer& lexer, Parse& parser) : lexer_(lexer), base_parser_(parser) {}

std::unique_ptr<AST::Statement> AIParser::parseAIStatement() {
    Token current = base_parser_.getCurrentToken();

    if (base_parser_.checkMatch(Token::Type::TRAIN) || base_parser_.checkMatch(Token::Type::AI_TRAIN)) {
        return parseTrainModel();
    } else if (base_parser_.checkMatch(Token::Type::PREDICT) || base_parser_.checkMatch(Token::Type::INFER)) {
        return parsePredict();
    } else if ((base_parser_.checkMatch(Token::Type::SHOW) &&
               base_parser_.checkPeekToken().type == Token::Type::MODELS) || base_parser_.checkMatch(Token::Type::SHOW_MODELS)) {
        return parseShowModels();
    } else if ((base_parser_.checkMatch(Token::Type::DROP) &&
               base_parser_.checkPeekToken().type == Token::Type::MODEL) || base_parser_.checkMatch(Token::Type::DROP_MODEL)) {
        return parseDropModel();
    } else if (base_parser_.checkMatch(Token::Type::EXPLAIN)) {
        return parseExplain();
    } else if (base_parser_.checkMatch(Token::Type::MODEL_METRICS)) {
        return parseModelMetrics();
    } else if (base_parser_.checkMatch(Token::Type::FEATURE_IMPORTANCE)) {
        return parseFeatureImportance();
    } else if (base_parser_.checkMatch(Token::Type::CREATE_MODEL)) {
        return parseCreateModel();
    } else if (base_parser_.checkMatch(Token::Type::DESCRIBE_MODEL) || (base_parser_.checkMatch(Token::Type::DESCRIBE) && base_parser_.checkPeekToken().type == Token::Type::MODEL)) {
        return parseDescribeModel();
    } else if (base_parser_.checkMatch(Token::Type::ANALYZE) &&
             base_parser_.checkPeekToken().type == Token::Type::DATA) {
        return parseAnalyzeData();
    } else if (base_parser_.checkMatch(Token::Type::CREATE) && base_parser_.checkPeekToken().type == Token::Type::PIPELINE) {
        return parseCreatePipeline();
    }

    throw ParseError(current.line, current.column, "Expected AI statement");
}

std::unique_ptr<AST::CreateModelStatement> AIParser::parseCreateModel() {
    auto stmt = std::make_unique<AST::CreateModelStatement>();

    // Parse CREATE [OR REPLACE] MODEL
    if (base_parser_.checkMatch(Token::Type::CREATE_OR_REPLACE)) {
        base_parser_.consumeToken(Token::Type::CREATE_OR_REPLACE);
        stmt->parameters["replace"] = "true";
    } else {
        base_parser_.consumeToken(Token::Type::CREATE_MODEL);
    }
        // Parse model name
    stmt->model_name = base_parser_.getCurrentToken().lexeme;
    if (!isValidModelName(stmt->model_name)) {
        throw ParseError(base_parser_.getCurrentToken().line,
                        base_parser_.getCurrentToken().column,
                        "Invalid model name: " + stmt->model_name);
    }
    base_parser_.consumeToken(Token::Type::IDENTIFIER);

    // Parse USING clause
    if (base_parser_.checkMatch(Token::Type::USING)) {
        base_parser_.consumeToken(Token::Type::USING);
        //stmt->model_name = base_parser_.getCurrentToken().lexeme;

        // Convert to uppercase for consistency
        std::transform(stmt->algorithm.begin(), stmt->algorithm.end(),stmt->algorithm.begin(), ::toupper);

        /*// Validate algorithm
        auto& algo_registry = esql::ai::AlgorithmRegistry::instance();
        if (!algo_registry.is_algorithm_supported(stmt->algorithm)) {
            throw ParseError(base_parser_.getCurrentToken().line,base_parser_.getCurrentToken().column,"Unsupported algorithm: " + stmt->algorithm + ". Supported: " +
                           [&algo_registry]() {
                               auto algos = algo_registry.get_supported_algorithms();
                               std::string result;
                               for (size_t i = 0; i < algos.size(); ++i) {
                                   if (i > 0) result += ", ";
                                   result += algos[i];
                               }
                               return result;
                           }());
        }*/
        base_parser_.consumeToken(base_parser_.getCurrentToken().type);
    } else {
        //stmt-> model_name = "LIGHTGBM";
    }

        // Parse FEATURES clause
    base_parser_.consumeToken(Token::Type::FEATURES);
    base_parser_.consumeToken(Token::Type::L_PAREN);

    do {
        if (base_parser_.checkMatch(Token::Type::COMMA)) {
            base_parser_.consumeToken(Token::Type::COMMA);
        }

        std::string feature_name = base_parser_.getCurrentToken().lexeme;
        base_parser_.consumeToken(Token::Type::IDENTIFIER);

        // Parse feature type if specified
        std::string feature_type = "AUTO";
        if (base_parser_.checkMatch(Token::Type::AS) ||
            base_parser_.checkMatch(Token::Type::COLON)) {
            base_parser_.consumeToken(base_parser_.getCurrentToken().type);

            if (base_parser_.checkMatchAny({
                Token::Type::INT, Token::Type::FLOAT, Token::Type::BOOL,
                Token::Type::TEXT, Token::Type::CATEGORICAL, Token::Type::NUMERIC
            })) {
                feature_type = base_parser_.getCurrentToken().lexeme;
                base_parser_.consumeToken(base_parser_.getCurrentToken().type);
            }
        }

        stmt->features.emplace_back(feature_name, feature_type);
    } while (base_parser_.checkMatch(Token::Type::COMMA));

        base_parser_.consumeToken(Token::Type::R_PAREN);

    // Parse TARGET clause
    base_parser_.consumeToken(Token::Type::TARGET);

    if (base_parser_.checkMatch(Token::Type::IDENTIFIER)) {
        // Target is a column name
        std::string target_column = base_parser_.getCurrentToken().lexeme;
        base_parser_.consumeToken(Token::Type::IDENTIFIER);

        // Parse target type if specified
        if (base_parser_.checkMatch(Token::Type::AS) ||
            base_parser_.checkMatch(Token::Type::COLON)) {
            base_parser_.consumeToken(base_parser_.getCurrentToken().type);
            if (base_parser_.checkMatchAny({
                Token::Type::CLASSIFICATION, Token::Type::REGRESSION,
                Token::Type::BINARY, Token::Type::MULTICLASS, Token::Type::CLUSTERING, Token::Type::RANKING
            })) {
                stmt->target_type = base_parser_.getCurrentToken().lexeme;
                base_parser_.consumeToken(base_parser_.getCurrentToken().type);
            }
        }

        stmt->parameters["target_column"] = target_column;
    } else if (base_parser_.checkMatchAny({
        Token::Type::CLASSIFICATION, Token::Type::REGRESSION,
        Token::Type::CLUSTERING, Token::Type::BINARY, Token::Type::MULTICLASS, Token::Type::RANKING
    })) {
        // Target type directly specified
        stmt->target_type = base_parser_.getCurrentToken().lexeme;
        base_parser_.consumeToken(base_parser_.getCurrentToken().type);
    }

        // Parse FROM clause if present
    if (base_parser_.checkMatch(Token::Type::FROM)) {
        base_parser_.consumeToken(Token::Type::FROM);
        std::string source_table = base_parser_.getCurrentToken().lexeme;
        base_parser_.consumeToken(Token::Type::IDENTIFIER);
        stmt->parameters["source_table"] = source_table;
    }

    // Parse WITH clause for parameters
    if (base_parser_.checkMatch(Token::Type::WITH)) {
        base_parser_.consumeToken(Token::Type::WITH);
        base_parser_.consumeToken(Token::Type::L_PAREN);

        stmt->parameters.merge(parseHyperparameters());

        base_parser_.consumeToken(Token::Type::R_PAREN);
    }

    return stmt;
}

std::unique_ptr<AST::TrainModelStatement> AIParser::parseTrainModel() {
    auto stmt = std::make_unique<AST::TrainModelStatement>();

    // TRAIN MODEL model_name
    if (base_parser_.checkMatch(Token::Type::AI_TRAIN)) {
        base_parser_.consumeToken(Token::Type::AI_TRAIN);
    } else {
        base_parser_.consumeToken(Token::Type::TRAIN);
        base_parser_.consumeToken(Token::Type::MODEL);
    }

    stmt->model_name = base_parser_.getCurrentToken().lexeme;
    if (!isValidModelName(stmt->model_name)) {
        throw ParseError(base_parser_.getCurrentToken().line,base_parser_.getCurrentToken().column,"Invalid model name: " + stmt->model_name);
    }
    base_parser_.consumeToken(Token::Type::IDENTIFIER);

    // USING algorithm
    if (base_parser_.checkMatch(Token::Type::USING)) {
        base_parser_.consumeToken(Token::Type::USING);
        stmt->algorithm = base_parser_.getCurrentToken().lexeme;
        base_parser_.consumeToken(base_parser_.getCurrentToken().type); // Could be LIGHTGBM, XGBOOST, etc.
    }

    // ON table_name
    if (base_parser_.checkMatch(Token::Type::TARGET)) {
        base_parser_.consumeToken(Token::Type::ON);
        stmt->source_table = base_parser_.getCurrentToken().lexeme;
        base_parser_.consumeToken(Token::Type::IDENTIFIER);
    }

    // TARGET column
    if (base_parser_.checkMatch(Token::Type::TARGET)) {
        base_parser_.consumeToken(Token::Type::TARGET);
        stmt->target_column = base_parser_.getCurrentToken().lexeme;
        base_parser_.consumeToken(Token::Type::IDENTIFIER);
    }

    // FEATURES (col1, col2, ...)
    base_parser_.consumeToken(Token::Type::FEATURES);
    base_parser_.consumeToken(Token::Type::L_PAREN);

    do {
        if (base_parser_.checkMatch(Token::Type::COMMA)) {
            base_parser_.consumeToken(Token::Type::COMMA);
        }
        stmt->feature_columns.push_back(base_parser_.getCurrentToken().lexeme);
        base_parser_.consumeToken(Token::Type::IDENTIFIER);
    } while (base_parser_.checkMatch(Token::Type::COMMA));

    base_parser_.consumeToken(Token::Type::R_PAREN);

    if (base_parser_.checkMatch(Token::Type::TARGET)) {
        base_parser_.consumeToken(Token::Type::TARGET);
        stmt->target_column = base_parser_.getCurrentToken().lexeme;
        base_parser_.consumeToken(Token::Type::IDENTIFIER);
    }

    if (base_parser_.checkMatch(Token::Type::FROM)) {
        base_parser_.consumeToken(Token::Type::FROM);
        stmt->source_table = base_parser_.getCurrentToken().lexeme;
        base_parser_.consumeToken(Token::Type::IDENTIFIER);
    }

    // Optional hyperparameters
    while (base_parser_.checkMatchAny({
        Token::Type::WITH, Token::Type::TEST_SPLIT, Token::Type::ITERATIONS,
        Token::Type::EPOCHS, Token::Type::BATCH_SIZE, Token::Type::LEARNING_RATE,
        Token::Type::HYPERPARAMETERS
    })) {
        if (base_parser_.checkMatch(Token::Type::WITH)) {
            base_parser_.consumeToken(Token::Type::WITH);

            // Parse hyperparameters
            if (base_parser_.checkMatch(Token::Type::HYPERPARAMETERS)) {
                base_parser_.consumeToken(Token::Type::HYPERPARAMETERS);
                base_parser_.consumeToken(Token::Type::L_PAREN);

                do {
                    if (base_parser_.checkMatch(Token::Type::COMMA)) {
                        base_parser_.consumeToken(Token::Type::COMMA);
                    }

                    std::string param_name = base_parser_.getCurrentToken().lexeme;
                    base_parser_.consumeToken(Token::Type::IDENTIFIER);
                    base_parser_.consumeToken(Token::Type::EQUAL);

                    std::string param_value = base_parser_.getCurrentToken().lexeme;
                    if (base_parser_.checkMatch(Token::Type::STRING_LITERAL)) {
                        // Remove quotes
                        if (param_value.size() >= 2 &&
                            ((param_value[0] == '\'' && param_value.back() == '\'') ||
                             (param_value[0] == '"' && param_value.back() == '"'))) {
                            param_value = param_value.substr(1, param_value.size() - 2);
                        }
                        base_parser_.consumeToken(Token::Type::STRING_LITERAL);
                    } else if (base_parser_.checkMatch(Token::Type::NUMBER_LITERAL)) {
                        base_parser_.consumeToken(Token::Type::NUMBER_LITERAL);
                    } else if (base_parser_.checkMatch(Token::Type::TRUE) ||
                              base_parser_.checkMatch(Token::Type::FALSE)) {
                        base_parser_.consumeToken(base_parser_.getCurrentToken().type);
                    }

                    stmt->hyperparameters[param_name] = param_value;
                } while (base_parser_.checkMatch(Token::Type::COMMA));

                base_parser_.consumeToken(Token::Type::R_PAREN);
            }
        } else if (base_parser_.checkMatch(Token::Type::TEST_SPLIT)) {
            base_parser_.consumeToken(Token::Type::TEST_SPLIT);
            base_parser_.consumeToken(Token::Type::EQUAL);

            if (base_parser_.checkMatch(Token::Type::NUMBER_LITERAL)) {
                try {
                    stmt->test_split = std::stof(base_parser_.getCurrentToken().lexeme);
                    base_parser_.consumeToken(Token::Type::NUMBER_LITERAL);
                } catch (...) {
                    throw ParseError(base_parser_.getCurrentToken().line,
                                    base_parser_.getCurrentToken().column,
                                    "Invalid test split value");
                }
            }
        } else if (base_parser_.checkMatch(Token::Type::ITERATIONS)) {
            base_parser_.consumeToken(Token::Type::ITERATIONS);
            base_parser_.consumeToken(Token::Type::EQUAL);

            if (base_parser_.checkMatch(Token::Type::NUMBER_LITERAL)) {
                try {
                    stmt->iterations = std::stoi(base_parser_.getCurrentToken().lexeme);
                    base_parser_.consumeToken(Token::Type::NUMBER_LITERAL);
                } catch (...) {
                    throw ParseError(base_parser_.getCurrentToken().line,
                                    base_parser_.getCurrentToken().column,
                                    "Invalid iterations value");
                }
            }
        }
    }

    // Optional WHERE clause
    if (base_parser_.checkMatch(Token::Type::WHERE)) {
        base_parser_.consumeToken(Token::Type::WHERE);

        // Store WHERE clause as string for now
        std::stringstream where_ss;
        while (!base_parser_.checkMatch(Token::Type::END_OF_INPUT) &&
               !base_parser_.checkMatch(Token::Type::WITH) &&
               !base_parser_.checkMatch(Token::Type::INTO) &&
               !base_parser_.checkMatch(Token::Type::SEMICOLON)) {
            where_ss << base_parser_.getCurrentToken().lexeme << " ";
            base_parser_.advanceToken();
        }
        stmt->where_clause = where_ss.str();
    }

    // Optional INTO table
    if (base_parser_.checkMatch(Token::Type::INTO)) {
        base_parser_.consumeToken(Token::Type::INTO);
        stmt->output_table = base_parser_.getCurrentToken().lexeme;
        base_parser_.consumeToken(Token::Type::IDENTIFIER);
    }

    // Optional SAVE MODEL
    if (base_parser_.checkMatch(Token::Type::SAVE_MODEL)) {
        base_parser_.consumeToken(Token::Type::SAVE_MODEL);
        stmt->save_model = true;
    } else if (base_parser_.checkMatch(Token::Type::DONT_SAVE)) {
        base_parser_.consumeToken(Token::Type::DONT_SAVE);
        stmt->save_model = false;
    }

    return stmt;
}

std::unique_ptr<AST::PredictStatement> AIParser::parsePredict() {
    auto stmt = std::make_unique<AST::PredictStatement>();

    // PREDICT USING model_name
    if (base_parser_.checkMatch(Token::Type::PREDICT)) {
        base_parser_.consumeToken(Token::Type::PREDICT);
    } else {
        base_parser_.consumeToken(Token::Type::INFER);
    }

    base_parser_.consumeToken(Token::Type::USING);
    stmt->model_name = base_parser_.getCurrentToken().lexeme;
    base_parser_.consumeToken(Token::Type::IDENTIFIER);

    // ON table_name
    base_parser_.consumeToken(Token::Type::ON);
    stmt->input_table = base_parser_.getCurrentToken().lexeme;
    base_parser_.consumeToken(Token::Type::IDENTIFIER);

    // Optional WHERE clause
    if (base_parser_.checkMatch(Token::Type::WHERE)) {
        base_parser_.consumeToken(Token::Type::WHERE);

        std::stringstream where_ss;
        while (!base_parser_.checkMatch(Token::Type::END_OF_INPUT) &&
               !base_parser_.checkMatch(Token::Type::INTO) &&
               !base_parser_.checkMatch(Token::Type::WITH) &&
               !base_parser_.checkMatch(Token::Type::LIMIT) &&
               !base_parser_.checkMatch(Token::Type::SEMICOLON)) {
            where_ss << base_parser_.getCurrentToken().lexeme << " ";
            base_parser_.advanceToken();
        }
        stmt->where_clause = where_ss.str();
    }

    // Optional OUTPUT columns
    if (base_parser_.checkMatch(Token::Type::OUTPUT)) {
        base_parser_.consumeToken(Token::Type::OUTPUT);

        if (base_parser_.checkMatch(Token::Type::L_PAREN)) {
            base_parser_.consumeToken(Token::Type::L_PAREN);

            do {
                if (base_parser_.checkMatch(Token::Type::COMMA)) {
                    base_parser_.consumeToken(Token::Type::COMMA);
                }

                std::string col_name = base_parser_.getCurrentToken().lexeme;
                base_parser_.consumeToken(Token::Type::IDENTIFIER);

                // Check for AS alias
                std::string alias = col_name;
                if (base_parser_.checkMatch(Token::Type::AS)) {
                    base_parser_.consumeToken(Token::Type::AS);
                    alias = base_parser_.getCurrentToken().lexeme;
                    base_parser_.consumeToken(Token::Type::IDENTIFIER);
                }

                stmt->output_columns.push_back(alias);
            } while (base_parser_.checkMatch(Token::Type::COMMA));

            base_parser_.consumeToken(Token::Type::R_PAREN);
        }
    }

    // Optional WITH PROBABILITIES/CONFIDENCE
    if (base_parser_.checkMatch(Token::Type::WITH)) {
        base_parser_.consumeToken(Token::Type::WITH);

        if (base_parser_.checkMatch(Token::Type::PROBABILITIES)) {
            base_parser_.consumeToken(Token::Type::PROBABILITIES);
            stmt->include_probabilities = true;
        } else if (base_parser_.checkMatch(Token::Type::CONFIDENCE)) {
            base_parser_.consumeToken(Token::Type::CONFIDENCE);
            stmt->include_confidence = true;
        }
    }

    // INTO output_table
    if (base_parser_.checkMatch(Token::Type::INTO)) {
        base_parser_.consumeToken(Token::Type::INTO);
        stmt->output_table = base_parser_.getCurrentToken().lexeme;
        base_parser_.consumeToken(Token::Type::IDENTIFIER);
    }

    // Optional LIMIT
    if (base_parser_.checkMatch(Token::Type::LIMIT)) {
        base_parser_.consumeToken(Token::Type::LIMIT);

        if (base_parser_.checkMatch(Token::Type::NUMBER_LITERAL)) {
            try {
                stmt->limit = std::stoul(base_parser_.getCurrentToken().lexeme);
                base_parser_.consumeToken(Token::Type::NUMBER_LITERAL);
            } catch (...) {
                throw ParseError(base_parser_.getCurrentToken().line,
                                base_parser_.getCurrentToken().column,
                                "Invalid limit value");
            }
        }
    }

    return stmt;
}

std::unique_ptr<AST::ShowModelsStatement> AIParser::parseShowModels() {
    auto stmt = std::make_unique<AST::ShowModelsStatement>();

    if (base_parser_.checkMatch(Token::Type::SHOW_MODELS)) {
        base_parser_.consumeToken(Token::Type::SHOW_MODELS);
    } else {
        base_parser_.consumeToken(Token::Type::SHOW);
        base_parser_.consumeToken(Token::Type::MODELS);
    }

    // Optional LIKE pattern
    if (base_parser_.checkMatch(Token::Type::LIKE)) {
        base_parser_.consumeToken(Token::Type::LIKE);
        stmt->pattern = base_parser_.getCurrentToken().lexeme;

        // Remove quotes if present
        if (stmt->pattern.size() >= 2 &&
            ((stmt->pattern[0] == '\'' && stmt->pattern.back() == '\'') ||
             (stmt->pattern[0] == '"' && stmt->pattern.back() == '"'))) {
            stmt->pattern = stmt->pattern.substr(1, stmt->pattern.size() - 2);
        }

        base_parser_.consumeToken(base_parser_.getCurrentToken().type);
    }

    // Optional DETAILED
    if (base_parser_.checkMatch(Token::Type::DETAILED)) {
        base_parser_.consumeToken(Token::Type::DETAILED);
        stmt->detailed = true;
    }

    return stmt;
}

std::unique_ptr<AST::DropModelStatement> AIParser::parseDropModel() {
    auto stmt = std::make_unique<AST::DropModelStatement>();

    if (base_parser_.checkMatch(Token::Type::DROP_MODEL)) {
           base_parser_.consumeToken(Token::Type::DROP_MODEL);
    } else {
        base_parser_.consumeToken(Token::Type::DROP);
        base_parser_.consumeToken(Token::Type::MODEL);
    }

    // Optional IF EXISTS
    if (base_parser_.checkMatch(Token::Type::IF)) {
        base_parser_.consumeToken(Token::Type::IF);
        base_parser_.consumeToken(Token::Type::EXISTS);
        stmt->if_exists = true;
    }

    stmt->model_name = base_parser_.getCurrentToken().lexeme;
    if (!isValidModelName(stmt->model_name)) {
        throw ParseError(base_parser_.getCurrentToken().line, base_parser_.getCurrentToken().column,"Invalid model name: " + stmt->model_name);
    }
    base_parser_.consumeToken(Token::Type::IDENTIFIER);

    return stmt;
}

std::unique_ptr<AST::AIFunctionCall> AIParser::parseAIFunctionWithOptions() {
    // This is a convenience wrapper around parseAIFunctionCall
    auto func_call = parseAIFunctionCall();

    if (!func_call) {
        return nullptr;
    }

    // Cast to AIFunctionCall (should always succeed if parseAIFunctionCall returned non-null)
    auto ai_func_call = dynamic_cast<AST::AIFunctionCall*>(func_call.get());
    if (!ai_func_call) {
        throw ParseError(base_parser_.getCurrentToken().line,
                        base_parser_.getCurrentToken().column,
                        "Expected AI function call");
    }

    func_call.release(); // Release ownership since we're transferring it
    return std::unique_ptr<AST::AIFunctionCall>(ai_func_call);
}

std::unordered_map<std::string, std::string> AIParser::parseAIOptions() {
    std::unordered_map<std::string, std::string> options;

    base_parser_.consumeToken(Token::Type::WITH);

    // Check if it's WITH (options) or just WITH keyword
    if (base_parser_.checkMatch(Token::Type::L_PAREN)) {
        base_parser_.consumeToken(Token::Type::L_PAREN);

        do {
            if (base_parser_.checkMatch(Token::Type::COMMA)) {
                base_parser_.consumeToken(Token::Type::COMMA);
            }

            std::string option_name = base_parser_.getCurrentToken().lexeme;
            base_parser_.consumeToken(Token::Type::IDENTIFIER);
            base_parser_.consumeToken(Token::Type::EQUAL);

            std::string option_value;
            if (base_parser_.checkMatch(Token::Type::STRING_LITERAL)) {
                option_value = base_parser_.getCurrentToken().lexeme;
                // Remove quotes
                if (option_value.size() >= 2 &&
                    ((option_value[0] == '\'' && option_value.back() == '\'') ||
                     (option_value[0] == '"' && option_value.back() == '"'))) {
                    option_value = option_value.substr(1, option_value.size() - 2);
                }
                base_parser_.consumeToken(Token::Type::STRING_LITERAL);
            } else if (base_parser_.checkMatch(Token::Type::NUMBER_LITERAL)) {
                option_value = base_parser_.getCurrentToken().lexeme;
                base_parser_.consumeToken(Token::Type::NUMBER_LITERAL);
            } else if (base_parser_.checkMatch(Token::Type::TRUE)) {
                option_value = "true";
                base_parser_.consumeToken(Token::Type::TRUE);
            } else if (base_parser_.checkMatch(Token::Type::FALSE)) {
                option_value = "false";
                base_parser_.consumeToken(Token::Type::FALSE);
            } else if (base_parser_.checkMatch(Token::Type::NULL_TOKEN)) {
                option_value = "NULL";
                base_parser_.consumeToken(Token::Type::NULL_TOKEN);
            } else {
                throw ParseError(base_parser_.getCurrentToken().line,
                                base_parser_.getCurrentToken().column,
                                "Expected option value");
            }

            options[option_name] = option_value;
        } while (base_parser_.checkMatch(Token::Type::COMMA));

        base_parser_.consumeToken(Token::Type::R_PAREN);
        } else {
            // Single option without parentheses (e.g., WITH PROBABILITY)
            if (base_parser_.checkMatch(Token::Type::PROBABILITY) || base_parser_.checkMatch(Token::Type::WITH_PROBABILITY)) {
                base_parser_.consumeToken(base_parser_.getCurrentToken().type);
                options["probability"] = "true";
            } else if (base_parser_.checkMatch(Token::Type::CONFIDENCE) || base_parser_.checkMatch(Token::Type::WITH_CONFIDENCE)) {
                base_parser_.consumeToken(base_parser_.getCurrentToken().type);
                options["confidence"] = "true";
            } else if (base_parser_.checkMatch(Token::Type::EXPLANATION) || base_parser_.checkMatch(Token::Type::WITH_EXPLANATION)) {
                base_parser_.consumeToken(base_parser_.getCurrentToken().type);
                options["explanation"] = "true";
            }
        }

    return options;
}

// Helper methods
AST::AIFunctionType AIParser::parseAIFunctionType() {
    Token current = base_parser_.getCurrentToken();

    switch(current.type) {
        case Token::Type::AI_PREDICT:
            return AST::AIFunctionType::PREDICT;
        case Token::Type::AI_PREDICT_CLASS:
            return AST::AIFunctionType::PREDICT_CLASS;
        case Token::Type::AI_PREDICT_VALUE:
            return AST::AIFunctionType::PREDICT_VALUE;
        case Token::Type::AI_PREDICT_PROBA:
            return AST::AIFunctionType::PREDICT_PROBA;
        case Token::Type::AI_PREDICT_CLUSTER:
            return AST::AIFunctionType::PREDICT_CLUSTER;
        case Token::Type::AI_PREDICT_ANOMALY:
            return AST::AIFunctionType::PREDICT_ANOMALY;
        case Token::Type::AI_EXPLAIN:
            return AST::AIFunctionType::EXPLAIN;
        case Token::Type::AI_TRAIN:
            return AST::AIFunctionType::TRAIN_MODEL;
        case Token::Type::AI_MODEL_METRICS:
            return AST::AIFunctionType::MODEL_METRICS;
        case Token::Type::AI_FEATURE_IMPORTANCE:
            return AST::AIFunctionType::FEATURE_IMPORTANCE;
        default:
            throw ParseError(current.line, current.column,
                           "Unknown AI function type: " + current.lexeme);
    }
}

std::vector<std::unique_ptr<AST::Expression>> AIParser::parseAIFunctionArguments() {
    std::vector<std::unique_ptr<AST::Expression>> arguments;

    // Parse first argument
    arguments.push_back(base_parser_.parseExpressionWrapper());

    // Parse additional arguments
    while (base_parser_.checkMatch(Token::Type::COMMA)) {
        base_parser_.consumeToken(Token::Type::COMMA);
        arguments.push_back(base_parser_.parseExpressionWrapper());
    }

    return arguments;
}

std::string AIParser::parseModelName() {
    if (!base_parser_.checkMatch(Token::Type::STRING_LITERAL)) {
        throw ParseError(base_parser_.getCurrentToken().line,
                        base_parser_.getCurrentToken().column,
                        "Expected model name string literal");
    }

    std::string model_name = base_parser_.getCurrentToken().lexeme;

    // Remove quotes
    if (model_name.size() >= 2 &&
        ((model_name[0] == '\'' && model_name.back() == '\'') ||
         (model_name[0] == '"' && model_name.back() == '"'))) {
        model_name = model_name.substr(1, model_name.size() - 2);
    }

    base_parser_.consumeToken(Token::Type::STRING_LITERAL);
        if (!isValidModelName(model_name)) {
        throw ParseError(base_parser_.getCurrentToken().line,
                        base_parser_.getCurrentToken().column,
                        "Invalid model name: " + model_name);
    }

    return model_name;
}

// Validation
bool AIParser::isValidAIFunctionInContext() const {
    Token current = base_parser_.getCurrentToken();

    // Check if current token is an AI function
    switch(current.type) {
        case Token::Type::AI_PREDICT:
        case Token::Type::AI_PREDICT_CLASS:
        case Token::Type::AI_PREDICT_VALUE:
        case Token::Type::AI_PREDICT_PROBA:
        case Token::Type::AI_PREDICT_CLUSTER:
        case Token::Type::AI_PREDICT_ANOMALY:
        case Token::Type::AI_EXPLAIN:
        case Token::Type::AI_TRAIN:
        case Token::Type::AI_MODEL_METRICS:
        case Token::Type::AI_FEATURE_IMPORTANCE:
            return true;
        default:
            return false;
    }
}

bool AIParser::isValidModelName(const std::string& name) const {
    if (name.empty() || name.length() > 128) {
        return false;
    }

    // First character must be letter or underscore
    if (!std::isalpha(name[0]) && name[0] != '_') {
        return false;
    }

    // Subsequent characters must be alphanumeric, underscore, or dash
    for (char c : name) {
        if (!std::isalnum(c) && c != '_' && c != '-') {
            return false;
        }
    }

       // Check for reserved keywords
    static const std::set<std::string> reserved_keywords = {
        "SELECT", "FROM", "WHERE", "INSERT", "UPDATE", "DELETE",
        "CREATE", "DROP", "TABLE", "DATABASE", "MODEL", "TRAIN",
        "PREDICT", "EXPLAIN", "SHOW", "MODELS", "WITH", "AS",
        "AND", "OR", "NOT", "NULL", "TRUE", "FALSE"
    };

    std::string upper_name = name;
    std::transform(upper_name.begin(), upper_name.end(),
                   upper_name.begin(), ::toupper);

    return reserved_keywords.find(upper_name) == reserved_keywords.end();
}

std::unique_ptr<AST::ExplainStatement> AIParser::parseExplain() {
    auto stmt = std::make_unique<AST::ExplainStatement>();

    base_parser_.consumeToken(Token::Type::EXPLAIN);
    base_parser_.consumeToken(Token::Type::MODEL);

    stmt->model_name = base_parser_.getCurrentToken().lexeme;
    base_parser_.consumeToken(Token::Type::IDENTIFIER);

    // FOR input_row
    base_parser_.consumeToken(Token::Type::FOR);

    if (base_parser_.checkMatch(Token::Type::L_PAREN)) {
        // Parse row values
        base_parser_.consumeToken(Token::Type::L_PAREN);

        std::vector<std::unique_ptr<AST::Expression>> row_values;
        do {
            if (base_parser_.checkMatch(Token::Type::COMMA)) {
                base_parser_.consumeToken(Token::Type::COMMA);
            }
            row_values.push_back(base_parser_.parseExpressionWrapper());
        } while (base_parser_.checkMatch(Token::Type::COMMA));

        base_parser_.consumeToken(Token::Type::R_PAREN);

        // Create a special expression for the row
        // This is simplified - in reality you'd need a RowExpression class
        if (!row_values.empty()) {
            // For simplicity, just use the first value
            stmt->input_row = std::move(row_values[0]);
        }
    } else {
        // Parse a single value
        stmt->input_row = base_parser_.parseExpressionWrapper();
    }

    // Optional WITH SHAP
    if (base_parser_.checkMatch(Token::Type::WITH)) {
        base_parser_.consumeToken(Token::Type::WITH);

        if (base_parser_.checkMatch(Token::Type::SHAP_VALUES)) {
            base_parser_.consumeToken(Token::Type::SHAP_VALUES);
            stmt->shap_values = true;
        }
    }

    return stmt;
}

std::unique_ptr<AST::ModelMetricsStatement> AIParser::parseModelMetrics() {
    auto stmt = std::make_unique<AST::ModelMetricsStatement>();

    base_parser_.consumeToken(Token::Type::MODEL_METRICS);
    base_parser_.consumeToken(Token::Type::FOR);

    stmt->model_name = base_parser_.getCurrentToken().lexeme;
    base_parser_.consumeToken(Token::Type::IDENTIFIER);

    // Optional ON test_data
    if (base_parser_.checkMatch(Token::Type::ON)) {
        base_parser_.consumeToken(Token::Type::ON);
        stmt->test_data_table = base_parser_.getCurrentToken().lexeme;
        base_parser_.consumeToken(Token::Type::IDENTIFIER);
    }

    return stmt;
}

std::unique_ptr<AST::FeatureImportanceStatement> AIParser::parseFeatureImportance() {
    auto stmt = std::make_unique<AST::FeatureImportanceStatement>();

    base_parser_.consumeToken(Token::Type::FEATURE_IMPORTANCE);
    base_parser_.consumeToken(Token::Type::FOR);

    stmt->model_name = base_parser_.getCurrentToken().lexeme;
    base_parser_.consumeToken(Token::Type::IDENTIFIER);

    // Optional TOP N
    if (base_parser_.checkMatch(Token::Type::TOP)) {
        base_parser_.consumeToken(Token::Type::TOP);

        if (base_parser_.checkMatch(Token::Type::NUMBER_LITERAL)) {
            try {
                stmt->top_n = std::stoi(base_parser_.getCurrentToken().lexeme);
                base_parser_.consumeToken(Token::Type::NUMBER_LITERAL);
            } catch (...) {
                throw ParseError(base_parser_.getCurrentToken().line,
                                base_parser_.getCurrentToken().column,
                                "Invalid top N value");
            }
        }
    }

    return stmt;
}

std::unique_ptr<AST::Expression> AIParser::parseAIFunctionCall() {
    // Save the current token for error reporting
    Token current = base_parser_.getCurrentToken();

    // Check if this is an AI function token
    if (!isValidAIFunctionInContext()) {
        return nullptr;
    }

    // Parse the function type from token
    AST::AIFunctionType func_type = parseAIFunctionType();

    // Consume the function token
    base_parser_.consumeToken(base_parser_.getCurrentToken().type);

    // Parse opening parenthesis
    if (!base_parser_.checkMatch(Token::Type::L_PAREN)) {
        throw ParseError(current.line, current.column,
                       "Expected '(' after AI function");
    }
    base_parser_.consumeToken(Token::Type::L_PAREN);

    // Parse model name (must be a string literal)
    std::string model_name = parseModelName();

    // Parse function arguments if any
    std::vector<std::unique_ptr<AST::Expression>> arguments;
    if (base_parser_.checkMatch(Token::Type::COMMA)) {
        base_parser_.consumeToken(Token::Type::COMMA);
        arguments = parseAIFunctionArguments();
    }

        // Parse closing parenthesis
    if (!base_parser_.checkMatch(Token::Type::R_PAREN)) {
        throw ParseError(base_parser_.getCurrentToken().line,
                        base_parser_.getCurrentToken().column,
                        "Expected ')' to close AI function");
    }
    base_parser_.consumeToken(Token::Type::R_PAREN);

    // Parse WITH options if present
    std::unordered_map<std::string, std::string> options;
    if (base_parser_.checkMatch(Token::Type::WITH)) {
        options = parseAIOptions();
    }

    // Parse alias if present
    std::unique_ptr<AST::Expression> alias = nullptr;
    if (base_parser_.checkMatch(Token::Type::AS)) {
        base_parser_.consumeToken(Token::Type::AS);
        alias = base_parser_.parseIdentifierWrapper();
    }

        // Create and return the AI function call
    return std::make_unique<AST::AIFunctionCall>(
        func_type, model_name, std::move(arguments), std::move(alias), options
    );
}

std::unique_ptr<AST::Expression> AIParser::parseAIFunction() {
    Token current = base_parser_.getCurrentToken();

    // Check for AI scalar functions with underscore syntax
    std::string token_str = current.lexeme;
    std::transform(token_str.begin(), token_str.end(), token_str.begin(), ::toupper);

    // Handle PREDICT_USING_model_name, PROBABILITY_USING_model_name, etc.
    size_t using_pos = token_str.find("USING_");
    if (using_pos != std::string::npos && using_pos > 0) {
        // Extract AI function type
        std::string ai_type_str = token_str.substr(0, using_pos);

        // Remove trailing underscore if present
        if (!ai_type_str.empty() && ai_type_str.back() == '_') {
            ai_type_str.pop_back();
        }

                // Extract model name
        std::string model_name = current.lexeme.substr(using_pos + 6); // Skip "USING_"

        // Determine AI type
        AST::AIScalarExpression::AIType ai_type;
        if (ai_type_str == "PREDICT") {
            ai_type = AST::AIScalarExpression::AIType::PREDICT;
        } else if (ai_type_str == "PROBABILITY" || ai_type_str == "PROBABILITIES") {
            ai_type = AST::AIScalarExpression::AIType::PROBABILITY;
        } else if (ai_type_str == "CONFIDENCE") {
            ai_type = AST::AIScalarExpression::AIType::CONFIDENCE;
        } else if (ai_type_str == "ANOMALY" || ai_type_str == "ANOMALY_SCORE") {
            ai_type = AST::AIScalarExpression::AIType::ANOMALY_SCORE;
        } else if (ai_type_str == "CLUSTER" || ai_type_str == "CLUSTER_ID") {
            ai_type = AST::AIScalarExpression::AIType::CLUSTER_ID;
        } else if (ai_type_str == "FORECAST") {
            ai_type = AST::AIScalarExpression::AIType::FORECAST_VALUE;
        } else if (ai_type_str == "RESIDUAL") {
            ai_type = AST::AIScalarExpression::AIType::RESIDUAL;
        } else if (ai_type_str == "INFLUENCE") {
            ai_type = AST::AIScalarExpression::AIType::INFLUENCE;
        } else {
            // Not an AI function, return nullptr
            return nullptr;
        }

        // Consume the function name
        base_parser_.consumeToken(current.type);

        // Parse opening parenthesis
        if (!base_parser_.checkMatch(Token::Type::L_PAREN)) {
            throw ParseError(base_parser_.getCurrentToken().line,
                           base_parser_.getCurrentToken().column,
                           "Expected '(' after AI scalar function");
        }
        base_parser_.consumeToken(Token::Type::L_PAREN);

                // Parse arguments
        std::vector<std::unique_ptr<AST::Expression>> inputs;
        if (!base_parser_.checkMatch(Token::Type::R_PAREN)) {
            do {
                inputs.push_back(base_parser_.parseExpressionWrapper());
            } while (base_parser_.checkMatch(Token::Type::COMMA) &&
                    (base_parser_.consumeToken(Token::Type::COMMA), true));
        }

        // Parse closing parenthesis
        base_parser_.consumeToken(Token::Type::R_PAREN);

        // Parse alias if present
        std::unique_ptr<AST::Expression> alias = nullptr;
        if (base_parser_.checkMatch(Token::Type::AS)) {
            base_parser_.consumeToken(Token::Type::AS);
            alias = base_parser_.parseIdentifierWrapper();
        }

                // Create the scalar expression
        auto scalar_expr = std::make_unique<AST::AIScalarExpression>(
            ai_type, model_name, std::move(inputs)
        );

        std::vector<std::unique_ptr<AST::Expression>> args;
        args.push_back(std::move(scalar_expr));

        // Wrap with alias if needed
        if (alias) {
            // Create a function call wrapper with alias
            return std::make_unique<AST::FunctionCall>(
                current,
                std::move(args),
                std::move(alias)
            );
        }

        return scalar_expr;
    }

       // Handle model direct function calls: model_name(feature1, feature2, ...)
    if (base_parser_.checkMatch(Token::Type::IDENTIFIER)) {
        Token next = base_parser_.checkPeekToken();

        if (next.type == Token::Type::L_PAREN) {
            std::string model_name = current.lexeme;
            base_parser_.consumeToken(Token::Type::IDENTIFIER);
            base_parser_.consumeToken(Token::Type::L_PAREN);

            std::vector<std::unique_ptr<AST::Expression>> args;
            if (!base_parser_.checkMatch(Token::Type::R_PAREN)) {
                do {
                    args.push_back(base_parser_.parseExpressionWrapper());
                } while (base_parser_.checkMatch(Token::Type::COMMA) &&
                        (base_parser_.consumeToken(Token::Type::COMMA), true));
            }

            base_parser_.consumeToken(Token::Type::R_PAREN);

            std::unique_ptr<AST::Expression> alias = nullptr;
            if (base_parser_.checkMatch(Token::Type::AS)) {
                base_parser_.consumeToken(Token::Type::AS);
                alias = base_parser_.parseIdentifierWrapper();
            }

            return std::make_unique<AST::ModelFunctionCall>(
                model_name, std::move(args), std::move(alias)
            );
        }
    }

    // Not an AI function
    return nullptr;
}

/*std::unique_ptr<AST::Expression> AIParser::parseAIFunction() {
    Token current = base_parser_.getCurrentToken();

    // Check for model function calls
    if (base_parser_.checkMatch(Token::Type::IDENTIFIER)) {
        Token next = base_parser_.checkPeekToken();

        // Check if this looks like a model function call
        // e.g., model_name(feature1, feature2, ...)
        if (next.type == Token::Type::L_PAREN) {
            std::string model_name = current.lexeme;
            base_parser_.consumeToken(Token::Type::IDENTIFIER);
            base_parser_.consumeToken(Token::Type::L_PAREN);

            std::vector<std::unique_ptr<AST::Expression>> args;
            if (!base_parser_.checkMatch(Token::Type::R_PAREN)) {
                do {
                    args.push_back(base_parser_.parseExpressionWrapper());
                } while (base_parser_.checkMatch(Token::Type::COMMA) &&
                        (base_parser_.consumeToken(Token::Type::COMMA), true));
            }

            base_parser_.consumeToken(Token::Type::R_PAREN);

            std::unique_ptr<AST::Expression> alias = nullptr;
            if (base_parser_.checkMatch(Token::Type::AS)) {
                base_parser_.consumeToken(Token::Type::AS);
                alias = base_parser_.parseIdentifierWrapper();
            }

            return std::make_unique<AST::ModelFunctionCall>(
                model_name, std::move(args), std::move(alias)
            );
        }
    }

    // Check for AI scalar functions
    // e.g., PREDICT_USING_model_name(features...)
    std::string token_str = current.lexeme;
    std::transform(token_str.begin(), token_str.end(), token_str.begin(), ::toupper);

    if (token_str.find("PREDICT_USING_") == 0 ||
        token_str.find("PROBABILITY_USING_") == 0 ||
        token_str.find("CONFIDENCE_USING_") == 0) {

        // Parse AI scalar expression
        size_t underscore_pos = token_str.find('_');
        size_t using_pos = token_str.find("USING_");

        if (using_pos != std::string::npos) {
            std::string ai_type_str = token_str.substr(0, underscore_pos);
            std::string model_name = current.lexeme.substr(using_pos + 6); // Skip "USING_"

            AST::AIScalarExpression::AIType ai_type;
            if (ai_type_str == "PREDICT") {
                ai_type = AST::AIScalarExpression::AIType::PREDICT;
            } else if (ai_type_str == "PROBABILITY") {
                ai_type = AST::AIScalarExpression::AIType::PROBABILITY;
            } else if (ai_type_str == "CONFIDENCE") {
                ai_type = AST::AIScalarExpression::AIType::CONFIDENCE;
            } else {
                throw ParseError(current.line, current.column,
                               "Unknown AI function type: " + ai_type_str);
            }

            base_parser_.consumeToken(base_parser_.getCurrentToken().type);
            base_parser_.consumeToken(Token::Type::L_PAREN);

            std::vector<std::unique_ptr<AST::Expression>> inputs;
            if (!base_parser_.checkMatch(Token::Type::R_PAREN)) {
                do {
                    inputs.push_back(base_parser_.parseExpressionWrapper());
                } while (base_parser_.checkMatch(Token::Type::COMMA) &&
                        (base_parser_.consumeToken(Token::Type::COMMA), true));
            }

            base_parser_.consumeToken(Token::Type::R_PAREN);

            return std::make_unique<AST::AIScalarExpression>(
                ai_type, model_name, std::move(inputs)
            );
        }
    }

    // Not an AI function, return nullptr to let base parser handle it
    return nullptr;
}*/

std::unordered_map<std::string, std::string> AIParser::parseHyperparameters() {
    std::unordered_map<std::string, std::string> params;

    // Check if we're already inside WITH clause
    bool has_lparen = false;
    if (base_parser_.checkMatch(Token::Type::L_PAREN)) {
        base_parser_.consumeToken(Token::Type::L_PAREN);
        has_lparen = true;
    }

    do {
        if (base_parser_.checkMatch(Token::Type::COMMA)) {
            base_parser_.consumeToken(Token::Type::COMMA);
        }

        std::string param_name = base_parser_.getCurrentToken().lexeme;
        base_parser_.consumeToken(Token::Type::IDENTIFIER);
        base_parser_.consumeToken(Token::Type::EQUAL);
        std::string param_value;
        if (base_parser_.checkMatch(Token::Type::STRING_LITERAL)) {
            param_value = base_parser_.getCurrentToken().lexeme;
            // Remove quotes
            if (param_value.size() >= 2 &&
                ((param_value[0] == '\'' && param_value.back() == '\'') ||
                 (param_value[0] == '"' && param_value.back() == '"'))) {
                param_value = param_value.substr(1, param_value.size() - 2);
            }
            base_parser_.consumeToken(Token::Type::STRING_LITERAL);
        } else if (base_parser_.checkMatch(Token::Type::NUMBER_LITERAL)) {
            param_value = base_parser_.getCurrentToken().lexeme;
            base_parser_.consumeToken(Token::Type::NUMBER_LITERAL);
        } else if (base_parser_.checkMatch(Token::Type::TRUE)) {
            param_value = "true";
            base_parser_.consumeToken(Token::Type::TRUE);
        } else if (base_parser_.checkMatch(Token::Type::FALSE)) {
            param_value = "false";
            base_parser_.consumeToken(Token::Type::FALSE);
        } else {
            throw ParseError(base_parser_.getCurrentToken().line,
                            base_parser_.getCurrentToken().column,
                            "Expected hyperparameter value");
        }

        params[param_name] = param_value;
    } while (base_parser_.checkMatch(Token::Type::COMMA));

    if (has_lparen && base_parser_.checkMatch(Token::Type::R_PAREN)) {
        base_parser_.consumeToken(Token::Type::R_PAREN);
    }

    return params;
}

std::unique_ptr<AST::DescribeModelStatement> AIParser::parseDescribeModel() {
    auto stmt = std::make_unique<AST::DescribeModelStatement>();

    // Parse DESCRIBE MODEL
    if (base_parser_.checkMatch(Token::Type::DESCRIBE_MODEL)) {
        base_parser_.consumeToken(Token::Type::DESCRIBE_MODEL);
    } else if (base_parser_.checkMatch(Token::Type::DESCRIBE)) {
        base_parser_.consumeToken(Token::Type::DESCRIBE);
        base_parser_.consumeToken(Token::Type::MODEL);
    } else {
        throw ParseError(base_parser_.getCurrentToken().line,
                        base_parser_.getCurrentToken().column,
                        "Expected DESCRIBE MODEL");
    }

    // Get model name
    stmt->model_name = base_parser_.getCurrentToken().lexeme;
    if (!isValidModelName(stmt->model_name)) {
        throw ParseError(base_parser_.getCurrentToken().line,base_parser_.getCurrentToken().column,"Invalid model name: " + stmt->model_name);
    }
    base_parser_.consumeToken(Token::Type::IDENTIFIER);

    // Check for EXTENDED keyword
    if (base_parser_.checkMatch(Token::Type::EXTENDED)) {
        base_parser_.consumeToken(Token::Type::EXTENDED);
        stmt->extended = true;
    }

    return stmt;
}

std::unique_ptr<AST::AnalyzeDataStatement> AIParser::parseAnalyzeData() {
    auto stmt = std::make_unique<AST::AnalyzeDataStatement>();

    // Parse ANALYZE DATA
    base_parser_.consumeToken(Token::Type::ANALYZE);
    base_parser_.consumeToken(Token::Type::DATA);

    // Get table name
    stmt->table_name = base_parser_.getCurrentToken().lexeme;
    base_parser_.consumeToken(Token::Type::IDENTIFIER);

    // Parse optional TARGET clause
    if (base_parser_.checkMatch(Token::Type::TARGET)) {
        base_parser_.consumeToken(Token::Type::TARGET);
        stmt->target_column = base_parser_.getCurrentToken().lexeme;
        base_parser_.consumeToken(Token::Type::IDENTIFIER);
    }

        // Parse FEATURES clause
    if (base_parser_.checkMatch(Token::Type::FEATURES)) {
        base_parser_.consumeToken(Token::Type::FEATURES);
        base_parser_.consumeToken(Token::Type::L_PAREN);

        do {
            if (base_parser_.checkMatch(Token::Type::COMMA)) {
                base_parser_.consumeToken(Token::Type::COMMA);
            }
            stmt->feature_columns.push_back(base_parser_.getCurrentToken().lexeme);
            base_parser_.consumeToken(Token::Type::IDENTIFIER);
        } while (base_parser_.checkMatch(Token::Type::COMMA));

        base_parser_.consumeToken(Token::Type::R_PAREN);
    }

    // Parse TYPE clause
    if (base_parser_.checkMatch(Token::Type::TYPE)) {
        base_parser_.consumeToken(Token::Type::TYPE);

        if (base_parser_.checkMatchAny({
            Token::Type::CORRELATION, Token::Type::IMPORTANCE,
            Token::Type::CLUSTERING, Token::Type::OUTLIER,
            Token::Type::DISTRIBUTION, Token::Type::SUMMARY
        })) {
            stmt->analysis_type = base_parser_.getCurrentToken().lexeme;
            base_parser_.consumeToken(base_parser_.getCurrentToken().type);
        } else {
            throw ParseError(base_parser_.getCurrentToken().line,
                           base_parser_.getCurrentToken().column,
                           "Expected analysis type (CORRELATION, IMPORTANCE, etc.)");
        }
    }

    // Parse WITH options
    if (base_parser_.checkMatch(Token::Type::WITH)) {
        base_parser_.consumeToken(Token::Type::WITH);

            if (base_parser_.checkMatch(Token::Type::L_PAREN)) {
            base_parser_.consumeToken(Token::Type::L_PAREN);

            do {
                if (base_parser_.checkMatch(Token::Type::COMMA)) {
                    base_parser_.consumeToken(Token::Type::COMMA);
                }

                std::string param_name = base_parser_.getCurrentToken().lexeme;
                base_parser_.consumeToken(Token::Type::IDENTIFIER);
                base_parser_.consumeToken(Token::Type::EQUAL);

                std::string param_value;
                if (base_parser_.checkMatch(Token::Type::STRING_LITERAL)) {
                    param_value = base_parser_.getCurrentToken().lexeme;
                    // Remove quotes
                    if (param_value.size() >= 2 &&
                        ((param_value[0] == '\'' && param_value.back() == '\'') || (param_value[0] == '"' && param_value.back() == '"'))) {
                        param_value = param_value.substr(1, param_value.size() - 2);
                    }
                    base_parser_.consumeToken(Token::Type::STRING_LITERAL);
                } else if (base_parser_.checkMatch(Token::Type::NUMBER_LITERAL)) {
                    param_value = base_parser_.getCurrentToken().lexeme;
                    base_parser_.consumeToken(Token::Type::NUMBER_LITERAL);
                } else if (base_parser_.checkMatch(Token::Type::TRUE)) {
                    param_value = "true";
                    base_parser_.consumeToken(Token::Type::TRUE);
                } else if (base_parser_.checkMatch(Token::Type::FALSE)) {
                    param_value = "false";
                    base_parser_.consumeToken(Token::Type::FALSE);
                } else {
                    throw ParseError(base_parser_.getCurrentToken().line,base_parser_.getCurrentToken().column,"Expected option value");
                }

                stmt->options[param_name] = param_value;
            } while (base_parser_.checkMatch(Token::Type::COMMA));

            base_parser_.consumeToken(Token::Type::R_PAREN);
        }
    }

    return stmt;
}

std::unique_ptr<AST::CreatePipelineStatement> AIParser::parseCreatePipeline() {
    auto stmt = std::make_unique<AST::CreatePipelineStatement>();

    // Parse CREATE PIPELINE
    base_parser_.consumeToken(Token::Type::CREATE);
    base_parser_.consumeToken(Token::Type::PIPELINE);

    // Get pipeline name
    stmt->pipeline_name = base_parser_.getCurrentToken().lexeme;
    base_parser_.consumeToken(Token::Type::IDENTIFIER);

    // Parse STEPS clause
    if (base_parser_.checkMatch(Token::Type::STEPS)) {
        base_parser_.consumeToken(Token::Type::STEPS);
        base_parser_.consumeToken(Token::Type::L_PAREN);

        do {
            if (base_parser_.checkMatch(Token::Type::COMMA)) {
                base_parser_.consumeToken(Token::Type::COMMA);
            }

            std::string step_type = base_parser_.getCurrentToken().lexeme;
            base_parser_.consumeToken(Token::Type::IDENTIFIER);

                        // Parse step configuration
            std::string step_config;
            if (base_parser_.checkMatch(Token::Type::L_PAREN)) {
                base_parser_.consumeToken(Token::Type::L_PAREN);

                // Read configuration string
                if (base_parser_.checkMatch(Token::Type::STRING_LITERAL)) {
                    step_config = base_parser_.getCurrentToken().lexeme;
                    // Remove quotes
                    if (step_config.size() >= 2 &&
                        ((step_config[0] == '\'' && step_config.back() == '\'') ||
                         (step_config[0] == '"' && step_config.back() == '"'))) {
                        step_config = step_config.substr(1, step_config.size() - 2);
                    }
                    base_parser_.consumeToken(Token::Type::STRING_LITERAL);
                } else {
                    // Could be a JSON-like configuration
                    std::stringstream config_ss;
                    while (!base_parser_.checkMatch(Token::Type::R_PAREN) &&
                           !base_parser_.checkMatch(Token::Type::END_OF_INPUT)) {
                        config_ss << base_parser_.getCurrentToken().lexeme << " ";
                        base_parser_.advanceToken();
                    }
                    step_config = config_ss.str();
                    // Trim trailing space
                    if (!step_config.empty() && step_config.back() == ' ') {
                        step_config.pop_back();
                    }
                }

                base_parser_.consumeToken(Token::Type::R_PAREN);
            }
            stmt->steps.emplace_back(step_type, step_config);
        } while (base_parser_.checkMatch(Token::Type::COMMA));

        base_parser_.consumeToken(Token::Type::R_PAREN);
    }

    // Parse WITH parameters
    if (base_parser_.checkMatch(Token::Type::WITH)) {
        base_parser_.consumeToken(Token::Type::WITH);

        if (base_parser_.checkMatch(Token::Type::L_PAREN)) {
            base_parser_.consumeToken(Token::Type::L_PAREN);
            stmt->parameters = parseHyperparameters();
            base_parser_.consumeToken(Token::Type::R_PAREN);
        }
    }

    return stmt;
}

std::unique_ptr<AST::BatchAIStatement> AIParser::parseBatchAI() {
    auto stmt = std::make_unique<AST::BatchAIStatement>();

    // Parse BEGIN BATCH
    base_parser_.consumeToken(Token::Type::BEGIN);
    base_parser_.consumeToken(Token::Type::BATCH);

    // Parse optional PARALLEL clause
    if (base_parser_.checkMatch(Token::Type::PARALLEL)) {
        base_parser_.consumeToken(Token::Type::PARALLEL);
        stmt->parallel = true;

        if (base_parser_.checkMatch(Token::Type::NUMBER_LITERAL)) {
            try {
                stmt->max_concurrent = std::stoi(base_parser_.getCurrentToken().lexeme);
                base_parser_.consumeToken(Token::Type::NUMBER_LITERAL);
            } catch (...) {
                throw ParseError(base_parser_.getCurrentToken().line,base_parser_.getCurrentToken().column,"Invalid max concurrent value");
            }
        }
    }

       // Parse ON ERROR clause
    if (base_parser_.checkMatch(Token::Type::ON)) {
        base_parser_.consumeToken(Token::Type::ON);
        base_parser_.consumeToken(Token::Type::ERROR);

        if (base_parser_.checkMatch(Token::Type::STOP)) {
            stmt->on_error = "STOP";
            base_parser_.consumeToken(Token::Type::STOP);
        } else if (base_parser_.checkMatch(Token::Type::CONTINUE)) {
            stmt->on_error = "CONTINUE";
            base_parser_.consumeToken(Token::Type::CONTINUE);
        } else if (base_parser_.checkMatch(Token::Type::ROLLBACK)) {
            stmt->on_error = "ROLLBACK";
            base_parser_.consumeToken(Token::Type::ROLLBACK);
        }
    }

    // Parse batch statements
    while (!base_parser_.checkMatch(Token::Type::END) &&!base_parser_.checkMatch(Token::Type::END_OF_INPUT)) {
                // Parse individual AI statement
        /*auto ai_stmt = parseAIStatement();
        stmt->statements.push_back(std::move(ai_stmt));*/

        auto parsed_stmt = parseAIStatement();

        // Convert unique_ptr<Statement> to unique_ptr<AIStatement>
        if (auto ai_stmt = dynamic_cast<AST::AIStatement*>(parsed_stmt.get())) {
            parsed_stmt.release(); // Release ownership
            stmt->statements.push_back(std::unique_ptr<AST::AIStatement>(ai_stmt));
        } else {
            throw ParseError(base_parser_.getCurrentToken().line,
                            base_parser_.getCurrentToken().column,
                            "Expected AI statement in batch");
        }

        // Consume semicolon if present
        if (base_parser_.checkMatch(Token::Type::SEMICOLON)) {
            base_parser_.consumeToken(Token::Type::SEMICOLON);
        }

        // Check if next token is END
        if (base_parser_.checkMatch(Token::Type::END)) {
            break;
        }
    }

    // Parse END BATCH
    if (base_parser_.checkMatch(Token::Type::END)) {
        base_parser_.consumeToken(Token::Type::END);
        base_parser_.consumeToken(Token::Type::BATCH);
    } else {
        throw ParseError(base_parser_.getCurrentToken().line,base_parser_.getCurrentToken().column,"Expected END BATCH");
    }

    return stmt;
}

std::vector<std::string> AIParser::parseAnalysisSections() {
    std::vector<std::string> sections;

    if (base_parser_.checkMatch(Token::Type::SECTIONS)) {
        base_parser_.consumeToken(Token::Type::SECTIONS);
        base_parser_.consumeToken(Token::Type::L_PAREN);

        do {
            if (base_parser_.checkMatch(Token::Type::COMMA)) {
                base_parser_.consumeToken(Token::Type::COMMA);
            }

            sections.push_back(base_parser_.getCurrentToken().lexeme);
            base_parser_.consumeToken(Token::Type::IDENTIFIER);
        } while (base_parser_.checkMatch(Token::Type::COMMA));

        base_parser_.consumeToken(Token::Type::R_PAREN);
    }

    return sections;
}

bool AIParser::validateFeatureList(const std::vector<std::string>& features) const {
    if (features.empty()) {
        return false;
    }

    std::set<std::string> unique_features;
    for (const auto& feature : features) {
        // Feature names must be valid identifiers
        if (!isValidModelName(feature)) {
            return false;
        }

        // Check for duplicates
        if (unique_features.find(feature) != unique_features.end()) {
            return false;
        }
        unique_features.insert(feature);
    }

    return true;
}
