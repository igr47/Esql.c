#include "ai_grammer.h"
#include "ai_parser.h"
#include "parser.h"
#include "scanner.h"
#include <sstream>
#include <algorithm>

AIParser::AIParser(Lexer& lexer, Parse& parser) : lexer_(lexer), base_parser_(parser) {}

std::unique_ptr<AST::Statement> AIParser::parseAIStatement() {
    Token current = base_parser_.getCurrentToken();

    if (base_parser_.checkMatch(Token::Type::TRAIN)) {
        return parseTrainModel();
    } else if (base_parser_.checkMatch(Token::Type::PREDICT) || base_parser_.checkMatch(Token::Type::INFER)) {
        return parsePredict();
    } else if (base_parser_.checkMatch(Token::Type::SHOW) &&
               base_parser_.checkPeekToken().type == Token::Type::MODELS) {
        return parseShowModels();
    } else if (base_parser_.checkMatch(Token::Type::DROP) &&
               base_parser_.checkPeekToken().type == Token::Type::MODEL) {
        return parseDropModel();
    } else if (base_parser_.checkMatch(Token::Type::EXPLAIN)) {
        return parseExplain();
    } else if (base_parser_.checkMatch(Token::Type::MODEL_METRICS)) {
        return parseModelMetrics();
    } else if (base_parser_.checkMatch(Token::Type::FEATURE_IMPORTANCE)) {
        return parseFeatureImportance();
    }

    throw ParseError(current.line, current.column, "Expected AI statement");
}

std::unique_ptr<AST::TrainModelStatement> AIParser::parseTrainModel() {
    auto stmt = std::make_unique<AST::TrainModelStatement>();

    // TRAIN MODEL model_name
    base_parser_.consumeToken(Token::Type::TRAIN);
    base_parser_.consumeToken(Token::Type::MODEL);

    stmt->model_name = base_parser_.getCurrentToken().lexeme;
    base_parser_.consumeToken(Token::Type::IDENTIFIER);

    // USING algorithm
    base_parser_.consumeToken(Token::Type::USING);
    stmt->algorithm = base_parser_.getCurrentToken().lexeme;
    base_parser_.consumeToken(base_parser_.getCurrentToken().type); // Could be LIGHTGBM, XGBOOST, etc.

    // ON table_name
    base_parser_.consumeToken(Token::Type::ON);
    stmt->source_table = base_parser_.getCurrentToken().lexeme;
    base_parser_.consumeToken(Token::Type::IDENTIFIER);

    // TARGET column
    base_parser_.consumeToken(Token::Type::TARGET);
    stmt->target_column = base_parser_.getCurrentToken().lexeme;
    base_parser_.consumeToken(Token::Type::IDENTIFIER);

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

    base_parser_.consumeToken(Token::Type::SHOW);
    base_parser_.consumeToken(Token::Type::MODELS);

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

    base_parser_.consumeToken(Token::Type::DROP);
    base_parser_.consumeToken(Token::Type::MODEL);

    // Optional IF EXISTS
    if (base_parser_.checkMatch(Token::Type::IF)) {
        base_parser_.consumeToken(Token::Type::IF);
        base_parser_.consumeToken(Token::Type::EXISTS);
        stmt->if_exists = true;
    }

    stmt->model_name = base_parser_.getCurrentToken().lexeme;
    base_parser_.consumeToken(Token::Type::IDENTIFIER);

    return stmt;
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

std::unique_ptr<AST::Expression> AIParser::parseAIFunction() {
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
}
