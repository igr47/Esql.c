
#ifndef AI_PARSER_H
#define AI_PARSER_H

#include "parser.h"
#include "ai_grammer.h"
#include <memory>
#include <vector>
#include <set>

class AIParser {
private:
    Lexer& lexer_;
    Parse& base_parser_;  // Reference to existing parser

public:
    explicit AIParser(Lexer& lexer, Parse& parser);

    // Main parsing method for AI statements
    std::unique_ptr<AST::Statement> parseAIStatement();

    // Individual AI statement parsers
    std::unique_ptr<AST::TrainModelStatement> parseTrainModel();
    std::unique_ptr<AST::PredictStatement> parsePredict();
    std::unique_ptr<AST::ShowModelsStatement> parseShowModels();
    std::unique_ptr<AST::DropModelStatement> parseDropModel();
    std::unique_ptr<AST::ExplainStatement> parseExplain();
    std::unique_ptr<AST::ModelMetricsStatement> parseModelMetrics();
    std::unique_ptr<AST::FeatureImportanceStatement> parseFeatureImportance();
    std::unique_ptr<AST::CreateModelStatement> parseCreateModel();
    std::unordered_map<std::string, std::string> parseAIOptions();
    std::unique_ptr<AST::AIFunctionCall> parseAIFunctionWithOptions();


    // Parse AI functions in expressions (like PREDICT_USING_model(...))
    std::unique_ptr<AST::Expression> parseAIFunction();
    std::unique_ptr<AST::Expression> parseAIFunctionCall();

    // Helper methods
    std::unordered_map<std::string, std::string> parseHyperparameters();
    std::vector<std::string> parseColumnList();
    std::string parseWhereClause();

private:
    AST::AIFunctionType parseAIFunctionType();
    std::vector<std::unique_ptr<AST::Expression>> parseAIFunctionArguments();
    std::string parseModelName();
    bool isValidAIFunctionInContext() const;
    bool isValidModelName(const std::string& name) const;
    // Helper for checking if next token matches
    bool lookaheadMatches(const std::vector<Token::Type>& types);

    // Helper for parsing key-value pairs
    std::pair<std::string, std::string> parseKeyValuePair();
};

#endif // AI_PARSER_H
