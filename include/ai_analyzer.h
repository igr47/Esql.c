
#ifndef AI_ANALYZER_H
#define AI_ANALYZER_H

#include "analyzer.h"
#include "ai_grammer.h"
#include "data_extractor.h"
#include "ai/model_registry.h"
#include "database.h"
#include <memory>
#include <unordered_set>
#include <set>

class AIAnalyzer {
private:
    Database& db_;
    fractal::DiskStorage& storage_;
    esql::DataExtractor data_extractor_;

public:
    AIAnalyzer(Database& db, fractal::DiskStorage& storage);

    // Main analysis method
    void analyze(std::unique_ptr<AST::Statement>& stmt);

    // Individual analysis methods
    void analyzeTrainModel(AST::TrainModelStatement& stmt);
    void analyzePredict(AST::PredictStatement& stmt);
    void analyzeShowModels(AST::ShowModelsStatement& stmt);
    void analyzeDropModel(AST::DropModelStatement& stmt);
    void analyzeModelMetrics(AST::ModelMetricsStatement& stmt);
    void analyzeExplain(AST::ExplainStatement& stmt);
    void analyzeFeatureImportance(AST::FeatureImportanceStatement& stmt);

    // Helper validation methods
    bool validateAlgorithm(const std::string& algorithm);
    bool validateHyperparameters(const std::unordered_map<std::string, std::string>& params);
    bool validateFeatureColumns(const std::string& table, const std::vector<std::string>& columns);
    bool validateTargetColumn(const std::string& table, const std::string& column);
    bool validateTestSplit(float split);
    bool validateIterations(int iterations);

    // Model-specific validation
    bool validateModelExists(const std::string& model_name, bool required = true);
    bool validateModelCompatibility(const std::string& model_name,
                                   const std::string& table_name);

    // Data validation
    std::vector<std::string> checkMissingFeatures(const std::string& model_name,
                                                 const std::string& table_name);
    bool checkDataQuality(const std::string& table,
                         const std::string& target_column,
                         const std::vector<std::string>& feature_columns);

    // Utility methods
    bool isValidIdentifier(const std::string& name);
    std::string getCurrentDatabase() const;

private:
    // Internal helper methods
    void validateModelName(const std::string& name);
    void validateOutputTable(const std::string& table_name, bool allow_create = true);
    void checkForCircularDependencies(const std::string& model_name,
                                     const std::string& source_table);

    // Resource validation
    bool hasSufficientMemory(size_t estimated_usage);
    bool hasSufficientDiskSpace(const std::string& model_name);

    // Schema validation
    bool validateSchemaCompatibility(const esql::ai::ModelSchema& schema,
                                    const std::string& table_name);
};

#endif // AI_ANALYZER_H
