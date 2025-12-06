#include "execution_engine_includes/executionengine_main.h"
#include "database.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <map>
#include <set>

// Check if query has analytical functions
bool ExecutionEngine::hasWindowFunctions(const AST::SelectStatement& stmt) {
    auto checkExpression = [&](const std::unique_ptr<AST::Expression>& expr) -> bool {
        if (!expr) return false;
        
        if (dynamic_cast<const AST::WindowFunction*>(expr.get())) {
            return true;
        }

                // Check nested expressions
        if (auto* binaryOp = dynamic_cast<const AST::BinaryOp*>(expr.get())) {
            return hasWindowFunctions(binaryOp);
        }

        return false;
    };

    for (const auto& col : stmt.columns) {
        if (checkExpression(col)) return true;
    }

    return false;
}

bool ExecutionEngine::hasWindowFunctions(const AST::Expression* expr) {
    if (!expr) return false;

    if (dynamic_cast<const AST::WindowFunction*>(expr)) {
        return true;
    }

    if (auto* binaryOp = dynamic_cast<const AST::BinaryOp*>(expr)) {
        return hasWindowFunctions(binaryOp->left.get()) ||
               hasWindowFunctions(binaryOp->right.get());
    }

    if (auto* aggregate = dynamic_cast<const AST::AggregateExpression*>(expr)) {
        return aggregate->argument && hasWindowFunctions(aggregate->argument.get());
    }

    return false;
}

bool ExecutionEngine::hasStatisticalFunctions(const AST::SelectStatement& stmt) {
    for (const auto& col : stmt.columns) {
        if (dynamic_cast<const AST::StatisticalExpression*>(col.get())) {
            return true;
        }
    }
    return false;
}

// Process all analytical functions
std::vector<std::unordered_map<std::string, std::string>> ExecutionEngine::processAnalyticalFunctions(const std::vector<std::unordered_map<std::string, std::string>>& data, const std::vector<std::unique_ptr<AST::Expression>>& columns) {

    auto result = data;

    // First pass: window functions
    result = processWindowFunctions(result, columns);

    // Second pass: statistical functions
    result = processStatisticalFunctions(result, columns);

    // Third pass: date functions
    result = processDateFunctions(result, columns);

    return result;
}
// Window Function Processing
std::vector<std::unordered_map<std::string, std::string>> ExecutionEngine::processWindowFunctions(std::vector<std::unordered_map<std::string, std::string>>& data, const std::vector<std::unique_ptr<AST::Expression>>& columns) {
    
    for (size_t colIdx = 0; colIdx < columns.size(); ++colIdx) {
        if (auto* windowFunc = dynamic_cast<const AST::WindowFunction*>(columns[colIdx].get())) {
            //std::string resultColumn = columns[colIdx]->toString();
            std::string resultColumn;
            if (windowFunc->alias) {
                resultColumn = windowFunc->alias->toString();
            } else {
                resultColumn = columns[colIdx]->toString();
            }
            
            // Partition data
            auto partitions = partitionData(data, windowFunc->partitionBy);

            for (auto& partition : partitions) {
                // Sort partition if ORDER BY specified
                if (!windowFunc->orderBy.empty()) {
                    partition = sortPartition(partition, windowFunc->orderBy);
                }

                // Apply window function
                applyWindowFunction(partition, windowFunc, resultColumn);
            }

            // Recombine partitions
            data = recombinePartitions(partitions);
        }
    }

    return data;
}

// Statistical Function Processing
std::vector<std::unordered_map<std::string, std::string>> ExecutionEngine::processStatisticalFunctions(const std::vector<std::unordered_map<std::string, std::string>>& data, const std::vector<std::unique_ptr<AST::Expression>>& columns) {
    
    auto result = data;
    
    for (size_t colIdx = 0; colIdx < columns.size(); ++colIdx) {
        if (auto* statFunc = dynamic_cast<const AST::StatisticalExpression*>(columns[colIdx].get())) {
            //std::string resultColumn = columns[colIdx]->toString();
            // Use alias if available
            std::string resultColumn;
            if (statFunc->alias) {
                resultColumn = statFunc->alias->toString();
            } else {
                resultColumn = columns[colIdx]->toString();
            }
            
            switch(statFunc->type) {
                case AST::StatisticalExpression::StatType::STDDEV:
                    applyStdDev(result, statFunc, resultColumn);
                    break;
                case AST::StatisticalExpression::StatType::VARIANCE:
                    applyVariance(result, statFunc, resultColumn);
                    break;
                case AST::StatisticalExpression::StatType::PERCENTILE:
                    applyPercentile(result, statFunc, resultColumn);
                    break;
                case AST::StatisticalExpression::StatType::CORRELATION:
                    applyCorrelation(result, statFunc, resultColumn);
                    break;
                case AST::StatisticalExpression::StatType::REGRESSION:
                    applyRegression(result, statFunc, resultColumn);
                    break;
            }
        }
    }

    return result;
}

// Date Function Processing
std::vector<std::unordered_map<std::string, std::string>> ExecutionEngine::processDateFunctions(const std::vector<std::unordered_map<std::string, std::string>>& data, const std::vector<std::unique_ptr<AST::Expression>>& columns) {

    auto result = data;

    for (size_t colIdx = 0; colIdx < columns.size(); ++colIdx) {
        if (auto* funcCall = dynamic_cast<const AST::FunctionCall*>(columns[colIdx].get())) {
            std::string funcName = funcCall->function.lexeme;

                        // Determine result column name
            std::string resultColumn;
            if (funcCall->alias) {
                resultColumn = funcCall->alias->toString();
            } else {
                resultColumn = columns[colIdx]->toString();
            }

            if (funcName == "JULIANDAY") {
                applyJulianDay(result, funcCall, /*columns[colIdx]->toString()*/resultColumn);
            } else if (funcName == "SUBSTR") {
                applySubstr(result, funcCall, /*columns[colIdx]->toString()*/resultColumn);
            }
        } else if (auto* dateFunc = dynamic_cast<const AST::DateFunction*>(columns[colIdx].get())) {
                       // Determine result column name
            std::string resultColumn;
            if (dateFunc->alias) {
                resultColumn = dateFunc->alias->toString();
            } else {
                resultColumn = columns[colIdx]->toString();
            }

            if (dateFunc->function.type == Token::Type::JULIANDAY) {
                applyJulianDay(result, dateFunc, /*columns[colIdx]->toString()*/resultColumn);
            }
        }
    }

    return result;
}

// Partition data for window functions
std::vector<std::vector<std::unordered_map<std::string, std::string>>> ExecutionEngine::partitionData(const std::vector<std::unordered_map<std::string, std::string>>& data, const std::vector<std::unique_ptr<AST::Expression>>& partitionBy) {

    if (partitionBy.empty()) {
        return {data};
    }

    std::map<std::vector<std::string>, std::vector<std::unordered_map<std::string, std::string>>> partitions;

    for (const auto& row : data) {
        std::vector<std::string> partitionKey;
        for (const auto& expr : partitionBy) {
            partitionKey.push_back(evaluateExpression(expr.get(), row));
        }
        partitions[partitionKey].push_back(row);
    }

    std::vector<std::vector<std::unordered_map<std::string, std::string>>> result;
    for (auto& [key, partition] : partitions) {
        result.push_back(std::move(partition));
    }

    return result;
}

// Sort partition for window functions
std::vector<std::unordered_map<std::string, std::string>> ExecutionEngine::sortPartition(std::vector<std::unordered_map<std::string, std::string>>& partition,const std::vector<std::pair<std::unique_ptr<AST::Expression>, bool>>& orderBy) {

    std::sort(partition.begin(), partition.end(),
        [&](const auto& a, const auto& b) {
            for (const auto& [expr, ascending] : orderBy) {
                std::string valA = evaluateExpression(expr.get(), a);
                std::string valB = evaluateExpression(expr.get(), b);

                if (valA != valB) {
                    if (ascending) {
                        return valA < valB;
                    } else {
                        return valA > valB;
                    }
                }
            }
            return false;
        });

    return partition;
}

// Apply window function to partition
void ExecutionEngine::applyWindowFunction(std::vector<std::unordered_map<std::string, std::string>>& partition,const AST::WindowFunction* windowFunc,const std::string& resultColumn) {

    std::string funcName = windowFunc->function.lexeme;

    if (funcName == "ROW_NUMBER") {
        applyRowNumber(partition, windowFunc, resultColumn);
    }
    else if (funcName == "RANK") {
        applyRank(partition, windowFunc, resultColumn);
    }
    else if (funcName == "DENSE_RANK") {
        applyDenseRank(partition, windowFunc, resultColumn);
    }
    else if (funcName == "NTILE") {
        applyNTile(partition, windowFunc, resultColumn);
    }
    else if (funcName == "LAG") {
        applyLag(partition, windowFunc, resultColumn);
    }
    else if (funcName == "LEAD") {
        applyLead(partition, windowFunc, resultColumn);
    }
    else if (funcName == "FIRST_VALUE") {
        applyFirstValue(partition, windowFunc, resultColumn);
    }
    else if (funcName == "LAST_VALUE") {
        applyLastValue(partition, windowFunc, resultColumn);
    }
}

// Implementationof specific window functions
void ExecutionEngine::applyRowNumber(std::vector<std::unordered_map<std::string, std::string>>& partition,const AST::WindowFunction* windowFunc,const std::string& resultColumn) {
    for (size_t i = 0; i < partition.size(); ++i) {
        partition[i][resultColumn] = std::to_string(i + 1);
    }
}

void ExecutionEngine::applyRank(std::vector<std::unordered_map<std::string, std::string>>& partition,const AST::WindowFunction* windowFunc,const std::string& resultColumn) {
    
    if (windowFunc->orderBy.empty()) {
        throw std::runtime_error("RANK requires ORDER BY clause");
    }
    
    std::vector<std::vector<std::string>> orderValues;
    for (const auto& row : partition) {
        std::vector<std::string> rowValues;
        for (const auto& [expr, _] : windowFunc->orderBy) {
            rowValues.push_back(evaluateExpression(expr.get(), row));
        }
        orderValues.push_back(rowValues);
    }

    std::vector<int> ranks(partition.size(), 1);

    for (size_t i = 1; i < partition.size(); ++i) {
        int rank = i + 1;
        // Check if current row has same order values as previous row
        bool sameAsPrevious = true;
        for (size_t j = 0; j < orderValues[i].size(); ++j) {
            if (orderValues[i][j] != orderValues[i-1][j]) {
                sameAsPrevious = false;
                break;
            }
        }

        if (sameAsPrevious) {
            ranks[i] = ranks[i-1];
        } else {
            ranks[i] = rank;
        }
    }

    for (size_t i = 0; i < partition.size(); ++i) {
        partition[i][resultColumn] = std::to_string(ranks[i]);
    }
}
void ExecutionEngine::applyDenseRank(std::vector<std::unordered_map<std::string, std::string>>& partition,const AST::WindowFunction* windowFunc,const std::string& resultColumn) {

    if (windowFunc->orderBy.empty()) {
        throw std::runtime_error("DENSE_RANK requires ORDER BY clause");
    }

    std::vector<std::vector<std::string>> orderValues;
    for (const auto& row : partition) {
        std::vector<std::string> rowValues;
        for (const auto& [expr, _] : windowFunc->orderBy) {
            rowValues.push_back(evaluateExpression(expr.get(), row));
        }

        orderValues.push_back(rowValues);
    }

    std::vector<int> ranks(partition.size(), 1);
    int currentRank = 1;

    for (size_t i = 1; i < partition.size(); ++i) {
        bool sameAsPrevious = true;
        for (size_t j = 0; j < orderValues[i].size(); ++j) {
            if (orderValues[i][j] != orderValues[i-1][j]) {
                sameAsPrevious = false;
                break;
            }
        }

        if (!sameAsPrevious) {
            currentRank++;
        }

        if (!sameAsPrevious) {
            currentRank++;
        }
        ranks[i] = currentRank;
    }

    for (size_t i = 0; i < partition.size(); ++i) {
        partition[i][resultColumn] = std::to_string(ranks[i]);
    }
}

void ExecutionEngine::applyNTile(std::vector<std::unordered_map<std::string, std::string>>& partition,const AST::WindowFunction* windowFunc,const std::string& resultColumn) {

    if (!windowFunc->argument) {
        throw std::runtime_error("NTILE requires bucket count");
    }

    int n = std::stoi(evaluateExpression(windowFunc->argument.get(), partition[0]));
    if (n <= 0) {
        throw std::runtime_error("NTILE bucket count must be positive");
    }

    size_t partitionSize = partition.size();
    size_t baseSize = partitionSize / n;
    size_t remainder = partitionSize % n;

    size_t currentRow = 0;
    for (int bucket = 1; bucket <= n; ++bucket) {
        size_t bucketSize = baseSize + (bucket <= remainder ? 1 : 0);

        for (size_t i = 0; i < bucketSize; ++i) {
            partition[currentRow++][resultColumn] = std::to_string(bucket);
        }
    }
}

void ExecutionEngine::applyLag(std::vector<std::unordered_map<std::string, std::string>>& partition,const AST::WindowFunction* windowFunc,const std::string& resultColumn) {
    
    for (size_t i = 0; i < partition.size(); ++i) {
        if (i == 0) {
            partition[i][resultColumn] = "NULL";
        } else {
            std::string arg = evaluateExpression(windowFunc->argument.get(), partition[i-1]);
            partition[i][resultColumn] = arg;
        }
    }
}

void ExecutionEngine::applyLead(std::vector<std::unordered_map<std::string, std::string>>& partition,const AST::WindowFunction* windowFunc,const std::string& resultColumn) {

    for (size_t i = 0; i < partition.size(); ++i) {
        if (i == partition.size() - 1) {
            partition[i][resultColumn] = "NULL";
        } else {
            std::string arg = evaluateExpression(windowFunc->argument.get(), partition[i+1]);
            partition[i][resultColumn] = arg;
        }
    }
}

void ExecutionEngine::applyRegression(std::vector<std::unordered_map<std::string, std::string>>& data,const AST::StatisticalExpression* statFunc,const std::string& resultColumn) {
    
    std::vector<double> xValues, yValues;
    for (const auto& row : data) {
        std::string xStr = evaluateExpression(statFunc->argument.get(), row);
        std::string yStr = evaluateExpression(statFunc->argument2.get(), row);
        
        if (isNumericString(xStr) && isNumericString(yStr) && 
            xStr != "NULL" && yStr != "NULL") {
            xValues.push_back(std::stod(xStr));
            yValues.push_back(std::stod(yStr));
        }

    }

    if (xValues.size() < 2) {
        for (auto& row : data) row[resultColumn] = "NULL";
        return;
    }

    double slope = calculateRegressionSlope(xValues, yValues);

    for (auto& row : data) {
        row[resultColumn] = std::to_string(slope);
    }
}

double ExecutionEngine::calculateRegressionSlope(
    const std::vector<double>& x, const std::vector<double>& y) {

    double sumX = std::accumulate(x.begin(), x.end(), 0.0);
    double sumY = std::accumulate(y.begin(), y.end(), 0.0);
    double sumXY = 0.0, sumX2 = 0.0;

    for (size_t i = 0; i < x.size(); ++i) {
        sumXY += x[i] * y[i];
        sumX2 += x[i] * x[i];
    }

    double n = x.size();
    double numerator = n * sumXY - sumX * sumY;
    double denominator = n * sumX2 - sumX * sumX;

    return denominator == 0 ? 0 : numerator / denominator;
}

void ExecutionEngine::applyFirstValue(std::vector<std::unordered_map<std::string, std::string>>& partition,const AST::WindowFunction* windowFunc,const std::string& resultColumn) {

    if (!windowFunc->argument) {
        throw std::runtime_error("FIRST_VALUE requires argument");
    }

    std::string firstValue = evaluateExpression(windowFunc->argument.get(), partition[0]);
    for (auto& row : partition) {
        row[resultColumn] = firstValue;
    }
}

void ExecutionEngine::applyLastValue(std::vector<std::unordered_map<std::string, std::string>>& partition,const AST::WindowFunction* windowFunc,const std::string& resultColumn) {

    if (!windowFunc->argument) {
        throw std::runtime_error("LAST_VALUE requires argument");
    }

    std::string lastValue = evaluateExpression(windowFunc->argument.get(), partition.back());
    for (auto& row : partition) {
        row[resultColumn] = lastValue;
    }
}

// Statistical function implementations
void ExecutionEngine::applyStdDev(std::vector<std::unordered_map<std::string, std::string>>& data,const AST::StatisticalExpression* statFunc,const std::string& resultColumn) {
    
    std::vector<double> values;
    for (const auto& row : data) {
        std::string valStr = evaluateExpression(statFunc->argument.get(), row);
        if (isNumericString(valStr) && valStr != "NULL") {
            values.push_back(std::stod(valStr));
        }
    }

    if (values.empty()) {
        for (auto& row : data) row[resultColumn] = "NULL";
        return;
    }

    double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    double sumSq = 0.0;
    for (double val : values) {
        sumSq += (val - mean) * (val - mean);
    }

    double stddev = std::sqrt(sumSq / values.size());

    for (auto& row : data) {
        row[resultColumn] = std::to_string(stddev);
    }
}

void ExecutionEngine::applyVariance(std::vector<std::unordered_map<std::string, std::string>>& data,const AST::StatisticalExpression* statFunc,const std::string& resultColumn) {

    std::vector<double> values;
    for (const auto& row : data) {
        std::string valStr = evaluateExpression(statFunc->argument.get(), row);
        if (isNumericString(valStr) && valStr != "NULL") {
            values.push_back(std::stod(valStr));
        }
    }

    if (values.empty()) {
        for (auto& row : data) row[resultColumn] = "NULL";
        return;
    }
    double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    double variance = 0.0;
    for (double val : values) {
        variance += (val - mean) * (val - mean);
    }
    variance /= values.size();

    for (auto& row : data) {
        row[resultColumn] = std::to_string(variance);
    }
}

void ExecutionEngine::applyPercentile(std::vector<std::unordered_map<std::string, std::string>>& data,const AST::StatisticalExpression* statFunc,const std::string& resultColumn) {
    
    std::vector<double> values;
    for (const auto& row : data) {
        std::string valStr = evaluateExpression(statFunc->argument.get(), row);
        if (isNumericString(valStr) && valStr != "NULL") {
            values.push_back(std::stod(valStr));
        }
    }
    
    if (values.empty()) {
        for (auto& row : data) row[resultColumn] = "NULL";
        return;
    }

    std::sort(values.begin(), values.end());
    double percentile = statFunc->percentileValue;

    if (percentile < 0 || percentile > 1) {
        throw std::runtime_error("Percentile must be between 0 and 1");
    }

    // Linear interpolation method
    double position = percentile * (values.size() - 1);
    size_t index = static_cast<size_t>(position);
    double fraction = position - index;

    double percentileValue;
    if (index == values.size() - 1) {
        percentileValue = values[index];
    } else {
        percentileValue = values[index] + fraction * (values[index + 1] - values[index]);
    }

    for (auto& row : data) {
        row[resultColumn] = std::to_string(percentileValue);
    }
}

void ExecutionEngine::applyCorrelation(std::vector<std::unordered_map<std::string, std::string>>& data,const AST::StatisticalExpression* statFunc,const std::string& resultColumn) {

    std::vector<double> xValues, yValues;
    for (const auto& row : data) {
        std::string xStr = evaluateExpression(statFunc->argument.get(), row);
        std::string yStr = evaluateExpression(statFunc->argument2.get(), row);

        if (isNumericString(xStr) && isNumericString(yStr) &&
            xStr != "NULL" && yStr != "NULL") {
            xValues.push_back(std::stod(xStr));
            yValues.push_back(std::stod(yStr));
        }
    }

    if (xValues.size() < 2) {
        for (auto& row : data) row[resultColumn] = "NULL";
        return;
    }

    double corr = calculateCorrelation(xValues, yValues);

    for (auto& row : data) {
        row[resultColumn] = std::to_string(corr);
    }
}


double ExecutionEngine::calculateCorrelation(
    const std::vector<double>& x, const std::vector<double>& y) {

    double sumX = std::accumulate(x.begin(), x.end(), 0.0);
    double sumY = std::accumulate(y.begin(), y.end(), 0.0);
    double sumXY = 0.0, sumX2 = 0.0, sumY2 = 0.0;

    for (size_t i = 0; i < x.size(); ++i) {
        sumXY += x[i] * y[i];
        sumX2 += x[i] * x[i];
        sumY2 += y[i] * y[i];
    }

    double n = x.size();
    double numerator = n * sumXY - sumX * sumY;
    double denominator = std::sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));

    return denominator == 0 ? 0 : numerator / denominator;
}

// Date function Iplemementation
void ExecutionEngine::applyJulianDay(std::vector<std::unordered_map<std::string, std::string>>& data,const AST::DateFunction* dateFunc, const std::string& resultColumn) {
    
    for (auto& row : data) {
        std::string dateStr = evaluateExpression(dateFunc->argument.get(), row);
        
        try {
            // Handle special case for "now" string
            if (dateStr == "now" || dateStr == "\"now\"") {
                DateTime now = DateTime::now();
                double julianDay = now.toJulianDay();
                row[resultColumn] = std::to_string(julianDay);
            } else {
                DateTime dt(dateStr);
                double julianDay = dt.toJulianDay();
                row[resultColumn] = std::to_string(julianDay);
            }
        } catch (...) {
            row[resultColumn] = "NULL";
        }
    }
}

void ExecutionEngine::applyJulianDay(std::vector<std::unordered_map<std::string, std::string>>& data,const AST::FunctionCall* funcCall,const std::string& resultColumn) {

    for (auto& row : data) {
        std::string dateStr = evaluateExpression(funcCall->arguments[0].get(), row);

        try {
            if (dateStr == "now" || dateStr == "\"now\"") {
                DateTime now = DateTime::now();
                double julianDay = now.toJulianDay();
                row[resultColumn] = std::to_string(julianDay);
            } else {
                DateTime dt(dateStr);
                double julianDay = dt.toJulianDay();
                row[resultColumn] = std::to_string(julianDay);
            }
        } catch (...) {
            row[resultColumn] = "NULL";
        }
    }
}

// string function implementations
void ExecutionEngine::applySubstr(std::vector<std::unordered_map<std::string, std::string>>& data,const AST::FunctionCall* funcCall,const std::string& resultColumn) {
    
    for (auto& row : data) {
        std::string str = evaluateExpression(funcCall->arguments[0].get(), row);
        int start = std::stoi(evaluateExpression(funcCall->arguments[1].get(), row));
        int length = funcCall->arguments.size() > 2 ? 
                     std::stoi(evaluateExpression(funcCall->arguments[2].get(), row)) : 
                     str.length() - start + 1;

        if (start < 1 || start > str.length() || length < 0) {
            row[resultColumn] = "";
        } else {
            size_t startIdx = start - 1;
            size_t endIdx = std::min(startIdx + length, str.length());
            row[resultColumn] = str.substr(startIdx, endIdx - startIdx);
        }
    }
}


// Join executions
std::vector<std::unordered_map<std::string, std::string>> ExecutionEngine::executeJoins(const std::vector<std::unordered_map<std::string, std::string>>& leftData, const std::vector<std::unique_ptr<AST::JoinClause>>& joins) {
    
    auto result = leftData;
    
    for (const auto& join : joins) {
        std::string tableName;
        if (auto* ident = dynamic_cast<AST::Identifier*>(join->table.get())) {
            tableName = ident->token.lexeme;
        }
        
        auto rightData = storage.getTableData(db.currentDatabase(), tableName);

        switch (join->type) {
            case AST::JoinClause::INNER:
                result = executeInnerJoin(result, rightData, join->condition.get());
                break;
            case AST::JoinClause::LEFT:
                result = executeLeftJoin(result, rightData, join->condition.get());
                break;
            case AST::JoinClause::RIGHT:
                result = executeRightJoin(result, rightData, join->condition.get());
                break;
            case AST::JoinClause::FULL:
                result = executeFullJoin(result, rightData, join->condition.get());
                break;
       }
    }

    return result;
}

std::vector<std::unordered_map<std::string, std::string>> ExecutionEngine::executeInnerJoin(const std::vector<std::unordered_map<std::string, std::string>>& leftData,const std::vector<std::unordered_map<std::string, std::string>>& rightData,const AST::Expression* condition) {

    std::vector<std::unordered_map<std::string, std::string>> result;

    for (const auto& leftRow : leftData) {
        for (const auto& rightRow : rightData) {
            // Merge rows
            auto mergedRow = leftRow;
            mergedRow.insert(rightRow.begin(), rightRow.end());

            // Check condition
            if (evaluateWhereClause(condition, mergedRow)) {
                result.push_back(mergedRow);
            }
        }
    }

    return result;
}


std::vector<std::unordered_map<std::string, std::string>> ExecutionEngine::executeLeftJoin(const std::vector<std::unordered_map<std::string, std::string>>& leftData,const std::vector<std::unordered_map<std::string, std::string>>& rightData, const AST::Expression* condition) {
    
    std::vector<std::unordered_map<std::string, std::string>> result;

    for (const auto& leftRow : leftData) {
        bool matched = false;
        
        for (const auto& rightRow : rightData) {
            // Merge rows
            auto mergedRow = leftRow;
            mergedRow.insert(rightRow.begin(), rightRow.end());

                        // Check condition
            if (evaluateWhereClause(condition, mergedRow)) {
                result.push_back(mergedRow);
                matched = true;
            }
        }

        // If no match found, add left row with NULLs for right columns
        if (!matched) {
            auto leftOnlyRow = leftRow;
            // Add NULL values for all right columns
            if (!rightData.empty()) {
                for (const auto& [key, _] : rightData[0]) {
                    leftOnlyRow[key] = "NULL";
                }
            }
            result.push_back(leftOnlyRow);
        }
    }
    return result;
}

std::vector<std::unordered_map<std::string, std::string>>ExecutionEngine::executeRightJoin(const std::vector<std::unordered_map<std::string, std::string>>& leftData, const std::vector<std::unordered_map<std::string, std::string>>& rightData, const AST::Expression* condition) {

    // Right join is essentially left join with tables swapped
    auto swappedResult = executeLeftJoin(rightData, leftData, condition);

    // But we need to maintain the original column order expectations
    std::vector<std::unordered_map<std::string, std::string>> result;

    for (auto& row : swappedResult) {
        std::unordered_map<std::string, std::string> reorderedRow;
        // Add left columns first
        for (const auto& [key, _] : leftData.empty() ? std::unordered_map<std::string, std::string>{} : leftData[0]) {
            if (row.find(key) != row.end()) {
                reorderedRow[key] = row[key];
            }
        }
        // Add right columns
        for (const auto& [key, _] : rightData.empty() ? std::unordered_map<std::string, std::string>{} : rightData[0]) {
            if (row.find(key) != row.end()) {
                reorderedRow[key] = row[key];
            }
        }
        result.push_back(reorderedRow);
    }

    return result;
}

std::vector<std::unordered_map<std::string, std::string>> ExecutionEngine::executeFullJoin(const std::vector<std::unordered_map<std::string, std::string>>& leftData, const std::vector<std::unordered_map<std::string, std::string>>& rightData, const AST::Expression* condition) {

    std::vector<std::unordered_map<std::string, std::string>> result;
    std::set<std::string> matchedLeftKeys;
    std::set<std::string> matchedRightKeys;

    // Get all column names
    std::set<std::string> allColumns;
    if (!leftData.empty()) {
        for (const auto& [key, _] : leftData[0]) allColumns.insert(key);
    }

    if (!rightData.empty()) {
        for (const auto& [key, _] : rightData[0]) allColumns.insert(key);
    }

    // First do left join and track matches
    for (size_t i = 0; i < leftData.size(); ++i) {
        const auto& leftRow = leftData[i];
        bool matched = false;

        for (size_t j = 0; j < rightData.size(); ++j) {
            const auto& rightRow = rightData[j];

            auto mergedRow = leftRow;
            mergedRow.insert(rightRow.begin(), rightRow.end());

            if (evaluateWhereClause(condition, mergedRow)) {
                result.push_back(mergedRow);
                matched = true;
                matchedLeftKeys.insert("left_" + std::to_string(i));
                matchedRightKeys.insert("right_" + std::to_string(j));
            }
        }

        if (!matched) {
            auto leftOnlyRow = leftRow;
            for (const auto& col : allColumns) {
                if (leftOnlyRow.find(col) == leftOnlyRow.end()) {
                    leftOnlyRow[col] = "NULL";
                }
            }

            result.push_back(leftOnlyRow);
        }
    }

    // Add unmatched right rows
    for (size_t j = 0; j < rightData.size(); ++j) {
        if (matchedRightKeys.find("right_" + std::to_string(j)) == matchedRightKeys.end()) {
            auto rightOnlyRow = rightData[j];
            for (const auto& col : allColumns) {
                if (rightOnlyRow.find(col) == rightOnlyRow.end()) {
                    rightOnlyRow[col] = "NULL";
                }
            }
            result.push_back(rightOnlyRow);
        }
    }
    return result;
}


// Recombine partitions
std::vector<std::unordered_map<std::string, std::string>>
ExecutionEngine::recombinePartitions(
    const std::vector<std::vector<std::unordered_map<std::string, std::string>>>& partitions) {

    std::vector<std::unordered_map<std::string, std::string>> result;

    for (const auto& partition : partitions) {
        result.insert(result.end(), partition.begin(), partition.end());
    }

    return result;
}

std::string ExecutionEngine::evaluateCaseExpression(const AST::CaseExpression* caseExpr,
                                                  const std::unordered_map<std::string, std::string>& row) {
    // Simple CASE: CASE expr WHEN value THEN result...
    if (caseExpr->caseExpression) {
        std::string caseValue = evaluateExpression(caseExpr->caseExpression.get(), row);
        //std::cout << "DEBUG EVAL CASE: Simple CASE, caseValue = " << caseValue << std::endl;

        for (const auto& [condition, result] : caseExpr->whenClauses) {
            std::string whenValue = evaluateExpression(condition.get(), row);
            //std::cout << "DEBUG EVAL CASE: Checking WHEN value = " << whenValue << std::endl;

            // Handle different comparison scenarios
            bool match = false;
            if (isNumericString(caseValue) && isNumericString(whenValue)) {
                match = (std::stod(caseValue) == std::stod(whenValue));
            } else {
                match = (caseValue == whenValue);
            }

            if (match) {
                return evaluateExpression(result.get(), row);
            }
        }
    }

    // Searched CASE: CASE WHEN condition THEN result...
    else {
         //std::cout << "DEBUG EVAL CASE: Searched CASE" << std::endl;
        for (const auto& [condition, result] : caseExpr->whenClauses) {
            std::string condResult = evaluateExpression(condition.get(), row);
            //std::cout << "DEBUG EVAL CASE: Condition result = " << condResult << std::endl;
            bool conditionTrue = (condResult == "true" || condResult == "1" ||
                                 condResult == "TRUE" || condResult == "t");
            if (conditionTrue) {
                return evaluateExpression(result.get(), row);
            }
        }
    }

    // ELSE clause
    if (caseExpr->elseClause) {
        return evaluateExpression(caseExpr->elseClause.get(), row);
    }

    return "NULL";
}

void ExecutionEngine::applyYear(std::vector<std::unordered_map<std::string, std::string>>& data,
                              const AST::FunctionCall* funcCall,
                              const std::string& resultColumn) {

    for (auto& row : data) {
        std::string dateStr = evaluateExpression(funcCall->arguments[0].get(), row);

        try {
            DateTime dt(dateStr);
            int year = dt.getYear();
            row[resultColumn] = std::to_string(year);
        } catch (...) {
            row[resultColumn] = "NULL";
        }
    }
}

void ExecutionEngine::applyMonth(std::vector<std::unordered_map<std::string, std::string>>& data,
                               const AST::FunctionCall* funcCall,
                               const std::string& resultColumn) {

    for (auto& row : data) {
        std::string dateStr = evaluateExpression(funcCall->arguments[0].get(), row);

        try {
            DateTime dt(dateStr);
            int month = dt.getMonth();
            row[resultColumn] = std::to_string(month);
        } catch (...) {
            row[resultColumn] = "NULL";
        }
    }
}

void ExecutionEngine::applyDay(std::vector<std::unordered_map<std::string, std::string>>& data,
                             const AST::FunctionCall* funcCall,
                             const std::string& resultColumn) {

    for (auto& row : data) {
        std::string dateStr = evaluateExpression(funcCall->arguments[0].get(), row);

        try {
            DateTime dt(dateStr);
            int day = dt.getDay();
            row[resultColumn] = std::to_string(day);
        } catch (...) {
            row[resultColumn] = "NULL";
        }
    }
}

void ExecutionEngine::applyNow(std::vector<std::unordered_map<std::string, std::string>>& data,
                             const AST::FunctionCall* funcCall,
                             const std::string& resultColumn) {

    DateTime now = DateTime::now();
    for (auto& row : data) {
        row[resultColumn] = now.toString();
    }
}

// String function implementations
void ExecutionEngine::applyConcat(std::vector<std::unordered_map<std::string, std::string>>& data,
                                const AST::FunctionCall* funcCall,
                                const std::string& resultColumn) {

    for (auto& row : data) {
        std::string result;
        for (const auto& arg : funcCall->arguments) {
            result += evaluateExpression(arg.get(), row);
        }
        row[resultColumn] = result;
    }
}

void ExecutionEngine::applyLength(std::vector<std::unordered_map<std::string, std::string>>& data,
                                const AST::FunctionCall* funcCall,
                                const std::string& resultColumn) {

    for (auto& row : data) {
        std::string str = evaluateExpression(funcCall->arguments[0].get(), row);
        row[resultColumn] = std::to_string(str.length());
    }
}

// In execute_analytical.cpp, replace executeAnalyticalSelect with this:

ExecutionEngine::ResultSet ExecutionEngine::executeAnalyticalSelect(AST::SelectStatement& stmt) {
    auto tableName = dynamic_cast<AST::Identifier*>(stmt.from.get())->token.lexeme;
    auto data = storage.getTableData(db.currentDatabase(), tableName);

    // Apply WHERE clause first
    std::vector<std::unordered_map<std::string, std::string>> filteredData;
    for (const auto& row : data) {
        if (!stmt.where || evaluateWhereClause(stmt.where.get(), row)) {
            filteredData.push_back(row);
        }
    }

    // Apply JOINs if present
    if (!stmt.joins.empty()) {
        filteredData = executeJoins(filteredData, stmt.joins);
    }

    ResultSet result;
    result.distinct = stmt.distinct;

    // Separate analytical functions from regular aggregates
    std::vector<std::unique_ptr<AST::Expression>> analyticalCols;
    std::vector<std::unique_ptr<AST::Expression>> aggregateCols;
    std::vector<std::unique_ptr<AST::Expression>> regularCols;
    
    for (const auto& col : stmt.columns) {
        if (dynamic_cast<const AST::StatisticalExpression*>(col.get()) ||
            dynamic_cast<const AST::WindowFunction*>(col.get()) ||
            dynamic_cast<const AST::DateFunction*>(col.get())) {
            analyticalCols.push_back(col->clone());
        } else if (dynamic_cast<const AST::AggregateExpression*>(col.get())) {
            aggregateCols.push_back(col->clone());
        } else {
            regularCols.push_back(col->clone());
        }
    }

    // Build column names
    for (const auto& col : stmt.columns) {
        std::string colName = col->toString();
        
        // Handle aliases for different expression types
        if (auto* statExpr = dynamic_cast<const AST::StatisticalExpression*>(col.get())) {
            if (statExpr->alias) {
                colName = statExpr->alias->toString();
            }
        }
        else if (auto* dateFunc = dynamic_cast<const AST::DateFunction*>(col.get())) {
            if (dateFunc->alias) {
                colName = dateFunc->alias->toString();
            }
        }
        else if (auto* funcCall = dynamic_cast<const AST::FunctionCall*>(col.get())) {
            if (funcCall->alias) {
                colName = funcCall->alias->toString();
            }
        }
        else if (auto* aggExpr = dynamic_cast<const AST::AggregateExpression*>(col.get())) {
            if (aggExpr->argument2) { // argument2 is used as alias in aggregates
                colName = aggExpr->argument2->toString();
            }
        }
        else if (auto* caseExpr = dynamic_cast<const AST::CaseExpression*>(col.get())) {
            if (!caseExpr->alias.empty()) {
                colName = caseExpr->alias;
            }
        }
        else if (auto* windowFunc = dynamic_cast<const AST::WindowFunction*>(col.get())) {
            if (windowFunc->alias) {
                colName = windowFunc->alias->toString();
            }
        }
        
        result.columns.push_back(colName);
    }

    // If we have only analytical functions (no regular aggregates)
    if (aggregateCols.empty()) {
        // Process analytical functions
        auto processedData = processAnalyticalFunctions(filteredData, stmt.columns);
        
        bool hasStatistical = hasStatisticalFunctions(stmt);
        bool hasWindow = hasWindowFunctions(stmt);

        if (hasStatistical && !hasWindow) {
            // Statistical functions only: single result row
            if (!processedData.empty()) {
                std::vector<std::string> resultRow;
                for (const auto& col : stmt.columns) {
                    std::string displayName = col->toString();
                    std::string aliasName = displayName;
                    
                    // Get alias if present
                    if (auto* statExpr = dynamic_cast<const AST::StatisticalExpression*>(col.get())) {
                        if (statExpr->alias) {
                            aliasName = statExpr->alias->toString();
                        }
                    }
                    
                    auto it = processedData[0].find(displayName);
                    if (it != processedData[0].end()) {
                        resultRow.push_back(it->second);
                    } else {
                        // Try with alias
                        it = processedData[0].find(aliasName);
                        if (it != processedData[0].end()) {
                            resultRow.push_back(it->second);
                        } else {
                            resultRow.push_back("NULL");
                        }
                    }
                }
                result.rows.push_back(resultRow);
            }
        } else {
            // Window functions or mixed: multiple result rows
            for (const auto& row : processedData) {
                std::vector<std::string> resultRow;
                for (size_t i = 0; i < result.columns.size(); ++i) {
                    const auto& colName = result.columns[i];
                    const auto& col = stmt.columns[i];
                    
                    // Try different keys to find the value
                    auto it = row.find(colName);
                    if (it != row.end()) {
                        resultRow.push_back(it->second);
                    } else {
                        // Try with expression string
                        std::string exprStr = col->toString();
                        it = row.find(exprStr);
                        if (it != row.end()) {
                            resultRow.push_back(it->second);
                        } else {
                            resultRow.push_back("NULL");
                        }
                    }
                }
                result.rows.push_back(resultRow);
            }
        }
    } 
    // If we have regular aggregates (with or without analytical functions)
    else {
        // First, calculate regular aggregates on the entire filtered dataset
        std::unordered_map<std::string, std::string> aggregateResults;
        
        // Group all data together (single group for aggregates)
        std::vector<std::vector<std::unordered_map<std::string, std::string>>> singleGroup = {filteredData};
        
        // Calculate each aggregate
        for (const auto& col : aggregateCols) {
            if (auto* aggExpr = dynamic_cast<const AST::AggregateExpression*>(col.get())) {
                std::string aggValue = calculateAggregate(aggExpr, filteredData);
                std::string aggKey = col->toString();
                
                aggregateResults[aggKey] = aggValue;
                
                // Also store by alias if present
                if (aggExpr->argument2) {
                    aggregateResults[aggExpr->argument2->toString()] = aggValue;
                }
            }
        }
        
        // Now process analytical functions if any
        if (!analyticalCols.empty() || !regularCols.empty()) {
            // Create a combined columns list for processing
            std::vector<std::unique_ptr<AST::Expression>> allCols;
            
            // Add regular columns first
            for (const auto& col : regularCols) {
                allCols.push_back(col->clone());
            }
            
            // Add analytical columns
            for (const auto& col : analyticalCols) {
                allCols.push_back(col->clone());
            }
            
            // Process analytical functions
            auto processedData = processAnalyticalFunctions(filteredData, allCols);
            
            // Determine output type
            bool hasWindow = hasWindowFunctions(stmt);
            bool onlyStatistical = hasStatisticalFunctions(stmt) && !hasWindow;
            
            if (onlyStatistical) {
                // Statistical functions only: create single result row
                std::vector<std::string> resultRow;
                
                for (const auto& col : stmt.columns) {
                    std::string colKey = col->toString();
                    std::string aliasKey = colKey;
                    
                    // Get alias if present
                    if (auto* aggExpr = dynamic_cast<const AST::AggregateExpression*>(col.get())) {
                        if (aggExpr->argument2) {
                            aliasKey = aggExpr->argument2->toString();
                        }
                    } else if (auto* statExpr = dynamic_cast<const AST::StatisticalExpression*>(col.get())) {
                        if (statExpr->alias) {
                            aliasKey = statExpr->alias->toString();
                        }
                    }
                    
                    // First check aggregate results
                    auto aggIt = aggregateResults.find(colKey);
                    if (aggIt != aggregateResults.end()) {
                        resultRow.push_back(aggIt->second);
                        continue;
                    }
                    
                    aggIt = aggregateResults.find(aliasKey);
                    if (aggIt != aggregateResults.end()) {
                        resultRow.push_back(aggIt->second);
                        continue;
                    }
                    
                    // Then check analytical results
                    if (!processedData.empty()) {
                        auto analIt = processedData[0].find(colKey);
                        if (analIt != processedData[0].end()) {
                            resultRow.push_back(analIt->second);
                            continue;
                        }
                        
                        analIt = processedData[0].find(aliasKey);
                        if (analIt != processedData[0].end()) {
                            resultRow.push_back(analIt->second);
                            continue;
                        }
                    }
                    
                    resultRow.push_back("NULL");
                }
                
                result.rows.push_back(resultRow);
            } else {
                // Window functions present: need to combine aggregates with window results
                for (const auto& row : processedData) {
                    std::vector<std::string> resultRow;
                    
                    for (const auto& col : stmt.columns) {
                        std::string colKey = col->toString();
                        std::string aliasKey = colKey;
                        
                        // Get alias
                        if (auto* aggExpr = dynamic_cast<const AST::AggregateExpression*>(col.get())) {
                            if (aggExpr->argument2) {
                                aliasKey = aggExpr->argument2->toString();
                            }
                        } else if (auto* statExpr = dynamic_cast<const AST::StatisticalExpression*>(col.get())) {
                            if (statExpr->alias) {
                                aliasKey = statExpr->alias->toString();
                            }
                        }
                        
                        // First check if it's in the current row (for window functions)
                        auto rowIt = row.find(colKey);
                        if (rowIt != row.end()) {
                            resultRow.push_back(rowIt->second);
                            continue;
                        }
                        
                        rowIt = row.find(aliasKey);
                        if (rowIt != row.end()) {
                            resultRow.push_back(rowIt->second);
                            continue;
                        }
                        
                        // Then check aggregate results (same for all rows)
                        auto aggIt = aggregateResults.find(colKey);
                        if (aggIt != aggregateResults.end()) {
                            resultRow.push_back(aggIt->second);
                            continue;
                        }
                        
                        aggIt = aggregateResults.find(aliasKey);
                        if (aggIt != aggregateResults.end()) {
                            resultRow.push_back(aggIt->second);
                            continue;
                        }
                        
                        resultRow.push_back("NULL");
                    }
                    
                    result.rows.push_back(resultRow);
                }
            }
        } else {
            // Only aggregates, no analytical functions
            std::vector<std::string> resultRow;
            for (const auto& col : stmt.columns) {
                std::string colKey = col->toString();
                std::string aliasKey = colKey;
                
                if (auto* aggExpr = dynamic_cast<const AST::AggregateExpression*>(col.get())) {
                    if (aggExpr->argument2) {
                        aliasKey = aggExpr->argument2->toString();
                    }
                }
                
                auto it = aggregateResults.find(colKey);
                if (it != aggregateResults.end()) {
                    resultRow.push_back(it->second);
                } else {
                    it = aggregateResults.find(aliasKey);
                    if (it != aggregateResults.end()) {
                        resultRow.push_back(it->second);
                    } else {
                        resultRow.push_back("NULL");
                    }
                }
            }
            result.rows.push_back(resultRow);
        }
    }

    // Apply ORDER BY if present
    if (stmt.orderBy && !result.rows.empty() && result.rows.size() > 1) {
        std::vector<std::unordered_map<std::string, std::string>> dataForSorting;
        for (const auto& rowVec : result.rows) {
            std::unordered_map<std::string, std::string> rowMap;
            for (size_t i = 0; i < result.columns.size(); ++i) {
                rowMap[result.columns[i]] = rowVec[i];
            }
            dataForSorting.push_back(rowMap);
        }

        auto sortedData = sortResult(dataForSorting, stmt.orderBy.get());

        // Convert back
        result.rows.clear();
        for (const auto& rowMap : sortedData) {
            std::vector<std::string> rowVec;
            for (const auto& colName : result.columns) {
                rowVec.push_back(rowMap.at(colName));
            }
            result.rows.push_back(rowVec);
        }
    }

    // Apply DISTINCT if specified
    if (stmt.distinct) {
        result.rows = applyDistinct(result.rows);
    }

    // Apply LIMIT and OFFSET
    if (stmt.limit || stmt.offset) {
        size_t offset = 0;
        size_t limit = result.rows.size();

        if (stmt.offset) {
            try {
                std::string offsetStr = evaluateExpression(stmt.offset.get(), {});
                offset = std::stoul(offsetStr);
            } catch (...) {
                throw std::runtime_error("Invalid OFFSET value");
            }
        }

        if (stmt.limit) {
            try {
                std::string limitStr = evaluateExpression(stmt.limit.get(), {});
                limit = std::stoul(limitStr);
            } catch (...) {
                throw std::runtime_error("Invalid LIMIT value");
            }
        }

        if (offset >= result.rows.size()) {
            result.rows.clear();
        } else {
            size_t end = std::min(offset + limit, result.rows.size());
            result.rows = std::vector<std::vector<std::string>>(
                result.rows.begin() + offset,
                result.rows.begin() + end
            );
        }
    }

    return result;
}


ExecutionEngine::ResultSet ExecutionEngine::executeWithCTE(AST::SelectStatement& stmt) {
    // Execute CTEs first and store results
    std::unordered_map<std::string, ResultSet> cteResults;

    if (stmt.withClause) {
        for (const auto& cte : stmt.withClause->ctes) {
            //cteResults[cte.name] = execute(*cte.query);
            auto* selectQuery = dynamic_cast<AST::SelectStatement*>(cte.query.get());
            if (selectQuery) {
                cteResults[cte.name] = executeSelect(*selectQuery);
            }
        }
    }

    // Now execute main query, replacing CTE references
    // This is a simplified implementation
    // In a full implementation, you'd need to substitute CTE references in the main query

    return executeSelect(stmt);
}
