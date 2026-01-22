#include "plotter_includes/plotter.h"
#include <regex>
#include <algorithm>

namespace Visualization {

    // Helper functions for data type detection
    bool Plotter::isNumericColumn(const std::vector<std::string>& values) {
        if (values.empty()) return false;

        size_t numericCount = 0;
        for (const auto& val : values) {
            try {
                std::stod(val);
                numericCount++;
            } catch (...) {
                // Not numeric
            }
        }

        return (static_cast<double>(numericCount) / values.size()) > 0.8;
    }

    bool Plotter::isIntegerColumn(const std::vector<std::string>& values) {
        if (values.empty()) return false;

        size_t integerCount = 0;
        for (const auto& val : values) {
            try {
                std::stoi(val);
                integerCount++;
            } catch (...) {
                // Not integer
            }
        }

        return (static_cast<double>(integerCount) / values.size()) > 0.8;
    }

    bool Plotter::isBooleanColumn(const std::vector<std::string>& values) {
        if (values.empty()) return false;

        size_t booleanCount = 0;
        for (const auto& val : values) {
            std::string lowerVal = val;
            std::transform(lowerVal.begin(), lowerVal.end(), lowerVal.begin(), ::tolower);
            if (lowerVal == "true" || lowerVal == "false" ||
                lowerVal == "1" || lowerVal == "0" ||
                lowerVal == "yes" || lowerVal == "no" ||
                lowerVal == "on" || lowerVal == "off" ||
                lowerVal == "t" || lowerVal == "f") {
                booleanCount++;
            }
        }

        return (static_cast<double>(booleanCount) / values.size()) > 0.8;
    }

    bool Plotter::isDateColumn(const std::vector<std::string>& values) {
        if (values.empty()) return false;

        // Simple date pattern matching (YYYY-MM-DD)
        size_t dateCount = 0;
        std::regex datePattern(R"(^\d{4}-\d{2}-\d{2}$)");

        for (const auto& val : values) {
            if (std::regex_match(val, datePattern)) {
                dateCount++;
            }
        }

        return (static_cast<double>(dateCount) / values.size()) > 0.8;
    }

    bool Plotter::isDateTimeColumn(const std::vector<std::string>& values) {
        if (values.empty()) return false;

        // Simple datetime pattern matching (YYYY-MM-DD HH:MM:SS)
        size_t datetimeCount = 0;
        std::regex datetimePattern(R"(^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$)");

        for (const auto& val : values) {
            if (std::regex_match(val, datetimePattern)) {
                datetimeCount++;
            }
        }

        return (static_cast<double>(datetimeCount) / values.size()) > 0.8;
    }

    // Data conversion functions
    std::vector<double> Plotter::convertToNumeric(const std::vector<std::string>& values) {
        std::vector<double> numericValues;
        numericValues.reserve(values.size());

        for (const auto& val : values) {
            try {
                numericValues.push_back(std::stod(val));
            } catch (...) {
                numericValues.push_back(std::numeric_limits<double>::quiet_NaN());
            }
        }

        return numericValues;
    }

    std::vector<int> Plotter::convertToInteger(const std::vector<std::string>& values) {
        std::vector<int> intValues;
        intValues.reserve(values.size());

        for (const auto& val : values) {
            try {
                intValues.push_back(std::stoi(val));
            } catch (...) {
                intValues.push_back(0); // Default for invalid integers
            }
        }

        return intValues;
    }

    std::vector<bool> Plotter::convertToBoolean(const std::vector<std::string>& values) {
        std::vector<bool> boolValues;
        boolValues.reserve(values.size());

        for (const auto& val : values) {
            std::string lowerVal = val;
            std::transform(lowerVal.begin(), lowerVal.end(), lowerVal.begin(), ::tolower);

            if (lowerVal == "true" || lowerVal == "1" || lowerVal == "yes" || lowerVal == "on" || lowerVal == "t") {
                boolValues.push_back(true);
            } else {
                boolValues.push_back(false);
            }
        }

        return boolValues;
    }

    // Data validation
    void Plotter::validateNumericData(const std::vector<double>& data, const std::string& columnName) {
        if (data.empty()) {
            throw std::runtime_error("Empty data for column: " + columnName);
        }

        size_t nanCount = 0;
        for (double v : data) {
            if (std::isnan(v)) nanCount++;
        }

        if (nanCount == data.size()) {
            throw std::runtime_error("All values are NaN for column: " + columnName);
        }
    }

    void Plotter::validateCategoricalData(const std::vector<std::string>& data, const std::string& columnName) {
        if (data.empty()) {
            throw std::runtime_error("Empty data for column: " + columnName);
        }

        size_t emptyCount = 0;
        for (const auto& v : data) {
            if (v.empty()) emptyCount++;
        }

        if (emptyCount == data.size()) {
            throw std::runtime_error("All values are empty for column: " + columnName);
        }
    }

} // namespace Visualization
