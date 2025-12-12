#pragma once
#ifndef DATUM_H
#define DATUM_H

#include <string>
#include <vector>
#include <memory>
#include <variant>
#include <optional>
#include <iostream>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <chrono>
#include <ctime>
#include <cstring>

namespace esql {

/**
 * Datum - A type-safe container for database values
 * Similar to PostgreSQL's Datum
 */
class Datum {
public:
    enum class Type {
        NULL_VALUE,    // NULL
        INTEGER,       // 64-bit integer
        FLOAT,         // 32-bit float
        DOUBLE,        // 64-bit double
        BOOLEAN,       // bool
        STRING,        // std::string
        BLOB,          // binary data
        DATE,          // Date only
        DATETIME,      // Date + Time
        TIMESTAMP,     // Unix timestamp
        INT_ARRAY,     // Array of integers
        FLOAT_ARRAY,   // Array of floats
        STRING_ARRAY,  // Array of strings
        ROW_ID,        // Database row ID
        UNKNOWN        // Unknown type
    };

private:
    using VariantType = std::variant<
        std::monostate,        // NULL
        int64_t,               // INTEGER
        float,                 // FLOAT
        double,                // DOUBLE
        bool,                  // BOOLEAN
        std::string,           // STRING
        std::vector<uint8_t>,  // BLOB
        std::chrono::system_clock::time_point, // DATETIME/TIMESTAMP
        std::vector<int64_t>,  // INT_ARRAY
        std::vector<float>,    // FLOAT_ARRAY
        std::vector<std::string>, // STRING_ARRAY
        uint32_t               // ROW_ID
    >;

    VariantType value_;
    Type type_;

public:
    // Constructors
    Datum() : type_(Type::NULL_VALUE), value_(std::monostate{}) {}

    // Type-specific constructors
    explicit Datum(int64_t val) : type_(Type::INTEGER), value_(val) {}
    explicit Datum(int32_t val) : type_(Type::INTEGER), value_(static_cast<int64_t>(val)) {}
    explicit Datum(float val) : type_(Type::FLOAT), value_(val) {}
    explicit Datum(double val) : type_(Type::DOUBLE), value_(val) {}
    explicit Datum(bool val) : type_(Type::BOOLEAN), value_(val) {}
    explicit Datum(const char* val) : type_(Type::STRING), value_(std::string(val)) {}
    explicit Datum(const std::string& val) : type_(Type::STRING), value_(val) {}
    explicit Datum(std::string&& val) : type_(Type::STRING), value_(std::move(val)) {}
    explicit Datum(const std::vector<uint8_t>& val) : type_(Type::BLOB), value_(val) {}
    explicit Datum(std::chrono::system_clock::time_point val) : type_(Type::DATETIME), value_(val) {}
    explicit Datum(const std::vector<int64_t>& val) : type_(Type::INT_ARRAY), value_(val) {}
    explicit Datum(const std::vector<float>& val) : type_(Type::FLOAT_ARRAY), value_(val) {}
    explicit Datum(const std::vector<std::string>& val) : type_(Type::STRING_ARRAY), value_(val) {}
    explicit Datum(uint32_t row_id) : type_(Type::ROW_ID), value_(row_id) {}

    // Factory methods
    static Datum create_null() { return Datum(); }
    static Datum create_int(int64_t val) { return Datum(val); }
    static Datum create_float(float val) { return Datum(val); }
    static Datum create_double(double val) { return Datum(val); }
    static Datum create_bool(bool val) { return Datum(val); }
    static Datum create_string(const std::string& val) { return Datum(val); }
    static Datum create_string(const char* val) { return Datum(val); }
    static Datum create_blob(const std::vector<uint8_t>& val) { return Datum(val); }
    static Datum create_datetime(std::chrono::system_clock::time_point val) { return Datum(val); }
    static Datum create_date(int year, int month, int day);
    static Datum create_int_array(const std::vector<int64_t>& val) { return Datum(val); }
    static Datum create_float_array(const std::vector<float>& val) { return Datum(val); }
    static Datum create_string_array(const std::vector<std::string>& val) { return Datum(val); }
    static Datum create_row_id(uint32_t val) { return Datum(val); }

    // Type checking
    Type type() const { return type_; }
    bool is_null() const { return type_ == Type::NULL_VALUE; }
    bool is_integer() const { return type_ == Type::INTEGER; }
    bool is_float() const { return type_ == Type::FLOAT; }
    bool is_double() const { return type_ == Type::DOUBLE; }
    bool is_boolean() const { return type_ == Type::BOOLEAN; }
    bool is_string() const { return type_ == Type::STRING; }
    bool is_blob() const { return type_ == Type::BLOB; }
    bool is_datetime() const { return type_ == Type::DATETIME || type_ == Type::TIMESTAMP || type_ == Type::DATE; }
    bool is_array() const {
        return type_ == Type::INT_ARRAY || type_ == Type::FLOAT_ARRAY || type_ == Type::STRING_ARRAY;
    }
    bool is_row_id() const { return type_ == Type::ROW_ID; }

    // Value accessors with type safety
    int64_t as_int() const {
        if (type_ == Type::INTEGER) {
            return std::get<int64_t>(value_);
        } else if (type_ == Type::FLOAT) {
            return static_cast<int64_t>(std::get<float>(value_));
        } else if (type_ == Type::DOUBLE) {
            return static_cast<int64_t>(std::get<double>(value_));
        } else if (type_ == Type::BOOLEAN) {
            return std::get<bool>(value_) ? 1 : 0;
        }
        throw std::runtime_error("Datum is not convertible to integer");
    }

    float as_float() const {
        if (type_ == Type::FLOAT) {
            return std::get<float>(value_);
        } else if (type_ == Type::INTEGER) {
            return static_cast<float>(std::get<int64_t>(value_));
        } else if (type_ == Type::DOUBLE) {
            return static_cast<float>(std::get<double>(value_));
        } else if (type_ == Type::BOOLEAN) {
            return std::get<bool>(value_) ? 1.0f : 0.0f;
        }
        throw std::runtime_error("Datum is not convertible to float");
    }

    double as_double() const {
        if (type_ == Type::DOUBLE) {
            return std::get<double>(value_);
        } else if (type_ == Type::INTEGER) {
            return static_cast<double>(std::get<int64_t>(value_));
        } else if (type_ == Type::FLOAT) {
            return static_cast<double>(std::get<float>(value_));
        } else if (type_ == Type::BOOLEAN) {
            return std::get<bool>(value_) ? 1.0 : 0.0;
        }
        throw std::runtime_error("Datum is not convertible to double");
    }

    bool as_bool() const {
        if (type_ == Type::BOOLEAN) {
            return std::get<bool>(value_);
        } else if (type_ == Type::INTEGER) {
            return std::get<int64_t>(value_) != 0;
        } else if (type_ == Type::FLOAT) {
            return std::get<float>(value_) != 0.0f;
        } else if (type_ == Type::DOUBLE) {
            return std::get<double>(value_) != 0.0;
        } else if (type_ == Type::STRING) {
            const std::string& s = std::get<std::string>(value_);
            return !(s.empty() || s == "false" || s == "0" || s == "no");
        }
        throw std::runtime_error("Datum is not convertible to boolean");
    }

    const std::string& as_string() const {
        if (type_ == Type::STRING) {
            return std::get<std::string>(value_);
        }
        throw std::runtime_error("Datum is not a string");
    }

    std::string as_string_convert() const {
        switch (type_) {
            case Type::NULL_VALUE: return "NULL";
            case Type::INTEGER: return std::to_string(std::get<int64_t>(value_));
            case Type::FLOAT: return std::to_string(std::get<float>(value_));
            case Type::DOUBLE: return std::to_string(std::get<double>(value_));
            case Type::BOOLEAN: return std::get<bool>(value_) ? "true" : "false";
            case Type::STRING: return std::get<std::string>(value_);
            case Type::BLOB: return "[BLOB]";
            case Type::DATETIME:
            case Type::TIMESTAMP:
            case Type::DATE: {
                auto tp = std::get<std::chrono::system_clock::time_point>(value_);
                auto tt = std::chrono::system_clock::to_time_t(tp);
                char buf[64];
                std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&tt));
                return buf;
            }
            case Type::INT_ARRAY: return "[INT_ARRAY]";
            case Type::FLOAT_ARRAY: return "[FLOAT_ARRAY]";
            case Type::STRING_ARRAY: return "[STRING_ARRAY]";
            case Type::ROW_ID: return "ROW_ID:" + std::to_string(std::get<uint32_t>(value_));
            default: return "[UNKNOWN]";
        }
    }

    const std::vector<uint8_t>& as_blob() const {
        if (type_ == Type::BLOB) {
            return std::get<std::vector<uint8_t>>(value_);
        }
        throw std::runtime_error("Datum is not a blob");
    }

    std::chrono::system_clock::time_point as_datetime() const {
        if (type_ == Type::DATETIME || type_ == Type::TIMESTAMP || type_ == Type::DATE) {
            return std::get<std::chrono::system_clock::time_point>(value_);
        }
        throw std::runtime_error("Datum is not a datetime");
    }

    const std::vector<int64_t>& as_int_array() const {
        if (type_ == Type::INT_ARRAY) {
            return std::get<std::vector<int64_t>>(value_);
        }
        throw std::runtime_error("Datum is not an integer array");
    }

    const std::vector<float>& as_float_array() const {
        if (type_ == Type::FLOAT_ARRAY) {
            return std::get<std::vector<float>>(value_);
        }
        throw std::runtime_error("Datum is not a float array");
    }

    const std::vector<std::string>& as_string_array() const {
        if (type_ == Type::STRING_ARRAY) {
            return std::get<std::vector<std::string>>(value_);
        }
        throw std::runtime_error("Datum is not a string array");
    }

    uint32_t as_row_id() const {
        if (type_ == Type::ROW_ID) {
            return std::get<uint32_t>(value_);
        }
        throw std::runtime_error("Datum is not a row ID");
    }

    // String representation
    std::string to_string() const {
        return as_string_convert();
    }

    // Type name
    static const char* type_name(Type type) {
        switch (type) {
            case Type::NULL_VALUE: return "NULL";
            case Type::INTEGER: return "INTEGER";
            case Type::FLOAT: return "FLOAT";
            case Type::DOUBLE: return "DOUBLE";
            case Type::BOOLEAN: return "BOOLEAN";
            case Type::STRING: return "STRING";
            case Type::BLOB: return "BLOB";
            case Type::DATE: return "DATE";
            case Type::DATETIME: return "DATETIME";
            case Type::TIMESTAMP: return "TIMESTAMP";
            case Type::INT_ARRAY: return "INT_ARRAY";
            case Type::FLOAT_ARRAY: return "FLOAT_ARRAY";
            case Type::STRING_ARRAY: return "STRING_ARRAY";
            case Type::ROW_ID: return "ROW_ID";
            default: return "UNKNOWN";
        }
    }

    const char* type_name() const {
        return type_name(type_);
    }

    // Comparison operators
    bool operator==(const Datum& other) const {
        if (type_ != other.type_) return false;

        switch (type_) {
            case Type::NULL_VALUE: return true;
            case Type::INTEGER: return std::get<int64_t>(value_) == std::get<int64_t>(other.value_);
            case Type::FLOAT: return std::get<float>(value_) == std::get<float>(other.value_);
            case Type::DOUBLE: return std::get<double>(value_) == std::get<double>(other.value_);
            case Type::BOOLEAN: return std::get<bool>(value_) == std::get<bool>(other.value_);
            case Type::STRING: return std::get<std::string>(value_) == std::get<std::string>(other.value_);
            case Type::BLOB: return std::get<std::vector<uint8_t>>(value_) == std::get<std::vector<uint8_t>>(other.value_);
            case Type::DATETIME:
            case Type::TIMESTAMP:
            case Type::DATE: return std::get<std::chrono::system_clock::time_point>(value_) ==
                                    std::get<std::chrono::system_clock::time_point>(other.value_);
            case Type::INT_ARRAY: return std::get<std::vector<int64_t>>(value_) == std::get<std::vector<int64_t>>(other.value_);
            case Type::FLOAT_ARRAY: return std::get<std::vector<float>>(value_) == std::get<std::vector<float>>(other.value_);
            case Type::STRING_ARRAY: return std::get<std::vector<std::string>>(value_) == std::get<std::vector<std::string>>(other.value_);
            case Type::ROW_ID: return std::get<uint32_t>(value_) == std::get<uint32_t>(other.value_);
            default: return false;
        }
    }

    bool operator!=(const Datum& other) const {
        return !(*this == other);
    }

    // Hash support
    /*size_t hash() const {
        std::hash<std::string> hasher;
        return hasher(to_string());
    }*/

    // Debug output
    friend std::ostream& operator<<(std::ostream& os, const Datum& d) {
        os << d.to_string();
        return os;
    }
};

// Helper functions for creating common types
inline Datum create_date(int year, int month, int day) {
    std::tm tm = {};
    tm.tm_year = year - 1900;
    tm.tm_mon = month - 1;
    tm.tm_mday = day;
    tm.tm_hour = 0;
    tm.tm_min = 0;
    tm.tm_sec = 0;

    auto tt = std::mktime(&tm);
    if (tt == -1) {
        throw std::runtime_error("Invalid date");
    }

    return Datum::create_datetime(std::chrono::system_clock::from_time_t(tt));
}

inline Datum create_timestamp(time_t timestamp) {
    return Datum::create_datetime(std::chrono::system_clock::from_time_t(timestamp));
}

// Hash specialization for use in unordered containers
/*struct DatumHash {
    size_t operator()(const Datum& d) const {
        return d.hash();
    }
};*/

struct DatumHash {
    size_t operator()(const Datum& d) const {
        // Create a string representation and hash that
        std::string repr = d.to_string();
        std::hash<std::string> hasher;
        return hasher(repr);
    }
};

// Less-than comparison for use in ordered containers
struct DatumLess {
    bool operator()(const Datum& a, const Datum& b) const {
        return a.to_string() < b.to_string();
    }
};

} // namespace esql

#endif // DATUM_H
