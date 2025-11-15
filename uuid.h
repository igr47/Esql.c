#pragma once
#ifndef UUID_H
#define UUID_H

#include <string>
#include <random>
#include <sstream>
#include <iomanip>

class UUID {
    private:
        std::string value;

    public:
        // Generate new UUID
        UUID() {
            static std::random_device rd;
            static std::mt19937_64 gen(rd());
            static std::uniform_int_distribution<uint64_t> dis;

            uint64_t part1 = dis(gen);
            uint64_t part_2 = dis(gen);

            std::ostringstream oss;
            oss << std::hex << std::setfill('0') << std::setw(16) << part1 << std::setw(16) << part_2;

            std::string uuid = oss.str();

            value = uuid.substr(0,8) + "-" + uuid.substr(8,4) + "-" + uuid.substr(12,4) + "-" + uuid.substr(16,4) + "-" + uuid.substr(20,12);
        }

        // Create from existing strin
        UUID(const std::string& str) : value(str) {
            // Basic validation
            if (str.length() != 36 || str[8] != '-' || str[13] != '-' || str[18] != '-' || str[23] != '-') {
                throw std::runtime_error("Invalid UUID format: " + str);
            }
        }

        // Convert to string
        std::string toString() const {
            return value;
        }

        // Comparison operators
        bool operator==(const UUID& other) const {
            return value == other.value;
        }

        bool operator<(const UUID& other) const {
            return value < other.value;
        }

        // Check if valid
        bool isValid() const {
            return value.length() == 36;
        }

        // Static factory method
        static UUID generate() {
            return UUID();
        }
        
        static UUID fromString(const std::string& str) {
            return UUID(str);
        }
};

#endif


