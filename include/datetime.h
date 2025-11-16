#pragma once
#ifndef DATETIME_H
#define DATETIME_H

#include <chrono>
#include <string>
#include <ctime>
#include <sstream>
#include <iomanip>

class DateTime {
    private:
        std::chrono::system_clock::time_point time_point;
        bool is_date_only;

    public:
        DateTime(bool date_only = false) : is_date_only(date_only) {
            time_point = std::chrono::system_clock::now();
        }

        // Construct from time point
        DateTime(std::chrono::system_clock::time_point tp, bool date_only = false) : time_point(tp), is_date_only(date_only) {}
        // Constructor from string (ISO format: "YYYY-MM-DD" or "YYYY-MM-DD HH:MM;;SS")
        DateTime(const std::string& str) {
            std::tm tm = {};
            std::istringstream ss(str);

            if (str.length() == 10) {
                ss >> std::get_time(&tm,"%Y-%m-%d");
                is_date_only = true;
            } else {
                ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
                is_date_only = false;
            }

            if (ss.fail()) {
                throw std::runtime_error("Invalid datetime format: " + str);
            }

            time_point = std::chrono::system_clock::from_time_t(std::mktime(&tm));
        }

        // Constructor from your components
        DateTime(int year, int month,int day,int hour = 0, int minute = 0,int second = 0) : is_date_only(hour == 0 && minute == 0 && second == 0) {
            std::tm tm = {};
            tm.tm_year = year - 1900;
            tm.tm_mon = month - 1;
            tm.tm_mday = day;
            tm.tm_hour = hour;
            tm.tm_min = minute;
            tm.tm_sec = second;

            time_point = std::chrono::system_clock::from_time_t(std::mktime(&tm));
        }

        // Convert to string representation
        std::string toString() const {
            auto time_t = std::chrono::system_clock::to_time_t(time_point);
            std::tm tm = *std::localtime(&time_t);

            std::ostringstream oss;
            if (is_date_only) {
                oss << std::put_time(&tm, "%Y-%m-%d");
            } else {
                oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
            }
            return oss.str();
        }

        // Convert to time_t
        std::time_t toTime() const {
            return std::chrono::system_clock::to_time_t(time_point);
        }

        // Get internal time_point
        std::chrono::system_clock::time_point getTimePoint() const {
            return time_point;
        }

        // Check if this is date only
        bool isDateOnly() const {
            return is_date_only;
        }

        // Comparison opeartions
        bool operator==(const DateTime& other) const{
            return time_point == other.time_point;
        }

        bool operator<(const DateTime& other) const {
            return time_point < other.time_point;
        }

        bool operator>(const DateTime& other) const {
            return time_point > other.time_point;
        }

        // Arithmetic operations
        DateTime addDays(int days) const {
            auto new_time = time_point + std::chrono::hours(24 + days);
            return DateTime(new_time, is_date_only);
        }

        DateTime addHours(int hours) const {
            auto new_time = time_point + std::chrono::hours(hours);
            return DateTime(new_time, false);
        }

        DateTime addMinutes(int minutes) const {
            auto new_time = time_point + std::chrono::minutes(minutes);
            return DateTime(new_time, false); // Adding minutes makes it datetime
        }

        static DateTime now() {
            return DateTime(false);
        }

        static DateTime today() {
            return DateTime(true);
        }

        static DateTime fromString(const std::string& str) {
            return DateTime(str);
        }

        // Get individual components
        int getYear() const {
            auto time_t = std::chrono::system_clock::to_time_t(time_point);
            std::tm tm = *std::localtime(&time_t);
            return tm.tm_year + 1900;
        }

        int getMonth() const {
            auto time_t = std::chrono::system_clock::to_time_t(time_point);
            std::tm tm = *std::localtime(&time_t);
            return tm.tm_mon + 1;
        }

        int getDay() const {
            auto time_t = std::chrono::system_clock::to_time_t(time_point);
            std::tm tm = *std::localtime(&time_t);
            return tm.tm_mday;
        }

        int getHour() const {
            auto time_t = std::chrono::system_clock::to_time_t(time_point);
            std::tm tm = *std::localtime(&time_t);
            return tm.tm_hour;
        }

        int getMinute() const {
            auto time_t = std::chrono::system_clock::to_time_t(time_point);
            std::tm tm = *std::localtime(&time_t);
            return tm.tm_min;
        }

        int getSecond() const {
            auto time_t = std::chrono::system_clock::to_time_t(time_point);
            std::tm tm = *std::localtime(&time_t);
            return tm.tm_sec;
        }
};

#endif

