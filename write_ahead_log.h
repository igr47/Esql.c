#pragma once
#ifndef WRITE_AHEAD_LOG_H
#define WRITE_AHEAD_LOG_H

#include "common_types.h"
#include <fstream>
#include <vector>
#include <map>
#include <string>

// Forward declarations
namespace fractal {
    class DatabaseFile;
    class BufferPool;
}

namespace fractal {

    class WriteAheadLog {
        public:
            enum class LogType : uint8_t {  
                PAGE_UPDATE = 1,
                TRANSACTION_BEGIN = 2,
                TRANSACTION_COMMIT = 3,
                TRANSACTION_ROLLBACK = 4,
                CHECKPOINT = 5,
                MESSAGE_BUFFER = 6, 
                TREE_OPERATION = 7
            };

           struct LogRecord {
              uint64_t lsn; // Log sequence number
              LogType type;
              uint64_t transaction_id;
              uint32_t page_id;
              uint32_t data_length;
              uint64_t timestamp;
              char data[0]; // Flexible array member

              static size_t size_with_data(uint32_t data_len) {
                  return sizeof(LogRecord) + data_len;
              }
           };

           struct RecoveryInfo {
              uint64_t last_checkpoint_lsn;
              std::vector<uint64_t> active_transactions;
              std::map<uint64_t, std::vector<const LogRecord*>> transaction_changes;  
              size_t total_records_recovered;
              size_t pages_restored;
           };

      private: 
           std::string filename;
           mutable std::fstream log_file;
           uint64_t next_lsn{1};
           uint64_t flushed_lsn{0};
           uint64_t last_checkpoint_lsn{0};

           // Statistics
           size_t total_records{0};
           size_t bytes_written{0};
           size_t transaction_count{0};

      public:
           explicit WriteAheadLog(const std::string& filename);
           ~WriteAheadLog();

           uint64_t log_page_update(uint64_t transaction_id, uint32_t page_id, const char* old_data, const char* new_data, uint32_t data_length);  
           uint64_t log_message_buffer(uint64_t transaction_id, uint32_t page_id, const char* message_data, uint32_t message_length);
           uint64_t log_tree_operation(uint64_t transaction_id, const char* operation_data, uint32_t operation_length);  
           void log_transaction_begin(uint64_t transaction_id);
           void log_transaction_commit(uint64_t transaction_id);
           void log_transaction_rollback(uint64_t transaction_id);
           uint64_t log_checkpoint();

           // Recovery methods
           RecoveryInfo recover(DatabaseFile* db_file, BufferPool* buffer_pool);
           void recover_transaction(uint64_t transaction_id, DatabaseFile* db_file, BufferPool* buffer_pool);

           // Maintenance and management
           void flush();
           void truncate(uint64_t up_to_lsn);
           bool create_checkpoint();  
           void archive_log(const std::string& archive_path);

           // Statistics and monitoring
           size_t size() const;
           size_t record_count() const { return total_records; }
           uint64_t get_last_checkpoint_lsn() const { return last_checkpoint_lsn; }  
           uint64_t get_flushed_lsn() const { return flushed_lsn; }
           void print_stats() const;

           // Utility methods
           bool verify_log_integrity() const;
           std::vector<uint64_t> get_active_transactions() const;

      private:
           void write_record(const LogRecord* record);
           bool read_record(LogRecord* record) const;
           void ensure_file_open() const;
           void update_statistics(size_t record_size);
           bool validate_record(const LogRecord* record) const;
           void apply_log_record(const LogRecord* record, DatabaseFile* db_file, BufferPool* buffer_pool);
    };
}

#endif
