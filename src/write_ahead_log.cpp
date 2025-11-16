#include "write_ahead_log.h"
#include "database_file.h"
#include "buffer_pool.h"
#include <iostream>
#include <string>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <chrono>
#include <algorithm>

namespace fractal {

    WriteAheadLog::WriteAheadLog(const std::string& filename) : filename(filename) {
        ensure_file_open();
        std::cout << "WriteAheadLog initialized: " << filename << std::endl;
    }

    WriteAheadLog::~WriteAheadLog() {
        std::cout << "WriteAheadLog destructor started: " << filename << std::endl;
        try {
            if (log_file.is_open()) {
                // Check if file is in good state before closing
                if (log_file.good()) {
                    log_file.flush();
                }
                log_file.close();
                std::cout << "WAL file closed: " << filename << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error closing WAL file: " << e.what() << std::endl;
        }
        std::cout << "WriteAheadLog destroyed. Total records: " << total_records << ", Bytes written: " << bytes_written << std::endl;
    }

    uint64_t WriteAheadLog::log_page_update(uint64_t transaction_id, uint32_t page_id, const char* old_data, const char* new_data, uint32_t data_length) {
        uint64_t lsn = next_lsn++;
        size_t record_size = LogRecord::size_with_data(data_length * 2);

        std::vector<char> buffer(record_size);
        LogRecord* record = reinterpret_cast<LogRecord*>(buffer.data());

        record->lsn = lsn;
        record->type = LogType::PAGE_UPDATE;
        record->transaction_id = transaction_id;
        record->page_id = page_id;
        record->data_length = data_length * 2;
        record->timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

        // Store both old and new data for redo/undo
        std::memcpy(record->data, old_data, data_length);
        std::memcpy(record->data + data_length, new_data, data_length);

        write_record(record);
        update_statistics(record_size);

        return lsn;
    }

    uint64_t WriteAheadLog::log_message_buffer(uint64_t transaction_id, uint32_t page_id, const char* message_data, uint32_t message_length) {
        uint64_t lsn = next_lsn++;
        size_t record_size = LogRecord::size_with_data(message_length);

        std::vector<char> buffer(record_size);
        LogRecord* record = reinterpret_cast<LogRecord*>(buffer.data());

        record->lsn = lsn;
        record->type = LogType::MESSAGE_BUFFER;
        record->transaction_id = transaction_id;
        record->page_id = page_id;
        record->data_length = message_length;
        record->timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        std::memcpy(record->data, message_data, message_length);

        write_record(record);
        update_statistics(record_size);

        return lsn;
    }

    uint64_t WriteAheadLog::log_tree_operation(uint64_t transaction_id, const char* operation_data, uint32_t operation_length) {
        uint64_t lsn = next_lsn++;
        size_t record_size = LogRecord::size_with_data(operation_length);

        std::vector<char> buffer(record_size);
        LogRecord* record = reinterpret_cast<LogRecord*>(buffer.data());

        record->lsn = lsn;
        record->type = LogType::TREE_OPERATION;
        record->transaction_id = transaction_id;
        record->page_id = 0; // Not Page specific
        record->data_length = operation_length;
        record->timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

        std::memcpy(record->data, operation_data, operation_length);

        write_record(record);
        update_statistics(record_size);

        return lsn;
    }

    void WriteAheadLog::log_transaction_begin(uint64_t transaction_id) {
        LogRecord record {};
        record.lsn = next_lsn++;
        record.type = LogType::TRANSACTION_BEGIN;
        record.transaction_id = transaction_id;
        record.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

        write_record(&record);
        update_statistics(sizeof(LogRecord));
        transaction_count++;
    }

    void WriteAheadLog::log_transaction_commit(uint64_t transaction_id) {
        LogRecord record{};
        record.lsn = next_lsn++;
        record.type = LogType::TRANSACTION_COMMIT;
        record.transaction_id = transaction_id;
        record.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

        write_record(&record);
        update_statistics(sizeof(LogRecord));
        flush(); // Force flush on commit for durability
    }

    void WriteAheadLog::log_transaction_rollback(uint64_t transaction_id) {
        LogRecord record {};
        record.lsn = next_lsn++;
        record.type = LogType::TRANSACTION_ROLLBACK;
        record.transaction_id = transaction_id;
        record.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

        write_record(&record);
        update_statistics(sizeof(LogRecord));
    }

    uint64_t WriteAheadLog::log_checkpoint() {
        LogRecord record{};
        record.lsn = next_lsn++;
        record.type = LogType::CHECKPOINT;
        record.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

        write_record(&record);
        update_statistics(sizeof(LogRecord));
        flush();

        last_checkpoint_lsn = record.lsn;

        return record.lsn;
    }

    WriteAheadLog::RecoveryInfo WriteAheadLog::recover(DatabaseFile* db_file, BufferPool* buffer_pool) {
        RecoveryInfo info{};

        if (!log_file.is_open()) {
            std::cout << "WAL: Log file not open, no recovery needed" << std::endl;
            return info;
        }

        log_file.seekg(0, std::ios::beg);

        std::cout << "WAL: Starting recovery process..." << std::endl;

        std::vector<uint64_t> active_transactions;
        std::map<uint64_t, std::vector<const LogRecord*>> transaction_changes;

        try {
            while (log_file.peek() != EOF) {
                LogRecord header;
                if (!read_record(&header)) {
                    break;
                }

                // Read full record with data if needed
                std::vector<char> full_record(LogRecord::size_with_data(header.data_length));
                log_file.seekg(-static_cast<std::streamoff>(sizeof(LogRecord)), std::ios::cur);
                log_file.read(full_record.data(), full_record.size());

                if (!log_file) {
                    std::cerr << "WAL: Failed to read complete log record" << std::endl;
                    break;
                }

                LogRecord* record = reinterpret_cast<LogRecord*>(full_record.data());

                if (!validate_record(record)) {
                    std::cerr << "WAL: Invalid log record at LSN " << record->lsn << std::endl;
                    continue;
                }

                switch (record->type) {
                    case LogType::TRANSACTION_BEGIN:
                        active_transactions.push_back(record->transaction_id);
                        break;

                    case LogType::TRANSACTION_COMMIT:
                        active_transactions.erase(std::remove(active_transactions.begin(), active_transactions.end(), record->transaction_id), active_transactions.end());
                        transaction_changes.erase(record->transaction_id);
                        break;

                    case LogType::TRANSACTION_ROLLBACK:
                        active_transactions.erase(std::remove(active_transactions.begin(), active_transactions.end(), record->transaction_id), active_transactions.end());
                        transaction_changes.erase(record->transaction_id);
                        break;

                    case LogType::CHECKPOINT:
                        info.last_checkpoint_lsn = record->lsn;
                        break;

                   case LogType::PAGE_UPDATE:
                   case LogType::MESSAGE_BUFFER:
                   case LogType::TREE_OPERATION:
                        transaction_changes[record->transaction_id].push_back(record);
                        break;
                }

                info.total_records_recovered++;
            }

            // Redo all updates from committed transactions
            for (auto& [txn_id, records] : transaction_changes) {
                if (std::find(active_transactions.begin(), active_transactions.end(), txn_id) == active_transactions.end()) {
                    // Transaction was committed, redo updates
                    for (const LogRecord* record : records) {
                        apply_log_record(record, db_file, buffer_pool);
                        info.pages_restored++;
                    }
                }
            }

            info.active_transactions = active_transactions;
            info.transaction_changes = transaction_changes;

            std::cout << "WAL: Recovery completed. Records: " << info.total_records_recovered << ", Pages restored: " << info.pages_restored << ", Active transactions: " << active_transactions.size() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "WAL: Recovery error: " << e.what() << std::endl;
        }

        return info;
    }

    void WriteAheadLog::recover_transaction(uint64_t transaction_id, DatabaseFile* db_file, BufferPool* buffer_pool) {
        if (!log_file.is_open()) return;

        log_file.seekg(0, std::ios::beg);

        std::vector<const LogRecord*> transaction_records;

        try{
            while (log_file.peek() != EOF) {
                LogRecord header;
                if (!read_record(&header)) break;

                std::vector<char> full_record(LogRecord::size_with_data(header.data_length));
                log_file.seekg(-static_cast<std::streamoff>(sizeof(LogRecord)), std::ios::cur);
                log_file.read(full_record.data(), full_record.size());

                if (!log_file) break;

                LogRecord* record = reinterpret_cast<LogRecord*>(full_record.data());

                if (record->transaction_id == transaction_id && validate_record(record)) {
                    transaction_records.push_back(record);
                }
            }

            // Apply records in order
            for (const LogRecord* record : transaction_records) {
                apply_log_record(record, db_file, buffer_pool);
            }

            std::cout << "WAL: Recovered transaction " << transaction_id << " with " << transaction_records.size() << " records " << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "WAL: Transaction recovery error: " << e.what() << std::endl;
        }
    }

    void WriteAheadLog::flush() {
        if (log_file.is_open()) {
            log_file.flush();
            flushed_lsn = next_lsn;
        }
    }

    void WriteAheadLog::truncate(uint64_t up_to_lsn) {
        // Early return if nothing to truncate
        if (up_to_lsn == 0) return;

        // Flush and close current file properly
        if (log_file.is_open()) {
            log_file.flush();
            log_file.close();
        }

        // CReate temporary file
        std::string temp_filename = filename + ".tmp";
        std::ofstream temp_file(temp_filename, std::ios::binary);
        if (!temp_file) {
            // Reopen original file before throwing
            ensure_file_open();
            throw std::runtime_error("Failed to create temporary log file");
        }

        // Reopen original file for reading
        std::ifstream src_file(filename, std::ios::binary);
        if (!src_file) {
            temp_file.close();
            std::filesystem::remove(temp_filename);
            ensure_file_open();
            throw std::runtime_error("Failed to open source log file for trancation");
        }

        try {
            bool found_cut_point = false;

            while (src_file.peek() != EOF) {
                LogRecord header;
                src_file.read(reinterpret_cast<char*>(&header),sizeof(LogRecord));

                if (!src_file || src_file.gcount() != sizeof(LogRecord)) {
                    break;
                }

                // Read full record
                std::vector<char> full_record(LogRecord::size_with_data(header.data_length));
                src_file.seekg(-static_cast<std::streamoff>(sizeof(LogRecord)), std::ios::cur);
                src_file.read(full_record.data(), full_record.size());

                if (!src_file) break;

                LogRecord* record = reinterpret_cast<LogRecord*>(full_record.data());

                if (record->lsn > up_to_lsn) {
                    temp_file.write(full_record.data(), full_record.size());
                    if (!temp_file) {
                        throw std::runtime_error("Failed to write to temporary log file");
                    }
                }
            }

            src_file.close();
            temp_file.close();

            // Replace files
            std::filesystem::remove(filename);
            std::filesystem::rename(temp_filename, filename);
        
        } catch (const std::exception& e) {
            src_file.close();
            temp_file.close();
            std::filesystem::remove(temp_filename);
            ensure_file_open();
            throw;
        }

        // Reopen the truncated file
        ensure_file_open();
        std::cout << "WAL: Truncated log up to LSN " << up_to_lsn << std::endl;
    }

    /*void WriteAheadLog::truncate(uint64_t up_to_lsn) {
        // Create temporary file
        std::string temp_filename = filename + ".tmp";
        std::ofstream temp_file(temp_filename, std::ios::binary);

        if (!temp_file) {
            throw std::runtime_error("Failed to create temporary log file");
        }

        // Copy records after lsn
        log_file.seekg(0, std::ios::beg);

        try {
            while (log_file.peek() != EOF) {
                LogRecord header;
                if (!read_record(&header)) break;

                std::vector<char> full_record(LogRecord::size_with_data(header.data_length));

                log_file.seekg(-static_cast<std::streamoff>(sizeof(LogRecord)), std::ios::cur);
                log_file.read(full_record.data(), full_record.size());

                if (!log_file) break;

                LogRecord* record = reinterpret_cast<LogRecord*>(full_record.data());

                if (record->lsn > up_to_lsn) {
                    temp_file.write(full_record.data(), full_record.size());
                }
            }
        } catch (...) {
            temp_file.close();
            std::filesystem::remove(temp_filename);
            throw;
        }

        temp_file.close();
        log_file.close();

        // Replace original file 
        std::filesystem::remove(filename);
        std::filesystem::rename(temp_filename, filename);

        // Re open file
        ensure_file_open();

        std::cout << "WAL: Truncated log up to LSN " << up_to_lsn << std::endl;
    }*/

    bool WriteAheadLog::create_checkpoint() {
        try {
            log_checkpoint();
            truncate(last_checkpoint_lsn);
            return true;
        } catch (const std::exception& e) {
            std::cerr << "WAL: Checkpoint creation failed: " << e.what() << std::endl;
            return false;
        }
    }

    void WriteAheadLog::archive_log(const std::string& archive_path) {
        if (!log_file.is_open()) return;

        // Simplified implementation
        std::ifstream src(filename, std::ios::binary);
        std::ofstream dst(archive_path, std::ios::binary);

        dst << src.rdbuf();
        
        std::cout << "WAL: Archived log to " << archive_path << std::endl;
    }

    size_t WriteAheadLog::size() const {
        if (log_file.is_open()) {
            log_file.seekg(0, std::ios::end); 
            return log_file.tellg();
        }
        return 0;
    }

    void WriteAheadLog::print_stats() const {
        std::cout << "=== WriteAheadLog Statistics ===" << std::endl;
        std::cout << "File: " << filename << std::endl;
        std::cout << "Size: " << size() << " bytes" << std::endl;
        std::cout << "Total records: " << total_records << std::endl;
        std::cout << "Bytes written: " << bytes_written << std::endl;
        std::cout << "Transactions: " << transaction_count << std::endl;
        std::cout << "Last checkpoint LSN: " << last_checkpoint_lsn << std::endl;
        std::cout << "Flushed LSN: " << flushed_lsn << std::endl;
        std::cout << "Next LSN: " << next_lsn << std::endl;
        std::cout << "================================" << std::endl;
    }

    bool WriteAheadLog::verify_log_integrity() const {
        if (!log_file.is_open()) return true;

        log_file.seekg(0, std::ios::beg);

        try {
            while (log_file.peek() != EOF) {
                LogRecord header;
                if(!read_record(&header)) return false;

                if (!validate_record(&header)) {
                    return false;
                }

                // Skip data
                log_file.seekg(header.data_length, std::ios::cur);
            }
            return true;
        } catch (...) {
            return false;
        }
    }

    std::vector<uint64_t> WriteAheadLog::get_active_transactions() const {
        std::vector<uint64_t> active_txns;
        if (!log_file.is_open()) return active_txns;

        log_file.seekg(0, std::ios::beg);

        try {
            while (log_file.peek() != EOF) {
                LogRecord record;
                if (!read_record(&record)) break;

                if (record.type == LogType::TRANSACTION_BEGIN) {
                    active_txns.push_back(record.transaction_id);
                } else if (record.type == LogType::TRANSACTION_COMMIT || 
                          record.type == LogType::TRANSACTION_ROLLBACK) {
                    active_txns.erase(std::remove(active_txns.begin(), active_txns.end(), record.transaction_id), active_txns.end());
                }
            }
        } catch (...) {
            // Ignore errors in this simplified version
        }

        return active_txns;
    }

    void WriteAheadLog::write_record(const LogRecord* record) {
        ensure_file_open();

        log_file.write(reinterpret_cast<const char*>(record), LogRecord::size_with_data(record->data_length));

        if (!log_file) {
            throw std::runtime_error("Failed to write WAL record");
        }
    }

    bool WriteAheadLog::read_record(LogRecord* record) const {
        ensure_file_open();

        log_file.read(reinterpret_cast<char*>(record), sizeof(LogRecord));

        if (!log_file || log_file.gcount() != sizeof(LogRecord)) {
                return false;
        }

        // Read data if present
        if (record->data_length > 0) {
            log_file.read(record->data, record->data_length);
        }

        return static_cast<bool>(log_file);
    }

    void WriteAheadLog::ensure_file_open() const {
        if (!log_file.is_open()) {
            log_file.open(filename, std::ios::binary | std::ios::in | std::ios::out | std::ios::app);
            if (!log_file.is_open()) {
                throw std::runtime_error("Failed to open WAL file: " + filename);
            }
        }
    }

    void WriteAheadLog::update_statistics(size_t record_size) { 
        total_records++;
        bytes_written += record_size;
    }

    bool WriteAheadLog::validate_record(const LogRecord* record) const {
        if (record->lsn == 0) return false;
        if (record->type < LogType::PAGE_UPDATE || record->type > LogType::TREE_OPERATION) return false;
        if (record->data_length > 1024 * 1024) return false;

        return true;
    }

    void WriteAheadLog::apply_log_record(const LogRecord* record, DatabaseFile* db_file, BufferPool* buffer_pool) {
        try {
            switch (record->type) {
                case LogType::PAGE_UPDATE: {
                         Page page;
                         db_file->read_page(record->page_id, &page);

                         // Apply the new data (stored in second half of data)
                         std::memcpy(page.data, record->data + record->data_length / 2, record->data_length / 2);
                         page.update_checksum();
                         db_file->write_page(record->page_id, &page);
                         break;
                 }
                 case LogType::MESSAGE_BUFFER:
                         // Message buffer records are applied during normal message processing
                         // no direct update done here
                         break;
                 case LogType::TREE_OPERATION:
                         // Tree operations will be handled by the B+Tree during recovery
                         break;
                 default:
                         // Other record types do not require direct page updates
                         break;
            }
        } catch (const std::exception& e) {
            std::cerr << "WAL: Failed to apply log record " << record->lsn << ": " << e.what() << std::endl;
        }
    }
}
