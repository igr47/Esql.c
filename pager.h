#pragma once
#ifndef PAGER_H
#define PAGER_H

#include "storagemanager.h"
#include "database_format.h"
#include <filesystem>
#include <fstream>
#include <atomic>
#include <shared_mutex>

namespace fs = std::filesystem;

class RobustPager {
private:
    std::fstream file;
    std::string filename;
    DatabaseFileHeader header;
    std::atomic<bool> recovery_mode{false};
    mutable std::shared_mutex file_mutex;
    std::atomic<bool> initialization_in_progress{false};

public:
    RobustPager() = default;
    ~RobustPager() { close(); }

    bool open(const std::string& fname, bool create_if_missing = true);
    void close();

    // Safe read/write operations with validation
    void read_page(uint32_t page_id, Node* node);
    void write_page(uint32_t page_id, const Node* node);

    // Recovery operations
    bool needs_recovery() const { return recovery_mode; }
    bool recover_database();
    bool validate_database_integrity();

    // Page allocation
    uint32_t allocate_page();
    void release_page(uint32_t page_id);

    // Table directory operations
    bool register_table(const std::string& table_name, uint32_t table_id, uint32_t root_page_id);
    bool unregister_table(const std::string& table_name);
    TableDirectory read_table_directory();
    bool update_table_directory(const TableDirectory& directory);
    TableDirectoryEntry* find_table_entry(const std::string& table_name);
    uint32_t get_next_table_id();

    // Utility functions
    bool is_open() const { return file.is_open(); }
    std::string get_filename() const { return filename; }
    void flush() { 
        std::shared_lock lock(file_mutex);
        if (file.is_open()) file.flush(); 
    }

private:
    void initialize_new_database();
    bool read_header();
    bool write_header();
    void write_page_unsafe(uint32_t page_id, const Node* node);
    bool validate_page(uint32_t page_id, const Node* node);
    void recover_corrupted_page(uint32_t page_id);
    void read_page_recovery(uint32_t page_id, Node* node);
    void recover_schema();
    void validate_all_pages();
    void read_page_unsafe(uint32_t page_id, Node* node);
    void read_page_recovery_unsafe(uint32_t page_id, Node* node);
    
    // Safe initialization without recursion
    void initialize_new_database_direct();

    // Table directory helpers
    bool write_table_directory_unsafe(const TableDirectory& directory);
    TableDirectory read_table_directory_unsafe();
};

#endif
