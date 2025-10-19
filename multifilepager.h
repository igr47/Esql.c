#pragma once
#ifndef MULTI_FILE_PAGER_H
#define MULTI_FILE_PAGER_H

#include "storagemanager.h"
#include <filesystem>
#include <map>
#include <memory>
#include <atomic>
#include <shared_mutex>

namespace fs = std::filesystem;

const uint32_t FREE_PAGE_MAGIC = 0x46524545;

class DatabaseFile {
private:
    std::string filename;
    mutable std::fstream file;
    std::atomic<uint32_t> next_page_id{0};
    std::atomic<uint32_t> free_list_head{0};
    
public:
    DatabaseFile(const std::string& fname);
    ~DatabaseFile();
    
    bool open(bool create_if_missing = false);
    void close();
    void read_page(uint32_t page_id, Node* node) const;
    void write_page(uint32_t page_id, const Node* node);
    uint32_t allocate_page();
    void flush();
    size_t get_file_size() const;
    bool is_open() const { return file.is_open(); }
    std::string get_filename() const { return filename; }
    
    // Free page management for individual database files
    void initialize_free_page_management();
    uint32_t allocate_from_free_list();
    void release_page(uint32_t page_id);
    std::fstream& get_file_stream() { return file; }
    
private:
    void write_free_page_header(uint32_t page_id, uint32_t next_free);
    FreePageHeader read_free_page_header(uint32_t page_id);
};

class MultiFilePager {
private:
    std::string base_path;
    std::map<std::string, std::unique_ptr<DatabaseFile>> database_files;
    std::string registry_file;
    
    // Atomic operations for thread-safe access
    std::atomic<bool> registry_loaded{false};
    
    // Helper methods
    bool create_database_file_internal(const std::string& db_name);
    void write_registry_internal();
    void read_registry_internal();
    bool validate_page_type(const std::string& db_name, uint32_t page_id, PageType expected_type);
    
public:
    MultiFilePager(const std::string& base_path = "databases/");
    ~MultiFilePager();
    
    // Database file management
    DatabaseFile* get_or_create_database_file(const std::string& db_name);
    DatabaseFile* get_database_file(const std::string& db_name);
    bool create_database_file(const std::string& db_name);
    void close_database_file(const std::string& db_name);
    bool database_exists(const std::string& db_name) const;
    std::vector<std::string> list_databases() const;
    
    // Page operations delegate to specific database files
    void read_page(const std::string& db_name, uint32_t page_id, Node* node);
    void write_page(const std::string& db_name, uint32_t page_id, const Node* node);
    uint32_t allocate_page(const std::string& db_name);
    void release_page(const std::string& db_name, uint32_t page_id);
    
    // Maintenance
    void flush_all();
    void checkpoint_all();
  
    uint64_t write_data_block(const std::string& db_name, const std::string& data);
    std::string read_data_block(const std::string& db_name, uint64_t offset, uint32_t length);
};

#endif // MULTI_FILE_PAGER_H
