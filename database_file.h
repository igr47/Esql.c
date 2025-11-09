#pragma once
#ifndef DATABASE_FILE_H
#define DATABASE_FILE_H

#include "common_types.h"
#include "database_schema.h"
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace fractal {

class DatabaseFile {
private:
    struct TableDataBlockInfo {
        uint64_t start_offset;
        uint64_t end_offset;
        uint64_t current_offset;
        uint32_t block_size;
        uint64_t next_free_block;

        TableDataBlockInfo() : start_offset(0), end_offset(0), current_offset(0), block_size(4096), next_free_block(0) {}
        TableDataBlockInfo(uint64_t start, uint64_t end, uint32_t blk_size = 4096) 
            : start_offset(start), end_offset(end), current_offset(start), block_size(blk_size), next_free_block(start) {}
        
        uint64_t allocate_block(uint32_t size) {
            uint64_t allocated = current_offset;
            current_offset += ((size + block_size - 1) / block_size) * block_size;
            if (current_offset > end_offset) {
                throw std::runtime_error("Data block region full for table");
            }
            return allocated;
        }
        
        bool can_allocate(uint32_t size) const {
            return (current_offset + ((size + block_size - 1) / block_size) * block_size) <= end_offset;
        }
    };

    struct TableRange {
        uint32_t start_page;
        uint32_t end_page;
        uint32_t original_table_id;

        TableRange() : start_page(0), end_page(0), original_table_id(0) {}
        TableRange(uint32_t start,uint32_t end, uint32_t table_id = 0) : start_page(start), end_page(end), original_table_id(table_id) {}

        size_t size() const  {return end_page - start_page + 1; }
    };

    std::vector<TableRange> available_table_ranges;
    std::unordered_map<uint32_t, TableRange> table_id_to_range;

    std::string filename;
    mutable std::fstream file;
    bool destroyed = false;
    
    DatabaseHeader db_header;
    TableDirectoryPage table_directory;
    
    std::unordered_map<uint32_t, TableDataBlockInfo> table_data_blocks;
    std::unordered_map<std::string, uint32_t> table_name_to_id;
    
    std::vector<uint32_t> free_pages;
    uint32_t next_available_page;
    
    uint64_t read_count{0};
    uint64_t write_count{0};
    uint64_t sync_count{0};

public:
    explicit DatabaseFile(const std::string& filename);
    ~DatabaseFile();

    // File operations
    void create();
    void open();
    void close();
    void sync();
    void remove();

    // Page operations
    void read_page(uint32_t page_id, Page* page);
    void write_page(uint32_t page_id, const Page* page);
    uint32_t allocate_page(PageType type);
    void deallocate_page(uint32_t page_id);
    void free_page(uint32_t page_id);
    
    // Data block operations
    uint64_t allocate_data_block(uint32_t table_id, uint32_t size);
    void read_data_block(uint64_t offset, char* data, uint32_t size);
    void write_data_block(uint64_t offset, const char* data, uint32_t size);
    void extend_data_region(uint32_t table_id, uint64_t additional_size);
    uint32_t allocate_new_table_range(uint32_t table_id);
    uint32_t allocate_table_page_range(uint32_t table_id);

    // Table management
    uint32_t create_table(const std::string& table_name);
    void drop_table(const std::string& table_name);
    uint32_t get_table_id(const std::string& table_name) const;
    uint32_t get_table_root_page(const std::string& table_name) const;
    void set_table_root_page(const std::string& table_name, uint32_t root_page_id);
    std::vector<std::string> get_table_names() const;
    std::vector<DatabaseSchema::Table> get_all_tables() const;
    void defragment();
    //uint32_t find_available_low_page();

    
    // Metadata operations
    void read_header();
    void write_header();
    void read_table_directory();
    void write_table_directory();
    
    // Space management
    uint32_t get_total_pages() const { return db_header.total_pages; }
    uint32_t get_free_page_count() const { return static_cast<uint32_t>(free_pages.size()); }
    uint64_t get_file_size() const;
    
    // Statistics
    uint64_t get_read_count() const { return read_count; }
    uint64_t get_write_count() const { return write_count; }
    uint64_t get_sync_count() const { return sync_count; }
    void print_stats() const;
    
    // Utility methods
    bool exists() const;
    bool is_open() const { return file.is_open(); }
    const std::string& get_filename() const { return filename; }
    
    // Page range management for tables
    uint32_t get_table_start_page(uint32_t table_id) const;
    uint32_t get_table_end_page(uint32_t table_id) const;
    bool is_page_in_table_range(uint32_t page_id, uint32_t table_id) const;
    void debug_page_access(uint32_t page_id);

private:
    void initialize_free_pages();
    void load_free_page_list();
    void save_free_page_list();
    void extend_file(uint32_t additional_pages = 1000);
    void validate_page_id(uint32_t page_id) const;
    void ensure_file_open() const;
    off_t calculate_file_offset(uint32_t page_id) const;
    
    void initialize_table_data_blocks();
    uint64_t calculate_data_block_offset(uint32_t table_id) const;
};

} // namespace fractal

#endif // DATABASE_FILE_H
