#pragma once
#ifndef DATABASE_FORMAT_H
#define DATABASE_FORMAT_H

#include <cstdint>
#include <string>
#include <vector>
#include <cstring>

namespace DatabaseFormat {
    const uint32_t TABLE_PAGES_PER_TABLE = 10000;
    const uint32_t TABLE_DATA_REGION_SIZE = 1024 * 1024 * 10; // 10MB per table
    const uint32_t SCHEMA_PAGE_ID = 1;
    const uint32_t DATA_START_PAGE = 2;
    const uint32_t MAX_TABLES = 100;
}

// Database file structure with table isolation
struct DatabaseFileHeader {
    char magic[8] = {'E', 'S', 'Q', 'L', 'D', 'B', '2', '1'};
    uint32_t version = 3;
    uint32_t page_size = 16384; // 16KB
    uint64_t schema_root = 1; // Schema always at page 1
    uint64_t data_root =2; // Data starts at page 2
    uint64_t free_list_head = 0;
    uint32_t header_check_sum = 0;
    uint32_t flags = 0;
    uint64_t created_timestamp = 0;
    uint64_t modified_timestamp = 0;
    char database_name[64] = {0};
    uint32_t reserved[96] = {0}; // Padding to 512 bytes
    //uint32_t reserved[4] = {0};

    // Calculate header checksum (excludes the checksum field itself)
    uint32_t calculate_checksum() const {
        const uint8_t* data = reinterpret_cast<const uint8_t*>(this) + sizeof(header_check_sum);
        size_t length = sizeof(DatabaseFileHeader) - sizeof(header_check_sum);
        uint32_t sum = 0x1234567;
        for (size_t i = 0; i <length; ++i) {
            sum = (sum << 3) ^ (sum >> 27) ^ data[i];
        }
        return sum;
    }

    bool validate() const{
        // Check magic number
        if (memcmp(magic, "ESQLDB21", 8) != 0) {
            std::cerr << "Invalid magic number" << std::endl;
            return false;
        }

        // Check page size is reasonable
        if (page_size != 16384) {
            std::cerr << "Invalid page size: " << page_size << std::endl;
            return false;
        }

        // Verify checksum
        uint32_t calculated = calculate_checksum();
        if (header_check_sum != calculated) {
            std::cerr << "Checksu, mismatch: Stored=" << header_check_sum << ", calculated=" << calculated << std::endl;
            return false;
        }
        //return memcmp(magic, "ESQLDB21", 8) == 0 && header_check_sum == calculate_checksum() && page_size == 16384;
        return true;
    }
};

static_assert(sizeof(DatabaseFileHeader) == 512, "Header must be 512 bytes");

struct TableDirectoryEntry {
    char table_name[64] = {0};
    uint32_t table_id = 0;
    uint32_t root_page_id = 0;
    uint64_t data_start_offset = 0;
    uint64_t data_end_offset = 0;
    uint32_t schema_checksum = 0;
    uint32_t flags = 0;

    bool validate() const {
        return table_id > 0 && root_page_id >= DatabaseFormat::DATA_START_PAGE;
    }
};

struct TableDirectory {
    uint32_t num_tables = 0;
    uint32_t next_table_id = 1;
    TableDirectoryEntry entries[DatabaseFormat::MAX_TABLES];

    TableDirectoryEntry* find_table(const std::string& table_name) {
        for (uint32_t i = 0; i< num_tables; ++i) {
            if (strcmp(entries[i].table_name, table_name.c_str()) == 0) {
                return &entries[i];
            }
        }

        return nullptr;
    }

    TableDirectoryEntry* find_table_by_id(uint32_t table_id) {
        for (uint32_t i = 0; i < num_tables; ++i) {
            if (entries[i].table_id == table_id) {
                return &entries[i];
            }
        }
        return nullptr;
    }

    bool add_table(const std::string& table_name, uint32_t table_id, uint32_t root_page_id) {
        if (num_tables >= DatabaseFormat::MAX_TABLES) return false;

        TableDirectoryEntry& entry = entries[num_tables++];
        strncpy(entry.table_name, table_name.c_str(), sizeof(entry.table_name) - 1);
        entry.table_name[sizeof(entry.table_name) - 1] = '\0';
        entry.table_id = table_id;
        entry.root_page_id = root_page_id;

        // Calculate data region for this table
        entry.data_start_offset = table_id * DatabaseFormat::TABLE_DATA_REGION_SIZE;
        entry.data_end_offset = entry.data_start_offset + DatabaseFormat::TABLE_DATA_REGION_SIZE;

        return true;
    }

    bool remove_table(const std::string& table_name) {
        for (uint32_t i = 0; i < num_tables; ++i) {
            if (strcmp(entries[i].table_name, table_name.c_str()) == 0) {
                // Shift remaining entries
                for (uint32_t j = i; j < num_tables - 1; ++j) {
                    entries[j] = entries[j + 1];
                }
                num_tables--;
                return true;
            }
        }
        return false;
    }
};

static_assert(sizeof(TableDirectory) <= 16384, "TableDirectory must fit in one page");

struct TablePageRange {
    uint32_t start_page;
    uint32_t end_page;
    uint32_t next_page;
};

/*namespace DatabaseFormat {
    const uint32_t TABLE_PAGES_PER_TABLE = 10000;
    const uint32_t TABLE_DATA_REGION_SIZE = 1024 * 1024 * 10;
    
    inline uint32_t calculate_checksum(const void* data, size_t length) {
        // Simple checksum implementation
        const uint8_t* bytes = static_cast<const uint8_t*>(data);
        uint32_t sum = 0;
        for (size_t i = 0; i < length; ++i) {
            sum = (sum << 3) ^ bytes[i];
        }
        return sum;
    }
    
    
    inline uint32_t table_id_to_start_page(uint32_t table_id) {
        // Each table gets a dedicated range starting from page 2
        return 2 + (table_id * TABLE_PAGES_PER_TABLE);
    }

    inline bool validate_table_access(uint32_t table_id, uint32_t page_id) {
        uint32_t start_page = table_id_to_start_page(table_id);
        return page_id >= start_page && page_id < start_page + TABLE_PAGES_PER_TABLE;
    }

}*/



#endif
