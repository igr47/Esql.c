#pragma once
#ifndef COMMON_TYPES_H
#define COMMON_TYPES_H

#include <cstdint>
#include <string>

namespace fractal {

    // Constants
    constexpr size_t PAGE_SIZE = 16384; // 16 KB
    constexpr size_t BUFFER_POOL_SIZE = 10000;
    constexpr size_t BPTREE_ORDER = 256;
    constexpr size_t MAX_MESSAGES = 50;
    constexpr uint32_t DATABASE_MAGIC = 0x44424631; // "DBF1"
    constexpr uint32_t TABLE_PAGES_START = 1000;
    constexpr uint32_t USER_TABLE_PAGES_START = 2000; // First table starts at 2000
    constexpr uint32_t TABLE_PAGE_RANGE_SIZE = 1000;
    constexpr uint32_t MAX_SCHEMA_PAGES = 100;
    constexpr uint32_t SCHEMA_PAGE_START = 1000;

    

    // Page Types
    enum class PageType : uint8_t {
        METADATA = 0,
        FREE_PAGE = 1,
        TABLE_HEADER = 2,
        LEAF_NODE = 3,
        INTERNAL_NODE = 4,
        DATA_PAGE = 5
    };

    // Message types for fractal tree
    enum class MessageType : uint8_t {
        INSERT = 1,
        UPDATE = 2,
        DELETE = 3
    };

    // Message structure for buffered updates
    struct Message {
        MessageType type;
        int64_t key;
        uint64_t value_offset;
        uint32_t value_length;
        uint64_t version;
        uint64_t timestamp;

        static constexpr size_t SIZE = sizeof(MessageType) + sizeof(int64_t) + sizeof(uint64_t) + sizeof(uint32_t) + sizeof(uint64_t) + sizeof(uint64_t);
    };

    // Key-Value pair for leaf nodes
    struct KeyValue {
        int64_t key;
        uint64_t value_offset;  
        uint32_t value_length;

        static constexpr size_t SIZE = sizeof(int64_t) + sizeof(uint64_t) + sizeof(uint32_t);
    };

    // Page header
    struct PageHeader {
        uint32_t checksum;
        PageType type;
        uint32_t page_id;
        uint32_t parent_id;
        uint32_t lsn;
        uint16_t free_space;
        uint16_t flags;
        uint16_t key_count;
        uint16_t message_count;
        uint32_t next_page;
        uint32_t prev_page;
        uint64_t timestamp;
        uint32_t database_id;

        static constexpr size_t SIZE = 52; // Sum of all fields
    };


    // Complete page structure 
    struct Page {
        PageHeader header;
        char data[PAGE_SIZE - PageHeader::SIZE];

        void initialize(uint32_t page_id, PageType type, uint32_t db_id);
        void update_checksum();
        bool validate_checksum() const;

        // Space management
        bool can_add_message() const;
        bool can_add_key_value() const;
        bool is_full() const;
        bool is_underfull() const;
        size_t get_used_space() const;
        size_t get_available_space() const;
    };

    // Database file header
    struct DatabaseHeader {
        uint32_t magic;
        uint32_t version;
        uint32_t page_size;
        uint32_t total_pages;
        uint32_t first_free_page;
        uint32_t table_count;
        uint64_t created_timestamp;
        uint64_t last_checkpoint;
        char database_name[64];
        char reserved[PAGE_SIZE - 104]; // PAGE_SIZE - 104 bytes for header
    };

    struct TableDirectoryEntry {
        char table_name[64];
        uint32_t table_id;
        uint32_t root_page_id;
        uint64_t data_start_offset;
        uint64_t data_end_offset;
        uint64_t record_count;
        uint64_t created_timestamp;
        uint64_t last_modified;

        static constexpr size_t SIZE = 64 + sizeof(uint32_t) * 2 + sizeof(uint64_t) * 5; 
    };

    // Table directory page (page 1) 
    struct TableDirectoryPage {
        uint32_t num_tables;
        uint32_t next_table_id;
        TableDirectoryEntry entries[100]; // Support 100 tables per database

        static constexpr size_t SIZE = sizeof(uint32_t) * 2 + 100 * TableDirectoryEntry::SIZE; 
    };
} // namespace fractal

#endif
