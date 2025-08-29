#ifndef FRACTAL_BPTREE_H
#define FRACTAL_BPTREE_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <map>
#include <mutex>
#include <atomic>
#include <zstd.h>
#include <algorithm>
#include <queue>
#include <thread>
#include <optional>
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__x86_64__) || defined(_M_x64)
#include <immintrin.h>
#endif


#if defined(__ANDROID__) || defined(__arm__) || defined(__aarch64__)
#define ARM_PLATFORM 1
#else
#define ARM_PLATFORM 0
#endif

#ifdef __linux__
#include <sys/auxv.h>
#include <asm/hwcap.h>
#ifdef HAS_NUMA
#include <numa.h>
#endif
#endif

// Rename to avoid conflict with system PAGE_SIZE macro
const size_t BPTREE_PAGE_SIZE = 16384; // 16KB, matching MySQL InnoDB
const size_t BPTREE_ORDER = 1000; // Approx. 1000 keys per node
const size_t KEY_SIZE = 8; // int64_t keys
const size_t MAX_KEY_PREFIX = 4; // For prefix compression
const size_t MAX_MESSAGES = 100; // Max messages per node
const size_t DATA_BLOCK_SIZE = 4096; // Size of data blocks for variable-length values

// Page types
enum class PageType : uint8_t {
    INTERNAL, LEAF, DATA_BLOCK, METADATA
};

// Message types for Fractal Tree
enum class MessageType : uint8_t {
    INSERT, UPDATE, DELETE
};

// Message structure for buffered updates
struct Message {
    MessageType type;
    int64_t key;
    uint64_t value_offset; // Offset to variable-length value in data block
    uint32_t value_length;
    uint64_t version; // For MVCC - ADDED THIS FIELD
};

// Key-value pair for leaf nodes
struct KeyValue {
    int64_t key;
    uint64_t value_offset; // Offset to variable-length value
    uint32_t value_length;
};

// Page header
struct PageHeader {
    PageType type;
    uint32_t page_id;
    uint32_t parent_page_id;
    uint32_t num_keys;
    uint32_t num_messages; // Number of buffered messages
    uint32_t next_page_id; // For leaf node linking
    uint32_t prev_page_id;
    uint64_t version; // For MVCC
};

// Node structure
struct Node {
    PageHeader header;
    char data[BPTREE_PAGE_SIZE - sizeof(PageHeader)]; // Keys, pointers/values, messages
};

// Pager class for disk I/O
class Pager {
private:
    std::fstream file;
    std::string filename;

public:
    Pager(const std::string& fname);
    ~Pager();
    void read_page(uint32_t page_id, Node* node);
    void write_page(uint32_t page_id, const Node* node);
    uint32_t allocate_page();
    uint64_t write_data_block(const std::string& data);
    std::string read_data_block(uint64_t offset, uint32_t length);
    void test_zstd();
};

// Write-Ahead Log for durability
class WriteAheadLog {
private:
    std::fstream log_file;
    std::mutex log_mutex;
    std::string filename;

public:
    WriteAheadLog(const std::string& fname);
    ~WriteAheadLog();
    void log_operation(const std::string& op, uint32_t page_id, const Node* node);
    void checkpoint(Pager& pager, uint32_t metadata_page_id);
    void recover(Pager& pager, uint32_t& metadata_page_id);
};

// Buffer Pool for caching pages
class BufferPool {
private:
    std::map<uint32_t, Node> cache;
    std::mutex cache_mutex;
    size_t max_pages;

public:
    BufferPool(size_t max_size);
    Node* get_page(Pager& pager, uint32_t page_id);
    void write_page(Pager& pager, WriteAheadLog& wal, uint32_t page_id, const Node* node);
    void evict_page(uint32_t page_id);
    void flush_all();
    size_t cache_size() const { return cache.size(); } // ADDED PUBLIC ACCESSOR
};

// Fractal B+ Tree Storage Engine
class FractalBPlusTree {
private:
    Pager& pager;
    WriteAheadLog& wal;
    BufferPool& buffer_pool;
    std::atomic<uint32_t> root_page_id;
    std::map<uint64_t, std::vector<Node>> versioned_nodes; // For MVCC
    std::mutex version_mutex;
    std::string table_name;
    std::map<std::string, uint32_t> secondary_indexes; // Secondary index name to root_page_id
    void compress_keys_neon(char* dest, const int64_t* keys, uint32_t num_keys);
    void decompress_keys_neon(int64_t* keys, const char* src, uint32_t num_keys);
    void compress_node_neon(Node* node);
    void decompress_node_neon(Node* node);

    // CPU feature detection
    bool has_neon_support() const;
    bool neon_supported;

    // Choose the best implementation based on CPU capabilities
    void compress_keys_optimized(char* dest, const int64_t* keys, uint32_t num_keys);
    void decompress_keys_optimized(int64_t* keys, const char* src, uint32_t num_keys);
    void compress_node_optimized(Node* node);
    void decompress_node_optimized(Node* node);

    // Memory management
    size_t memory_usage();
    bool memory_usage_high();

    // Node operations
    Node* get_node(uint32_t page_id);
    void write_node(uint32_t page_id, const Node* node, uint64_t transaction_id);
    void compress_node(Node* node);
    void decompress_node(Node* node);
    void compress_keys(char* dest, const int64_t* keys, uint32_t num_keys);
    void decompress_keys(int64_t* keys, const char* src, uint32_t num_keys);

    // Tree operations
    bool is_node_full(const Node* node);
    bool is_node_underfull(const Node* node);
    void compact_messages(Node* node);
    void split_node(uint32_t page_id, Node* node, uint32_t parent_page_id, uint64_t transaction_id);
    void merge_nodes(uint32_t page_id, Node* node, uint32_t parent_page_id, uint64_t transaction_id);
    void flush_messages(Node* node, uint32_t page_id, uint64_t transaction_id);
    void adaptive_flush(Node* node, uint32_t page_id, uint64_t transaction_id);
    uint32_t find_child_index(const Node* node, int64_t key);

    // Secondary index operations
    void update_secondary_index(const std::string& index_name, int64_t primary_key, 
                               const std::string& secondary_key, uint64_t transaction_id, 
                               MessageType op_type);

    // Transactional memory fallback
    bool begin_transaction();
    void end_transaction();
    void transaction_fallback();

public:
    FractalBPlusTree(Pager& p, WriteAheadLog& w, BufferPool& bp, 
                    const std::string& tname, uint32_t root_id);
    
    // Basic operations
    void create();
    void drop();
    
    // CRUD operations
    void insert(int64_t key, const std::string& value, uint64_t transaction_id, 
               const std::map<std::string, std::string>& secondary_keys = {});
    void update(int64_t key, const std::string& value, uint64_t transaction_id, 
               const std::map<std::string, std::string>& secondary_keys = {});
    void remove(int64_t key, uint64_t transaction_id, 
               const std::map<std::string, std::string>& secondary_keys = {});
    
    // Query operations
    std::string select(int64_t key, uint64_t transaction_id);
    std::vector<std::pair<int64_t, std::string>> select_range(int64_t start_key, 
                                                             int64_t end_key, 
                                                             uint64_t transaction_id);
    
    // Index operations
    void create_secondary_index(const std::string& index_name);
    std::vector<int64_t> select_by_secondary(const std::string& index_name, 
                                            const std::string& secondary_key, 
                                            uint64_t transaction_id);
    
    // Bulk operations
    void bulk_load(std::vector<std::pair<int64_t, std::string>>& data, uint64_t transaction_id);
    
    // Maintenance
    void checkpoint();
    
    // Utility methods
    uint32_t get_root_page_id() const { return root_page_id.load(); }
};

#endif // FRACTAL_BPTREE_H
