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
#include <unordered_set>
#include <unordered_map>
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
//#include <asm/hwcap.h>
#ifdef HAS_NUMA
#include <numa.h>
#endif
#endif

class MultiFilePager;

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
    uint32_t database_id; //Database owner ID
    uint32_t checksum; //For data integrity
    uint64_t timestamp; //For recovery
};

struct FreePageHeader {
	uint32_t magic_number;  //0x46524545 = "Free"
	uint32_t next_free_page;
	uint32_t prev_free_page;
	uint32_t free_region_size; //Number of consecutive free pages
};

struct DatabasePageRange {
	uint32_t database_id;
	std::string database_name;
	uint32_t start_page;
	uint32_t end_page;
	uint32_t current_page;
	std::vector<uint32_t> free_pages;
};

// Node structure
struct Node {
    PageHeader header;
    char data[BPTREE_PAGE_SIZE - sizeof(PageHeader)]; // Keys, pointers/values, messages
    // Helper methods
    void set_database_id(uint32_t db_id) {header.database_id = db_id; }
    uint32_t get_database_id() const {return header.database_id; }
    void update_checksum() { header.checksum = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count(); }
};

// Pager class for disk I/O
class Pager {
private:
    mutable std::fstream file;
    std::string filename;
    std::atomic<uint32_t> next_high_page{1000};
    std::mutex page_mutex;

    //Free page management
    uint32_t free_list_head{0};
    uint32_t free_list_tail{0};
    std::unordered_set<uint32_t> free_pages;

    //Database page ranges
    std::unordered_map<std::string, DatabasePageRange> database_ranges;
    std::unordered_map<uint32_t,std::string> page_to_database_map;
    std::atomic<uint32_t> next_database_id{1};

    //Recovery information
    uint32_t last_checkpoint_pge{0};

public:
    Pager(const std::string& fname);
    ~Pager();
    void read_page(uint32_t page_id, Node* node) const;
    void write_page(uint32_t page_id, const Node* node);

    // Enhanced page allocation
    uint32_t allocate_page();
    uint32_t get_database_id(const std::string& db_name) const;
    uint32_t allocate_high_page();
    uint32_t allocate_page_for_database(const std::string& dbName);
    void update_page_ownership(uint32_t page_id, const std::string& db_name);

    // Free page management
    void release_page(uint32_t page_id);
    void release_pages(const std::vector<uint32_t>& page_ids);
    void defragment_free_space();
    
    //Database Isolation
    void register_database(const std::string& db_name, uint32_t start_page = 0);
    void unregister_database(const std::string& db_name);
    bool validate_page_ownership(uint32_t page_id, const std::string& expected_db) const;
    std::vector<uint32_t> get_database_pages(const std::string& dbName) const;

    //Data block operations
    uint64_t write_data_block(const std::string& data);
    std::string read_data_block(uint64_t offset, uint32_t length);

    //Recovery and maintanance
    void initialize_free_page_management();
    void rebuild_free_list();
    void checkpoint();
    void recover();
    std::string get_database_stats() const;

    //Utility
    void test_zstd();
    uint32_t get_total_pages() const;
    uint32_t get_free_pages_count() const;

private:
    void initialize_page_range(const std::string& db_name);
    uint32_t allocate_from_free_list();
    void add_to_free_list(uint32_t page_id);
    void remove_from_free_list(uint32_t page_id);
    bool is_page_free(uint32_t page_id) const;
    uint32_t calculate_checksum(const Node* node) const;
    bool verify_checksum(const Node* node) const;
    void write_free_page_header(uint32_t page_id, uint32_t next_free, uint32_t prev_free, uint32_t region_size = 1);
    FreePageHeader read_free_page_header(uint32_t page_id);
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
    void checkpoint(MultiFilePager& multi_pager,const std::string& db_name, uint32_t metadata_page_id);
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
    Node* get_page(MultiFilePager& pager, const std::string& db_name, uint32_t page_id);
    void write_page(MultiFilePager& multi_pager, const std::string& db_name, WriteAheadLog& wal, uint32_t page_id, const Node* node);
    void evict_page(uint32_t page_id);
    void flush_all();
    size_t cache_size() const { return cache.size(); } // ADDED PUBLIC ACCESSOR
};

// Fractal B+ Tree Storage Engine
class FractalBPlusTree {
private:
    //Pager& pager;
    MultiFilePager& multi_pager;
    WriteAheadLog& wal;
    BufferPool& buffer_pool;
    std::string database_name;
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
    void write_node(const std::string& db_name, uint32_t page_id, const Node* node, uint64_t transaction_id);
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
    size_t calculate_actual_data_size(uint64_t transaction_id);

public:
    FractalBPlusTree(MultiFilePager& mp, WriteAheadLog& w, BufferPool& bp, 
                    const std::string& db_name, const std::string& tname, uint32_t root_id);
    
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
