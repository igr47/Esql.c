#ifndef STORAGE_MANAGER_H
#define STORAGE_MANAGER_H

#include <cstdint>
#include <vector>
#include <memory>
#include <unordered_map>
#include <string>
#include <list>
#include <fstream>
#include <mutex>
#include <array>

inline constexpr uint32_t DB_PAGE_SIZE = 1024;

class Page {
public:
    uint32_t page_id;
    bool dirty;
    std::array<uint8_t, DB_PAGE_SIZE> data;
    Page(uint32_t id);
};

class Pager {
private:
    std::string filename;
    std::fstream file;
    std::unordered_map<uint32_t, std::shared_ptr<Page>> page_cache;
    std::mutex cache_mutex;
    uint32_t next_page_id = 0;
public:
    explicit Pager(const std::string& db_file);
    ~Pager();

    std::shared_ptr<Page> get_page(uint32_t page_id);
    uint32_t allocate_page();
    void mark_dirty(uint32_t page_id);
    void flush_all();
public:
    bool read_page_from_disk(uint32_t page_id, std::array<uint8_t, DB_PAGE_SIZE>& buffer);
    bool write_page_to_disk(uint32_t page_id, const std::array<uint8_t, DB_PAGE_SIZE>& buffer);
    void load_metadata();
    void save_metadata();
};

class BufferPool {
private:
    Pager& pager;
    size_t capacity;
    std::unordered_map<uint32_t, std::shared_ptr<Page>> pool;
    std::list<uint32_t> lru_list;
    std::unordered_map<uint32_t, std::list<uint32_t>::iterator> lru_map;
    std::mutex mutex;
public:
    BufferPool(Pager& pager, size_t capacity = 1000);
    std::shared_ptr<Page> fetch_page(uint32_t page_id);
    void flush_all();
};

class WriteAheadLog {
private:
    std::string log_file_path;
    std::fstream log_file;
    std::mutex mutex;
public:
    explicit WriteAheadLog(const std::string& path);
    void log_transaction_begin(uint64_t tx_id);
    void log_page_write(uint64_t tx_id, uint32_t page_id, 
                       const std::array<uint8_t, DB_PAGE_SIZE>& old_data,
                       const std::array<uint8_t, DB_PAGE_SIZE>& new_data);
    void log_transaction_commit(uint64_t tx_id);
    void log_check_point();
    void recover();
};

class BPlusTree {
private:
    Pager& pager;
    BufferPool& buffer_pool;
    uint32_t root_page_id;
    uint32_t order;
    struct Node {
        bool is_leaf;
        uint32_t num_keys;
        std::vector<uint32_t> keys;
        std::vector<uint32_t> children;
        std::vector<std::vector<uint8_t>> values;
    };
public:
    BPlusTree(Pager& pager, BufferPool& buffer_pool, uint32_t order = 100);
    void insert(uint32_t key, const std::vector<uint8_t>& value);
    std::vector<uint8_t> search(uint32_t key);
private:
    void insert_non_full(Node& node, uint32_t key, const std::vector<uint8_t>& value);
    void split_child(Node& parent, uint32_t index, Node& child);
    void serialize_node(const Node& node, std::array<uint8_t, DB_PAGE_SIZE>& buffer);
    void deserialize_node(const std::array<uint8_t, DB_PAGE_SIZE>& buffer, Node& node);
};

#endif
