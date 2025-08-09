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
#include <stdexcept>

inline constexpr uint32_t DB_PAGE_SIZE = 4096;
inline constexpr size_t DEFAULT_BUFFER_POOL_CAPACITY = 1000;
inline constexpr uint32_t DEFAULT_BPTREE_ORDER = 100;

class StorageException : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class FileIOException : public StorageException {
public:
    using StorageException::StorageException;
};

class IntegrityViolation : public StorageException {
public:
    using StorageException::StorageException;
};

class Page {
public:
    uint32_t page_id;
    bool dirty;
    std::array<uint8_t, DB_PAGE_SIZE> data;

    explicit Page(uint32_t id) noexcept;/* : page_id(id), dirty(false) {
        data.fill(0);
    }*/
    Page(const Page&) = delete;
    Page& operator=(const Page&) = delete;
};

class Pager {
private:
    std::string filename;
    std::fstream file;
    std::unordered_map<uint32_t, std::shared_ptr<Page>> page_cache;
    std::mutex cache_mutex;
    uint32_t next_page_id = 1; // Start at 1 (0 is metadata)

    void ensure_file_open();
    bool safe_seek(uint32_t page_id) noexcept;

public:
    explicit Pager(const std::string& db_file);
    ~Pager() noexcept;

    Pager(const Pager&) = delete;
    Pager& operator=(const Pager&) = delete;

    std::shared_ptr<Page> get_page(uint32_t page_id);
    uint32_t allocate_page();
    void mark_dirty(uint32_t page_id);
    void flush_all() noexcept;
    bool read_page_from_disk(uint32_t page_id, std::array<uint8_t, DB_PAGE_SIZE>& buffer) noexcept;
    bool write_page_to_disk(uint32_t page_id, const std::array<uint8_t, DB_PAGE_SIZE>& buffer) noexcept;
    void load_metadata();
    void save_metadata() noexcept;
    uint32_t get_next_page_id() const { return next_page_id; }
};

class BufferPool {
private:
    Pager& pager;
    const size_t capacity;
    std::unordered_map<uint32_t, std::shared_ptr<Page>> pool;
    std::list<uint32_t> lru_list;
    std::unordered_map<uint32_t, std::list<uint32_t>::iterator> lru_map;
    std::mutex mutex;

    void evict_page() noexcept;

public:
    BufferPool(Pager& pager, size_t capacity = DEFAULT_BUFFER_POOL_CAPACITY) noexcept;
    BufferPool(const BufferPool&) = delete;
    BufferPool& operator=(const BufferPool&) = delete;

    std::shared_ptr<Page> fetch_page(uint32_t page_id);
    void flush_all() noexcept;
};

class WriteAheadLog {
private:
    std::string log_file_path;
    std::fstream log_file;
    std::mutex mutex;

    void ensure_file_open();

public:
    explicit WriteAheadLog(const std::string& path);
    ~WriteAheadLog() noexcept;

    WriteAheadLog(const WriteAheadLog&) = delete;
    WriteAheadLog& operator=(const WriteAheadLog&) = delete;

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
    const uint32_t order;

    struct Node {
        bool is_leaf;
        uint32_t num_keys;
        std::vector<uint32_t> keys;
        std::vector<uint32_t> children;
        std::vector<std::vector<uint8_t>> values;
    };

    void validate_node(const Node& node) const;

public:
    BPlusTree(Pager& pager, BufferPool& buffer_pool, uint32_t order = DEFAULT_BPTREE_ORDER);
    BPlusTree(Pager& pager, BufferPool& buffer_pool, uint32_t root_page_id,bool existing_key);
    BPlusTree(const BPlusTree&) = delete;
    BPlusTree& operator=(const BPlusTree&) = delete;

    void insert(uint32_t key, const std::vector<uint8_t>& value);
    std::vector<uint32_t> getAllKeys() const;
    std::vector<uint8_t> search(uint32_t key);
    uint32_t get_root_page_id() const { return root_page_id; }

private:
    void collectKeys(uint32_t page_id,std::vector<uint32_t>& keys) const;
    void insert_non_full(Node& node, uint32_t key, const std::vector<uint8_t>& value);
    void split_child(Node& parent, uint32_t index, Node& child);
    void serialize_node(const Node& node, std::array<uint8_t, DB_PAGE_SIZE>& buffer) const;
    void deserialize_node(const std::array<uint8_t, DB_PAGE_SIZE>& buffer, Node& node) const;
};

#endif // STORAGE_MANAGER_H
