#include "storagemanager.h"
#include <stdexcept>
#include <thread>
#include <sstream>

// Cross-platform transactional memory detection
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define HAS_HARDWARE_TM 1
#else
#define HAS_HARDWARE_TM 0
#endif

//ARM PLATFORMS. Mobile.
namespace {
    bool detect_neon_support() {
#if defined(__ARM_NEON) || defined(__aarch64__)
        // For ARM platforms, check if NEON is available
        #ifdef __linux__
        unsigned long hwcap = getauxval(AT_HWCAP);
        #ifdef HWCAP_NEON
        return (hwcap & HWCAP_NEON) != 0;
        #else
        return true; // Assume NEON is available on AArch64
        #endif
        #else
        return true; // Assume NEON on ARM platforms
        #endif
#else
        return false; // Not an ARM platform
#endif
    }
}

// Pager implementation
Pager::Pager(const std::string& fname) : filename(fname) {
    file.open(filename, std::ios::binary | std::ios::in | std::ios::out);
    if (!file.is_open()) {
        // Create the file if it doesn't exist
        file.open(filename, std::ios::binary | std::ios::out);
        if (!file) {
            throw std::runtime_error("Failed to create database file: " + filename);
        }
        file.close();
        file.open(filename, std::ios::binary | std::ios::in | std::ios::out);
    }
    
    if (!file) {
        throw std::runtime_error("Failed to open database file: " + filename);
    }
}

Pager::~Pager() {
    file.close();
}

void Pager::read_page(uint32_t page_id, Node* node) {
    file.seekg(page_id * BPTREE_PAGE_SIZE, std::ios::beg);
    file.read(reinterpret_cast<char*>(node), BPTREE_PAGE_SIZE);
    if (!file) throw std::runtime_error("Failed to read page " + std::to_string(page_id));
}

/*void Pager::write_page(uint32_t page_id, const Node* node) {
    file.seekp(page_id * BPTREE_PAGE_SIZE, std::ios::beg);
    if (!file) {
        throw std::runtime_error("Failed to seek to page " + std::to_string(page_id));
    }
    
    file.write(reinterpret_cast<const char*>(node), BPTREE_PAGE_SIZE);
    if (!file) {
        throw std::runtime_error("Failed to write page " + std::to_string(page_id));
    }
    
    file.flush();
    if (!file) {
        throw std::runtime_error("Failed to flush page " + std::to_string(page_id));
    }
}*/

void Pager::write_page(uint32_t page_id, const Node* node) {
    try {
        file.seekp(page_id * BPTREE_PAGE_SIZE, std::ios::beg);
        if (!file) {
            // Try to create the file if seeking fails
            file.clear();
            file.open(filename, std::ios::binary | std::ios::out | std::ios::app);
            if (!file) {
                throw std::runtime_error("Failed to open file for writing");
            }
            file.seekp(page_id * BPTREE_PAGE_SIZE, std::ios::beg);
        }

        file.write(reinterpret_cast<const char*>(node), BPTREE_PAGE_SIZE);
        if (!file) {
            throw std::runtime_error("Failed to write page " + std::to_string(page_id));
        }

        file.flush();
    } catch (const std::exception& e) {
        throw std::runtime_error("Pager write error: " + std::string(e.what()));
    }
}

uint32_t Pager::allocate_page() {
    file.seekp(0, std::ios::end);
    uint32_t page_id = file.tellp() / BPTREE_PAGE_SIZE;
    Node node = {};
    write_page(page_id, &node);
    return page_id;
}

uint64_t Pager::write_data_block(const std::string& data) {
    file.seekp(0, std::ios::end);
    uint64_t offset = file.tellp();
    std::vector<char> compressed(ZSTD_compressBound(data.size()));
    size_t compressed_size = ZSTD_compress(compressed.data(), compressed.size(), 
                                          data.data(), data.size(), 1);
    if (ZSTD_isError(compressed_size)) {
        throw std::runtime_error("Zstd compression failed for data block");
    }
    file.write(compressed.data(), compressed_size);
    file.flush();
    return offset;
}

std::string Pager::read_data_block(uint64_t offset, uint32_t length) {
    file.seekg(offset, std::ios::beg);
    std::vector<char> compressed(length);
    file.read(compressed.data(), length);
    
    // Use ZSTD_getFrameContentSize instead of deprecated ZSTD_getDecompressedSize
    unsigned long long const decompressed_size = ZSTD_getFrameContentSize(compressed.data(), length);
    if (decompressed_size == ZSTD_CONTENTSIZE_ERROR || decompressed_size == ZSTD_CONTENTSIZE_UNKNOWN) {
        throw std::runtime_error("Zstd decompression failed for data block: invalid frame");
    }
    
    std::vector<char> decompressed(decompressed_size);
    size_t actual_decompressed_size = ZSTD_decompress(decompressed.data(), decompressed.size(), 
                                                     compressed.data(), length);
    if (ZSTD_isError(actual_decompressed_size)) {
        throw std::runtime_error("Zstd decompression failed for data block");
    }
    return std::string(decompressed.data(), actual_decompressed_size);
}

// WriteAheadLog implementation
WriteAheadLog::WriteAheadLog(const std::string& fname) : filename(fname) {
    log_file.open(fname + ".wal", std::ios::binary | std::ios::app);
    if (!log_file.is_open()) throw std::runtime_error("Failed to open WAL");
}

WriteAheadLog::~WriteAheadLog() {
    log_file.close();
}

void WriteAheadLog::log_operation(const std::string& op, uint32_t page_id, const Node* node) {
    std::lock_guard<std::mutex> lock(log_mutex);
    log_file.write(op.c_str(), op.size());
    log_file.write("\0", 1);
    log_file.write(reinterpret_cast<const char*>(&page_id), sizeof(page_id));
    if (node) {
        log_file.write(reinterpret_cast<const char*>(node), BPTREE_PAGE_SIZE);
    }
    log_file.flush();
}

void WriteAheadLog::checkpoint(Pager& pager, uint32_t metadata_page_id) {
    log_operation("CHECKPOINT", metadata_page_id, nullptr);
}

void WriteAheadLog::recover(Pager& pager, uint32_t& metadata_page_id) {
    log_file.seekg(0, std::ios::beg);
    uint32_t last_checkpoint_page_id = 0;
    while (log_file) {
        char op_buf[16];
        log_file.read(op_buf, 16);
        std::string op(op_buf, strnlen(op_buf, 16));
        if (op.empty()) break;
        uint32_t page_id;
        log_file.read(reinterpret_cast<char*>(&page_id), sizeof(page_id));
        if (op == "CHECKPOINT") {
            last_checkpoint_page_id = page_id;
            continue;
        }
        Node node;
        log_file.read(reinterpret_cast<char*>(&node), BPTREE_PAGE_SIZE);
        if (log_file) {
            pager.write_page(page_id, &node);
        }
    }
    metadata_page_id = last_checkpoint_page_id;
    log_file.close();
    std::remove((filename + ".wal").c_str());
}

// BufferPool implementation
BufferPool::BufferPool(size_t max_size) : max_pages(max_size) {
#ifdef HAS_NUMA
    if (numa_available() >= 0) {
        numa_set_preferred(-1); // Use local NUMA node
    }
#endif
}

Node* BufferPool::get_page(Pager& pager, uint32_t page_id) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    if (cache.find(page_id) != cache.end()) {
        return &cache[page_id];
    }
    Node node;
    pager.read_page(page_id, &node);
    if (cache.size() >= max_pages) {
        cache.erase(cache.begin());
    }
    cache[page_id] = node;
    return &cache[page_id];
}

void BufferPool::write_page(Pager& pager, WriteAheadLog& wal, uint32_t page_id, const Node* node) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    cache[page_id] = *node;
    wal.log_operation("WRITE", page_id, node);
    pager.write_page(page_id, node);
}

void BufferPool::evict_page(uint32_t page_id) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    cache.erase(page_id);
}

void BufferPool::flush_all() {
    std::lock_guard<std::mutex> lock(cache_mutex);
    cache.clear();
}

// FractalBPlusTree implementation
FractalBPlusTree::FractalBPlusTree(Pager& p, WriteAheadLog& w, BufferPool& bp, const std::string& tname, uint32_t root_id): pager(p), wal(w), buffer_pool(bp), table_name(tname), root_page_id(root_id) ,neon_supported(detect_neon_support()){
    Node root;
    pager.read_page(root_id, &root);
    if (root.header.type != PageType::INTERNAL && root.header.type != PageType::LEAF) {
        root.header.type = PageType::LEAF;
        root.header.page_id = root_id;
        root.header.num_keys = 0;
        root.header.num_messages = 0;
        root.header.version = 0;
        write_node(root_id, &root, 0);
    }
}

bool FractalBPlusTree::has_neon_support() const {
    return neon_supported;
}

// Transactional memory helpers
bool FractalBPlusTree::begin_transaction() {
#if HAS_HARDWARE_TM
    return _xbegin() == _XBEGIN_STARTED;
#else
    return false; // Fallback to mutex on non-x86 platforms
#endif
}

void FractalBPlusTree::end_transaction() {
#if HAS_HARDWARE_TM
    _xend();
#endif
}

void FractalBPlusTree::transaction_fallback() {
    // Fallback implementation using mutex
    std::lock_guard<std::mutex> lock(version_mutex);
}

// NEON-optimized key compression
void FractalBPlusTree::compress_keys_neon(char* dest, const int64_t* keys, uint32_t num_keys) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    if (num_keys == 0) return;

    // Store first key normally
    memcpy(dest, &keys[0], sizeof(int64_t));

    // Process remaining keys with NEON
    uint32_t i = 1;

    // Process 2 keys at a time (NEON can handle 2x64-bit integers)
    for (; i + 1 < num_keys; i += 2) {
        // Load current and previous keys
        int64x2_t current_keys = vld1q_s64(&keys[i]);
        int64x2_t prev_keys = vld1q_s64(&keys[i - 1]);

        // Calculate deltas
        int64x2_t deltas = vsubq_s64(current_keys, prev_keys);

        // Store deltas (each delta is stored in MAX_KEY_PREFIX bytes)
        vst1_s64(reinterpret_cast<int64_t*>(dest + i * MAX_KEY_PREFIX),
                 vget_low_s64(deltas));
        vst1_s64(reinterpret_cast<int64_t*>(dest + (i + 1) * MAX_KEY_PREFIX),
                 vget_high_s64(deltas));
    }

    // Handle remaining single key
    if (i < num_keys) {
        int64_t delta = keys[i] - keys[i - 1];
        memcpy(dest + i * MAX_KEY_PREFIX, &delta, MAX_KEY_PREFIX);
    }
#else
    // Fallback to scalar implementation
    compress_keys(dest, keys, num_keys);
#endif
}

// NEON-optimized key decompression
void FractalBPlusTree::decompress_keys_neon(int64_t* keys, const char* src, uint32_t num_keys) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    if (num_keys == 0) return;

    // Load first key
    memcpy(&keys[0], src, sizeof(int64_t));

    // Process remaining keys with NEON
    uint32_t i = 1;

    // Process 2 keys at a time
    for (; i + 1 < num_keys; i += 2) {
        // Load deltas
        int64x2_t deltas = vcombine_s64(
            vld1_s64(reinterpret_cast<const int64_t*>(src + i * MAX_KEY_PREFIX)),
            vld1_s64(reinterpret_cast<const int64_t*>(src + (i + 1) * MAX_KEY_PREFIX))
        );

        // Load previous keys
        int64x2_t prev_keys = vld1q_s64(&keys[i - 1]);

        // Calculate current keys: current = previous + delta
        // We need to create a vector with [keys[i-1], keys[i]] for the first pair
        // and [keys[i], keys[i+1]] for the second pair, but this requires shuffling
        int64x2_t current_keys;

        // For the first key in the pair: keys[i] = keys[i-1] + delta1
        // For the second key in the pair: keys[i+1] = keys[i] + delta2
        int64_t temp_key = keys[i - 1] + vgetq_lane_s64(deltas, 0);
        keys[i] = temp_key;
        keys[i + 1] = temp_key + vgetq_lane_s64(deltas, 1);
    }

    // Handle remaining single key
    if (i < num_keys) {
        int64_t delta;
        memcpy(&delta, src + i * MAX_KEY_PREFIX, MAX_KEY_PREFIX);
        keys[i] = keys[i - 1] + delta;
    }
#else
    // Fallback to scalar implementation
    decompress_keys(keys, src, num_keys);
#endif
}

// NEON-optimized node compression
void FractalBPlusTree::compress_node_neon(Node* node) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    char temp[BPTREE_PAGE_SIZE];
    memcpy(temp, node->data, BPTREE_PAGE_SIZE - sizeof(PageHeader));

    // Use NEON-accelerated compression if available
    size_t compressed_size = ZSTD_compress(node->data, BPTREE_PAGE_SIZE - sizeof(PageHeader),
                                          temp, BPTREE_PAGE_SIZE - sizeof(PageHeader), 1);

    if (ZSTD_isError(compressed_size)) {
        throw std::runtime_error("Zstd compression failed");
    }

    node->header.num_keys |= (1U << 31);
#else
    compress_node(node);
#endif
}

// NEON-optimized node decompression
void FractalBPlusTree::decompress_node_neon(Node* node) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    if (!(node->header.num_keys & (1U << 31))) return;

    char temp[BPTREE_PAGE_SIZE];
    size_t decompressed_size = ZSTD_decompress(temp, BPTREE_PAGE_SIZE - sizeof(PageHeader),
                                              node->data, BPTREE_PAGE_SIZE - sizeof(PageHeader));

    if (ZSTD_isError(decompressed_size)) {
        throw std::runtime_error("Zstd decompression failed");
    }

    memcpy(node->data, temp, decompressed_size);
    node->header.num_keys &= ~(1U << 31);
#else
    decompress_node(node);
#endif
}

void FractalBPlusTree::compress_keys_optimized(char* dest, const int64_t* keys, uint32_t num_keys) {
    if (has_neon_support()) {
        compress_keys_neon(dest, keys, num_keys);
    } else {
        compress_keys(dest, keys, num_keys);
    }
}

void FractalBPlusTree::decompress_keys_optimized(int64_t* keys, const char* src, uint32_t num_keys) {
    if (has_neon_support()) {
        decompress_keys_neon(keys, src, num_keys);
    } else {
        decompress_keys(keys, src, num_keys);
    }
}

void FractalBPlusTree::compress_node_optimized(Node* node) {
    if (has_neon_support()) {
        compress_node_neon(node);
    } else {
        compress_node(node);
    }
}

void FractalBPlusTree::decompress_node_optimized(Node* node) {
    if (has_neon_support()) {
        decompress_node_neon(node);
    } else {
        decompress_node(node);
    }
}

size_t FractalBPlusTree::memory_usage() {
    return versioned_nodes.size() * sizeof(Node) + buffer_pool.cache_size() * BPTREE_PAGE_SIZE;
}

bool FractalBPlusTree::memory_usage_high() {
    return memory_usage() > 1024 * 1024 * 1024; // 1GB threshold
}

Node* FractalBPlusTree::get_node(uint32_t page_id) {
    return buffer_pool.get_page(pager, page_id);
}

void FractalBPlusTree::write_node(uint32_t page_id, const Node* node, uint64_t transaction_id) {
    {
        std::lock_guard<std::mutex> lock(version_mutex);
        versioned_nodes[transaction_id].push_back(*node);
    }
    buffer_pool.write_page(pager, wal, page_id, node);
}

void FractalBPlusTree::compress_node(Node* node) {
    char temp[BPTREE_PAGE_SIZE];
    memcpy(temp, node->data, BPTREE_PAGE_SIZE - sizeof(PageHeader));
    size_t compressed_size = ZSTD_compress(node->data, BPTREE_PAGE_SIZE - sizeof(PageHeader),
                                          temp, BPTREE_PAGE_SIZE - sizeof(PageHeader), 1);
    if (ZSTD_isError(compressed_size)) {
        throw std::runtime_error("Zstd compression failed");
    }
    node->header.num_keys |= (1U << 31);
}

void FractalBPlusTree::decompress_node(Node* node) {
    if (!(node->header.num_keys & (1U << 31))) return;
    char temp[BPTREE_PAGE_SIZE];
    size_t decompressed_size = ZSTD_decompress(temp, BPTREE_PAGE_SIZE - sizeof(PageHeader),
                                              node->data, BPTREE_PAGE_SIZE - sizeof(PageHeader));
    if (ZSTD_isError(decompressed_size)) {
        throw std::runtime_error("Zstd decompression failed");
    }
    memcpy(node->data, temp, decompressed_size);
    node->header.num_keys &= ~(1U << 31);
}

void FractalBPlusTree::compress_keys(char* dest, const int64_t* keys, uint32_t num_keys) {
    memcpy(dest, &keys[0], sizeof(int64_t));
    for (uint32_t i = 1; i < num_keys; ++i) {
        int64_t delta = keys[i] - keys[i - 1];
        memcpy(dest + i * MAX_KEY_PREFIX, &delta, MAX_KEY_PREFIX);
    }
}

void FractalBPlusTree::decompress_keys(int64_t* keys, const char* src, uint32_t num_keys) {
    memcpy(&keys[0], src, sizeof(int64_t));
    for (uint32_t i = 1; i < num_keys; ++i) {
        int64_t delta;
        memcpy(&delta, src + i * MAX_KEY_PREFIX, MAX_KEY_PREFIX);
        keys[i] = keys[i - 1] + delta;
    }
}

bool FractalBPlusTree::is_node_full(const Node* node) {
    return node->header.num_keys >= BPTREE_ORDER || node->header.num_messages >= MAX_MESSAGES;
}

bool FractalBPlusTree::is_node_underfull(const Node* node) {
    return node->header.num_keys < BPTREE_ORDER / 2 && node->header.type == PageType::LEAF;
}

void FractalBPlusTree::compact_messages(Node* node) {
    Message messages[MAX_MESSAGES];
    memcpy(messages, node->data + node->header.num_keys * MAX_KEY_PREFIX +
           (node->header.type == PageType::INTERNAL ? (BPTREE_ORDER + 1) * sizeof(uint32_t) : 0),
           sizeof(Message) * node->header.num_messages);

    std::map<int64_t, Message> compacted;
    for (uint32_t i = 0; i < node->header.num_messages; ++i) {
        if (messages[i].type != MessageType::DELETE) {
            compacted[messages[i].key] = messages[i];
        } else {
            compacted.erase(messages[i].key);
        }
    }

    node->header.num_messages = compacted.size();
    uint32_t i = 0;
    for (const auto& kv : compacted) {
        messages[i++] = kv.second;
    }

    memcpy(node->data + node->header.num_keys * MAX_KEY_PREFIX +
           (node->header.type == PageType::INTERNAL ? (BPTREE_ORDER + 1) * sizeof(uint32_t) : 0),
           messages, sizeof(Message) * node->header.num_messages);
}

void FractalBPlusTree::split_node(uint32_t page_id, Node* node, uint32_t parent_page_id, uint64_t transaction_id) {
    decompress_node_optimized(node);
    compact_messages(node);

    Node* parent = parent_page_id ? get_node(parent_page_id) : nullptr;
    Node new_node = {};
    new_node.header.type = node->header.type;
    new_node.header.page_id = pager.allocate_page();
    new_node.header.parent_page_id = parent_page_id;
    new_node.header.version = transaction_id;

    uint32_t mid = node->header.num_keys / 2;
    int64_t keys[BPTREE_ORDER];
    decompress_keys_optimized(keys, node->data, node->header.num_keys);

    new_node.header.num_keys = node->header.num_keys - mid - 1;
    node->header.num_keys = mid;

    if (node->header.type == PageType::INTERNAL) {
        uint32_t children[BPTREE_ORDER + 1];
        memcpy(children, node->data + node->header.num_keys * MAX_KEY_PREFIX,
               sizeof(uint32_t) * (node->header.num_keys + 1));

        compress_keys_optimized(new_node.data, &keys[mid + 1], new_node.header.num_keys);
        memcpy(new_node.data + new_node.header.num_keys * MAX_KEY_PREFIX,
               &children[mid + 1], sizeof(uint32_t) * (new_node.header.num_keys + 1));
        memcpy(node->data + node->header.num_keys * MAX_KEY_PREFIX,
               children, sizeof(uint32_t) * (node->header.num_keys + 1));

        Message messages[MAX_MESSAGES];
        memcpy(messages, node->data + node->header.num_keys * MAX_KEY_PREFIX +
               (BPTREE_ORDER + 1) * sizeof(uint32_t), sizeof(Message) * node->header.num_messages);

        uint32_t msg_mid = node->header.num_messages / 2;
        new_node.header.num_messages = node->header.num_messages - msg_mid;
        node->header.num_messages = msg_mid;

        memcpy(new_node.data + new_node.header.num_keys * MAX_KEY_PREFIX +
               (new_node.header.num_keys + 1) * sizeof(uint32_t),
               &messages[msg_mid], sizeof(Message) * new_node.header.num_messages);
    } else {
        KeyValue kvs[BPTREE_ORDER];
        memcpy(kvs, node->data + node->header.num_keys * MAX_KEY_PREFIX,
               sizeof(KeyValue) * node->header.num_keys);

        compress_keys_optimized(new_node.data, &keys[mid + 1], new_node.header.num_keys);
        memcpy(new_node.data + new_node.header.num_keys * MAX_KEY_PREFIX,
               &kvs[mid + 1], sizeof(KeyValue) * new_node.header.num_keys);

        new_node.header.next_page_id = node->header.next_page_id;
        new_node.header.prev_page_id = page_id;

        if (node->header.next_page_id) {
            Node* next = get_node(node->header.next_page_id);
            next->header.prev_page_id = new_node.header.page_id;
            compress_node_optimized(next);
            write_node(next->header.page_id, next, transaction_id);
        }
        node->header.next_page_id = new_node.header.page_id;
    }

    compress_node_optimized(node);
    compress_node_optimized(&new_node);
    write_node(new_node.header.page_id, &new_node, transaction_id);

    if (!parent) {
        parent = new Node();
        parent->header.type = PageType::INTERNAL;
        parent->header.page_id = pager.allocate_page();
        parent->header.version = transaction_id;
        root_page_id.store(parent->header.page_id);
    }

    decompress_node_optimized(parent);
    int64_t promoted_key = keys[mid];
    uint32_t children[BPTREE_ORDER + 1];
    memcpy(children, parent->data + parent->header.num_keys * MAX_KEY_PREFIX,
           sizeof(uint32_t) * parent->header.num_keys);

    parent->header.num_keys++;
    compress_keys_optimized(parent->data, &promoted_key, 1);
    memcpy(parent->data + parent->header.num_keys * MAX_KEY_PREFIX,
           &new_node.header.page_id, sizeof(uint32_t));
    memcpy(parent->data + parent->header.num_keys * MAX_KEY_PREFIX + sizeof(uint32_t),
           children, sizeof(uint32_t) * parent->header.num_keys);

    compress_node_optimized(parent);
    write_node(parent->header.page_id, parent, transaction_id);
    write_node(page_id, node, transaction_id);
}

void FractalBPlusTree::merge_nodes(uint32_t page_id, Node* node, uint32_t parent_page_id, uint64_t transaction_id) {
    if (!parent_page_id) return;

    Node* parent = get_node(parent_page_id);
    decompress_node_optimized(parent);

    int64_t parent_keys[BPTREE_ORDER];
    decompress_keys_optimized(parent_keys, parent->data, parent->header.num_keys);

    uint32_t children[BPTREE_ORDER + 1];
    memcpy(children, parent->data + parent->header.num_keys * MAX_KEY_PREFIX,
           sizeof(uint32_t) * (parent->header.num_keys + 1));

    uint32_t node_index = 0;
    for (; node_index <= parent->header.num_keys; ++node_index) {
        if (children[node_index] == page_id) break;
    }

    Node* sibling = nullptr;
    uint32_t sibling_page_id = 0;
    bool merge_with_left = false;

    if (node_index > 0) {
        sibling_page_id = children[node_index - 1];
        sibling = get_node(sibling_page_id);
        merge_with_left = true;
    } else if (node_index < parent->header.num_keys) {
        sibling_page_id = children[node_index + 1];
        sibling = get_node(sibling_page_id);
    }

    if (!sibling) return;
    decompress_node_optimized(sibling);

    int64_t node_keys[BPTREE_ORDER], sibling_keys[BPTREE_ORDER];
    decompress_keys_optimized(node_keys, node->data, node->header.num_keys);
    decompress_keys_optimized(sibling_keys, sibling->data, sibling->header.num_keys);

    if (node->header.num_keys + sibling->header.num_keys < BPTREE_ORDER) {
        if (merge_with_left) {
            memcpy(&sibling_keys[sibling->header.num_keys], node_keys,
                   sizeof(int64_t) * node->header.num_keys);
            sibling->header.num_keys += node->header.num_keys;

            if (node->header.type == PageType::LEAF) {
                KeyValue node_kvs[BPTREE_ORDER], sibling_kvs[BPTREE_ORDER];
                memcpy(sibling_kvs, sibling->data + sibling->header.num_keys * MAX_KEY_PREFIX,
                       sizeof(KeyValue) * sibling->header.num_keys);
                memcpy(node_kvs, node->data + node->header.num_keys * MAX_KEY_PREFIX,
                       sizeof(KeyValue) * node->header.num_keys);

                memcpy(&sibling_kvs[sibling->header.num_keys], node_kvs,
                       sizeof(KeyValue) * node->header.num_keys);
                memcpy(sibling->data + sibling->header.num_keys * MAX_KEY_PREFIX,
                       sibling_kvs, sizeof(KeyValue) * sibling->header.num_keys);

                sibling->header.next_page_id = node->header.next_page_id;
                if (node->header.next_page_id) {
                    Node* next = get_node(node->header.next_page_id);
                    next->header.prev_page_id = sibling_page_id;
                    compress_node_optimized(next);
                    write_node(next->header.page_id, next, transaction_id);
                }
            }

            compress_keys_optimized(sibling->data, sibling_keys, sibling->header.num_keys);
            compress_node_optimized(sibling);
            write_node(sibling_page_id, sibling, transaction_id);

            for (uint32_t i = node_index; i < parent->header.num_keys; ++i) {
                parent_keys[i - 1] = parent_keys[i];
                children[i] = children[i + 1];
            }
            parent->header.num_keys--;
        } else {
            memcpy(&node_keys[node->header.num_keys], sibling_keys,
                   sizeof(int64_t) * sibling->header.num_keys);
            node->header.num_keys += sibling->header.num_keys;

            if (node->header.type == PageType::LEAF) {
                KeyValue node_kvs[BPTREE_ORDER], sibling_kvs[BPTREE_ORDER];
                memcpy(node_kvs, node->data + node->header.num_keys * MAX_KEY_PREFIX,
                       sizeof(KeyValue) * node->header.num_keys);
                memcpy(sibling_kvs, sibling->data + sibling->header.num_keys * MAX_KEY_PREFIX,
                       sizeof(KeyValue) * sibling->header.num_keys);

                memcpy(&node_kvs[node->header.num_keys], sibling_kvs,
                       sizeof(KeyValue) * sibling->header.num_keys);
                memcpy(node->data + node->header.num_keys * MAX_KEY_PREFIX,
                       node_kvs, sizeof(KeyValue) * node->header.num_keys);

                node->header.next_page_id = sibling->header.next_page_id;
                if (sibling->header.next_page_id) {
                    Node* next = get_node(sibling->header.next_page_id);
                    next->header.prev_page_id = page_id;
                    compress_node_optimized(next);
                    write_node(next->header.page_id, next, transaction_id);
                }
            }

            compress_keys_optimized(node->data, node_keys, node->header.num_keys);
            compress_node_optimized(node);
            write_node(page_id, node, transaction_id);

            for (uint32_t i = node_index + 1; i < parent->header.num_keys; ++i) {
                parent_keys[i - 1] = parent_keys[i];
                children[i] = children[i + 1];
            }
            parent->header.num_keys--;
        }

        compress_keys_optimized(parent->data, parent_keys, parent->header.num_keys);
        memcpy(parent->data + parent->header.num_keys * MAX_KEY_PREFIX,
               children, sizeof(uint32_t) * (parent->header.num_keys + 1));
        compress_node_optimized(parent);
        write_node(parent_page_id, parent, transaction_id);

        if (parent->header.num_keys == 0) {
            root_page_id.store(merge_with_left ? sibling_page_id : page_id);
            write_node(root_page_id.load(), merge_with_left ? sibling : node, transaction_id);
        }
    }
}

void FractalBPlusTree::flush_messages(Node* node, uint32_t page_id, uint64_t transaction_id) {
    decompress_node_optimized(node);
    compact_messages(node);

    if (node->header.type == PageType::LEAF) {
        KeyValue kvs[BPTREE_ORDER];
        memcpy(kvs, node->data + node->header.num_keys * MAX_KEY_PREFIX,
               sizeof(KeyValue) * node->header.num_keys);

        Message messages[MAX_MESSAGES];
        memcpy(messages, node->data + node->header.num_keys * MAX_KEY_PREFIX +
               sizeof(KeyValue) * node->header.num_keys, sizeof(Message) * node->header.num_messages);

        for (uint32_t i = 0; i < node->header.num_messages; ++i) {
            bool found = false;
            for (uint32_t j = 0; j < node->header.num_keys; ++j) {
                if (kvs[j].key == messages[i].key) {
                    if (messages[i].type == MessageType::DELETE) {
                        for (uint32_t k = j; k < node->header.num_keys - 1; ++k) {
                            kvs[k] = kvs[k + 1];
                        }
                        node->header.num_keys--;
                    } else {
                        kvs[j].value_offset = messages[i].value_offset;
                        kvs[j].value_length = messages[i].value_length;
                    }
                    found = true;
                    break;
                }
            }
            if (!found && messages[i].type != MessageType::DELETE) {
                kvs[node->header.num_keys].key = messages[i].key;
                kvs[node->header.num_keys].value_offset = messages[i].value_offset;
                kvs[node->header.num_keys].value_length = messages[i].value_length;
                node->header.num_keys++;
            }
        }

        memcpy(node->data + node->header.num_keys * MAX_KEY_PREFIX,
               kvs, sizeof(KeyValue) * node->header.num_keys);
        node->header.num_messages = 0;
    }

    compress_node_optimized(node);
    write_node(page_id, node, transaction_id);
}

void FractalBPlusTree::adaptive_flush(Node* node, uint32_t page_id, uint64_t transaction_id) {
    if (memory_usage_high() || node->header.num_messages >= MAX_MESSAGES) {
        flush_messages(node, page_id, transaction_id);
    }
}

uint32_t FractalBPlusTree::find_child_index(const Node* node, int64_t key) {
    int64_t keys[BPTREE_ORDER];
    decompress_keys_optimized(keys, node->data, node->header.num_keys);

    uint32_t left = 0, right = node->header.num_keys;
    while (left < right) {
        uint32_t mid = left + (right - left) / 2;
        if (keys[mid] < key) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return left;
}

void FractalBPlusTree::update_secondary_index(const std::string& index_name, int64_t primary_key,
                                             const std::string& secondary_key, uint64_t transaction_id,
                                             MessageType op_type) {
    auto it = secondary_indexes.find(index_name);
    if (it == secondary_indexes.end()) return;

    uint32_t index_root = it->second;
    Node* node = get_node(index_root);

    // Simplified secondary index update
    // In a real implementation, this would traverse the index tree
    // and update the appropriate entries

    compress_node_optimized(node);
    write_node(index_root, node, transaction_id);
}

void FractalBPlusTree::create() {
    Node root;
    root.header.type = PageType::LEAF;
    root.header.page_id = 0;
    root.header.num_keys = 0;
    root.header.num_messages = 0;
    root.header.version = 0;
    write_node(0, &root, 0);
}

void FractalBPlusTree::drop() {
    buffer_pool.flush_all();
    std::remove(table_name.c_str());
    std::remove((table_name + ".wal").c_str());
}

void FractalBPlusTree::insert(int64_t key, const std::string& value, uint64_t transaction_id,
                             const std::map<std::string, std::string>& secondary_keys) {
    uint64_t value_offset = pager.write_data_block(value);
    uint32_t value_length = value.size();

    bool use_tm = begin_transaction();
    if (!use_tm) {
        transaction_fallback();
    }

    try {
        uint32_t current_page_id = root_page_id.load();
        Node* node = get_node(current_page_id);

        while (node->header.type == PageType::INTERNAL) {
            uint32_t child_index = find_child_index(node, key);
            uint32_t children[BPTREE_ORDER + 1];
            memcpy(children, node->data + node->header.num_keys * MAX_KEY_PREFIX,
                   sizeof(uint32_t) * (node->header.num_keys + 1));
            current_page_id = children[child_index];
            node = get_node(current_page_id);
        }

        decompress_node_optimized(node);

        if (is_node_full(node)) {
            split_node(current_page_id, node, node->header.parent_page_id, transaction_id);
            node = get_node(current_page_id);
            decompress_node_optimized(node);
        }

        Message msg;
        msg.type = MessageType::INSERT;
        msg.key = key;
        msg.value_offset = value_offset;
        msg.value_length = value_length;
        msg.version = transaction_id;

        Message messages[MAX_MESSAGES + 1];
        memcpy(messages, node->data + node->header.num_keys * MAX_KEY_PREFIX +
               sizeof(KeyValue) * node->header.num_keys, sizeof(Message) * node->header.num_messages);

        messages[node->header.num_messages++] = msg;
        memcpy(node->data + node->header.num_keys * MAX_KEY_PREFIX +
               sizeof(KeyValue) * node->header.num_keys, messages, sizeof(Message) * node->header.num_messages);

        compress_node_optimized(node);
        write_node(current_page_id, node, transaction_id);

        for (const auto& sk : secondary_keys) {
            update_secondary_index(sk.first, key, sk.second, transaction_id, MessageType::INSERT);
        }

        adaptive_flush(node, current_page_id, transaction_id);

        if (use_tm) {
            end_transaction();
        }
    } catch (...) {
        if (use_tm) {
            // Transaction would abort automatically on exception
        }
        throw;
    }
}

void FractalBPlusTree::update(int64_t key, const std::string& value, uint64_t transaction_id,
                             const std::map<std::string, std::string>& secondary_keys) {
    uint64_t value_offset = pager.write_data_block(value);
    uint32_t value_length = value.size();

    bool use_tm = begin_transaction();
    if (!use_tm) {
        transaction_fallback();
    }

    try {
        uint32_t current_page_id = root_page_id.load();
        Node* node = get_node(current_page_id);

        while (node->header.type == PageType::INTERNAL) {
            uint32_t child_index = find_child_index(node, key);
            uint32_t children[BPTREE_ORDER + 1];
            memcpy(children, node->data + node->header.num_keys * MAX_KEY_PREFIX,
                   sizeof(uint32_t) * (node->header.num_keys + 1));
            current_page_id = children[child_index];
            node = get_node(current_page_id);
        }

        decompress_node_optimized(node);

        Message msg;
        msg.type = MessageType::UPDATE;
        msg.key = key;
        msg.value_offset = value_offset;
        msg.value_length = value_length;
        msg.version = transaction_id;

        Message messages[MAX_MESSAGES + 1];
        memcpy(messages, node->data + node->header.num_keys * MAX_KEY_PREFIX +
               sizeof(KeyValue) * node->header.num_keys, sizeof(Message) * node->header.num_messages);

        messages[node->header.num_messages++] = msg;
        memcpy(node->data + node->header.num_keys * MAX_KEY_PREFIX +
               sizeof(KeyValue) * node->header.num_keys, messages, sizeof(Message) * node->header.num_messages);

        compress_node_optimized(node);
        write_node(current_page_id, node, transaction_id);

        for (const auto& sk : secondary_keys) {
            update_secondary_index(sk.first, key, sk.second, transaction_id, MessageType::UPDATE);
        }

        adaptive_flush(node, current_page_id, transaction_id);

        if (use_tm) {
            end_transaction();
        }
    } catch (...) {
        if (use_tm) {
            // Transaction would abort automatically on exception
        }
        throw;
    }
}

void FractalBPlusTree::remove(int64_t key, uint64_t transaction_id,
                             const std::map<std::string, std::string>& secondary_keys) {
    bool use_tm = begin_transaction();
    if (!use_tm) {
        transaction_fallback();
    }

    try {
        uint32_t current_page_id = root_page_id.load();
        Node* node = get_node(current_page_id);

        while (node->header.type == PageType::INTERNAL) {
            uint32_t child_index = find_child_index(node, key);
            uint32_t children[BPTREE_ORDER + 1];
            memcpy(children, node->data + node->header.num_keys * MAX_KEY_PREFIX,
                   sizeof(uint32_t) * (node->header.num_keys + 1));
            current_page_id = children[child_index];
            node = get_node(current_page_id);
        }

        decompress_node_optimized(node);

        Message msg;
        msg.type = MessageType::DELETE;
        msg.key = key;
        msg.version = transaction_id;

        Message messages[MAX_MESSAGES + 1];
        memcpy(messages, node->data + node->header.num_keys * MAX_KEY_PREFIX +
               sizeof(KeyValue) * node->header.num_keys, sizeof(Message) * node->header.num_messages);

        messages[node->header.num_messages++] = msg;
        memcpy(node->data + node->header.num_keys * MAX_KEY_PREFIX +
               sizeof(KeyValue) * node->header.num_keys, messages, sizeof(Message) * node->header.num_messages);

        compress_node_optimized(node);
        write_node(current_page_id, node, transaction_id);

        for (const auto& sk : secondary_keys) {
            update_secondary_index(sk.first, key, sk.second, transaction_id, MessageType::DELETE);
        }

        adaptive_flush(node, current_page_id, transaction_id);

        if (use_tm) {
            end_transaction();
        }
    } catch (...) {
        if (use_tm) {
            // Transaction would abort automatically on exception
        }
        throw;
    }
}

std::string FractalBPlusTree::select(int64_t key, uint64_t transaction_id) {
    uint32_t current_page_id = root_page_id.load();
    Node* node = get_node(current_page_id);

    while (node->header.type == PageType::INTERNAL) {
        uint32_t child_index = find_child_index(node, key);
        uint32_t children[BPTREE_ORDER + 1];
        memcpy(children, node->data + node->header.num_keys * MAX_KEY_PREFIX,
               sizeof(uint32_t) * (node->header.num_keys + 1));
        current_page_id = children[child_index];
        node = get_node(current_page_id);
    }

    decompress_node_optimized(node);

    // Check messages first (most recent updates)
    Message messages[MAX_MESSAGES];
    memcpy(messages, node->data + node->header.num_keys * MAX_KEY_PREFIX +
           sizeof(KeyValue) * node->header.num_keys, sizeof(Message) * node->header.num_messages);

    for (int32_t i = node->header.num_messages - 1; i >= 0; --i) {
        if (messages[i].key == key && messages[i].version <= transaction_id) {
            if (messages[i].type == MessageType::DELETE) {
                return "";
            }
            return pager.read_data_block(messages[i].value_offset, messages[i].value_length);
        }
    }

    // Check stored key-value pairs
    KeyValue kvs[BPTREE_ORDER];
    memcpy(kvs, node->data + node->header.num_keys * MAX_KEY_PREFIX,
           sizeof(KeyValue) * node->header.num_keys);

    for (uint32_t i = 0; i < node->header.num_keys; ++i) {
        if (kvs[i].key == key) {
            return pager.read_data_block(kvs[i].value_offset, kvs[i].value_length);
        }
    }

    return "";
}

std::vector<std::pair<int64_t, std::string>> FractalBPlusTree::select_range(int64_t start_key,
                                                                           int64_t end_key,
                                                                           uint64_t transaction_id) {
    std::vector<std::pair<int64_t, std::string>> result;

    uint32_t current_page_id = root_page_id.load();
    Node* node = get_node(current_page_id);

    while (node->header.type == PageType::INTERNAL) {
        uint32_t child_index = find_child_index(node, start_key);
        uint32_t children[BPTREE_ORDER + 1];
        memcpy(children, node->data + node->header.num_keys * MAX_KEY_PREFIX,
               sizeof(uint32_t) * (node->header.num_keys + 1));
        current_page_id = children[child_index];
        node = get_node(current_page_id);
    }

    while (node) {
        decompress_node_optimized(node);

        KeyValue kvs[BPTREE_ORDER];
        memcpy(kvs, node->data + node->header.num_keys * MAX_KEY_PREFIX,
               sizeof(KeyValue) * node->header.num_keys);

        Message messages[MAX_MESSAGES];
        memcpy(messages, node->data + node->header.num_keys * MAX_KEY_PREFIX +
               sizeof(KeyValue) * node->header.num_keys, sizeof(Message) * node->header.num_messages);

        std::map<int64_t, std::pair<uint64_t, uint32_t>> visible_data;

        // Process stored key-value pairs
        for (uint32_t i = 0; i < node->header.num_keys; ++i) {
            if (kvs[i].key >= start_key && kvs[i].key <= end_key) {
                visible_data[kvs[i].key] = {kvs[i].value_offset, kvs[i].value_length};
            }
        }

        // Process messages (overwrite stored data)
        for (int32_t i = node->header.num_messages - 1; i >= 0; --i) {
            if (messages[i].key >= start_key && messages[i].key <= end_key &&
                messages[i].version <= transaction_id) {
                if (messages[i].type == MessageType::DELETE) {
                    visible_data.erase(messages[i].key);
                } else {
                    visible_data[messages[i].key] = {messages[i].value_offset, messages[i].value_length};
                }
            }
        }

        // Add visible data to result
        for (const auto& kv : visible_data) {
            result.emplace_back(kv.first, pager.read_data_block(kv.second.first, kv.second.second));
        }

        if (node->header.next_page_id && kvs[node->header.num_keys - 1].key <= end_key) {
            current_page_id = node->header.next_page_id;
            node = get_node(current_page_id);
        } else {
            break;
        }
    }

    return result;
}

void FractalBPlusTree::create_secondary_index(const std::string& index_name) {
    uint32_t index_root = pager.allocate_page();
    Node root;
    root.header.type = PageType::LEAF;
    root.header.page_id = index_root;
    root.header.num_keys = 0;
    root.header.num_messages = 0;
    root.header.version = 0;
    write_node(index_root, &root, 0);
    secondary_indexes[index_name] = index_root;
}

std::vector<int64_t> FractalBPlusTree::select_by_secondary(const std::string& index_name,
                                                          const std::string& secondary_key,
                                                          uint64_t transaction_id) {
    auto it = secondary_indexes.find(index_name);
    if (it == secondary_indexes.end()) return {};

    // Simplified implementation - in a real system, this would traverse
    // the secondary index and return matching primary keys
    return {};
}

void FractalBPlusTree::bulk_load(std::vector<std::pair<int64_t, std::string>>& data,
                                uint64_t transaction_id) {
    std::sort(data.begin(), data.end());

    uint32_t current_page_id = root_page_id.load();
    Node* node = get_node(current_page_id);

    for (const auto& kv : data) {
        if (is_node_full(node)) {
            split_node(current_page_id, node, node->header.parent_page_id, transaction_id);
            node = get_node(current_page_id);
        }

        uint64_t value_offset = pager.write_data_block(kv.second);

        KeyValue kvs[BPTREE_ORDER + 1];
        memcpy(kvs, node->data + node->header.num_keys * MAX_KEY_PREFIX,
               sizeof(KeyValue) * node->header.num_keys);

        kvs[node->header.num_keys].key = kv.first;
        kvs[node->header.num_keys].value_offset = value_offset;
        kvs[node->header.num_keys].value_length = kv.second.size();

        node->header.num_keys++;
        memcpy(node->data + node->header.num_keys * MAX_KEY_PREFIX,
               kvs, sizeof(KeyValue) * node->header.num_keys);

        compress_node_optimized(node);
        write_node(current_page_id, node, transaction_id);
    }
}

void FractalBPlusTree::checkpoint() {
    wal.checkpoint(pager, root_page_id.load());
}
