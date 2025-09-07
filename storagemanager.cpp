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
/*#if defined(__ARM_NEON) || defined(__aarch64__)
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
#else*/
        return false; // Not an ARM platform
//#endif
    }
}

// Pager implementation
Pager::Pager(const std::string& fname) : filename(fname) {
    // Try to open existing file first
    file.open(filename, std::ios::binary | std::ios::in | std::ios::out);

    if (!file.is_open()) {
        // Create the file if it doesn't exist
        std::ofstream create_file(filename, std::ios::binary);
        if (!create_file) {
            throw std::runtime_error("Failed to create database file: " + filename);
        }
        create_file.close();

        // Now open for both input and output
        file.open(filename, std::ios::binary | std::ios::in | std::ios::out);
        if (!file) {
            throw std::runtime_error("Failed to open database file: " + filename);
        }

        // Ensure the file has at least one page (for metadata)
        file.seekp(BPTREE_PAGE_SIZE - 1, std::ios::beg);
        file.put('\0');
        file.flush();

        // Initialize page 0
        Node metadata_page = {};
        metadata_page.header.type = PageType::METADATA;
        metadata_page.header.page_id = 0;
        write_page(0, &metadata_page);
    }

    try {
        test_zstd();
    } catch (const std::exception& e) {
        std::cerr << "Warning: Zstd test failed: " << e.what() << std::endl;
        // Continue anyway, but compression will be disabled
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



void Pager::write_page(uint32_t page_id, const Node* node) {
    try {
        // Ensure file is open
        if (!file.is_open()) {
            file.open(filename, std::ios::binary | std::ios::in | std::ios::out);
            if (!file) {
                throw std::runtime_error("Failed to open file for writing");
            }
        }

        // Extend file if needed
        uint64_t required_size = (page_id + 1) * BPTREE_PAGE_SIZE;
        file.seekp(0, std::ios::end);
        uint64_t current_size = file.tellp();

        if (required_size > current_size) {
            file.seekp(required_size - 1, std::ios::beg);
            file.put('\0');
            file.flush();
        }

        file.seekp(page_id * BPTREE_PAGE_SIZE, std::ios::beg);
        if (!file) {
            throw std::runtime_error("Failed to seek to page " + std::to_string(page_id));
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
    
    // Create the new page by extending the file
    uint64_t new_size = (page_id + 1) * BPTREE_PAGE_SIZE;
    file.seekp(new_size - 1, std::ios::beg);
    file.put('\0');
    file.flush();
    
    // Initialize the new page
    Node node = {};
    node.header.page_id = page_id;
    write_page(page_id, &node);
    
    return page_id;
}


uint64_t Pager::write_data_block(const std::string& data) {
    file.seekp(0, std::ios::end);
    uint64_t offset = file.tellp();

    // Handle empty data
    if (data.empty()) {
        uint32_t original_size = 0;
        uint32_t compressed_size = 0;
        uint8_t compression_flag = 0; // 0 = uncompressed

        file.write(reinterpret_cast<const char*>(&original_size), sizeof(original_size));
        file.write(reinterpret_cast<const char*>(&compressed_size), sizeof(compressed_size));
        file.write(reinterpret_cast<const char*>(&compression_flag), sizeof(compression_flag));
        file.flush();
        return offset;
    }

    uint32_t original_size = static_cast<uint32_t>(data.size());
    uint32_t compressed_size;
    uint8_t compression_flag;

    // For small data or debugging, skip compression
    if (data.size() <= 64) { // Don't compress small data
        compression_flag = 0;
        compressed_size = original_size;

        file.write(reinterpret_cast<const char*>(&original_size), sizeof(original_size));
        file.write(reinterpret_cast<const char*>(&compressed_size), sizeof(compressed_size));
        file.write(reinterpret_cast<const char*>(&compression_flag), sizeof(compression_flag));
        file.write(data.data(), data.size());
    } else {
        // Use Zstd compression
        size_t max_compressed_size = ZSTD_compressBound(data.size());
        std::vector<char> compressed(max_compressed_size);

        size_t zstd_result = ZSTD_compress(
            compressed.data(), max_compressed_size,
            data.data(), data.size(),
            3  // Medium compression level
        );

        if (ZSTD_isError(zstd_result)) {
            // Fallback to uncompressed storage
            compression_flag = 0;
            compressed_size = original_size;

            file.write(reinterpret_cast<const char*>(&original_size), sizeof(original_size));
            file.write(reinterpret_cast<const char*>(&compressed_size), sizeof(compressed_size));
            file.write(reinterpret_cast<const char*>(&compression_flag), sizeof(compression_flag));
            file.write(data.data(), data.size());
        } else {
            // Successfully compressed
            compression_flag = 1;
            compressed_size = static_cast<uint32_t>(zstd_result);

            file.write(reinterpret_cast<const char*>(&original_size), sizeof(original_size));
            file.write(reinterpret_cast<const char*>(&compressed_size), sizeof(compressed_size));
            file.write(reinterpret_cast<const char*>(&compression_flag), sizeof(compression_flag));
            file.write(compressed.data(), compressed_size);
        }
    }

    file.flush();
    return offset;
}

std::string Pager::read_data_block(uint64_t offset, uint32_t expected_length) {
    if (offset == 0) {
        return ""; // No data stored at offset 0
    }

    file.seekg(offset, std::ios::beg);
    if (!file) {
        throw std::runtime_error("Failed to seek to offset " + std::to_string(offset));
    }

    // Read the metadata we stored
    uint32_t original_size, compressed_size;
    uint8_t compression_flag;

    file.read(reinterpret_cast<char*>(&original_size), sizeof(original_size));
    file.read(reinterpret_cast<char*>(&compressed_size), sizeof(compressed_size));
    file.read(reinterpret_cast<char*>(&compression_flag), sizeof(compression_flag));

    if (!file) {
        throw std::runtime_error("Failed to read data block metadata at offset " +
                                std::to_string(offset));
    }

    // Handle empty data block
    if (original_size == 0 && compressed_size == 0) {
        return "";
    }

    // Validate sizes
    if (original_size > 100 * 1024 * 1024) { // 100MB sanity check
        throw std::runtime_error("Suspiciously large original size: " +
                               std::to_string(original_size) + " at offset " +
                               std::to_string(offset));
    }

    if (compressed_size > 100 * 1024 * 1024) { // 100MB sanity check
        throw std::runtime_error("Suspiciously large compressed size: " +
                               std::to_string(compressed_size) + " at offset " +
                               std::to_string(offset));
    }

    // Check if data is compressed
    if (compression_flag == 0) {
        // Data is uncompressed
        std::string result(original_size, '\0');
        file.read(&result[0], original_size);

        if (!file || file.gcount() != static_cast<std::streamsize>(original_size)) {
            throw std::runtime_error("Failed to read uncompressed data at offset " +
                                    std::to_string(offset) + ", expected " +
                                    std::to_string(original_size) + " bytes, got " +
                                    std::to_string(file.gcount()) + " bytes");
        }

        return result;
    } else {
        // Data is compressed with Zstd
        std::vector<char> compressed(compressed_size);
        file.read(compressed.data(), compressed_size);

        if (!file || file.gcount() != static_cast<std::streamsize>(compressed_size)) {
            throw std::runtime_error("Failed to read compressed data at offset " +
                                    std::to_string(offset) + ", expected " +
                                    std::to_string(compressed_size) + " bytes, got " +
                                    std::to_string(file.gcount()) + " bytes");
        }

        std::vector<char> decompressed(original_size);
        size_t actual_decompressed_size = ZSTD_decompress(
            decompressed.data(), original_size,
            compressed.data(), compressed_size
        );

        if (ZSTD_isError(actual_decompressed_size)) {
            std::string error_msg = "Zstd decompression failed: " +
                std::string(ZSTD_getErrorName(actual_decompressed_size)) +
                " at offset " + std::to_string(offset) +
                ", original_size: " + std::to_string(original_size) +
                ", compressed_size: " + std::to_string(compressed_size) +
                ", Zstd version: " + ZSTD_versionString();

            // Try to dump some debug info
            std::cerr << error_msg << std::endl;

            // Try to read as uncompressed data as fallback
            try {
                file.seekg(offset + sizeof(original_size) + sizeof(compressed_size) + sizeof(compression_flag), std::ios::beg);
                std::string fallback_result(original_size, '\0');
                file.read(&fallback_result[0], original_size);
                if (file && file.gcount() == static_cast<std::streamsize>(original_size)) {
                    std::cerr << "Fallback to uncompressed read successful" << std::endl;
                    return fallback_result;
                }
            } catch (...) {
                // Ignore fallback errors
            }

            throw std::runtime_error(error_msg);
        }

        if (actual_decompressed_size != original_size) {
            throw std::runtime_error("Decompressed size mismatch: expected " +
                std::to_string(original_size) + ", got " + std::to_string(actual_decompressed_size));
        }

        return std::string(decompressed.data(), actual_decompressed_size);
    }
}
void Pager::test_zstd() {
    std::cout << "Testing Zstd compression..." << std::endl;
    
    std::string test_data = "ELVIS";
    
    std::cout << "Testing with data: '" << test_data << "'" << std::endl;
    
    // Test compression
    size_t compressed_size = ZSTD_compressBound(test_data.size());
    std::vector<char> compressed(compressed_size);

    size_t result = ZSTD_compress(
        compressed.data(), compressed_size,
        test_data.data(), test_data.size(),
        1
    );

    if (ZSTD_isError(result)) {
        std::string error_msg = "Zstd compression test failed: " + 
            std::string(ZSTD_getErrorName(result));
        throw std::runtime_error(error_msg);
    }

    // Test decompression
    std::vector<char> decompressed(test_data.size());
    size_t decompressed_size = ZSTD_decompress(
        decompressed.data(), test_data.size(),
        compressed.data(), result
    );

    if (ZSTD_isError(decompressed_size)) {
        std::string error_msg = "Zstd decompression test failed: " + 
            std::string(ZSTD_getErrorName(decompressed_size));
        throw std::runtime_error(error_msg);
    }

    // Verify data integrity
    std::string decompressed_str(decompressed.data(), decompressed_size);
    if (test_data != decompressed_str) {
        throw std::runtime_error("Data mismatch in test: expected '" + 
            test_data + "', got '" + decompressed_str + "'");
    }
    
    std::cout << "Zstd test passed successfully!" << std::endl;
    std::cout << "Zstd version: " << ZSTD_versionString() << std::endl;
    std::cout << "Original size: " << test_data.size() << std::endl;
    std::cout << "Compressed size: " << result << std::endl;
    std::cout << "Decompressed size: " << decompressed_size << std::endl;
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
    //return neon_supported;
    return false;
}

// Transactional memory helpers
bool FractalBPlusTree::begin_transaction() {
//#if HAS_HARDWARE_TM
    //return _xbegin() == _XBEGIN_STARTED;
//#else
    return false; // Fallback to mutex on non-x86 platforms
//#endif
}

void FractalBPlusTree::end_transaction() {
/*#if HAS_HARDWARE_TM
    _xend();
#endif*/
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
    /*if (has_neon_support()) {
        compress_keys_neon(dest, keys, num_keys);
    } else {*/
    compress_keys(dest, keys, num_keys);
}

void FractalBPlusTree::decompress_keys_optimized(int64_t* keys, const char* src, uint32_t num_keys) {
    /*if (has_neon_support()) {
        decompress_keys_neon(keys, src, num_keys);
    } else {*/
    decompress_keys(keys, src, num_keys);
}

void FractalBPlusTree::compress_node_optimized(Node* node) {
    /*if (has_neon_support()) {
        compress_node_neon(node);
    } else {*/
    compress_node(node);

}

void FractalBPlusTree::decompress_node_optimized(Node* node) {
    /*if (has_neon_support()) {
        decompress_node_neon(node);
    } else {*/
    decompress_node(node);
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
    // Create a copy of the node to avoid modifying the original
    Node node_copy = *node;
    
    if (node_copy.header.num_keys >= 10 && !(node_copy.header.num_keys & (1U << 31))) {
        compress_node_optimized(&node_copy);
    }
    
    {
        std::lock_guard<std::mutex> lock(version_mutex);
        versioned_nodes[transaction_id].push_back(node_copy);
    }
    
    // Write to buffer pool and WAL
    buffer_pool.write_page(pager, wal, page_id, &node_copy);
    
    //std::cout << "WRITE_NODE: Wrote page " << page_id << " with " << node_copy.header.num_keys 
              //<< " keys, compressed: " << ((node_copy.header.num_keys & (1U << 31)) ? "yes" : "no") 
              //<< std::endl;
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

void FractalBPlusTree::compress_node(Node* node) {
    // Don't compress already compressed nodes or very small nodes
    if (node->header.num_keys & (1U << 31) || node->header.num_keys < 10) {
        return;
    }

    char temp[BPTREE_PAGE_SIZE];
    memcpy(temp, node->data, BPTREE_PAGE_SIZE - sizeof(PageHeader));

    size_t compressed_size = ZSTD_compress(
        node->data, BPTREE_PAGE_SIZE - sizeof(PageHeader),
        temp, BPTREE_PAGE_SIZE - sizeof(PageHeader),
        1
    );

    if (ZSTD_isError(compressed_size)) {
        // If compression fails, restore original data
        memcpy(node->data, temp, BPTREE_PAGE_SIZE - sizeof(PageHeader));
        std::cerr << "Zstd node compression failed: " <<
            ZSTD_getErrorName(compressed_size) << ", keeping uncompressed" << std::endl;
    } else {
        node->header.num_keys |= (1U << 31);
    }
}

void FractalBPlusTree::decompress_node(Node* node) {
    if (!(node->header.num_keys & (1U << 31))) return;

    char temp[BPTREE_PAGE_SIZE];
    size_t decompressed_size = ZSTD_decompress(
        temp, BPTREE_PAGE_SIZE - sizeof(PageHeader),
        node->data, BPTREE_PAGE_SIZE - sizeof(PageHeader)
    );

    if (ZSTD_isError(decompressed_size)) {
        throw std::runtime_error("Zstd node decompression failed: " +
            std::string(ZSTD_getErrorName(decompressed_size)));
    }

    memcpy(node->data, temp, decompressed_size);
    node->header.num_keys &= ~(1U << 31);
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

        // Process messages in order (newest first)
        for (uint32_t i = 0; i < node->header.num_messages; ++i) {
            bool found = false;

            // Look for existing key-value pair
            for (uint32_t j = 0; j < node->header.num_keys; ++j) {
                if (kvs[j].key == messages[i].key) {
                    if (messages[i].type == MessageType::DELETE) {
                        // Remove the key-value pair
                        for (uint32_t k = j; k < node->header.num_keys - 1; ++k) {
                            kvs[k] = kvs[k + 1];
                        }
                        node->header.num_keys--;
                    } else if (messages[i].type == MessageType::UPDATE ||
                              messages[i].type == MessageType::INSERT) {
                        // Update the value
                        kvs[j].value_offset = messages[i].value_offset;
                        kvs[j].value_length = messages[i].value_length;
                    }
                    found = true;
                    break;
                }
            }

            // If not found and it's an INSERT/UPDATE, add new key-value pair
            if (!found && (messages[i].type == MessageType::INSERT ||
                          messages[i].type == MessageType::UPDATE)) {
                if (node->header.num_keys < BPTREE_ORDER) {
                    kvs[node->header.num_keys].key = messages[i].key;
                    kvs[node->header.num_keys].value_offset = messages[i].value_offset;
                    kvs[node->header.num_keys].value_length = messages[i].value_length;
                    node->header.num_keys++;
                }
            }
        }

        // Write back the updated key-value pairs
        memcpy(node->data + node->header.num_keys * MAX_KEY_PREFIX,
               kvs, sizeof(KeyValue) * node->header.num_keys);

        // Clear all messages
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
    memset(root.data, 0, sizeof(root.data));

    write_node(root_page_id.load(), &root, 0);
    //std::cout << "TREE_CREATE: Created new tree with root page " << root_page_id.load() << std::endl;

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

        // Traverse to the leaf node containing the key
        while (node->header.type == PageType::INTERNAL) {
            uint32_t child_index = find_child_index(node, key);
            uint32_t children[BPTREE_ORDER + 1];
            memcpy(children, node->data + node->header.num_keys * MAX_KEY_PREFIX,
                   sizeof(uint32_t) * (node->header.num_keys + 1));
            current_page_id = children[child_index];
            node = get_node(current_page_id);
        }

        decompress_node_optimized(node);

        // Check if we need to split due to too many messages
        if (is_node_full(node)) {
            split_node(current_page_id, node, node->header.parent_page_id, transaction_id);
            node = get_node(current_page_id);
            decompress_node_optimized(node);
        }

        Message msg;
        msg.type = MessageType::UPDATE;
        msg.key = key;
        msg.value_offset = value_offset;
        msg.value_length = value_length;
        msg.version = transaction_id;

        // Get existing messages
        Message messages[MAX_MESSAGES + 1];
        memcpy(messages, node->data + node->header.num_keys * MAX_KEY_PREFIX +
               sizeof(KeyValue) * node->header.num_keys, sizeof(Message) * node->header.num_messages);

        // Add new message
        messages[node->header.num_messages++] = msg;
        memcpy(node->data + node->header.num_keys * MAX_KEY_PREFIX +
               sizeof(KeyValue) * node->header.num_keys, messages, sizeof(Message) * node->header.num_messages);

        compress_node_optimized(node);
        write_node(current_page_id, node, transaction_id);

        // Update secondary indexes if any
        for (const auto& sk : secondary_keys) {
            update_secondary_index(sk.first, key, sk.second, transaction_id, MessageType::UPDATE);
        }

        // Force flush to ensure update is applied immediately
        flush_messages(node, current_page_id, transaction_id);

        //std::cout << "FractalBPlusTree UPDATE: key=" << key << ", value_size=" << value.size()
                  //<< ", txn=" << transaction_id << std::endl;

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
            try {
                return pager.read_data_block(messages[i].value_offset, messages[i].value_length);
            } catch (const std::exception& e) {
                std::cerr << "Error reading data block for message (key=" << key
                          << ", offset=" << messages[i].value_offset
                          << "): " << e.what() << std::endl;
                // Continue to check stored key-value pairs
                break;
            }
        }
    }

    // Check stored key-value pairs
    KeyValue kvs[BPTREE_ORDER];
    memcpy(kvs, node->data + node->header.num_keys * MAX_KEY_PREFIX,
           sizeof(KeyValue) * node->header.num_keys);

    for (uint32_t i = 0; i < node->header.num_keys; ++i) {
        if (kvs[i].key == key) {
            try {
                return pager.read_data_block(kvs[i].value_offset, kvs[i].value_length);
            } catch (const std::exception& e) {
                std::cerr << "Error reading data block for KV (key=" << key
                          << ", offset=" << kvs[i].value_offset
                          << "): " << e.what() << std::endl;
                return "";
            }
        }
    }

    return "";
}

std::vector<std::pair<int64_t, std::string>> FractalBPlusTree::select_range(int64_t start_key,int64_t end_key,
uint64_t transaction_id) {
    //std::cout << "SELECT_RANGE: Querying range [" << start_key << ", " << end_key << "] with transaction ID " << transaction_id << std::endl;
    
    uint32_t current_page_id = root_page_id.load();
    //std::cout << "SELECT_RANGE: Starting at root page ID " << current_page_id << std::endl;
    
    Node* node = get_node(current_page_id);
    decompress_node_optimized(node);
    
    //std::cout << "SELECT_RANGE: Root node type: " << (node->header.type == PageType::LEAF ? "LEAF" : "INTERNAL") << ", num_keys: " << node->header.num_keys << std::endl;
    std::vector<std::pair<int64_t, std::string>> result;

    //uint32_t current_page_id = root_page_id.load();
    //Node* node = get_node(current_page_id);

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
    if (data.empty()) {
        //std::cout << "BULK_LOAD: No data to load" << std::endl;
        return;
    }
    
    std::sort(data.begin(), data.end());
    //std::cout << "BULK_LOAD: Starting bulk load of " << data.size() << " rows" << std::endl;

    uint32_t current_page_id = root_page_id.load();
    Node* node = get_node(current_page_id);
    decompress_node_optimized(node);
    
    for (size_t i = 0; i < data.size(); i++) {
        const auto& kv = data[i];
        
        //std::cout << "BULK_LOAD: Processing row " << (i+1) << "/" << data.size() 
                  //<< ", key: " << kv.first << std::endl;
        
        if (is_node_full(node)) {
            //std::cout << "BULK_LOAD: Node full, splitting" << std::endl;
            split_node(current_page_id, node, node->header.parent_page_id, transaction_id);
            current_page_id = root_page_id.load();
            node = get_node(current_page_id);
            decompress_node_optimized(node);
        }
        
        // Process the key-value pair
        uint64_t value_offset = pager.write_data_block(kv.second);
        
        // Get existing keys and values
        int64_t keys[BPTREE_ORDER];
        KeyValue kvs[BPTREE_ORDER];
        
        if (node->header.num_keys > 0) {
            decompress_keys_optimized(keys, node->data, node->header.num_keys);
            memcpy(kvs, node->data + node->header.num_keys * MAX_KEY_PREFIX,
                   sizeof(KeyValue) * node->header.num_keys);
        }
        
        // Add new key-value pair
        keys[node->header.num_keys] = kv.first;
        kvs[node->header.num_keys].key = kv.first;
        kvs[node->header.num_keys].value_offset = value_offset;
        kvs[node->header.num_keys].value_length = kv.second.size();
        
        node->header.num_keys++;
        
        // Write back compressed keys and values
        compress_keys_optimized(node->data, keys, node->header.num_keys);
        memcpy(node->data + node->header.num_keys * MAX_KEY_PREFIX,
               kvs, sizeof(KeyValue) * node->header.num_keys);
        
        // Write the node
        write_node(current_page_id, node, transaction_id);
        
        // Refresh node pointer for next iteration
        if (i < data.size() - 1) {
            node = get_node(current_page_id);
            decompress_node_optimized(node);
        }
    }
    
    //std::cout << "BULK_LOAD: Completed successfully, inserted " << data.size() << " rows" << std::endl;
}
void FractalBPlusTree::checkpoint() {
    wal.checkpoint(pager, root_page_id.load());
}
