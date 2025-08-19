#include "storagemanager.h"
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <system_error>

// Page Implementation
Page::Page(uint32_t id) noexcept : page_id(id), dirty(false) {
    data.fill(0);
}

// Pager Implementation
void Pager::ensure_file_open() {
    if (!file.is_open()) {
        file.open(filename, std::ios::binary | std::ios::in | std::ios::out);
        if (!file) {
            file.open(filename, std::ios::binary | std::ios::out);
            file.close();
            file.open(filename, std::ios::binary | std::ios::in | std::ios::out);
            if (!file) {
                throw FileIOException("Failed to open database file: " + filename);
            }
        }
        load_metadata();
    }
}

bool Pager::safe_seek(uint32_t page_id) noexcept {
    try {
        file.seekg(page_id * DB_PAGE_SIZE);
        file.seekp(page_id * DB_PAGE_SIZE);
        return !file.fail();
    } catch (...) {
        return false;
    }
}
void Pager::free_page(uint32_t page_id) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    
    // 1. Validate page_id
    if (page_id == 0 || page_id >= next_page_id) {
        throw std::runtime_error("Attempt to free invalid page ID: " + 
                               std::to_string(page_id));
    }

    // 2. Remove from cache if present
    page_cache.erase(page_id);

    // 3. Add to free list for reuse
    free_pages.push_back(page_id);

    // 4. Optionally zero out the page on disk to prevent data leakage
    std::array<uint8_t, DB_PAGE_SIZE> empty_page;
    empty_page.fill(0);
    write_page_to_disk(page_id, empty_page);

    // 5. Update metadata if needed (e.g., tracking free pages)
    save_metadata();
}

Pager::Pager(const std::string& db_file) : filename(db_file) {
    try{
        ensure_file_open();
	std::cerr<<"Successfully opened databasefile: "<<filename<<"\n";
    }catch(const std::exception& e){
	    std::cerr<<"Failed to initialize page: "<<e.what()<<"\n";
    }
}

Pager::~Pager() noexcept {
    try {
        flush_all();
        if (file.is_open()) {
            file.close();
        }
    } catch (...) {}
}

std::shared_ptr<Page> Pager::get_page(uint32_t page_id) {
    if (page_id >= next_page_id) {
        throw StorageException("Invalid page ID: " + std::to_string(page_id) +
                              " (max allocated: " + std::to_string(next_page_id - 1) + ")");
    }
    std::lock_guard<std::mutex> lock(cache_mutex);
    if (auto it = page_cache.find(page_id); it != page_cache.end()) {
        return it->second;
    }
    auto page = std::make_shared<Page>(page_id);
    if (!read_page_from_disk(page_id, page->data)) {
        throw FileIOException("Failed to read page " + std::to_string(page_id));
    }
    page_cache[page_id] = page;
    return page;
}

uint32_t Pager::allocate_page() {
    std::lock_guard<std::mutex> lock(cache_mutex);
    uint32_t new_id = next_page_id++;
    try {
        auto page = std::make_shared<Page>(new_id);
        page_cache[new_id] = page;
        save_metadata();
        return new_id;
    } catch (...) {
        next_page_id--;
        throw;
    }
}

void Pager::mark_dirty(uint32_t page_id) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    if (auto it = page_cache.find(page_id); it != page_cache.end()) {
        it->second->dirty = true;
    }
}

void Pager::flush_all() noexcept {
    std::lock_guard<std::mutex> lock(cache_mutex);
    try {
        for (auto& [page_id, page] : page_cache) {
            if (page->dirty) {
                if (!write_page_to_disk(page_id, page->data)) {
                    continue;
                }
                page->dirty = false;
            }
        }
        save_metadata();
    } catch (...) {}
}

bool Pager::read_page_from_disk(uint32_t page_id, std::array<uint8_t, DB_PAGE_SIZE>& buffer) noexcept {
    try {
        ensure_file_open();
        if (!safe_seek(page_id)) {
            return false;
        }
        file.read(reinterpret_cast<char*>(buffer.data()), DB_PAGE_SIZE);
        return file.gcount() == DB_PAGE_SIZE;
    } catch (...) {
        return false;
    }
}

bool Pager::write_page_to_disk(uint32_t page_id, const std::array<uint8_t, DB_PAGE_SIZE>& buffer) noexcept {
    try {
        ensure_file_open();
        if (!safe_seek(page_id)) {
            return false;
        }
        file.write(reinterpret_cast<const char*>(buffer.data()), DB_PAGE_SIZE);
        file.flush();
        return !file.fail();
    } catch (...) {
        return false;
    }
}

void Pager::load_metadata() {
    std::array<uint8_t, DB_PAGE_SIZE> buffer;
    buffer.fill(0);
    if (read_page_from_disk(0, buffer)) {
        next_page_id = *reinterpret_cast<uint32_t*>(buffer.data());
        if (next_page_id < 1) {
            next_page_id = 1;
            *reinterpret_cast<uint32_t*>(buffer.data()) = next_page_id;
            write_page_to_disk(0, buffer);
        }
    } else {
        next_page_id = 1;
        *reinterpret_cast<uint32_t*>(buffer.data()) = next_page_id;
        write_page_to_disk(0, buffer);
    }
    auto page = std::make_shared<Page>(0);
    page->data = buffer;
    page_cache[0] = page;
}

void Pager::save_metadata() noexcept {
    std::array<uint8_t, DB_PAGE_SIZE> buffer;
    buffer.fill(0);
    *reinterpret_cast<uint32_t*>(buffer.data()) = next_page_id;
    write_page_to_disk(0, buffer);
}

// BufferPool Implementation
void BufferPool::evict_page() noexcept {
    if (lru_list.empty()) return;
    uint32_t victim = lru_list.back();
    lru_list.pop_back();
    lru_map.erase(victim);
    if (auto it = pool.find(victim); it != pool.end()) {
        if (it->second->dirty) {
            try {
                pager.write_page_to_disk(victim, it->second->data);
            } catch (...) {}
        }
        pool.erase(victim);
    }
}

BufferPool::BufferPool(Pager& pager, size_t capacity) noexcept
    : pager(pager), capacity(capacity > 0 ? capacity : DEFAULT_BUFFER_POOL_CAPACITY) {}

std::shared_ptr<Page> BufferPool::fetch_page(uint32_t page_id) {
    std::lock_guard<std::mutex> lock(mutex);
    if (auto it = pool.find(page_id); it != pool.end()) {
        lru_list.erase(lru_map[page_id]);
        lru_list.push_front(page_id);
        lru_map[page_id] = lru_list.begin();
        return it->second;
    }
    if (pool.size() >= capacity) {
        evict_page();
    }
    auto page = pager.get_page(page_id);
    pool[page_id] = page;
    lru_list.push_front(page_id);
    lru_map[page_id] = lru_list.begin();
    return page;
}

void BufferPool::flush_all() noexcept {
    std::lock_guard<std::mutex> lock(mutex);
    try {
        for (auto& [page_id, page] : pool) {
            if (page->dirty) {
                pager.write_page_to_disk(page_id, page->data);
                page->dirty = false;
            }
        }
    } catch (...) {}
}

// WriteAheadLog Implementation
void WriteAheadLog::ensure_file_open() {
    if (!log_file.is_open()) {
        log_file.open(log_file_path, std::ios::binary | std::ios::app | std::ios::in | std::ios::out);
        if (!log_file) {
            log_file.open(log_file_path, std::ios::binary | std::ios::out);
            log_file.close();
            log_file.open(log_file_path, std::ios::binary | std::ios::app | std::ios::in | std::ios::out);
            if (!log_file) {
                throw FileIOException("Failed to open WAL file: " + log_file_path);
            }
        }
    }
}

WriteAheadLog::WriteAheadLog(const std::string& path,BufferPool& buffer_pool,Pager& pager) : log_file_path(path),buffer_pool(buffer_pool),pager(pager) {
    ensure_file_open();
}

WriteAheadLog::~WriteAheadLog() noexcept {
    try {
        if (log_file.is_open()) {
            log_file.close();
        }
    } catch (...) {}
}

void WriteAheadLog::log_transaction_begin(uint64_t tx_id) {
    std::lock_guard<std::mutex> lock(mutex);
    ensure_file_open();
    log_file << "BEGIN," << tx_id << "\n";
    log_file.flush();
    if (log_file.fail()) {
        throw FileIOException("Failed to write transaction begin to WAL");
    }
}

void WriteAheadLog::log_page_write(uint64_t tx_id, uint32_t page_id,
                                   const std::array<uint8_t, DB_PAGE_SIZE>& old_data,
                                   const std::array<uint8_t, DB_PAGE_SIZE>& new_data) {
    std::lock_guard<std::mutex> lock(mutex);
    ensure_file_open();
    log_file << "UPDATE," << tx_id << "," << page_id << ",";
    log_file.write(reinterpret_cast<const char*>(old_data.data()), DB_PAGE_SIZE);
    log_file.write(reinterpret_cast<const char*>(new_data.data()), DB_PAGE_SIZE);
    log_file << "\n";
    log_file.flush();
    if (log_file.fail()) {
        throw FileIOException("Failed to write page update to WAL");
    }
}

void WriteAheadLog::log_transaction_commit(uint64_t tx_id) {
    std::lock_guard<std::mutex> lock(mutex);
    ensure_file_open();
    log_file << "COMMIT," << tx_id << "\n";
    log_file.flush();
    if (log_file.fail()) {
        throw FileIOException("Failed to write transaction commit to WAL");
    }
}

void WriteAheadLog::log_check_point() {
    std::lock_guard<std::mutex> lock(mutex);
    ensure_file_open();
    log_file << "CHECKPOINT\n";
    log_file.flush();
    if (log_file.fail()) {
        throw FileIOException("Failed to write checkpoint to WAL");
    }
}
void WriteAheadLog::reset_log(){
    log_file.close();
    log_file.open(log_file_path, std::ios::binary | std::ios::trunc | std::ios::out);
    log_file.close();
    log_file.open(log_file_path, std::ios::binary | std::ios::app | std::ios::in | std::ios::out);
}
void WriteAheadLog::recover() {
    std::lock_guard<std::mutex> lock(mutex);
    try {
        ensure_file_open();
        log_file.seekg(0);
        
        if (log_file.peek() == std::ifstream::traits_type::eof()) {
            return;
        }
        
        std::unordered_map<uint64_t, std::vector<std::function<void()>>> tx_ops;
        std::string line;
        
        while (std::getline(log_file, line)) {
            std::istringstream iss(line);
            std::string cmd;
            getline(iss, cmd, ',');
            
            if (cmd == "BEGIN") {
                uint64_t tx_id;
                iss >> tx_id;
                tx_ops[tx_id] = {};
            } 
            else if (cmd == "UPDATE") {
                uint64_t tx_id;
                uint32_t page_id;
                iss >> tx_id >> page_id;
                
                // Read old and new data
                std::array<uint8_t, DB_PAGE_SIZE> old_data, new_data;
                log_file.read(reinterpret_cast<char*>(old_data.data()), DB_PAGE_SIZE);
                log_file.read(reinterpret_cast<char*>(new_data.data()), DB_PAGE_SIZE);
                
                tx_ops[tx_id].push_back([this, page_id, new_data]() {
                    // Reapply the update
                    auto page = buffer_pool.fetch_page(page_id);
                    if (page) {
                        std::copy(new_data.begin(), new_data.end(), page->data.begin());
                        pager.mark_dirty(page_id);
                    }
                });
            }
            else if (cmd == "COMMIT") {
                uint64_t tx_id;
                iss >> tx_id;
                
                // Apply all operations for this transaction
                for (auto& op : tx_ops[tx_id]) {
                    op();
                }
                tx_ops.erase(tx_id);
            }
        }
        
        // Clear the log after recovery
        reset_log();
        
    } catch (...) {
        std::cerr << "WAL recovery failed\n";
    }
}

// BPlusTree Implementation
void BPlusTree::validate_node(const Node& node) const {
    if (node.num_keys > 2 * order - 1) {
        throw IntegrityViolation("Invalid node: too many keys");
    }
    if (!node.is_leaf && node.children.size() != node.num_keys + 1) {
        throw IntegrityViolation("Invalid node: children count mismatch");
    }
    if (node.is_leaf && node.values.size() != node.num_keys) {
        throw IntegrityViolation("Invalid node: values count mismatch");
    }
    if (!std::is_sorted(node.keys.begin(), node.keys.begin() + node.num_keys)) {
        throw IntegrityViolation("Invalid node: keys not sorted");
    }
}

BPlusTree::BPlusTree(Pager& pager, BufferPool& buffer_pool, uint32_t order)
    : pager(pager), buffer_pool(buffer_pool), order(order > 2 ? order : DEFAULT_BPTREE_ORDER) {
    try {
        // Ensure metadata page exists and is initialized
        if (pager.get_next_page_id() == 0) {
            pager.allocate_page(); // Page 0 for metadata
            pager.allocate_page(); // Page 1 for root
        }

        auto meta_page = buffer_pool.fetch_page(0);
        if (!meta_page) {
            throw std::runtime_error("Failed to fetch metadata page");
        }

        // Check if metadata is initialized (first 8 bytes: next_page_id and root_page_id)
        if (pager.get_next_page_id() <= 2) {
            root_page_id = 1; // Default root page
            *reinterpret_cast<uint32_t*>(meta_page->data.data() + sizeof(uint32_t)) = root_page_id;
            pager.mark_dirty(0);
            
            // Initialize empty root node
            auto root_page = buffer_pool.fetch_page(root_page_id);
            Node root;
            root.is_leaf = true;
            root.num_keys = 0;
            serialize_node(root, root_page->data);
            pager.mark_dirty(root_page_id);
            return;
        }

        // Read root page ID from metadata
        root_page_id = *reinterpret_cast<uint32_t*>(meta_page->data.data() + sizeof(uint32_t));
        
        // Validate root page exists
        if (root_page_id == 0 || root_page_id >= pager.get_next_page_id()) {
            throw std::runtime_error("Invalid root page ID in metadata");
        }

        // Verify root page contains valid node
        auto root_page = buffer_pool.fetch_page(root_page_id);
        if (!root_page) {
            throw std::runtime_error("Failed to fetch root page");
        }
        
        Node test_node;
        deserialize_node(root_page->data, test_node);

    } catch (const std::exception& e) {
        // If we get here, the index is corrupted - recreate it empty
        std::cerr << "Warning: B+Tree initialization failed: " << e.what() 
                  << ". Recreating empty index.\n";
        
        root_page_id = pager.allocate_page();
        auto meta_page = buffer_pool.fetch_page(0);
        *reinterpret_cast<uint32_t*>(meta_page->data.data() + sizeof(uint32_t)) = root_page_id;
        pager.mark_dirty(0);
        
        auto root_page = buffer_pool.fetch_page(root_page_id);
        Node root;
        root.is_leaf = true;
        root.num_keys = 0;
        serialize_node(root, root_page->data);
        pager.mark_dirty(root_page_id);
    }
}

BPlusTree::BPlusTree(Pager& pager, BufferPool& buffer_pool, uint32_t root_page_id,bool existing_key)
    : pager(pager), buffer_pool(buffer_pool), order(DEFAULT_BPTREE_ORDER), root_page_id(root_page_id) {
    if (root_page_id >= pager.get_next_page_id()) {
        throw std::runtime_error("Invalid root page ID: " + std::to_string(root_page_id));
    }
}

void BPlusTree::insert(uint32_t key, const std::vector<uint8_t>& value) {
    try {
        if (root_page_id >= pager.get_next_page_id()) {
            throw std::runtime_error("Invalid root page ID: " + std::to_string(root_page_id));
        }
        auto root_page = buffer_pool.fetch_page(root_page_id);
        if (!root_page) {
            throw std::runtime_error("Failed to fetch root page: " + std::to_string(root_page_id));
        }
        Node root;
        deserialize_node(root_page->data, root);
        if (root.num_keys == 2 * order - 1) {
            uint32_t new_root_id = pager.allocate_page();
            auto new_root_page = buffer_pool.fetch_page(new_root_id);
            if (!new_root_page) {
                throw std::runtime_error("Failed to fetch new root page");
            }
            Node new_root;
            new_root.is_leaf = false;
            new_root.num_keys = 0;
            new_root.children.push_back(root_page_id);
            split_child(new_root, 0, root);
            insert_non_full(new_root, key, value);
            root_page_id = new_root_id;
            serialize_node(new_root, new_root_page->data);
            pager.mark_dirty(new_root_id);
            auto meta_page = buffer_pool.fetch_page(0);
            if (!meta_page) {
                throw std::runtime_error("Failed to fetch metadata page");
            }
            *reinterpret_cast<uint32_t*>(meta_page->data.data() + sizeof(uint32_t)) = root_page_id;
            pager.mark_dirty(0);
        } else {
            insert_non_full(root, key, value);
            serialize_node(root, root_page->data);
            pager.mark_dirty(root_page_id);
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to insert into BPlusTree: " + std::string(e.what()));
    }
}
std::vector<uint32_t> BPlusTree::getAllKeys() const{
	std::vector<uint32_t> keys;
	collectKeys(root_page_id,keys);
	return keys;
}
void BPlusTree::collectKeys(uint32_t page_id,std::vector<uint32_t>& keys) const{
	auto page=buffer_pool.fetch_page(page_id);
	Node node;
	deserialize_node(page->data,node);
	if(node.is_leaf){
		keys.insert(keys.end(),node.keys.begin(),node.keys.begin()+node.num_keys);
	}else{
		for(uint32_t i=0;i<=node.num_keys;i++){
			collectKeys(node.children[i],keys);
		}
	}
}
void BPlusTree::update(uint32_t old_key, uint32_t new_key, const std::vector<uint8_t>& value) {
    std::lock_guard<std::mutex> lock(tree_mutex);

    // Remove old entry
    if (old_key != new_key) {
        remove(old_key);
    }

    // Insert new entry
    auto page = buffer_pool.fetch_page(root_page_id);
    Node root;
    deserialize_node(page->data, root);

    if (root.num_keys == 2 * order - 1) {
        // Handle root split
        uint32_t new_root_id = pager.allocate_page();
        auto new_root_page = buffer_pool.fetch_page(new_root_id);
        Node new_root;
        new_root.is_leaf = false;
        new_root.children.push_back(root_page_id);
        split_child(new_root, 0, root);

        root_page_id = new_root_id;
        insert_non_full(new_root, new_key, value);
        serialize_node(new_root, new_root_page->data);
        pager.mark_dirty(new_root_id);

        // Update metadata
        auto meta_page = buffer_pool.fetch_page(0);
        *reinterpret_cast<uint32_t*>(meta_page->data.data() + sizeof(uint32_t)) = root_page_id;
        pager.mark_dirty(0);
    } else {
        insert_non_full(root, new_key, value);
        serialize_node(root, page->data);
        pager.mark_dirty(root_page_id);
    }
}
void BPlusTree::remove(uint32_t key) {
    std::lock_guard<std::mutex> lock(tree_mutex);

    auto root_page = buffer_pool.fetch_page(root_page_id);
    Node root;
    deserialize_node(root_page->data, root);

    remove_from_node(root, key);

    // If root becomes empty and has children, make its first child the new root
    if (!root.is_leaf && root.num_keys == 0) {
        uint32_t old_root = root_page_id;
        root_page_id = root.children[0];

        // Update metadata
        auto meta_page = buffer_pool.fetch_page(0);
        *reinterpret_cast<uint32_t*>(meta_page->data.data() + sizeof(uint32_t)) = root_page_id;
        pager.mark_dirty(0);

        // Free old root page
        pager.free_page(old_root);
    } else {
        serialize_node(root, root_page->data);
        pager.mark_dirty(root_page_id);
    }
}
std::vector<uint8_t> BPlusTree::search(uint32_t key) {
    try {
        if (root_page_id >= pager.get_next_page_id()) {
            throw std::runtime_error("Invalid root page ID: " + std::to_string(root_page_id));
        }
        auto current_page = buffer_pool.fetch_page(root_page_id);
        if (!current_page) {
            throw std::runtime_error("Failed to fetch root page: " + std::to_string(root_page_id));
        }
        Node node;
        deserialize_node(current_page->data, node);
        while (!node.is_leaf) {
            auto it = std::lower_bound(node.keys.begin(), node.keys.begin() + node.num_keys, key);
            uint32_t index = it - node.keys.begin();
            if (index >= node.children.size()) {
                throw std::runtime_error("Invalid child index in BPlusTree search");
            }
            uint32_t child_page_id = node.children[index];
            if (child_page_id >= pager.get_next_page_id()) {
                throw std::runtime_error("Invalid child page ID: " + std::to_string(child_page_id));
            }
            current_page = buffer_pool.fetch_page(child_page_id);
            if (!current_page) {
                throw std::runtime_error("Failed to fetch child page: " + std::to_string(child_page_id));
            }
            deserialize_node(current_page->data, node);
        }
        auto it = std::lower_bound(node.keys.begin(), node.keys.begin() + node.num_keys, key);
        if (it != node.keys.begin() + node.num_keys && *it == key) {
            uint32_t index = it - node.keys.begin();
            return node.values[index];
        }
        return {};
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to search BPlusTree: " + std::string(e.what()));
    }
}

void BPlusTree::insert_non_full(Node& node, uint32_t key, const std::vector<uint8_t>& value) {
    validate_node(node);
    int i = node.num_keys - 1;
    if (node.is_leaf) {
        node.keys.resize(node.num_keys + 1);
        node.values.resize(node.num_keys + 1);
        while (i >= 0 && key < node.keys[i]) {
            node.keys[i + 1] = node.keys[i];
            node.values[i + 1] = std::move(node.values[i]);
            i--;
        }
        node.keys[i + 1] = key;
        node.values[i + 1] = value;
        node.num_keys++;
    } else {
        while (i >= 0 && key < node.keys[i]) {
            i--;
        }
        i++;
        if (i >= static_cast<int>(node.children.size())) {
            throw std::runtime_error("Invalid child index in insert_non_full");
        }
        auto child_page = buffer_pool.fetch_page(node.children[i]);
        if (!child_page) {
            throw std::runtime_error("Failed to fetch child page: " + std::to_string(node.children[i]));
        }
        Node child;
        deserialize_node(child_page->data, child);
        if (child.num_keys == 2 * order - 1) {
            split_child(node, i, child);
            if (key > node.keys[i]) {
                i++;
            }
            child_page = buffer_pool.fetch_page(node.children[i]);
            if (!child_page) {
                throw std::runtime_error("Failed to fetch child page after split: " + std::to_string(node.children[i]));
            }
            deserialize_node(child_page->data, child);
        }
        insert_non_full(child, key, value);
        serialize_node(child, child_page->data);
        pager.mark_dirty(node.children[i]);
    }
}
void BPlusTree::remove_from_node(Node& node, uint32_t key) {
    // 1. Find the key position
    auto it = std::lower_bound(node.keys.begin(), node.keys.begin() + node.num_keys, key);
    int idx = it - node.keys.begin();
    
    // 2. Key found in current node
    if (idx < node.num_keys && node.keys[idx] == key) {
        if (node.is_leaf) {
            // Case 1: Simple leaf node deletion
            node.keys.erase(node.keys.begin() + idx);
            node.values.erase(node.values.begin() + idx);
            node.num_keys--;
        } else {
            // Case 2: Internal node deletion
            uint32_t left_child_id = node.children[idx];
            uint32_t right_child_id = node.children[idx + 1];
            
            auto left_page = buffer_pool.fetch_page(left_child_id);
            auto right_page = buffer_pool.fetch_page(right_child_id);
            Node left_child, right_child;
            deserialize_node(left_page->data, left_child);
            deserialize_node(right_page->data, right_child);
            
            if (left_child.num_keys >= order) {
                // Case 2a: Borrow from left sibling
                uint32_t predecessor_key = left_child.keys[left_child.num_keys - 1];
                std::vector<uint8_t> predecessor_value;
                if (left_child.is_leaf) {
                    predecessor_value = left_child.values[left_child.num_keys - 1];
                }
                
                // Remove from left child
                left_child.keys.pop_back();
                if (left_child.is_leaf) left_child.values.pop_back();
                left_child.num_keys--;
                
                // Update current node
                node.keys[idx] = predecessor_key;
                
                // Serialize changes
                serialize_node(left_child, left_page->data);
                serialize_node(node, buffer_pool.fetch_page(node.children[idx])->data);
                pager.mark_dirty(left_child_id);
                pager.mark_dirty(node.children[idx]);
            } 
            else if (right_child.num_keys >= order) {
                // Case 2b: Borrow from right sibling
                uint32_t successor_key = right_child.keys[0];
                std::vector<uint8_t> successor_value;
                if (right_child.is_leaf) {
                    successor_value = right_child.values[0];
                }
                
                // Remove from right child
                right_child.keys.erase(right_child.keys.begin());
                if (right_child.is_leaf) right_child.values.erase(right_child.values.begin());
                right_child.num_keys--;
                
                // Update current node
                node.keys[idx] = successor_key;
                
                // Serialize changes
                serialize_node(right_child, right_page->data);
                serialize_node(node, buffer_pool.fetch_page(node.children[idx + 1])->data);
                pager.mark_dirty(right_child_id);
                pager.mark_dirty(node.children[idx + 1]);
            } 
            else {
                // Case 2c: Merge children
                // Move all elements from right to left
                left_child.keys.insert(left_child.keys.end(), 
                                      right_child.keys.begin(), 
                                      right_child.keys.begin() + right_child.num_keys);
                if (left_child.is_leaf) {
                    left_child.values.insert(left_child.values.end(),
                                           right_child.values.begin(),
                                           right_child.values.begin() + right_child.num_keys);
                } else {
                    left_child.children.insert(left_child.children.end(),
                                             right_child.children.begin(),
                                             right_child.children.begin() + right_child.num_keys + 1);
                }
                left_child.num_keys += right_child.num_keys;
                
                // Remove the key and right child pointer from current node
                node.keys.erase(node.keys.begin() + idx);
                node.children.erase(node.children.begin() + idx + 1);
                node.num_keys--;
                
                // Serialize changes
                serialize_node(left_child, left_page->data);
                serialize_node(node, buffer_pool.fetch_page(node.children[idx])->data);
                pager.mark_dirty(left_child_id);
                pager.mark_dirty(node.children[idx]);
                
                // Free the right child page
                pager.free_page(right_child_id);
            }
        }
    } 
    else if (!node.is_leaf) {
        // 3. Key not found - recurse into appropriate child
        uint32_t child_id = node.children[idx];
        auto child_page = buffer_pool.fetch_page(child_id);
        Node child;
        deserialize_node(child_page->data, child);
        
        // Check for underflow before recursing
        if (child.num_keys < order - 1) {
            handle_underflow(node, idx);
            // Need to re-fetch as tree structure may have changed
            child_page = buffer_pool.fetch_page(node.children[idx]);
            deserialize_node(child_page->data, child);
        }
        
        remove_from_node(child, key);
        serialize_node(child, child_page->data);
        pager.mark_dirty(child_id);
    }
}

void BPlusTree::handle_underflow(Node& parent, uint32_t child_idx) {
    uint32_t child_id = parent.children[child_idx];
    auto child_page = buffer_pool.fetch_page(child_id);
    Node child;
    deserialize_node(child_page->data, child);
    
    // Try to borrow from left sibling
    if (child_idx > 0) {
        uint32_t left_sib_id = parent.children[child_idx - 1];
        auto left_sib_page = buffer_pool.fetch_page(left_sib_id);
        Node left_sib;
        deserialize_node(left_sib_page->data, left_sib);
        
        if (left_sib.num_keys >= order) {
            // Rotate right
            if (child.is_leaf) {
                child.keys.insert(child.keys.begin(), left_sib.keys.back());
                child.values.insert(child.values.begin(), left_sib.values.back());
                parent.keys[child_idx - 1] = left_sib.keys.back();
            } else {
                child.keys.insert(child.keys.begin(), parent.keys[child_idx - 1]);
                child.children.insert(child.children.begin(), left_sib.children.back());
                parent.keys[child_idx - 1] = left_sib.keys.back();
            }
            
            left_sib.keys.pop_back();
            if (left_sib.is_leaf) left_sib.values.pop_back();
            else left_sib.children.pop_back();
            left_sib.num_keys--;
            child.num_keys++;
            
            serialize_node(left_sib, left_sib_page->data);
            serialize_node(child, child_page->data);
            pager.mark_dirty(left_sib_id);
            pager.mark_dirty(child_id);
            return;
        }
    }
    
    // Try to borrow from right sibling
    if (child_idx < parent.num_keys) {
        uint32_t right_sib_id = parent.children[child_idx + 1];
        auto right_sib_page = buffer_pool.fetch_page(right_sib_id);
        Node right_sib;
        deserialize_node(right_sib_page->data, right_sib);
        
        if (right_sib.num_keys >= order) {
            // Rotate left
            if (child.is_leaf) {
                child.keys.push_back(right_sib.keys.front());
                child.values.push_back(right_sib.values.front());
                parent.keys[child_idx] = right_sib.keys[1];
            } else {
                child.keys.push_back(parent.keys[child_idx]);
                child.children.push_back(right_sib.children.front());
                parent.keys[child_idx] = right_sib.keys.front();
            }
            
            right_sib.keys.erase(right_sib.keys.begin());
            if (right_sib.is_leaf) right_sib.values.erase(right_sib.values.begin());
            else right_sib.children.erase(right_sib.children.begin());
            right_sib.num_keys--;
            child.num_keys++;
            
            serialize_node(right_sib, right_sib_page->data);
            serialize_node(child, child_page->data);
            pager.mark_dirty(right_sib_id);
            pager.mark_dirty(child_id);
            return;
        }
    }
    
    // Must merge with a sibling
    if (child_idx > 0) {
        // Merge with left sibling
        uint32_t left_sib_id = parent.children[child_idx - 1];
        auto left_sib_page = buffer_pool.fetch_page(left_sib_id);
        Node left_sib;
        deserialize_node(left_sib_page->data, left_sib);
        
        // Move parent key down
        if (child.is_leaf) {
            left_sib.keys.insert(left_sib.keys.end(), child.keys.begin(), child.keys.end());
            left_sib.values.insert(left_sib.values.end(), child.values.begin(), child.values.end());
        } else {
            left_sib.keys.push_back(parent.keys[child_idx - 1]);
            left_sib.keys.insert(left_sib.keys.end(), child.keys.begin(), child.keys.end());
            left_sib.children.insert(left_sib.children.end(), child.children.begin(), child.children.end());
        }
        left_sib.num_keys += child.num_keys + (child.is_leaf ? 0 : 1);
        
        // Remove from parent
        parent.keys.erase(parent.keys.begin() + child_idx - 1);
        parent.children.erase(parent.children.begin() + child_idx);
        parent.num_keys--;
        
        serialize_node(left_sib, left_sib_page->data);
        serialize_node(parent, buffer_pool.fetch_page(parent.children[child_idx - 1])->data);
        pager.mark_dirty(left_sib_id);
        pager.mark_dirty(parent.children[child_idx - 1]);
        
        // Free the child page
        pager.free_page(child_id);
    } else {
        // Merge with right sibling
        uint32_t right_sib_id = parent.children[child_idx + 1];
        auto right_sib_page = buffer_pool.fetch_page(right_sib_id);
        Node right_sib;
        deserialize_node(right_sib_page->data, right_sib);
        
        // Move parent key down
        if (child.is_leaf) {
            child.keys.insert(child.keys.end(), right_sib.keys.begin(), right_sib.keys.end());
            child.values.insert(child.values.end(), right_sib.values.begin(), right_sib.values.end());
        } else {
            child.keys.push_back(parent.keys[child_idx]);
            child.keys.insert(child.keys.end(), right_sib.keys.begin(), right_sib.keys.end());
            child.children.insert(child.children.end(), right_sib.children.begin(), right_sib.children.end());
        }
        child.num_keys += right_sib.num_keys + (child.is_leaf ? 0 : 1);
        
        // Remove from parent
        parent.keys.erase(parent.keys.begin() + child_idx);
        parent.children.erase(parent.children.begin() + child_idx + 1);
        parent.num_keys--;
        
        serialize_node(child, child_page->data);
        serialize_node(parent, buffer_pool.fetch_page(parent.children[child_idx])->data);
        pager.mark_dirty(child_id);
        pager.mark_dirty(parent.children[child_idx]);
        
        // Free the right sibling page
        pager.free_page(right_sib_id);
    }
}
void BPlusTree::split_child(Node& parent, uint32_t index, Node& child) {
    validate_node(parent);
    validate_node(child);
    uint32_t new_child_id = pager.allocate_page();
    auto new_child_page = buffer_pool.fetch_page(new_child_id);
    if (!new_child_page) {
        throw std::runtime_error("Failed to fetch new child page: " + std::to_string(new_child_id));
    }
    Node new_child;
    new_child.is_leaf = child.is_leaf;
    new_child.num_keys = order - 1;
    new_child.keys.resize(order - 1);
    new_child.values.resize(child.is_leaf ? order - 1 : 0);
    new_child.children.resize(child.is_leaf ? 0 : order);
    std::copy(child.keys.begin() + order, child.keys.end(), new_child.keys.begin());
    if (child.is_leaf) {
        std::copy(child.values.begin() + order, child.values.end(), new_child.values.begin());
    } else {
        std::copy(child.children.begin() + order, child.children.end(), new_child.children.begin());
    }
    child.num_keys = order - 1;
    child.keys.resize(order - 1);
    if (child.is_leaf) {
        child.values.resize(order - 1);
    } else {
        child.children.resize(order);
    }
    parent.children.insert(parent.children.begin() + index + 1, new_child_id);
    parent.keys.insert(parent.keys.begin() + index, child.keys[order - 1]);
    parent.num_keys++;
    serialize_node(child, buffer_pool.fetch_page(parent.children[index])->data);
    serialize_node(new_child, new_child_page->data);
    pager.mark_dirty(parent.children[index]);
    pager.mark_dirty(new_child_id);
}

void BPlusTree::serialize_node(const Node& node, std::array<uint8_t, DB_PAGE_SIZE>& buffer) const {
    validate_node(node);
    buffer.fill(0);
    uint32_t offset = 0;
    *reinterpret_cast<bool*>(buffer.data() + offset) = node.is_leaf;
    offset += sizeof(bool);
    *reinterpret_cast<uint32_t*>(buffer.data() + offset) = node.num_keys;
    offset += sizeof(uint32_t);
    for (uint32_t i = 0; i < node.num_keys; i++) {
        *reinterpret_cast<uint32_t*>(buffer.data() + offset) = node.keys[i];
        offset += sizeof(uint32_t);
    }
    if (node.is_leaf) {
        for (uint32_t i = 0; i < node.num_keys; i++) {
            uint32_t value_size = static_cast<uint32_t>(node.values[i].size());
            *reinterpret_cast<uint32_t*>(buffer.data() + offset) = value_size;
            offset += sizeof(uint32_t);
            if (offset + value_size > DB_PAGE_SIZE) {
                throw std::runtime_error("Node serialization exceeds page size");
            }
            std::copy(node.values[i].begin(), node.values[i].end(), buffer.begin() + offset);
            offset += value_size;
        }
    } else {
        for (uint32_t i = 0; i <= node.num_keys; i++) {
            *reinterpret_cast<uint32_t*>(buffer.data() + offset) = node.children[i];
            offset += sizeof(uint32_t);
        }
    }
}

void BPlusTree::deserialize_node(const std::array<uint8_t, DB_PAGE_SIZE>& buffer, Node& node) const {
    uint32_t offset = 0;
    node.is_leaf = *reinterpret_cast<const bool*>(buffer.data() + offset);
    offset += sizeof(bool);
    node.num_keys = *reinterpret_cast<const uint32_t*>(buffer.data() + offset);
    offset += sizeof(uint32_t);
    if (node.num_keys > 2 * order - 1) {
        throw IntegrityViolation("Invalid node: too many keys during deserialization");
    }
    node.keys.resize(node.num_keys);
    for (uint32_t i = 0; i < node.num_keys; i++) {
        node.keys[i] = *reinterpret_cast<const uint32_t*>(buffer.data() + offset);
        offset += sizeof(uint32_t);
    }
    if (node.is_leaf) {
        node.values.resize(node.num_keys);
        for (uint32_t i = 0; i < node.num_keys; i++) {
            if (offset + sizeof(uint32_t) > DB_PAGE_SIZE) {
                throw IntegrityViolation("Invalid node: buffer too small for value size");
            }
            uint32_t value_size = *reinterpret_cast<const uint32_t*>(buffer.data() + offset);
            offset += sizeof(uint32_t);
            if (offset + value_size > DB_PAGE_SIZE) {
                throw IntegrityViolation("Invalid node: buffer too small for value data");
            }
            node.values[i].resize(value_size);
            std::copy(buffer.begin() + offset, buffer.begin() + offset + value_size, node.values[i].begin());
            offset += value_size;
        }
    } else {
        node.children.resize(node.num_keys + 1);
        for (uint32_t i = 0; i <= node.num_keys; i++) {
            if (offset + sizeof(uint32_t) > DB_PAGE_SIZE) {
                throw IntegrityViolation("Invalid node: buffer too small for child pointer");
            }
            node.children[i] = *reinterpret_cast<const uint32_t*>(buffer.data() + offset);
            offset += sizeof(uint32_t);
        }
    }
    validate_node(node);
}
