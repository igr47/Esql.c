#include "storagemanager.h"
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <iostream>
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

WriteAheadLog::WriteAheadLog(const std::string& path) : log_file_path(path) {
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

void WriteAheadLog::recover() {
    std::lock_guard<std::mutex> lock(mutex);
    try{
        ensure_file_open();
        log_file.seekg(0);
	//skip recovery if log is empty
	if(log_file.peek()==std::ifstream::traits_type::eof()){
		return;
	}
        std::string line;
        while (std::getline(log_file, line)) {
		std::cerr<<"Processing WAL entry:" <<line.substr(0,20)<<"------\n";
        // Simplified recovery; parse and apply changes as needed
        }
    }catch(...){
	    std::cerr<<"WAL recovery encountered an error\n";
    }
    //reset after recovery
    try{
        log_file.close();
        log_file.open(log_file_path, std::ios::binary | std::ios::trunc | std::ios::out);
        log_file.close();
        log_file.open(log_file_path, std::ios::binary | std::ios::app | std::ios::in | std::ios::out);
        log_check_point();
    }catch(...){
	    std::cerr<<"Failed to reset WAL\n";
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
        auto meta_page = buffer_pool.fetch_page(0);
        if (!meta_page) {
            throw std::runtime_error("Failed to fetch metadata page");
        }
        if (pager.get_next_page_id() > 1) {
            root_page_id = *reinterpret_cast<uint32_t*>(meta_page->data.data() + sizeof(uint32_t));
            if (root_page_id >= pager.get_next_page_id()) {
                throw std::runtime_error("Invalid root page ID: " + std::to_string(root_page_id));
            }
        } else {
            root_page_id = pager.allocate_page();
            auto root_page = buffer_pool.fetch_page(root_page_id);
            if (!root_page) {
                throw std::runtime_error("Failed to fetch root page");
            }
            Node root;
            root.is_leaf = true;
            root.num_keys = 0;
            serialize_node(root, root_page->data);
            pager.mark_dirty(root_page_id);
            *reinterpret_cast<uint32_t*>(meta_page->data.data() + sizeof(uint32_t)) = root_page_id;
            pager.mark_dirty(0);
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to initialize BPlusTree: " + std::string(e.what()));
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
