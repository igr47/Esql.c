#include "storagemanager.h"
#include <cstring>
#include <algorithm>

// Page implementation
Page::Page(uint32_t id) : page_id(id), dirty(false) {
    data.fill(0);
}

// Pager implementation
Pager::Pager(const std::string& db_file) : filename(db_file) {
    file.open(db_file, std::ios::binary | std::ios::in | std::ios::out);
    if (!file) {
        file.open(db_file, std::ios::binary | std::ios::out);
        file.close();
        file.open(db_file, std::ios::binary | std::ios::in | std::ios::out);
        
        if (file.seekg(0, std::ios::end).tellg() == 0) {
            load_metadata();
        }
    }
}

Pager::~Pager() {
    flush_all();
    file.close();
}

std::shared_ptr<Page> Pager::get_page(uint32_t page_id) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto it = page_cache.find(page_id);
    if (it != page_cache.end()) {
        return it->second;
    }

    auto page = std::make_shared<Page>(page_id);
    if (read_page_from_disk(page_id, page->data)) {
        page_cache[page_id] = page;
        return page;
    }

    return nullptr;
}

uint32_t Pager::allocate_page() {
    std::lock_guard<std::mutex> lock(cache_mutex);
    uint32_t new_id = next_page_id++;
    auto page = std::make_shared<Page>(new_id);
    page_cache[new_id] = page;
    save_metadata();
    return new_id;
}

void Pager::mark_dirty(uint32_t page_id) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    if (auto it = page_cache.find(page_id); it != page_cache.end()) {
        it->second->dirty = true;
    }
}

void Pager::flush_all() {
    std::lock_guard<std::mutex> lock(cache_mutex);
    for (auto& [page_id, page] : page_cache) {
        if (page->dirty) {
            write_page_to_disk(page_id, page->data);
            page->dirty = false;
        }
    }
    save_metadata();
}

bool Pager::read_page_from_disk(uint32_t page_id, std::array<uint8_t, DB_PAGE_SIZE>& buffer) {
    file.seekg(page_id * DB_PAGE_SIZE);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), DB_PAGE_SIZE)) {
        return false;
    }
    return true;
}

bool Pager::write_page_to_disk(uint32_t page_id, const std::array<uint8_t, DB_PAGE_SIZE>& buffer) {
    file.seekp(page_id * DB_PAGE_SIZE);
    file.write(reinterpret_cast<const char*>(buffer.data()), DB_PAGE_SIZE);
    file.flush();
    return true;
}

void Pager::load_metadata() {
    std::array<uint8_t, DB_PAGE_SIZE> buffer;
    if (read_page_from_disk(0, buffer)) {
        next_page_id = *reinterpret_cast<uint32_t*>(buffer.data());
    }
}

void Pager::save_metadata() {
    std::array<uint8_t, DB_PAGE_SIZE> buffer;
    *reinterpret_cast<uint32_t*>(buffer.data()) = next_page_id;
    write_page_to_disk(0, buffer);
}

// BufferPool implementation
BufferPool::BufferPool(Pager& pager, size_t capacity) 
    : pager(pager), capacity(capacity) {}

std::shared_ptr<Page> BufferPool::fetch_page(uint32_t page_id) {
    std::lock_guard<std::mutex> lock(mutex);
    
    // Check if page is in the buffer pool
    auto it = pool.find(page_id);
    if (it != pool.end()) {
        lru_list.erase(lru_map[page_id]);
        lru_list.push_front(page_id);
        lru_map[page_id] = lru_list.begin();
        return it->second;
    }

    // If buffer pool is full, evict least recently used page
    if (pool.size() >= capacity) {
        uint32_t victim = lru_list.back();
        lru_list.pop_back();
        lru_map.erase(victim);
        
        auto victim_page = pool[victim];
        if (victim_page->dirty) {
            pager.write_page_to_disk(victim, victim_page->data);
        }
        pool.erase(victim);
    }

    // Get page from pager
    auto page = pager.get_page(page_id);
    if (!page) {
        return nullptr;
    }

    // Add to buffer pool
    pool[page_id] = page;
    lru_list.push_front(page_id);
    lru_map[page_id] = lru_list.begin();
    return page;
}

void BufferPool::flush_all() {
    std::lock_guard<std::mutex> lock(mutex);
    for (auto& [page_id, page] : pool) {
        if (page->dirty) {
            pager.write_page_to_disk(page_id, page->data);
            page->dirty = false;
        }
    }
}

// WriteAheadLog implementation
WriteAheadLog::WriteAheadLog(const std::string& path) : log_file_path(path) {
    log_file.open(log_file_path, std::ios::binary | std::ios::app | std::ios::in | std::ios::out);
    if (!log_file) {
        log_file.open(log_file_path, std::ios::binary | std::ios::out);
        log_file.close();
        log_file.open(log_file_path, std::ios::binary | std::ios::app | std::ios::in | std::ios::out);
    }
}

void WriteAheadLog::log_transaction_begin(uint64_t tx_id) {
    std::lock_guard<std::mutex> lock(mutex);
    log_file << "BEGIN," << tx_id << "\n";
    log_file.flush();
}

void WriteAheadLog::log_page_write(uint64_t tx_id, uint32_t page_id, 
                                 const std::array<uint8_t, DB_PAGE_SIZE>& old_data,
                                 const std::array<uint8_t, DB_PAGE_SIZE>& new_data) {
    std::lock_guard<std::mutex> lock(mutex);
    log_file << "UPDATE," << tx_id << "," << page_id << ",";
    log_file.write(reinterpret_cast<const char*>(old_data.data()), DB_PAGE_SIZE);
    log_file.write(reinterpret_cast<const char*>(new_data.data()), DB_PAGE_SIZE);
    log_file << "\n";
    log_file.flush();
}

void WriteAheadLog::log_transaction_commit(uint64_t tx_id) {
    std::lock_guard<std::mutex> lock(mutex);
    log_file << "COMMIT," << tx_id << "\n";
    log_file.flush();
}

void WriteAheadLog::log_check_point() {
    std::lock_guard<std::mutex> lock(mutex);
    log_file << "CHECKPOINT\n";
    log_file.flush();
}

void WriteAheadLog::recover() {
    std::lock_guard<std::mutex> lock(mutex);
    log_file.seekg(0);
    std::string line;
    
    // Simplified recovery - in practice would need to parse each log entry
    while (std::getline(log_file, line)) {
        // Parse and apply log entries
    }
    
    log_file.close();
    log_file.open(log_file_path, std::ios::binary | std::ios::trunc | std::ios::out);
    log_file.close();
    log_file.open(log_file_path, std::ios::binary | std::ios::app | std::ios::in | std::ios::out);
}

// BPlusTree implementation
BPlusTree::BPlusTree(Pager& pager, BufferPool& buffer_pool, uint32_t order) 
    : pager(pager), buffer_pool(buffer_pool), order(order) {
    
    auto meta_page = buffer_pool.fetch_page(0);
    if (meta_page) {
        // Load root page id from metadata
        root_page_id = *reinterpret_cast<uint32_t*>(meta_page->data.data() + sizeof(uint32_t));
    } else {
        // Create new root leaf node
        root_page_id = pager.allocate_page();
        auto root_page = buffer_pool.fetch_page(root_page_id);
        
        Node root;
        root.is_leaf = true;
        root.num_keys = 0;
        serialize_node(root, root_page->data);
        pager.mark_dirty(root_page_id);
        
        // Update metadata
        meta_page = buffer_pool.fetch_page(0);
        *reinterpret_cast<uint32_t*>(meta_page->data.data() + sizeof(uint32_t)) = root_page_id;
        pager.mark_dirty(0);
    }
}

void BPlusTree::insert(uint32_t key, const std::vector<uint8_t>& value) {
    auto root_page = buffer_pool.fetch_page(root_page_id);
    Node root;
    deserialize_node(root_page->data, root);
    
    if (root.num_keys == 2 * order - 1) {
        // Root is full and needs to split
        uint32_t new_root_id = pager.allocate_page();
        auto new_root_page = buffer_pool.fetch_page(new_root_id);
        
        Node new_root;
        new_root.is_leaf = false;
        new_root.num_keys = 0;
        new_root.children.push_back(root_page_id);
        
        split_child(new_root, 0, root);
        insert_non_full(new_root, key, value);
        
        // Update root
        root_page_id = new_root_id;
        serialize_node(new_root, new_root_page->data);
        pager.mark_dirty(new_root_id);
        
        // Update metadata
        auto meta_page = buffer_pool.fetch_page(0);
        *reinterpret_cast<uint32_t*>(meta_page->data.data() + sizeof(uint32_t)) = root_page_id;
        pager.mark_dirty(0);
    } else {
        insert_non_full(root, key, value);
        serialize_node(root, root_page->data);
        pager.mark_dirty(root_page_id);
    }
}

std::vector<uint8_t> BPlusTree::search(uint32_t key) {
    auto current_page = buffer_pool.fetch_page(root_page_id);
    Node node;
    deserialize_node(current_page->data, node);
    
    while (!node.is_leaf) {
        uint32_t i = 0;
        while (i < node.num_keys && key > node.keys[i]) {
            i++;
        }
        current_page = buffer_pool.fetch_page(node.children[i]);
        deserialize_node(current_page->data, node);
    }
    
    for (uint32_t i = 0; i < node.num_keys; i++) {
        if (node.keys[i] == key) {
            return node.values[i];
        }
    }
    
    return {}; // Not found
}

void BPlusTree::insert_non_full(Node& node, uint32_t key, const std::vector<uint8_t>& value) {
    int i = node.num_keys - 1;
    
    if (node.is_leaf) {
        // Find position to insert new key
        while (i >= 0 && key < node.keys[i]) {
            node.keys[i + 1] = node.keys[i];
            node.values[i + 1] = node.values[i];
            i--;
        }
        node.keys[i + 1] = key;
        node.values[i + 1] = value;
        node.num_keys++;
    } else {
        // Find child to insert into
        while (i >= 0 && key < node.keys[i]) {
            i--;
        }
        i++;
        
        // Check if child is full
        auto child_page = buffer_pool.fetch_page(node.children[i]);
        Node child;
        deserialize_node(child_page->data, child);
        
        if (child.num_keys == 2 * order - 1) {
            split_child(node, i, child);
            if (key > node.keys[i]) {
                i++;
            }
        }
        
        // Refetch child as split would have changed it
        child_page = buffer_pool.fetch_page(node.children[i]);
        deserialize_node(child_page->data, child);
        insert_non_full(child, key, value);
        serialize_node(child, child_page->data);
        pager.mark_dirty(node.children[i]);
    }
}

void BPlusTree::split_child(Node& parent, uint32_t index, Node& child) {
    uint32_t new_child_id = pager.allocate_page();
    auto new_child_page = buffer_pool.fetch_page(new_child_id);
    Node new_child;
    
    new_child.is_leaf = child.is_leaf;
    new_child.num_keys = order - 1;
    
    // Copy second half of child to new child
    if (child.is_leaf) {
        new_child.keys.assign(child.keys.begin() + order, child.keys.end());
        new_child.values.assign(child.values.begin() + order, child.values.end());
    } else {
        new_child.keys.assign(child.keys.begin() + order, child.keys.end());
        new_child.children.assign(child.children.begin() + order, child.children.end());
    }
    
    child.num_keys = order - 1;
    
    // Make space in the parent for new child
    parent.children.insert(parent.children.begin() + index + 1, new_child_id);
    parent.keys.insert(parent.keys.begin() + index, child.keys[order - 1]);
    parent.num_keys++;
    
    // Save nodes
    serialize_node(child, buffer_pool.fetch_page(parent.children[index])->data);
    serialize_node(new_child, new_child_page->data);
    pager.mark_dirty(parent.children[index]);
    pager.mark_dirty(new_child_id);
}

void BPlusTree::serialize_node(const Node& node, std::array<uint8_t, DB_PAGE_SIZE>& buffer) {
    uint32_t offset = 0;
    
    // Serialize header
    *reinterpret_cast<bool*>(buffer.data() + offset) = node.is_leaf;
    offset += sizeof(bool);
    *reinterpret_cast<uint32_t*>(buffer.data() + offset) = node.num_keys;
    offset += sizeof(uint32_t);
    
    // Serialize keys
    for (uint32_t key : node.keys) {
        *reinterpret_cast<uint32_t*>(buffer.data() + offset) = key;
        offset += sizeof(uint32_t);
    }
    
    if (node.is_leaf) {
        // Serialize values for leaf nodes
        for (const auto& value : node.values) {
            uint32_t value_size = static_cast<uint32_t>(value.size());
            *reinterpret_cast<uint32_t*>(buffer.data() + offset) = value_size;
            offset += sizeof(uint32_t);
            std::copy(value.begin(), value.end(), buffer.begin() + offset);
            offset += value_size;
        }
    } else {
        // Serialize children for internal nodes
        for (uint32_t child : node.children) {
            *reinterpret_cast<uint32_t*>(buffer.data() + offset) = child;
            offset += sizeof(uint32_t);
        }
    }
}

void BPlusTree::deserialize_node(const std::array<uint8_t, DB_PAGE_SIZE>& buffer, Node& node) {
    uint32_t offset = 0;
    
    // Deserialize header
    node.is_leaf = *reinterpret_cast<const bool*>(buffer.data() + offset);
    offset += sizeof(bool);
    node.num_keys = *reinterpret_cast<const uint32_t*>(buffer.data() + offset);
    offset += sizeof(uint32_t);
    
    // Deserialize keys
    node.keys.resize(node.num_keys);
    for (uint32_t i = 0; i < node.num_keys; ++i) {
        node.keys[i] = *reinterpret_cast<const uint32_t*>(buffer.data() + offset);
        offset += sizeof(uint32_t);
    }
    
    if (node.is_leaf) {
        // Deserialize values for leaf nodes
        node.values.resize(node.num_keys);
        for (uint32_t i = 0; i < node.num_keys; ++i) {
            uint32_t value_size = *reinterpret_cast<const uint32_t*>(buffer.data() + offset);
            offset += sizeof(uint32_t);
            node.values[i].resize(value_size);
            std::copy(buffer.begin() + offset, buffer.begin() + offset + value_size, node.values[i].begin());
            offset += value_size;
        }
    } else {
        // Deserialize children for internal nodes
        node.children.resize(node.num_keys + 1);
        for (uint32_t i = 0; i <= node.num_keys; ++i) {
            node.children[i] = *reinterpret_cast<const uint32_t*>(buffer.data() + offset);
            offset += sizeof(uint32_t);
        }
    }
}
