#include "storagemanager.h"
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <system_error>

// ==================== Page Implementation ====================
Page::Page(uint32_t id) noexcept : page_id(id), dirty(false) {
    data.fill(0);
}

// ==================== Pager Implementation ====================
void Pager::ensure_file_open() {
    if (!file.is_open()) {
        file.open(filename, std::ios::binary | std::ios::in | std::ios::out);
        if (!file) {
            throw FileIOException("Failed to open database file: " + filename);
        }
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
    try {
        file.open(filename, std::ios::binary | std::ios::in | std::ios::out);
        if (!file) {
            // Create new file
            file.open(filename, std::ios::binary | std::ios::out);
            file.close();
            file.open(filename, std::ios::binary | std::ios::in | std::ios::out);
            if (!file) {
                throw FileIOException("Failed to create database file: " + filename);
            }
        }
        load_metadata();
    } catch (...) {
        if (file.is_open()) file.close();
        throw;
    }
}

Pager::~Pager() noexcept {
    try {
        flush_all();
        if (file.is_open()) {
            file.close();
        }
    } catch (...) {
        // Destructors must not throw
    }
}

std::shared_ptr<Page> Pager::get_page(uint32_t page_id) {
    if(page_id==0){
	    auto page=std::make_shared<Page>(0);
	    if(!read_page_from_disk(0,page->data)){
		    page->data.fill(0);
		    *reinterpret_cast<uint32_t*>(page->data.data())=1;
	        }
	    return page;
	}
    if (page_id >= next_page_id) {
        throw StorageException("Invalid page ID: " + std::to_string(page_id)+ "(max allocated:" +std::to_string(next_page_id-1)+")");
    }

    std::lock_guard<std::mutex> lock(cache_mutex);
    
    // Check cache first
    if (auto it = page_cache.find(page_id); it != page_cache.end()) {
        return it->second;
    }

    // Not in cache, read from disk
    auto page = std::make_shared<Page>(page_id);
    if (!read_page_from_disk(page_id, page->data)) {
        throw FileIOException("Failed to read page " + std::to_string(page_id) + " from disk");
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
        next_page_id--; // Rollback on failure
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
                    // Log error but continue flushing other pages
                    continue;
                }
                page->dirty = false;
            }
        }
        save_metadata();
    } catch (...) {
        // Swallow exceptions in flush
    }
}

bool Pager::read_page_from_disk(uint32_t page_id, std::array<uint8_t, DB_PAGE_SIZE>& buffer) noexcept {
    try {
        ensure_file_open();
        if (!safe_seek(page_id)) {
            return false;
        }
        
        file.read(reinterpret_cast<char*>(buffer.data()), DB_PAGE_SIZE);
        if (file.gcount() != DB_PAGE_SIZE) {
            return false;
        }
        return true;
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
    if (read_page_from_disk(0, buffer)) {
        next_page_id = *reinterpret_cast<uint32_t*>(buffer.data());
	if(next_page_id<=1){
		next_page_id=1;
		buffer.fill(0);
		*reinterpret_cast<uint32_t*>(buffer.data())=next_page_id;
		write_page_to_disk(0,buffer);
	}
    } else {
        next_page_id = 1; // Start with page 1 (0 is metadata)
	buffer.fill(0);
	*reinterpret_cast<uint32_t*>(buffer.data())=next_page_id;
	write_page_to_disk(0,buffer);
    }
    auto page=std::make_shared<Page>(0);
    page->data=buffer;
    page_cache[0]=page;
}

void Pager::save_metadata() noexcept {
    std::array<uint8_t, DB_PAGE_SIZE> buffer;
    buffer.fill(0);
    *reinterpret_cast<uint32_t*>(buffer.data()) = next_page_id;
    if (!write_page_to_disk(0, buffer)) {
        // Metadata save failed - critical error
    }
}
uint32_t Pager::get_next_page_id() const{return next_page_id;}

// ==================== BufferPool Implementation ====================
void BufferPool::evict_page() noexcept {
    if (lru_list.empty()) return;

    uint32_t victim = lru_list.back();
    lru_list.pop_back();
    lru_map.erase(victim);
    
    if (auto it = pool.find(victim); it != pool.end()) {
        if (it->second->dirty) {
            try {
                pager.write_page_to_disk(victim, it->second->data);
            } catch (...) {
                // Ignore write errors during eviction
            }
        }
        pool.erase(victim);
    }
}

BufferPool::BufferPool(Pager& pager, size_t capacity) noexcept 
    : pager(pager), capacity(capacity > 0 ? capacity : DEFAULT_BUFFER_POOL_CAPACITY) {}

std::shared_ptr<Page> BufferPool::fetch_page(uint32_t page_id) {
    std::lock_guard<std::mutex> lock(mutex);
    
    // Check if page is in the buffer pool
    if (auto it = pool.find(page_id); it != pool.end()) {
        lru_list.erase(lru_map[page_id]);
        lru_list.push_front(page_id);
        lru_map[page_id] = lru_list.begin();
        return it->second;
    }

    // If buffer pool is full, evict least recently used page
    if(page_id==1 && pager.get_next_page_id()==1){
	    auto page=std::make_shared<Page>(1);
	    page->dirty=true;
	    pool[1]=page;
	    lru_list.push_front(1);
	    lru_map[1]=lru_list.begin();
	    return page;
	}
    auto page=pager.get_page(page_id);
    pool[page_id]=page;
    lru_list.push_front(page_id);
    lru_map[page_id]=lru_list.begin();
    return page;
}
    /*if (pool.size() >= capacity) {
        evict_page();
    }

    // Get page from pager
    auto page = pager.get_page(page_id);
    if (!page) {
        throw StorageException("Failed to fetch page " + std::to_string(page_id) + " from pager");
    }

    // Add to buffer pool
    try {
        pool[page_id] = page;
        lru_list.push_front(page_id);
        lru_map[page_id] = lru_list.begin();
        return page;
    } catch (const StorageException& e) {
	if(page_id==1 && std::string(e.what()).find("Invalid page id")!=std::string::npos){
		pool[1]=page;
		lru_list.push_front(1);
		lru_map[1]=lru_list.begin();
		return page;
	}
        //pool.erase(page_id);
        //lru_map.erase(page_id);
        throw;
    }
}*/

void BufferPool::flush_all() noexcept {
    std::lock_guard<std::mutex> lock(mutex);
    try {
        for (auto& [page_id, page] : pool) {
            if (page->dirty) {
                pager.write_page_to_disk(page_id, page->data);
                page->dirty = false;
            }
        }
    } catch (...) {
        // Ignore flush errors
    }
}

// ==================== WriteAheadLog Implementation ====================
void WriteAheadLog::ensure_file_open() {
    if (!log_file.is_open()) {
        log_file.open(log_file_path, std::ios::binary | std::ios::app | std::ios::in | std::ios::out);
        if (!log_file) {
            throw FileIOException("Failed to open WAL file: " + log_file_path);
        }
    }
}

WriteAheadLog::WriteAheadLog(const std::string& path) : log_file_path(path) {
    try {
        log_file.open(log_file_path, std::ios::binary | std::ios::app | std::ios::in | std::ios::out);
        if (!log_file) {
            log_file.open(log_file_path, std::ios::binary | std::ios::out);
            log_file.close();
            log_file.open(log_file_path, std::ios::binary | std::ios::app | std::ios::in | std::ios::out);
            if (!log_file) {
                throw FileIOException("Failed to create WAL file: " + log_file_path);
            }
        }
    } catch (...) {
        if (log_file.is_open()) log_file.close();
        throw;
    }
}

WriteAheadLog::~WriteAheadLog() noexcept {
    try {
        if (log_file.is_open()) {
            log_file.close();
        }
    } catch (...) {
        // Destructors must not throw
    }
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
    ensure_file_open();
    log_file.seekg(0);
    
    std::string line;
    while (std::getline(log_file, line)) {
        // Simplified recovery - in practice would need to parse each log entry
        // and reapply changes to the database
    }
    
    // Truncate log after recovery
    log_file.close();
    log_file.open(log_file_path, std::ios::binary | std::ios::trunc | std::ios::out);
    log_file.close();
    log_file.open(log_file_path, std::ios::binary | std::ios::app | std::ios::in | std::ios::out);
}

// ==================== BPlusTree Implementation ====================
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
    
    auto meta_page = buffer_pool.fetch_page(0);
    //uint32_t stored_root_id=0;
    if(pager.get_next_page_id()>1){
	    root_page_id=*reinterpret_cast<uint32_t*>(meta_page->data.data()+sizeof(uint32_t));
	}else{
		root_page_id=pager.allocate_page();  
		auto root_page=buffer_pool.fetch_page(root_page_id);
		Node root;   
		root.is_leaf=true;
		root.num_keys=0; 
		serialize_node(root, root_page->data);
		pager.mark_dirty(root_page_id);  
		// Update metadata 
		meta_page = buffer_pool.fetch_page(0); 
		*reinterpret_cast<uint32_t*>(meta_page->data.data() + sizeof(uint32_t)) = root_page_id; 
		pager.mark_dirty(0);
	}
}

    

    /*} else {
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
    }*/


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
        // Find the first key greater than or equal to the search key
        auto it = std::lower_bound(node.keys.begin(), node.keys.begin() + node.num_keys, key);
        uint32_t index = it - node.keys.begin();
        
        // Follow the appropriate child pointer
        current_page = buffer_pool.fetch_page(node.children[index]);
        deserialize_node(current_page->data, node);
    }
    
    // Search for the key in the leaf node
    auto it = std::lower_bound(node.keys.begin(), node.keys.begin() + node.num_keys, key);
    if (it != node.keys.begin() + node.num_keys && *it == key) {
        uint32_t index = it - node.keys.begin();
        return node.values[index];
    }
    
    return {}; // Key not found
}

void BPlusTree::insert_non_full(Node& node, uint32_t key, const std::vector<uint8_t>& value) {
    validate_node(node);
    
    int i = node.num_keys - 1;
    if (node.is_leaf) {
        // Find position to insert new key
        while (i >= 0 && key < node.keys[i]) {
            node.keys[i + 1] = node.keys[i];
            if (node.is_leaf) {
                node.values[i + 1] = std::move(node.values[i]);
            }
            i--;
        }
        node.keys[i + 1] = key;
        if (node.is_leaf) {
            node.values[i + 1] = value;
        }
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
    validate_node(parent);
    validate_node(child);
    
    uint32_t new_child_id = pager.allocate_page();
    auto new_child_page = buffer_pool.fetch_page(new_child_id);
    Node new_child;
    
    new_child.is_leaf = child.is_leaf;
    new_child.num_keys = order - 1;
    
    // Copy second half of child to new child
    std::copy(child.keys.begin() + order, child.keys.end(), 
              std::back_inserter(new_child.keys));
    
    if (child.is_leaf) {
        std::copy(child.values.begin() + order, child.values.end(),
                  std::back_inserter(new_child.values));
    } else {
        std::copy(child.children.begin() + order, child.children.end(),
                  std::back_inserter(new_child.children));
    }
    
    // Update child's key count
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

void BPlusTree::serialize_node(const Node& node, std::array<uint8_t, DB_PAGE_SIZE>& buffer) const {
    validate_node(node);
    buffer.fill(0);
    
    uint32_t offset = 0;
    
    // Serialize header (is_leaf, num_keys)
    *reinterpret_cast<bool*>(buffer.data() + offset) = node.is_leaf;
    offset += sizeof(bool);
    *reinterpret_cast<uint32_t*>(buffer.data() + offset) = node.num_keys;
    offset += sizeof(uint32_t);
    
    // Serialize keys
    for (uint32_t i = 0; i < node.num_keys; i++) {
        *reinterpret_cast<uint32_t*>(buffer.data() + offset) = node.keys[i];
        offset += sizeof(uint32_t);
    }
    
    if (node.is_leaf) {
        // Serialize values for leaf nodes
        for (uint32_t i = 0; i < node.num_keys; i++) {
            uint32_t value_size = static_cast<uint32_t>(node.values[i].size());
            *reinterpret_cast<uint32_t*>(buffer.data() + offset) = value_size;
            offset += sizeof(uint32_t);
            
            std::copy(node.values[i].begin(), node.values[i].end(), 
                      buffer.begin() + offset);
            offset += value_size;
        }
    } else {
        // Serialize children for internal nodes
        for (uint32_t i = 0; i <= node.num_keys; i++) {
            *reinterpret_cast<uint32_t*>(buffer.data() + offset) = node.children[i];
            offset += sizeof(uint32_t);
        }
    }
}

void BPlusTree::deserialize_node(const std::array<uint8_t, DB_PAGE_SIZE>& buffer, Node& node) const {
    uint32_t offset = 0;
    
    // Deserialize header
    node.is_leaf = *reinterpret_cast<const bool*>(buffer.data() + offset);
    offset += sizeof(bool);
    node.num_keys = *reinterpret_cast<const uint32_t*>(buffer.data() + offset);
    offset += sizeof(uint32_t);
    
    // Deserialize keys
    node.keys.resize(node.num_keys);
    for (uint32_t i = 0; i < node.num_keys; i++) {
        node.keys[i] = *reinterpret_cast<const uint32_t*>(buffer.data() + offset);
        offset += sizeof(uint32_t);
    }
    
    if (node.is_leaf) {
        // Deserialize values for leaf nodes
        node.values.resize(node.num_keys);
        for (uint32_t i = 0; i < node.num_keys; i++) {
            uint32_t value_size = *reinterpret_cast<const uint32_t*>(buffer.data() + offset);
            offset += sizeof(uint32_t);
            
            node.values[i].resize(value_size);
            std::copy(buffer.begin() + offset, buffer.begin() + offset + value_size,
                      node.values[i].begin());
            offset += value_size;
        }
    } else {
        // Deserialize children for internal nodes
        node.children.resize(node.num_keys + 1);
        for (uint32_t i = 0; i <= node.num_keys; i++) {
            node.children[i] = *reinterpret_cast<const uint32_t*>(buffer.data() + offset);
            offset += sizeof(uint32_t);
        }
    }
    
    validate_node(node);
}
