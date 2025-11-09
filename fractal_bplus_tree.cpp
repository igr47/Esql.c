#include "fractal_bplus_tree.h"
#include "database_file.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cstring>
#include <stack>
#include <queue>
#include <limits>
#include <unordered_set>

namespace fractal {

    FractalBPlusTree::FractalBPlusTree(DatabaseFile* db_file, BufferPool* buffer_pool, WriteAheadLog* wal, const std::string& table_name, uint32_t root_page_id, uint32_t table_id) 
        : db_file(db_file), buffer_pool(buffer_pool), wal(wal), table_name(table_name), root_page_id(root_page_id), table_id(table_id) {
            
            //If table_id is 0, try to get it from db_file
            if (table_id == 0) {
                try {
                    table_id = db_file->get_table_id(table_name);
                } catch (...) {
                    // If table doesnt exist yet - it will be created
                    table_id = 0; 
                }
            }
            std::cout << "FractalBPlusTree initialized for table '" << table_name << "' with root page: " << root_page_id << std::endl;

        // If root_id is 0, we need to create the tree
        if (root_page_id == 0) {
            create();
        }
    }

    FractalBPlusTree::~FractalBPlusTree() {
        std::cout << "FractalBPlusTree destuctor called for table: " << table_name << std::endl;

        try {
            // Ensure messages are flushed before destruction
            if (root_page_id != 0) {
                flush_all_messages(0); // Using transaction ID 0 for clean up
            }
            std::cout << "FractalBPlusTree cleanup completed for table: " << table_name << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error in fractalBPlusTree destruction for " << table_name << ": " << e.what() << std::endl;
        }
    }

    void FractalBPlusTree::create() {

        if (root_page_id == 0) {
            // Allocate new root page 
            root_page_id = db_file->allocate_page(PageType::LEAF_NODE);
            Page* root = get_page(root_page_id, true);

            // Initialize root as leaf node 
            root->initialize(root_page_id, PageType::LEAF_NODE, 0);
            root->header.key_count = 0;
            root->header.message_count = 0;
            root->header.next_page = 0;
            root->header.prev_page = 0;

            // Log Creation
            wal->log_tree_operation(0, "CREATE_TREE", 11);

            release_page(root_page_id, true);

            std::cout << "Created new FractalBPlusTree with root page: " << root_page_id << std::endl;
        }
    }
    
    void FractalBPlusTree::drop() {
        std::cout << "FractalBPlusTree::drop() called for table '" << table_name << "'" << std::endl;

        if (root_page_id == 0) {
            std::cout << "Tree already dropped: "<< std::endl;
            return;
        }

        try {
            // Traversal to collect all pages
            std::vector<uint32_t> all_pages = collect_all_pages_safely();

            std::cout << "DEBUG: Collected " << all_pages.size() << " pages to free" << std::endl;

            // Free all pages
            for (uint32_t page_id : all_pages) {
                if(page_id != 0) {
                    try {
                        free_page(page_id, 0);
                        std::cout << "DEBUG: Freed page " << page_id << std::endl;
                    } catch (const std::exception& e) {
                        std::cerr << "WARNING: Failed to free page " << page_id << ": " << e.what() << std::endl;
                    }
                }
            }

            // Log the drop
            wal->log_tree_operation(0, "DROP_TREE",9);

            root_page_id = 0;
            total_messages = 0;
            memory_usage = 0;

            std::cout << "Dropped FractalBPLusTree for table '" << table_name << "'" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Failed to drop tree: " << e.what() << std::endl;
            // Fallback: atleast free the root page
            try {
                free_page(root_page_id, 0);
                root_page_id = 0;
            } catch (...) {
                // Ignore errors in fall back
            }
            throw;
        }
    }


    void FractalBPlusTree::insert(int64_t key, const std::string& value, uint64_t transaction_id) {
        std::cout << "=== DEBUG INSERT START ===" << std::endl;
        std::cout << "DEBUG: INSERT key=" << key << ", value='" << value << "', txn=" << transaction_id << std::endl;

        try {
            // Write data block first
            uint64_t value_offset = write_data_block(value);
            uint32_t value_length = value.size();
            std::cout << "DEBUG: Value written to offset " << value_offset << ", length " << value_length << std::endl;

            // Find appropriate leaf node 
            Page* leaf = find_leaf_page(key, transaction_id, true);
            std::cout << "DEBUG: Found leaf page " << leaf->header.page_id<< " with " << leaf->header.key_count << " keys, " << leaf->header.message_count << " messages" << std::endl;

            // Add insert message (fractal tree optimisation - buffer instead of immediate apply)
            add_message_to_node(leaf, MessageType::INSERT, key, value_offset, value_length, transaction_id);


            // Check if we need to split
            if (is_node_full(leaf)) {
                 std::cout << "DEBUG: Node is full, splitting..." << std::endl;
                split_node(leaf, leaf->header.page_id, transaction_id);
            } else {
                std::cout << "DEBUG: Node not full, adaptive flush..." << std::endl;
                adaptive_flush(leaf, leaf->header.page_id, transaction_id);
            }

            release_page(leaf->header.page_id, true);


            std::cout << "INSERT Key= " << key << ", valuesize=" << value_length << ", Txn= " << transaction_id << std::endl;
        } catch (const std::exception& e) {
            //LockManager::clear_thread_locks();
            //DeadlockDetector::clear_thread_locks(std::this_thread::get_id());
            std::cerr << "INSERT failed: " << e.what() << std::endl;
            throw;
        }
    }

    void FractalBPlusTree::update(int64_t key, const std::string& value, uint64_t transaction_id) {

        try {
            // Write data block
            uint64_t value_offset = write_data_block(value);
            uint32_t value_length = value.size();

            // Find leaf node
            Page* leaf = find_leaf_page(key, transaction_id, true);

            // Check if we need to handle this as INSERS vs UPDATE 
            bool key_exists = false;

            // Check base e keys
            int64_t* keys = reinterpret_cast<int64_t*>(leaf->data);
            for (uint32_t i = 0; i < leaf->header.key_count; ++i) {
                if (keys[i] == key) {
                    key_exists = true;
                    break;
                }
            }

            // Check messages for the key
            if (!key_exists) {
                Message* messages = get_messages(leaf);
                for (uint32_t i = 0; i < leaf->header.message_count; ++i) {
                    if (messages[i].key == key && (messages[i].type == MessageType::INSERT || messages[i].type == MessageType::UPDATE)) {
                        key_exists = true;
                        break;
                    }
                }
            }

            // Use insert if key doesn't exist, UPDATE if it does
            MessageType msg_type = key_exists ? MessageType::UPDATE :MessageType::INSERT;

            // Add update
            add_message_to_node(leaf,/* MessageType::UPDATE*/msg_type, key, value_offset, value_length, transaction_id);

            adaptive_flush(leaf, leaf->header.page_id, transaction_id);

            release_page(leaf->header.page_id, true);

            std::cout << "UPDATE Key=" << key << ", ValueSize=" << value_length << ", Txn=" << transaction_id << std::endl;
        } catch (const std::exception& e) {
            //LockManager::clear_thread_locks();
            //DeadlockDetector::clear_thread_locks(std::this_thread::get_id());
            std::cerr << "UPDATE failed: " << e.what() << std::endl;
            throw;
        }
    }

    void FractalBPlusTree::remove(int64_t key, uint64_t transaction_id) {

        try {
            // Find leaf page
            Page* leaf = find_leaf_page(key, transaction_id, true);

            bool key_exists = false;

            // Check base keys
            int64_t* keys = reinterpret_cast<int64_t*>(leaf->data);
            for (uint32_t i = 0; i < leaf->header.key_count; ++i) {
                if (keys[i] == key) {
                    key_exists = true;
                    break;
                }
            }

            // Check messages for the key (non-delete messages)
            if (!key_exists) {
                Message* messages = get_messages(leaf);
                for (uint32_t i = 0; i < leaf->header.message_count; ++i) {
                    if (messages[i].key == key && (messages[i].type == MessageType::INSERT || messages[i].type == MessageType::UPDATE)) {
                        key_exists = true;
                        break;
                    }
                }
            }

            if (!key_exists) {
                release_page(leaf->header.page_id, false);
            }
            // Add delete messages 
            add_message_to_node(leaf, MessageType::DELETE, key, 0, 0, transaction_id);

            adaptive_flush(leaf, leaf->header.page_id, transaction_id);
            release_page(leaf->header.page_id, true);

            std::cout << "DELETE: Key=" << key << ", Txn=" << transaction_id << std::endl;
        } catch (const std::exception& e) {
            //LockManager::clear_thread_locks();
            //DeadlockDetector::clear_thread_locks(std::this_thread::get_id());
            std::cerr << "DELETE failed: " << e.what() << std::endl;
            throw;
        }
    }


    std::string FractalBPlusTree::select(int64_t key, uint64_t transaction_id) {
        Page* leaf = find_leaf_page(key, transaction_id, false);
        
        // Get all data from the node
        int64_t* keys = reinterpret_cast<int64_t*>(leaf->data);
        KeyValue* kvs = get_key_values(leaf);
        Message* messages = get_messages(leaf);
        
        // Build the current state by applying messages to key-values
        std::string current_value = "";
        //std::map<int64_t, std::string> key_state;
        bool value_found = false;
        bool key_exists = false;
        bool key_deleted = false;
        uint64_t latest_version = 0;
        
        // check if key exists in key-values (base state)
        for (uint32_t i = 0; i < leaf->header.key_count; ++i) {
            if (keys[i] == key) {
                current_value = read_data_block(kvs[i].value_offset, kvs[i].value_length);
                //value_found = true;
                key_exists = true;
                latest_version = 0; // Base data has version 0
                break;
            }
        }
        
        // Apply messages in chronological order (oldest to newest)
        for (uint32_t i = 0; i < leaf->header.message_count; ++i) {
            if (messages[i].key == key && messages[i].version <= transaction_id) {
                bool should_apply = false;
                if (transaction_id == 0) {
                    should_apply = true;
                } else {
                    should_apply = (messages[i].version <= transaction_id);
                }
                
                if (should_apply) {
                    if (messages[i].version >= latest_version) {
                        latest_version = messages[i].version;
                        
                        switch (messages[i].type) {
                            case MessageType::INSERT:
                            case MessageType::UPDATE:
                                current_value = read_data_block(messages[i].value_offset, messages[i].value_length);
                                //value_found = true;
                                key_exists = true;
                                key_deleted = true;
                                break;
                            case MessageType::DELETE:
                                current_value = "";
                                //value_found = false;
                                key_exists = false;
                                key_deleted = true;
                                break;
                        }
                    }
                }
            }
        }
        
        release_page(leaf->header.page_id, false);
        
        return key_exists ? current_value : "";
    }

    std::vector<std::pair<int64_t, std::string>> FractalBPlusTree::select_range(int64_t start_key, int64_t end_key, uint64_t transaction_id) {
        std::vector<std::pair<int64_t, std::string>> results;

        // Use a map to track the latest state across all pages
        std::map<int64_t, std::string> global_state;

        Page* current = find_leaf_page(start_key, transaction_id, false);

        while (current != nullptr) {
            // Get node data
            int64_t* keys = reinterpret_cast<int64_t*>(current->data);
            KeyValue* kvs = get_key_values(current);
            Message* messages = get_messages(current);

            // Process base keys first
            for (uint32_t i = 0; i < current->header.key_count; ++i) {
                if (keys[i] >= start_key && keys[i] <= end_key) {
                    try {
                        global_state[keys[i]] = read_data_block(kvs[i].value_offset, kvs[i].value_length);
                    } catch (const std::exception& e) {
                        std::cerr << "Warning: Failed to read base data for key " << keys[i] << ": " << e.what() << std::endl;
                    }
                }
            }

            // Apply messages in order
            for (uint32_t i = 0; i < current->header.message_count; ++i) {
                int64_t msg_key = messages[i].key;

                bool should_apply = (transaction_id == 0) || (messages[i].version <= transaction_id);

                if (msg_key >= start_key && msg_key <= end_key && should_apply) {
                    try {
                        switch (messages[i].type) {
                            case MessageType::INSERT:
                            case MessageType::UPDATE:
                                global_state[msg_key] = read_data_block(messages[i].value_offset, messages[i].value_length);
                                break;
                            case MessageType::DELETE:
                                global_state.erase(msg_key);
                                break;
                        }
                    } catch (const std::exception& e) {
                        std::cerr << "Warning: Failed to apply message for key " << msg_key << ": " << e.what() << std::endl;
                    }
                }
            }

            // Move to next leaf
            uint32_t next_page = current->header.next_page;
            release_page(current->header.page_id, false);

            if (next_page != 0) {
                current = get_page(next_page, false);
            } else {
                current = nullptr;
            }
        }

        // Convert map to results
        for (const auto& [key, value] : global_state) {
            results.emplace_back(key, value);
        }

        return results;
    }

    

    std::vector<std::pair<int64_t, std::string>> FractalBPlusTree::scan_all(uint64_t transaction_id) {
        return select_range(std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max(), transaction_id);
    }

    void FractalBPlusTree::bulk_load(const std::vector<std::pair<int64_t, std::string>>& data, uint64_t transaction_id) {

        if (data.empty()) {
            std::cout << "BULK_LOAD: No data to load" << std::endl;
            return;
        }

        std::cout << "BULK_LOAD: Starting bulk load of " << data.size() << " rows" << std::endl;

        // Pre allocate data region for bulk load
        size_t total_data_size = 0;
        for (const auto& item : data) {
            total_data_size = item.second.size();
        }

        // Extend data region if needed
        try {
            db_file->extend_data_region(table_id, total_data_size + (1024 * 1024)); // Extra 1MB buffer
        } catch (const std::exception& e) {
            std::cout << "BULK_LOAD: Note - " << e.what() << std::endl;
        }

        // Sort data by key
        std::vector<std::pair<int64_t, std::string>> sorted_data = data;
        std::sort(sorted_data.begin(), sorted_data.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });

        // Clear existing tree if it exists
        if (root_page_id != 0) {
            drop();
        }

        create(); // Create new tree
        
        // Bulk load into leaf nodes
        Page* current_leaf = get_page(root_page_id, true);
        size_t data_index = 0;
        size_t leaf_count = 1;

        while (data_index < sorted_data.size()) {
            // Fill current leaf
            while (data_index < sorted_data.size() && current_leaf->header.key_count < BPTREE_ORDER) {
                const auto& [key, value] = sorted_data[data_index];
                uint64_t value_offset = write_data_block(value);

                insert_key_value_into_leaf(current_leaf, key, value_offset, value.size());
                data_index++;
            }

            if (data_index < sorted_data.size()) {
                // Create new leaf
                uint32_t new_leaf_id = allocate_new_page(PageType::LEAF_NODE, transaction_id);
                Page* new_leaf = get_page(new_leaf_id, true);
                leaf_count++;

                // Link leaves
                new_leaf->header.prev_page = current_leaf->header.page_id;
                current_leaf->header.next_page = new_leaf_id;

                release_page(current_leaf->header.page_id, true);
                current_leaf = new_leaf;
            }
        }

        release_page(current_leaf->header.page_id, true);

        // Build internal nodes bottom up
        build_internal_nodes_from_leaves(transaction_id);

        std::cout << "BULK_LOAD: Completed. Loaded " << sorted_data.size() << " records into " << leaf_count << " leaf nodes " << std::endl;

    }

    void FractalBPlusTree::checkpoint() {

        flush_all_messages(0); // Use transaction ID 0 for checkpoint
        wal->log_checkpoint();
        std::cout << "CHECKPOINT: Completed for table '" << table_name << "'" << std::endl;

    }

    void FractalBPlusTree::flush_all_messages(uint64_t transaction_id) {
        std::queue<uint32_t> page_queue;
        page_queue.push(root_page_id);

        size_t total_flushed = 0;

        while (!page_queue.empty()) {
            uint32_t page_id = page_queue.front();
            page_queue.pop();

            Page* node = get_page(page_id, true);

            if (node->header.type == PageType::LEAF_NODE && node->header.message_count > 0) {
                flush_messages_in_node(node, page_id, transaction_id);
                total_flushed += node->header.message_count;
            }

            if (node->header.type == PageType::INTERNAL_NODE) {
                // Add to children queue
                const uint32_t* children = get_child_pointers(node);
                for (uint32_t i = 0; i <= node->header.key_count; ++i) {
                    page_queue.push(children[i]);
                }
            }

            release_page(page_id, true);
        }

        if (total_flushed > 0) {
            std::cout << "FLUSH_ALL_MESSAGES: Flushed " << total_flushed << " messages" << std::endl;
        }
    }

    void FractalBPlusTree::defragment_tree(uint64_t transaction_id) {

        std::cout << "DEFRAGMENT_TREE: Starting defragmentation..." << std::endl;

        // First flush all messages
        flush_all_messages(transaction_id);

        // Then rebuild the tree structure
        auto all_data = scan_all(transaction_id);
        bulk_load(all_data, transaction_id);

        std::cout << "DEFRAGMENT_TREE: Completed" << std::endl;
        
    }

    size_t FractalBPlusTree::get_tree_height() const {

        if (root_page_id == 0) return 0;

        size_t height = 1;
        Page* current = get_page(root_page_id, false);

        while (current->header.type == PageType::INTERNAL_NODE) {
            const uint32_t* children = get_child_pointers(current);
            uint32_t first_child = children[0];
            release_page(current->header.page_id, false);

            current = get_page(first_child, false);
            height++;
        }

        release_page(current->header.page_id, false);
        return height;
    }

    size_t FractalBPlusTree::get_total_pages() const {

        std::vector<uint32_t> all_pages;
        collect_all_leaf_pages(all_pages);
        return all_pages.size();
    }

    std::string FractalBPlusTree::get_tree_stats() const {
        
        std::stringstream ss;
        ss << "=== FractalBPlusTree Statistics ===" << std::endl;
        ss << "Table: " << table_name << std::endl;
        ss << "Root Page ID: " << root_page_id.load() << std::endl;
        ss << "Tree Height: " << get_tree_height() << std::endl;
        ss << "Total Pages: " << get_total_pages() << std::endl;
        ss << "Total Messages Buffered: " << total_messages.load() << std::endl;
        ss << "Memory Usage: " << memory_usage.load() << " bytes" << std::endl;
        ss << "===================================" << std::endl;
        
        return ss.str();
    }

    void FractalBPlusTree::print_tree_structure() const {

        std::stringstream ss;
        ss << "=== FractalBPlusTree structure ===" << std::endl;
        print_subtree(root_page_id, 0, ss);
        ss << "==================================" << std::endl;

        std::cout << ss.str();
    }

    Page* FractalBPlusTree::find_leaf_page(int64_t key, uint64_t transaction_id, bool exclusive) {
        std::cout << "DEBUG: find_leaf_page(key=" << key << ", txn=" << transaction_id << ", exclusive=" << exclusive << ")" << std::endl;

        if (root_page_id == 0) {
            throw std::runtime_error("Tree is empty");
        }

         std::cout << "DEBUG: Starting from root page: " << root_page_id << std::endl;

        std::vector<uint32_t> page_chain;
        Page* current = get_page(root_page_id, exclusive);
        page_chain.push_back(root_page_id);

        // Traverse down to leaf
        while (current->header.type == PageType::INTERNAL_NODE) {
            std::cout << "DEBUG: Internal node " << current->header.page_id << " with " << current->header.key_count << " keys" << std::endl;

            uint32_t child_index = find_child_index(current, key);
            const uint32_t* children = get_child_pointers(current);

            uint32_t child_page_id = children[child_index];
            std::cout << "DEBUG: Following child pointer at index " << child_index << " to page " << child_page_id << std::endl;
            Page* child = get_page(child_page_id, exclusive);
            page_chain.push_back(child_page_id);

            if (page_chain.size() > 1) {
                release_page(page_chain[page_chain.size() - 3], false);
            }
            current = child;
        }
        
        std::cout << "DEBUG: Found leaf page: " << current->header.page_id << std::endl;

        // Release all but the last page (the leaf we're returning)
        for (size_t i = 0; i < page_chain.size() - 1; ++i) {
            release_page(page_chain[i], false);
        }

        return current;
    }

    void FractalBPlusTree::split_node(Page* node, uint32_t page_id, uint64_t transaction_id) {
        std::cout << "SPLIT_NODE: Page " << page_id << " is full, splitting..." << std::endl;

        // Create new node
        uint32_t new_page_id = allocate_new_page(node->header.type, transaction_id);
        Page* new_node = get_page(new_page_id, true);

        // Initialize new node
        new_node->initialize(new_page_id, node->header.type, node->header.database_id);
        new_node->header.parent_id = node->header.parent_id;

        if (node->header.type == PageType::LEAF_NODE) {
            // Split leaf node
            uint32_t split_point = node->header.key_count / 2;
            int64_t split_key = reinterpret_cast<int64_t*>(node->data)[split_point];

            // Get keys and values from original node
            int64_t* keys = reinterpret_cast<int64_t*>(node->data);
            KeyValue* kvs = get_key_values(node);

            // Move second half to new node
            for (uint32_t i = split_point; i < node->header.key_count; ++i) {
                insert_key_value_into_leaf(new_node, keys[i], kvs[i].value_offset, kvs[i].value_length);
            }

            // Update original node
            node->header.key_count = split_point;

            // Update leaf links
            new_node->header.next_page = node->header.next_page;
            new_node->header.prev_page = page_id;
            node->header.next_page = new_page_id;

            if (new_node->header.next_page != 0) {
                Page* next_leaf = get_page(new_node->header.next_page, true);
                next_leaf->header.prev_page = new_page_id;
                release_page(next_leaf->header.page_id, true);
            }

            // Update parent or create new root
            if(node->header.parent_id == 0) {
                create_new_root(page_id, new_page_id, split_key, transaction_id);
            } else {
                update_parent_after_split(node->header.parent_id, page_id, new_page_id, split_key, transaction_id);
            }
        } else {
            // Split internal node
            uint32_t split_point = node->header.key_count / 2;
            int64_t* keys = reinterpret_cast<int64_t*>(node->data);
            const uint32_t* children = get_child_pointers(node);
            int64_t split_key = keys[split_point];

            // Move second half to new node
            for (uint32_t i = split_point + 1; i < node->header.key_count; ++i) {
                insert_key_into_internal(new_node, keys[i], children[i]);
            }

            // Add the last child pointer
            set_child_pointers(new_node, new_node->header.key_count, children[node->header.key_count]);

            // Update original node
            node->header.key_count = split_point;

            // Update parent or create new root
            if (node->header.parent_id == 0) {
                create_new_root(page_id, new_page_id, split_key, transaction_id);
            } else {
                update_parent_after_split(node->header.parent_id, page_id, new_page_id, split_key, transaction_id);
            }

            // Update parent references for children of new node
            update_children_parent_references(new_node, new_page_id, transaction_id);
        }

        // Log the split operation
        wal->log_tree_operation(transaction_id, "SPLIT_NODE", 10);
        release_page(new_page_id, true);
        std::cout << "SPLIT_NODE: Completed. New page: " << new_page_id << std::endl;
    }

    void FractalBPlusTree::merge_nodes(Page* left, Page* right, uint64_t transaction_id) {
        if (left->header.parent_id == 0) {
            return; // Can't merge root
        }

        std::cout << "MERGE_NODES: Page " << left->header.page_id << " and "  << right->header.page_id << " merging... " << std::endl;

        // Get parent node
        Page* parent = get_page(left->header.parent_id, true);

        if (left->header.type == PageType::LEAF_NODE) {
            merge_leaf_nodes(left, right, parent, transaction_id);
        } else {
            merge_internal_nodes(left, right, parent, transaction_id);
        }

        release_page(parent->header.page_id, true);
    }

    void FractalBPlusTree::merge_leaf_nodes(Page* left, Page* right, Page* parent, uint64_t transaction_id) {
        uint32_t left_page_id = left->header.page_id;
        uint32_t right_page_id = right->header.page_id;

        // Check if merge is possible
        size_t total_keys = left->header.key_count + right->header.key_count;
        size_t total_messages = left->header.message_count + right->header.message_count;

        if (total_keys > BPTREE_ORDER || total_messages > MAX_MESSAGES) {
            std::cout << "MERGE_LEAF_NODE: Cannot merge - would exceed capacity " << std::endl;
           return;
        }

       // Flush messages from both nodes first
       if (left->header.message_count > 0) {
          flush_messages_in_node(left, left_page_id, transaction_id);
       }
      if (right->header.message_count > 0) {
         flush_messages_in_node(right, right_page_id, transaction_id);
      }

      // Get data pointers
      int64_t* left_keys = reinterpret_cast<int64_t*>(left->data);
      KeyValue* left_kvs = get_key_values(left);
      int64_t* right_keys = reinterpret_cast<int64_t*>(right->data);
      KeyValue* right_kvs = get_key_values(right);

      // Calculate available space
      size_t left_used = left->header.key_count * (sizeof(int64_t) + sizeof(KeyValue));
      size_t right_used = right->header.key_count * (sizeof(int64_t) + sizeof(KeyValue));
      size_t available_space = PAGE_SIZE - sizeof(PageHeader) - left_used;

      // Determine how many keys to move
      uint32_t keys_to_move = 0;
      while (keys_to_move < right->header.key_count && (keys_to_move * (sizeof(int64_t) + sizeof(KeyValue))) < available_space) {
          keys_to_move++;
      }

      if (keys_to_move == 0) {
          std::cout << "MERGE_LEAF_NODES: No keys to move" << std::endl;
          return;
      }

      // Move keys and values from right to left
      for (uint32_t i = 0; i < keys_to_move; ++i) {
          uint32_t left_index = left->header.key_count + i;
          uint32_t right_index = i;

          left_keys[left_index] = right_keys[right_index];
          left_kvs[left_index] = right_kvs[right_index];
      }

      left->header.key_count += keys_to_move;

      // Remove moved keys from right node
      if (keys_to_move < right->header.key_count) {
          // Some keys remain in the right node
          uint32_t remaining_keys = right->header.key_count - keys_to_move;
          std::memmove(right_keys, &right_keys[keys_to_move], remaining_keys * sizeof(int64_t));
          std::memmove(right_kvs, &right_kvs[keys_to_move], remaining_keys * sizeof(KeyValue));
          right->header.key_count = remaining_keys;
      } else {
          // All keys moved - right node becomes empty
          right->header.key_count = 0;

          // Update leaf chain links
          left->header.next_page = right->header.next_page;
          if (right->header.next_page != 0){
              Page* next_leaf = get_page(right->header.next_page, true);
              next_leaf->header.prev_page = left_page_id;
              release_page(next_leaf->header.page_id, true);
          }

          // Remove right node from parent and free it
          remove_node_from_parent(parent, right_page_id, transaction_id);
          free_page(right_page_id, transaction_id);
      }

      // Update parent's key if this was a complete merge
      if (right->header.key_count == 0) {
          update_parent_key_after_merge(parent, left_page_id, right_page_id, transaction_id);
      }

      std::cout << "MERGE_LEAF_NODES: Moved " << keys_to_move << " keys from " << right_page_id << " to " << left_page_id << std::endl;

      // Log the merge operation
      wal->log_tree_operation(transaction_id, "MERGE_LEAF_NODES", 15);
    }

    void FractalBPlusTree::merge_internal_nodes(Page* left, Page* right, Page* parent, uint64_t transaction_id) {
        uint32_t left_page_id = left->header.page_id;
        uint32_t right_page_id = right->header.page_id;

        // Check if merge is possible
        size_t total_keys = left->header.key_count + right->header.key_count + 1; // +1 for separator key
        if (total_keys > BPTREE_ORDER) {
            std::cout << "MERGE_INTERNAL_NODES: Cannot merge - would exceed capacity" << std::endl;
            return;
        }

        // Find separator key in parent
        int64_t separator_key = find_separator_key(parent, left_page_id, right_page_id);

        // Add separator key to left node
        insert_key_into_internal(left, separator_key, 0); // Child pointer will be updated

        // Get data pointers
        int64_t* left_keys = reinterpret_cast<int64_t*>(left->data);
        uint32_t* left_children = get_child_pointers(left);
        int64_t* right_keys = reinterpret_cast<int64_t*>(right->data);
        const uint32_t* right_children = get_child_pointers(right);

        // Move all keys and children from right to left
        for (uint32_t i = 0; i < right->header.key_count; ++i) {
            insert_key_into_internal(left, right_keys[i], right_children[i]);
        }

        // Add the last child pointer from right node
        set_child_pointers(left, left->header.key_count, right_children[right->header.key_count]);

        // Update parent references for moved children
        update_children_parent_references(left, left_page_id, transaction_id);

        // Remove right node from parent and free it
        remove_node_from_parent(parent, right_page_id, transaction_id);
        free_page(right_page_id, transaction_id);

        std::cout << "MERGE_INTERNAL_NODES: Merged " << right_page_id << " into " << left_page_id << std::endl;

        // Log the merge operation
        wal->log_tree_operation(transaction_id, "MERGE_INTERNAL_NODES", 19);
    }

    void FractalBPlusTree::redistribute_keys(Page* left, Page* right, uint64_t transaction_id) {
        if (left->header.type == PageType::LEAF_NODE) {
            redistribute_leaf_keys(left, right, transaction_id);
        } else {
            redistribute_internal_keys(left, right, transaction_id);
        }
    }

    void FractalBPlusTree::redistribute_leaf_keys(Page* left, Page* right, uint64_t transaction_id) {
        uint32_t total_keys = left->header.key_count + right->header.key_count;
        uint32_t target_keys = total_keys / 2;

        if (left->header.key_count < target_keys) {
            redistribute_right_to_left(left, right, target_keys, transaction_id);
        } else {
            redistribute_left_to_right(left, right, target_keys, transaction_id);
        }

        // Update parent separator key
        update_parent_separator_key(left, right, transaction_id);
    }

    void FractalBPlusTree::redistribute_right_to_left(Page* left, Page* right, uint32_t target_keys, uint64_t transaction_id) {
        uint32_t keys_needed = target_keys - left->header.key_count;
        if (keys_needed == 0) return;

        // Get data pointers
        int64_t* left_keys = reinterpret_cast<int64_t*>(left->data);
        KeyValue* left_kvs = get_key_values(left);
        int64_t* right_keys = reinterpret_cast<int64_t*>(right->data);
        KeyValue* right_kvs = get_key_values(right);

        // Move keys from right to left
        for (uint32_t i = 0; i < keys_needed; ++i) {
            uint32_t left_index = left->header.key_count + i;
            uint32_t right_index = i;

            left_keys[left_index] = right_keys[right_index];
            left_kvs[left_index] = right_kvs[right_index];
        }

        // Update counts
        left->header.key_count += keys_needed;

        // Remove moved keys from right
        uint32_t remaining_keys = right->header.key_count - keys_needed;
        std::memmove(right_keys, &right_keys[keys_needed], remaining_keys * sizeof(int64_t));
        std::memmove(right_kvs, &right_kvs[keys_needed], remaining_keys * sizeof(KeyValue));
        right->header.key_count = remaining_keys;

        std::cout << "REDISTRIBUTE_RIGHT_TO_LEFT: Moved " << keys_needed << " keys" << std::endl;
    }

    void FractalBPlusTree::redistribute_left_to_right(Page* left, Page* right, uint32_t target_keys, uint64_t transaction_id) {
        uint32_t keys_to_move = left->header.key_count - target_keys;
        if (keys_to_move == 0) return;

        // Get data pointers
        int64_t* left_keys = reinterpret_cast<int64_t*>(left->data);
        KeyValue* left_kvs = get_key_values(left);
        int64_t* right_keys = reinterpret_cast<int64_t*>(right->data);
        KeyValue* right_kvs = get_key_values(right);

        // Make space in right node
        std::memmove(&right_keys[keys_to_move], right_keys, right->header.key_count * sizeof(int64_t));
        std::memmove(&right_kvs[keys_to_move], right_kvs, right->header.key_count * sizeof(KeyValue));

        // Move keys from left to right
        for (uint32_t i = 0; i < keys_to_move; ++i) {
            uint32_t left_index = left->header.key_count - keys_to_move + i;
            uint32_t right_index = i;

            right_keys[right_index] = left_keys[left_index];
            right_kvs[right_index] = left_kvs[left_index];
        }

        // Update counts
        left->header.key_count -= keys_to_move;
        right->header.key_count += keys_to_move;

        std::cout << "REDISTRIBUTE_LEFT_TO_RIGHT: Moved " << keys_to_move << " keys" << std::endl;
    }

    void FractalBPlusTree::add_message_to_node(Page* node, MessageType type, int64_t key, uint64_t value_offset, uint32_t value_length, uint64_t transaction_id) {
        if (node->header.message_count >= MAX_MESSAGES) {
            flush_messages_in_node(node, node->header.page_id, transaction_id);
        }

        Message* messages = get_messages(node);
        uint32_t message_index = node->header.message_count;

        messages[message_index].type = type;
        messages[message_index].key = key;
        messages[message_index].value_offset = value_offset;
        messages[message_index].value_length = value_length;
        messages[message_index].version = transaction_id;
        messages[message_index].timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

        node->header.message_count++;
        total_messages++;
        memory_usage += sizeof(Message);

        std::string message_data = std::to_string(static_cast<int>(type)) + ":" + std::to_string(key) + ":" + std::to_string(value_offset) + ":" + std::to_string(value_length);

        // Log message addition
        wal->log_message_buffer(transaction_id,node->header.page_id, message_data.c_str(), message_data.size());

        std::cout << "ADD_MESSAGE: Type=" << static_cast<int>(type) << ", Key=" << key << ", Txn=" << transaction_id << std::endl;
    }

    void FractalBPlusTree::flush_messages_in_node(Page* node, uint32_t page_id, uint64_t transaction_id) {
        if (node->header.message_count == 0) return;

        std::cout << "FLUSH_MESSAGES: Flushing " << node->header.message_count << " messages from page " << page_id << std::endl;

        // Apply messages to key-values
        apply_messages_to_key_values(node);

        // Reset message count
        total_messages -= node->header.message_count;
        memory_usage -= node->header.message_count * sizeof(Message);
        node->header.message_count = 0;

        // Log flush operation
        wal->log_tree_operation(transaction_id, "FLUSH_MESSAGES", 13);
    }


    void FractalBPlusTree::apply_messages_to_key_values(Page* node) {
        if (node->header.message_count == 0) return;
        
        Message* messages = get_messages(node);
   
        // Sort messages by key and version for deterministic application
        std::vector<Message> sorted_messages(messages, messages + node->header.message_count);
        std::sort(sorted_messages.begin(), sorted_messages.end(), [](const Message& a, const Message& b) {
                if (a.key != b.key) return a.key < b.key;
                return a.version < b.version; // Older versions first
         });
        
        // Get current key-values
        int64_t* keys = reinterpret_cast<int64_t*>(node->data);
        KeyValue* kvs = get_key_values(node);
   
        // Use a map to track the latest state of each key
        std::map<int64_t, std::pair<bool,KeyValue>> final_state;
        
        // Initialize with existing key value
        for (uint32_t i = 0; i < node->header.key_count; ++i) {
            if (keys[i] != kvs[i].value_offset > 0 && kvs[i].value_length > 0 && kvs[i].value_length < 65536) {
                final_state[keys[i]] = {true,kvs[i]};
                std::cout << "DEBUG: Initial valid  key " << keys[i] << " with offset " << kvs[i].value_offset << std::endl;
            } else {
                std::cout << "DEBUG: Skipping invalid base key " << keys[i] << " offset=" << kvs[i].value_offset<< " length=" << kvs[i].value_length << std::endl;
            }
        }
        
        // Apply messages in order
        for (const auto& message : sorted_messages) {
            std::cout << "DEBUG: Applying message key=" << message.key<< " type=" << static_cast<int>(message.type) << " offset=" << message.value_offset << "  length=" << message.value_length << std::endl;

            if (message.value_length > 65536) {
                std::cerr << "WARNING: Skipping message with invalid length: " << message.value_length << std::endl;
                continue;
            }
            switch (message.type) {
                case MessageType::INSERT:
                case MessageType::UPDATE:
                    final_state[message.key] = {true,KeyValue{message.key, message.value_offset, message.value_length}};
                    break;
                case MessageType::DELETE:
                    final_state.erase(message.key);
                    //final_state[message.key] = {false, KeyValue{}}; // Mark as deleted
                    break;
            }
        }
        
        // Clear and rebuild the node's key-values
        node->header.key_count = 0;

        // Only clear the key-value area, not the entire section
        size_t kv_area_size = BPTREE_ORDER * (sizeof(int64_t) + sizeof(KeyValue));
        std::memset(node->data, 0, kv_area_size);
        //std::memset(node->data, 0, PAGE_SIZE - sizeof(PageHeader));
        
        keys = reinterpret_cast<int64_t*>(node->data);
        kvs = get_key_values(node);
        
        uint32_t new_index = 0;
        for (const auto& [key, state] : final_state) {
            if (new_index >= BPTREE_ORDER) {
                throw std::runtime_error("Node overflow during message application");
            }

            /*if (kv.value_length == 0 || kv.value_length > 65536) {
                std::cerr << "WARNING: Skipping key " << key << " with invalid length " << kv.value_length << std::endl;
                    continue;
            }*/
            
            const auto& [is_valid, kv] = state;
            if (is_valid && kv.value_length > 0 && kv.value_length <= 65536) {
                keys[new_index] = key;
                kvs[new_index] = kv;
                std::cout << "DEBUG: Final key " << key << " at index " << new_index <<  " offset=" << kv.value_offset << " length=" << kv.value_length << std::endl;
                new_index++;
            }
            // Skip deled entries
        }
        
        node->header.key_count = new_index;
        
        // Clear messages after application
        total_messages -= node->header.message_count;
        memory_usage -= node->header.message_count * sizeof(Message);
        node->header.message_count = 0;
        std::cout << "DEBUG: apply_messages_to_key_values completed. Final key count: " << new_index << std::endl;
    }

    void FractalBPlusTree::apply_single_message(Page* node, const Message& message) {
        int64_t* keys = reinterpret_cast<int64_t*>(node->data);
        KeyValue* kvs = get_key_values(node);

        switch (message.type) {
            case MessageType::INSERT:
            case MessageType::UPDATE: {
                // Find existing key or insert new
                uint32_t key_index = 0;
                while (key_index < node->header.key_count && keys[key_index] < message.key) {
                    key_index++;
                }

                if (key_index < node->header.key_count && keys[key_index] == message.key) {
                    // Update existing key
                    kvs[key_index].value_offset = message.value_offset;
                    kvs[key_index].value_length = message.value_length;
                } else {
                    // Insert new key
                    if (node->header.key_count >= BPTREE_ORDER) {
                        throw std::runtime_error("Node full during message application");
                    }

                    // Make space for new key
                    std::memmove(&keys[key_index + 1], &keys[key_index], (node->header.key_count - key_index) * sizeof(int64_t));
                    std::memmove(&kvs[key_index + 1], &kvs[key_index], (node->header.key_count - key_index) * sizeof(KeyValue));

                    // Insert new key-value
                    keys[key_index] = message.key;
                    kvs[key_index].value_offset = message.value_offset;
                    kvs[key_index].value_length = message.value_length;
                    node->header.key_count++;
                }
                break;
            }
            case MessageType::DELETE: {
                // Find and remove key
                uint32_t key_index = 0;
                while (key_index < node->header.key_count && keys[key_index] != message.key) {
                    key_index++;
                }

                if (key_index < node->header.key_count) {
                    // Remove key
                    std::memmove(&keys[key_index], &keys[key_index + 1], (node->header.key_count - key_index - 1) * sizeof(int64_t));
                    std::memmove(&kvs[key_index], &kvs[key_index + 1], (node->header.key_count - key_index - 1) * sizeof(KeyValue));
                    node->header.key_count--;
                }
                break;
            }
        }
    }


    void FractalBPlusTree::compact_messages(Page* node) {
        if (node->header.message_count <= 1) return;
        
        Message* messages = get_messages(node);
        
        // Group messages by key and keep only the latest version
        std::map<int64_t, Message> latest_messages;
        for (uint32_t i = 0; i < node->header.message_count; ++i) {
            const Message& msg = messages[i];
            if (latest_messages.find(msg.key) == latest_messages.end() || msg.version > latest_messages[msg.key].version) {
                latest_messages[msg.key] = msg;
            }
        }
        
        // Copy back only the latest messages
        node->header.message_count = 0;
        Message* msg_ptr = get_messages(node);
   
        for (const auto& [key, msg] : latest_messages) {
            if (node->header.message_count < MAX_MESSAGES) {
                msg_ptr[node->header.message_count++] = msg;
            }
        }
        
        // Update statistics
        total_messages = node->header.message_count;
        memory_usage = node->header.message_count * sizeof(Message);
    }

    void FractalBPlusTree::insert_key_value_into_leaf(Page* leaf, int64_t key, uint64_t value_offset, uint32_t value_length) {
        if (value_length == 0 || value_length > 65536) {
            throw std::runtime_error("Invalid value length: " + std::to_string(value_length));
        }

        if (leaf->header.key_count >= BPTREE_ORDER) {
            throw std::runtime_error("Leaf node is full");
        }

        int64_t* keys = reinterpret_cast<int64_t*>(leaf->data);
        KeyValue* kvs = get_key_values(leaf);

        // Find insertion point
        uint32_t insert_index = 0;
        while (insert_index < leaf->header.key_count && keys[insert_index] < key) {
            insert_index++;
        }

        // Make space
        if (insert_index < leaf->header.key_count) {
            std::memmove(&keys[insert_index + 1], &keys[insert_index], (leaf->header.key_count - insert_index) * sizeof(int64_t));
            std::memmove(&kvs[insert_index + 1], &kvs[insert_index], (leaf->header.key_count - insert_index) * sizeof(KeyValue));
        }

        // Insert new key-value
        keys[insert_index] = key;
        kvs[insert_index].key = key;
        kvs[insert_index].value_offset = value_offset;
        kvs[insert_index].value_length = value_length;
        leaf->header.key_count++;

        std::cout << "DEBUG: insert_key_value_into_leaf - key: " << key << ", offset: " << value_offset<< ", length: " << value_length << std::endl;
    }

    void FractalBPlusTree::insert_key_into_internal(Page* internal, int64_t key, uint32_t child_page_id) {
        int64_t* keys = reinterpret_cast<int64_t*>(internal->data);
        uint32_t* children = get_child_pointers(internal);

        // Find insertion point for key
        uint32_t insert_index = 0;
        while (insert_index < internal->header.key_count && keys[insert_index] < key) {
            insert_index++;
        }

        // Make space for key
        std::memmove(&keys[insert_index + 1], &keys[insert_index], (internal->header.key_count - insert_index) * sizeof(int64_t));
        
        // Make space for child pointer (children array has key_count + 1 elements)
        std::memmove(&children[insert_index + 2], &children[insert_index + 1], (internal->header.key_count - insert_index) * sizeof(uint32_t));

        // Insert key and child pointer
        keys[insert_index] = key;
        children[insert_index + 1] = child_page_id;
        internal->header.key_count++;
    }

    void FractalBPlusTree::remove_key_from_leaf(Page* leaf, int64_t key) {
        int64_t* keys = reinterpret_cast<int64_t*>(leaf->data);
        KeyValue* kvs = get_key_values(leaf);

        // Find key index
        uint32_t key_index = 0;
        while (key_index < leaf->header.key_count && keys[key_index] != key) {
            key_index++;
        }

        if (key_index < leaf->header.key_count) {
            // Remove key
            std::memmove(&keys[key_index], &keys[key_index + 1], (leaf->header.key_count - key_index - 1) * sizeof(int64_t));
            std::memmove(&kvs[key_index], &kvs[key_index + 1], (leaf->header.key_count - key_index - 1) * sizeof(KeyValue));
            leaf->header.key_count--;
        }
    }

    void FractalBPlusTree::remove_key_from_internal(Page* internal, int64_t key) {
        int64_t* keys = reinterpret_cast<int64_t*>(internal->data);
        uint32_t* children = get_child_pointers(internal);

        // Find key index
        uint32_t key_index = 0;
        while (key_index < internal->header.key_count && keys[key_index] != key) {
            key_index++;
        }

        if (key_index < internal->header.key_count) {
            // Remove key
            std::memmove(&keys[key_index], &keys[key_index + 1], (internal->header.key_count - key_index - 1) * sizeof(int64_t));
            
            // Remove child pointer (children array has key_count + 1 elements)
            std::memmove(&children[key_index + 1], &children[key_index + 2], (internal->header.key_count - key_index - 1) * sizeof(uint32_t));
            
            internal->header.key_count--;
        }
    }

    Page* FractalBPlusTree::get_page(uint32_t page_id, bool exclusive) {
        return buffer_pool->get_page(page_id, exclusive);
    }

    Page* FractalBPlusTree::get_page(uint32_t page_id, bool exclusive) const {
        return buffer_pool->get_page(page_id, exclusive);
    }

    void FractalBPlusTree::release_page(uint32_t page_id, bool dirty) {
        buffer_pool->release_page(page_id, dirty);
    }

    void FractalBPlusTree::release_page(uint32_t page_id, bool dirty) const {
        buffer_pool->release_page(page_id, dirty);
    }

    uint32_t FractalBPlusTree::allocate_new_page(PageType type, uint64_t transaction_id) {
        uint32_t page_id = db_file->allocate_page(type);
        Page* page = get_page(page_id, true);
        page->initialize(page_id, type, 0); // database_id 0 for now
        release_page(page_id, true);

        //wal->log_page_operation(transaction_id, page_id, type);
        wal->log_tree_operation(transaction_id, "PAGE_ALLOC", 10);

        return page_id;
    }

    void FractalBPlusTree::free_page(uint32_t page_id, uint64_t transaction_id) {
        db_file->free_page(page_id);
        wal->log_tree_operation(transaction_id, "PAGE_FREE", 9);
    }

    uint32_t FractalBPlusTree::find_child_index(const Page* node, int64_t key) {
        const int64_t* keys = reinterpret_cast<const int64_t*>(node->data);
        
        uint32_t index = 0;
        while (index < node->header.key_count && key >= keys[index]) {
            index++;
        }
        return index;
    }

    uint32_t FractalBPlusTree::find_key_index(const Page* node, int64_t key) {
        const int64_t* keys = reinterpret_cast<const int64_t*>(node->data);
        
        for (uint32_t i = 0; i < node->header.key_count; ++i) {
            if (keys[i] == key) {
                return i;
            }
        }
        return node->header.key_count; // Not found
    }

    int64_t FractalBPlusTree::get_min_key(const Page* node) const {
        if (node->header.key_count == 0) {
            return std::numeric_limits<int64_t>::max();
        }
        const int64_t* keys = reinterpret_cast<const int64_t*>(node->data);
        return keys[0];
    }

    int64_t FractalBPlusTree::get_max_key(const Page* node) const {
        if (node->header.key_count == 0) {
            return std::numeric_limits<int64_t>::min();
        }
        const int64_t* keys = reinterpret_cast<const int64_t*>(node->data);
        return keys[node->header.key_count - 1];
    }

    bool FractalBPlusTree::is_node_full(const Page* node) const {
        if (node->header.type == PageType::LEAF_NODE) {
            size_t used_space = get_used_space(node);
            return used_space >= PAGE_SIZE * FLUSH_RATIO || node->header.key_count >= BPTREE_ORDER;
        } else {
            return node->header.key_count >= BPTREE_ORDER;
        }
    }

    bool FractalBPlusTree::is_node_underfull(const Page* node) const {
        if (node->header.type == PageType::LEAF_NODE) {
            return node->header.key_count < BPTREE_ORDER / 2;
        } else {
            return node->header.key_count < (BPTREE_ORDER + 1) / 2 - 1;
        }
    }

    size_t FractalBPlusTree::get_used_space(const Page* node) const {
        size_t space = sizeof(PageHeader);
        
        if (node->header.type == PageType::LEAF_NODE) {
            space += node->header.key_count * (sizeof(int64_t) + sizeof(KeyValue));
            space += node->header.message_count * sizeof(Message);
        } else {
            space += node->header.key_count * sizeof(int64_t);
            space += (node->header.key_count + 1) * sizeof(uint32_t);
        }
        
        return space;
    }

    size_t FractalBPlusTree::get_available_space(const Page* node) const {
        return PAGE_SIZE - get_used_space(node);
    }

    bool FractalBPlusTree::should_flush_messages(const Page* node) const {
        return node->header.message_count >= MAX_MESSAGES * FLUSH_RATIO || memory_usage_high();
    }


    void FractalBPlusTree::adaptive_flush(Page* node, uint32_t page_id, uint64_t transaction_id) {
        // Only flush under specific conditions to maintain performance
        bool should_flush = node->header.message_count >= MAX_MESSAGES || // At capacity
        (node->header.message_count >= 10) && (node->header.message_count > 0 && get_available_space(node) < PAGE_SIZE * 0.4) || // Low space
        memory_usage_high() || // System memory pressure
        transaction_id % 50 == 0;
    
        if (should_flush) {
            flush_messages_in_node(node, page_id, transaction_id);
        } else if (node->header.message_count > MAX_MESSAGES * 0.6) {
            // Compact if we have a good number of messages but not critical
            compact_messages(node);
        }
        // Otherwise, just keep buffering messages for performance
    }

    bool FractalBPlusTree::memory_usage_high() const {
        return memory_usage.load() >= MEMORY_THRESHOLD;
    }


    uint64_t FractalBPlusTree::write_data_block(const std::string& data) {
        try {
            // Ensure table_id is valid
            if (table_id == 0) {
                // Try to get table_id from db_file if not set
                try {
                    table_id = db_file->get_table_id(table_name);
                } catch (...) {
                    throw std::runtime_error("Invalid table_id for data block allocation");
                }
            }

            // Validation for data size
            if (data.size() > 65536) { // 1MB limit per value
                throw std::runtime_error("Data too large for storage: " + std::to_string(data.size()));
            }

            // Store the actual data length separately for verification
            uint32_t actual_length = data.size();

            uint64_t offset = db_file->allocate_data_block(table_id, sizeof(uint32_t) + actual_length);
        
            // First write the length
            db_file->write_data_block(offset, reinterpret_cast<const char*>(&actual_length), sizeof(uint32_t));
            // Write actual data
            db_file->write_data_block(offset + sizeof(uint32_t), data.data(), actual_length);

            std::cout << "DEBUG: write_data_block - offset: " << offset << ", actual_length: " << actual_length << ", table_id: " << table_id << std::endl;
        
            return offset;
        } catch (const std::exception& e) {
            std::cerr << "Failed to write data block for table " << table_name << " (id: " << table_id << "): " << e.what() << std::endl;
            throw;
        }
    }

    /*std::string FractalBPlusTree::read_data_block(uint64_t offset, uint32_t length) {
        std::cout << "DEBUG: read_data_block(offset=" << offset << ", length=" << length << ")" << std::endl;
        try {
            // Create buffer and read data
            std::vector<char> buffer(length);
            db_file->read_data_block(offset, buffer.data(), length);

            std::string result = std::string(buffer.data(), length);
            std::cout << "DEBUG: read_data_block returning: '" << result << "'" << std::endl;

            // Convert to string
            //return std::string(buffer.data(), length);
             return result;
        } catch (const std::exception& e) {
            std::cerr << "Faied to read data block at offset " << offset << ": " << e.what() << std::endl;
            throw;
        }
    }*/

    std::string FractalBPlusTree::read_data_block(uint64_t offset, uint32_t length) {
    std::cout << "DEBUG: read_data_block(offset=" << offset << ", length=" << length << ")" << std::endl;

    // Validate inputs
    if (length == 0 || length > 65536) {
        throw std::runtime_error("Invalid data length: " + std::to_string(length));
    }

    if (offset == 0) {
        throw std::runtime_error("Invalid data offset: 0");
    }

    try {
        // Read the actual length that we stored
        uint32_t actual_length;
        db_file->read_data_block(offset, reinterpret_cast<char*>(&actual_length), sizeof(uint32_t));

        if (actual_length > 65536) {
            throw std::runtime_error("Corrupted data length: " + std::to_string(actual_length));
        }

        if (actual_length != length) {
            std::cout << "WARNING: Length mismatch in read_data_block! "
                      << "stored_length=" << length << ", actual_length=" << actual_length << std::endl;
            // Use the smaller length for safety
            length = std::min(length, actual_length);
        }

        std::vector<char> buffer(length);
        db_file->read_data_block(offset + sizeof(uint32_t), buffer.data(), length);

        // Validate the data isn't all zeros
        bool all_zeros = std::all_of(buffer.begin(), buffer.end(), [](char c) { return c == 0; });
        if (all_zeros) {
            throw std::runtime_error("Read zero-filled data block at offset " + std::to_string(offset));
        }

        std::string result(buffer.data(), length);
        std::cout << "DEBUG: read_data_block returning string of length " << result.size() << std::endl;
        return result;
    } catch (const std::exception& e) {
        std::cerr << "Failed to read data block at offset " << offset << " length " << length << ": " << e.what() << std::endl;
        throw;
    }
}

    uint32_t FractalBPlusTree::find_left_sibling(uint32_t page_id, uint32_t parent_page_id) {
        Page* parent = get_page(parent_page_id, false);
        const uint32_t* children = get_child_pointers(parent);
        
        for (uint32_t i = 0; i <= parent->header.key_count; ++i) {
            if (children[i] == page_id && i > 0) {
                uint32_t left_sibling = children[i - 1];
                release_page(parent_page_id, false);
                return left_sibling;
            }
        }
        
        release_page(parent_page_id, false);
        return 0; // No left sibling
    }

    uint32_t FractalBPlusTree::find_right_sibling(uint32_t page_id, uint32_t parent_page_id) {
        Page* parent = get_page(parent_page_id, false);
        const uint32_t* children = get_child_pointers(parent);
        
        for (uint32_t i = 0; i <= parent->header.key_count; ++i) {
            if (children[i] == page_id && i < parent->header.key_count) {
                uint32_t right_sibling = children[i + 1];
                release_page(parent_page_id, false);
                return right_sibling;
            }
        }
        
        release_page(parent_page_id, false);
        return 0; // No right sibling
    }

    std::vector<uint32_t> FractalBPlusTree::collect_all_pages_safely() {
        std::vector<uint32_t> all_pages;
        if (root_page_id == 0) return all_pages;

        std::queue<uint32_t> page_queue;
        std::unordered_set<uint32_t> visited_pages;
    
        page_queue.push(root_page_id);
        visited_pages.insert(root_page_id);

        //Also track the table's expected page range for validation
        uint32_t table_range_start = root_page_id;
        uint32_t table_range_end = root_page_id + TABLE_PAGE_RANGE_SIZE -1;

        while (!page_queue.empty()) {
            uint32_t page_id = page_queue.front();
            page_queue.pop();

            if (page_id < table_range_start || page_id > table_range_end) {
                std::cerr << "WARNING: Page " << page_id << " outside expected table range "<< table_range_start << "-" << table_range_end << std::endl;
            }

            all_pages.push_back(page_id);


            try {
                Page* node = get_page(page_id, false);
            
                if (node->header.type == PageType::INTERNAL_NODE) {
                    const uint32_t* children = get_child_pointers(node);
                    for (uint32_t i = 0; i <= node->header.key_count; ++i) {
                        uint32_t child_id = children[i];
                        if (child_id != 0 && visited_pages.find(child_id) == visited_pages.end()) {
                            // Validate child is within table range
                            if (child_id >= table_range_start && child_id <= table_range_end) {
                                page_queue.push(child_id);
                                visited_pages.insert(child_id);
                            } else {
                                std::cerr << "WARNING: Skipping child page " << child_id<< " outside table range" << std::endl;
                            }
                        }
                    }
                }

                release_page(page_id, false);
            } catch (const std::exception& e) {
                std::cerr << "WARNING: Failed to process page " << page_id << " during collection: " << e.what() << std::endl;
                // Continue with other pages
            }
        }

        std::cout << "DEBUG: Collected " << all_pages.size() << " pages from tree (root: " << root_page_id << ", range: " << table_range_start << "-" << table_range_end << ")" << std::endl;

        return all_pages;
    }

    void FractalBPlusTree::collect_all_leaf_pages(std::vector<uint32_t>& leaf_pages) const {
        if (root_page_id == 0) return;

        std::queue<uint32_t> page_queue;
        page_queue.push(root_page_id);

        while (!page_queue.empty()) {
            uint32_t page_id = page_queue.front();
            page_queue.pop();

            Page* node = get_page(page_id, false);

            if (node->header.type == PageType::LEAF_NODE) {
                leaf_pages.push_back(page_id);
            } else {
                // Internal node - add children to queue
                const uint32_t* children = get_child_pointers(node);
                for (uint32_t i = 0; i <= node->header.key_count; ++i) {
                    page_queue.push(children[i]);
                }
            }

            release_page(page_id, false);
        }
    }

    uint32_t* FractalBPlusTree::get_child_pointers(Page* node) {
        return reinterpret_cast<uint32_t*>(node->data + node->header.key_count * sizeof(int64_t));
    }

    const uint32_t* FractalBPlusTree::get_child_pointers(const Page* node) const {
        return reinterpret_cast<const uint32_t*>(node->data + node->header.key_count * sizeof(int64_t));
    }

    void FractalBPlusTree::set_child_pointers(Page* node, uint32_t index, uint32_t child_page_id) {
        uint32_t* children = get_child_pointers(node);
        children[index] = child_page_id;
    }

    uint32_t FractalBPlusTree::get_child_pointer(const Page* node, uint32_t index) const {
        const uint32_t* children = get_child_pointers(node);
        return children[index];
    }

    KeyValue* FractalBPlusTree::get_key_values(Page* node) {
        return reinterpret_cast<KeyValue*>(node->data + (BPTREE_ORDER * sizeof(int64_t)));
    }

    const KeyValue* FractalBPlusTree::get_key_values(const Page* node) const {
        return reinterpret_cast<const KeyValue*>(node->data + (BPTREE_ORDER * sizeof(int64_t)));
    }

    Message* FractalBPlusTree::get_messages(Page* node) {
        return reinterpret_cast<Message*>(node->data + (BPTREE_ORDER * sizeof(int64_t)) + (BPTREE_ORDER * sizeof(KeyValue)));
    }

    const Message* FractalBPlusTree::get_messages(const Page* node) const {
        return reinterpret_cast<const Message*>(node->data + (BPTREE_ORDER * sizeof(int64_t)) + (BPTREE_ORDER * sizeof(KeyValue)));
    }

    void FractalBPlusTree::create_new_root(uint32_t left_child, uint32_t right_child, int64_t split_key, uint64_t transaction_id) {
        uint32_t new_root_id = allocate_new_page(PageType::INTERNAL_NODE, transaction_id);
        Page* new_root = get_page(new_root_id, true);

        // Initialize new root
        new_root->header.key_count = 1;
        new_root->header.parent_id = 0;

        // Set first key and child pointers
        int64_t* keys = reinterpret_cast<int64_t*>(new_root->data);
        keys[0] = split_key;

        uint32_t* children = get_child_pointers(new_root);
        children[0] = left_child;
        children[1] = right_child;

        // Update children's parent pointers
        Page* left_child_page = get_page(left_child, true);
        left_child_page->header.parent_id = new_root_id;
        release_page(left_child, true);

        Page* right_child_page = get_page(right_child, true);
        right_child_page->header.parent_id = new_root_id;
        release_page(right_child, true);

        // Update root pointer
        root_page_id = new_root_id;

        release_page(new_root_id, true);

        std::cout << "CREATE_NEW_ROOT: New root page " << new_root_id << " created" << std::endl;
    }

    void FractalBPlusTree::update_parent_after_split(uint32_t parent_page_id, uint32_t old_child, uint32_t new_child, int64_t split_key, uint64_t transaction_id) {
        Page* parent = get_page(parent_page_id, true);

        // Insert new child into parent
        insert_key_into_internal(parent, split_key, new_child);

        // Check if parent needs splitting
        if (is_node_full(parent)) {
            split_node(parent, parent_page_id, transaction_id);
        }

        release_page(parent_page_id, true);
    }

    void FractalBPlusTree::update_children_parent_references(Page* parent, uint32_t parent_page_id, uint64_t transaction_id) {
        const uint32_t* children = get_child_pointers(parent);
        
        for (uint32_t i = 0; i <= parent->header.key_count; ++i) {
            Page* child = get_page(children[i], true);
            child->header.parent_id = parent_page_id;
            release_page(children[i], true);
        }
    }

    void FractalBPlusTree::build_internal_nodes_from_leaves(uint64_t transaction_id) {
        // Collect all leaf pages
        std::vector<uint32_t> leaf_pages;
        collect_all_leaf_pages(leaf_pages);

        if (leaf_pages.empty()) return;

        std::vector<uint32_t> current_level = leaf_pages;
        
        while (current_level.size() > 1) {
            std::vector<uint32_t> parent_level;
            
            // Group current level nodes into internal nodes
            for (size_t i = 0; i < current_level.size(); i += BPTREE_ORDER) {
                size_t end = std::min(i + BPTREE_ORDER, current_level.size());
                std::vector<uint32_t> children(current_level.begin() + i, current_level.begin() + end);
                
                uint32_t parent_id = build_internal_node_for_children(children, transaction_id);
                parent_level.push_back(parent_id);
            }
            
            current_level = parent_level;
        }
        
        // Update root
        if (!current_level.empty()) {
            root_page_id = current_level[0];
        }
    }

    uint32_t FractalBPlusTree::build_internal_node_for_children(const std::vector<uint32_t>& children, uint64_t transaction_id) {
        if (children.empty()) {
            return 0;
        }

        uint32_t parent_id = allocate_new_page(PageType::INTERNAL_NODE, transaction_id);
        Page* parent = get_page(parent_id, true);

        parent->header.key_count = children.size() - 1;
        parent->header.parent_id = 0;

        int64_t* keys = reinterpret_cast<int64_t*>(parent->data);
        uint32_t* child_ptrs = get_child_pointers(parent);

        // Set child pointers
        for (size_t i = 0; i < children.size(); ++i) {
            child_ptrs[i] = children[i];
            
            // Update child's parent reference
            Page* child = get_page(children[i], true);
            child->header.parent_id = parent_id;
            release_page(children[i], true);
        }

        // Set keys (minimum keys from second child onwards)
        for (size_t i = 1; i < children.size(); ++i) {
            keys[i - 1] = get_min_key_from_subtree(children[i]);
        }

        release_page(parent_id, true);
        return parent_id;
    }

    int64_t FractalBPlusTree::get_min_key_from_subtree(uint32_t root_page_id) {
        Page* root = get_page(root_page_id, false);
        int64_t min_key = get_min_key(root);
        release_page(root_page_id, false);
        return min_key;
    }

    int64_t FractalBPlusTree::get_max_key_from_subtree(uint32_t root_page_id) {
        Page* root = get_page(root_page_id, false);
        int64_t max_key = get_max_key(root);
        release_page(root_page_id, false);
        return max_key;
    }

    void FractalBPlusTree::print_subtree(uint32_t page_id, int depth, std::stringstream& ss) const {
        std::string indent(depth * 2, ' ');
        
        Page* node = get_page(page_id, false);
        
        ss << indent << "Page " << page_id << " [" << (node->header.type == PageType::LEAF_NODE ? "LEAF" : "INTERNAL") << "]";
        ss << " Keys: " << node->header.key_count;
        ss << " Messages: " << node->header.message_count;
        
        if (node->header.type == PageType::LEAF_NODE) {
            ss << " Prev: " << node->header.prev_page << " Next: " << node->header.next_page;
        }
        ss << std::endl;
        
        if (node->header.type == PageType::INTERNAL_NODE) {
            const uint32_t* children = get_child_pointers(node);
            for (uint32_t i = 0; i <= node->header.key_count; ++i) {
                print_subtree(children[i], depth + 1, ss);
            }
        }
        
        release_page(page_id, false);
    }

    // Additional helper implementations for internal node operations

    void FractalBPlusTree::redistribute_internal_keys(Page* left, Page* right, uint64_t transaction_id) {
        // Get parent to find separator key
        Page* parent = get_page(left->header.parent_id, true);
        int64_t separator_key = find_separator_key(parent, left->header.page_id, right->header.page_id);
        
        uint32_t total_keys = left->header.key_count + right->header.key_count + 1; // +1 for separator
        uint32_t target_keys = total_keys / 2;
        
        if (left->header.key_count < target_keys) {
            redistribute_internal_right_to_left(left, right, separator_key, target_keys, transaction_id);
        } else {
            redistribute_internal_left_to_right(left, right, separator_key, target_keys, transaction_id);
        }
        
        update_parent_separator_after_internal_redistribute(parent, left, right, transaction_id);
        release_page(parent->header.page_id, true);
    }

    void FractalBPlusTree::redistribute_internal_right_to_left(Page* left, Page* right, int64_t separator_key, uint32_t target_keys, uint64_t transaction_id) {
        uint32_t keys_needed = target_keys - left->header.key_count;
        if (keys_needed == 0) return;

        // Add separator key to left
        insert_key_into_internal(left, separator_key, 0); // Child pointer will be updated

        // Get data pointers
        int64_t* left_keys = reinterpret_cast<int64_t*>(left->data);
        uint32_t* left_children = get_child_pointers(left);
        int64_t* right_keys = reinterpret_cast<int64_t*>(right->data);
        const uint32_t* right_children = get_child_pointers(right);

        // Move keys and children from right to left
        for (uint32_t i = 0; i < keys_needed - 1; ++i) { // -1 because we already added separator
            insert_key_into_internal(left, right_keys[i], right_children[i]);
        }

        // Add the last child pointer for the moved keys
        set_child_pointers(left, left->header.key_count, right_children[keys_needed - 1]);

        // Update parent references for moved children
        update_children_parent_references(left, left->header.page_id, transaction_id);

        // Remove moved keys from right
        uint32_t remaining_keys = right->header.key_count - (keys_needed - 1);
        std::memmove(right_keys, &right_keys[keys_needed - 1], remaining_keys * sizeof(int64_t));
        std::memmove(const_cast<uint32_t*>(right_children), &right_children[keys_needed], (remaining_keys + 1) * sizeof(uint32_t));
        right->header.key_count = remaining_keys;

        std::cout << "REDISTRIBUTE_INTERNAL_RIGHT_TO_LEFT: Moved " << keys_needed << " keys" << std::endl;
    }

    void FractalBPlusTree::redistribute_internal_left_to_right(Page* left, Page* right, int64_t separator_key, uint32_t target_keys, uint64_t transaction_id) {
        uint32_t keys_to_move = left->header.key_count - target_keys;
        if (keys_to_move == 0) return;

        // Make space in right node for separator and moved keys
        int64_t* right_keys = reinterpret_cast<int64_t*>(right->data);
        uint32_t* right_children = get_child_pointers(right);
        
        std::memmove(&right_keys[keys_to_move], right_keys, right->header.key_count * sizeof(int64_t));
        std::memmove(&right_children[keys_to_move + 1], right_children, (right->header.key_count + 1) * sizeof(uint32_t));

        // Get data from left
        int64_t* left_keys = reinterpret_cast<int64_t*>(left->data);
        const uint32_t* left_children = get_child_pointers(left);

        // Move separator key to right
        right_keys[keys_to_move - 1] = separator_key;

        // Move keys and children from left to right
        for (uint32_t i = 0; i < keys_to_move - 1; ++i) {
            uint32_t left_index = left->header.key_count - keys_to_move + i + 1;
            right_keys[i] = left_keys[left_index];
            right_children[i] = left_children[left_index + 1];
        }

        // Set the first child pointer in right to the last child from left
        right_children[keys_to_move - 1] = left_children[left->header.key_count - keys_to_move + 1];

        // Update counts
        left->header.key_count -= keys_to_move;
        right->header.key_count += keys_to_move;

        // Update parent references for moved children
        update_children_parent_references(right, right->header.page_id, transaction_id);

        std::cout << "REDISTRIBUTE_INTERNAL_LEFT_TO_RIGHT: Moved " << keys_to_move << " keys" << std::endl;
    }

    void FractalBPlusTree::update_parent_separator_key(Page* left, Page* right, uint64_t transaction_id) {
        if (left->header.parent_id == 0) return;

        Page* parent = get_page(left->header.parent_id, true);
        update_separator_in_parent(parent, left->header.page_id, right->header.page_id, get_min_key(right), transaction_id);
        release_page(parent->header.page_id, true);
    }

    void FractalBPlusTree::update_separator_in_parent(Page* parent, uint32_t left_child, uint32_t right_child, int64_t new_separator, uint64_t transaction_id) {
        int64_t* keys = reinterpret_cast<int64_t*>(parent->data);
        const uint32_t* children = get_child_pointers(parent);

        // Find the key that separates left_child and right_child
        for (uint32_t i = 0; i < parent->header.key_count; ++i) {
            if (children[i] == left_child && children[i + 1] == right_child) {
                keys[i] = new_separator;
                break;
            }
        }
    }

    void FractalBPlusTree::update_parent_separator_after_internal_redistribute(Page* parent, Page* left, Page* right, uint64_t transaction_id) {
        update_separator_in_parent(parent, left->header.page_id, right->header.page_id, get_min_key(right), transaction_id);
    }

    void FractalBPlusTree::remove_node_from_parent(Page* parent, uint32_t child_page_id, uint64_t transaction_id) {
        int64_t* keys = reinterpret_cast<int64_t*>(parent->data);
        uint32_t* children = get_child_pointers(parent);

        // Find child index
        uint32_t child_index = 0;
        while (child_index <= parent->header.key_count && children[child_index] != child_page_id) {
            child_index++;
        }

        if (child_index > parent->header.key_count) {
            std::cout << "REMOVE_NODE_FROM_PARENT: Child " << child_page_id << " not found in parent " << parent->header.page_id << std::endl;
            return;
        }

        if (child_index == 0) {
            // Removing first child - remove first key
            std::memmove(keys, &keys[1], (parent->header.key_count - 1) * sizeof(int64_t));
            std::memmove(children, &children[1], parent->header.key_count * sizeof(uint32_t));
        } else if (child_index == parent->header.key_count) {
            // Removing last child - remove last key
            // No need to move data, just reduce count
        } else {
            // Removing middle child - remove corresponding key
            std::memmove(&keys[child_index - 1], &keys[child_index], (parent->header.key_count - child_index) * sizeof(int64_t));
            std::memmove(&children[child_index], &children[child_index + 1], (parent->header.key_count - child_index) * sizeof(uint32_t));
        }

        parent->header.key_count--;

        // Check if parent becomes underfull
        if (is_node_underfull(parent) && parent->header.parent_id != 0) {
            Page* grandparent = get_page(parent->header.parent_id, true);
            uint32_t left_sibling = find_left_sibling(parent->header.page_id, grandparent->header.page_id);
            uint32_t right_sibling = find_right_sibling(parent->header.page_id, grandparent->header.page_id);

            if (left_sibling != 0) {
                Page* left_sib = get_page(left_sibling, true);
                if (!is_node_full(left_sib)) {
                    merge_nodes(left_sib, parent, transaction_id);
                } else {
                    redistribute_keys(left_sib, parent, transaction_id);
                }
                release_page(left_sibling, true);
            } else if (right_sibling != 0) {
                Page* right_sib = get_page(right_sibling, true);
                if (!is_node_full(right_sib)) {
                    merge_nodes(parent, right_sib, transaction_id);
                } else {
                    redistribute_keys(parent, right_sib, transaction_id);
                }
                release_page(right_sibling, true);
            }

            release_page(grandparent->header.page_id, true);
        }
    }

    int64_t FractalBPlusTree::find_separator_key(Page* parent, uint32_t left_child, uint32_t right_child) {
        int64_t* keys = reinterpret_cast<int64_t*>(parent->data);
        const uint32_t* children = get_child_pointers(parent);

        for (uint32_t i = 0; i < parent->header.key_count; ++i) {
            if (children[i] == left_child && children[i + 1] == right_child) {
                return keys[i];
            }
        }

        throw std::runtime_error("Separator key not found for children " + std::to_string(left_child) + " and " + std::to_string(right_child));
    }

    void FractalBPlusTree::update_parent_key_after_merge(Page* parent, uint32_t remaining_child, uint32_t removed_child, uint64_t transaction_id) {
        // The separator key for the merged node needs to be updated
        // For leaf nodes, we just need to remove the separator
        remove_node_from_parent(parent, removed_child, transaction_id);
    }
}
