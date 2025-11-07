#pragma once
#ifndef FRACTAL_BPLUS_TREE_H
#define FRACTAL_BPLUS_TREE_H

#include "buffer_pool.h"
#include "write_ahead_log.h"
#include "common_types.h"
#include "locking_policy.h"
#include "deadlock_detector.h"
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <atomic>
#include <shared_mutex>
#include <algorithm>
#include <queue>
#include <cstring>
#include <mutex>

namespace fractal {

    // Forward declaration
    class DatabaseFile;

    class FractalBPlusTree {
        private:
            struct Node {
                PageHeader header;
                char data[PAGE_SIZE - sizeof(PageHeader)];

                // Helper methods for node operations
                int64_t* keys() { return reinterpret_cast<int64_t*>(data); }
                const int64_t* keys() const { return reinterpret_cast<const int64_t*>(data); }

                KeyValue* key_values() { 
                    return reinterpret_cast<KeyValue*>(data + header.key_count * sizeof(int64_t)); 
                }
                
                const KeyValue* key_values() const {
                    return reinterpret_cast<const KeyValue*>(data + header.key_count * sizeof(int64_t)); 
                }

                Message* messages() {
                    return reinterpret_cast<Message*>(data + header.key_count * sizeof(int64_t) + header.key_count * sizeof(KeyValue));
                }
                
                const Message* messages() const {
                    return reinterpret_cast<const Message*>(data + header.key_count * sizeof(int64_t) + header.key_count * sizeof(KeyValue));
                }

                uint32_t* child_pointers() {
                    return reinterpret_cast<uint32_t*>(data + header.key_count * sizeof(int64_t));
                }

                const uint32_t* child_pointers() const {
                    return reinterpret_cast<const uint32_t*>(data + header.key_count * sizeof(int64_t));
                }
            };

            DatabaseFile* db_file;
            BufferPool* buffer_pool;
            WriteAheadLog* wal;
            std::string table_name;
            uint32_t table_id;
            std::atomic<uint32_t> root_page_id;

            // Fractal tree specific
            mutable HierarchicalMutex<LockLevel::TREE> tree_mutex;
            std::atomic<size_t> total_messages{0};
            std::atomic<size_t> memory_usage{0};

            // Constants
            static constexpr size_t MEMORY_THRESHOLD = 1024 * 1024 * 1024;
            static constexpr double FLUSH_RATIO = 0.8;

       public:
            FractalBPlusTree(DatabaseFile* db_file, BufferPool* buffer_pool, WriteAheadLog* wal, const std::string& table_name, uint32_t root_page_id = 0, uint32_t table_id = 0);
            ~FractalBPlusTree();

            // Core operations
            void create();
            void drop();

            // CRUD Operations with message buffering
            void insert(int64_t key, const std::string& value, uint64_t transaction_id);
            void update(int64_t key, const std::string& value, uint64_t transaction_id);
            void remove(int64_t key, uint64_t transaction_id);
            std::string select(int64_t key, uint64_t transaction_id);

            // Range queries
            std::vector<std::pair<int64_t, std::string>> select_range(int64_t start_key, int64_t end_key, uint64_t transaction_id);
            // Scan operations
            std::vector<std::pair<int64_t, std::string>> scan_all(uint64_t transaction_id);

            // Bulk operations
            void bulk_load(const std::vector<std::pair<int64_t, std::string>>& data, uint64_t transaction_id);

            // Maintenance
            void checkpoint();
            void flush_all_messages(uint64_t transaction_id);
            void defragment_tree(uint64_t transaction_id);

            // Statistics and info
            uint32_t get_root_page_id() const { return root_page_id.load(); }
            size_t get_message_count() const { return total_messages.load(); }
            size_t get_memory_usage() const { return memory_usage.load(); }
            size_t get_tree_height() const;
            size_t get_total_pages() const;
            std::string get_tree_stats() const;
            void print_tree_structure() const;

       private:
            // Tree operations 
            Page* find_leaf_page(int64_t key, uint64_t transaction_id, bool exclusive = false);
            void split_node(Page* node, uint32_t page_id, uint64_t transaction_id);
            void merge_nodes(Page* left, Page* right, uint64_t transaction_id);
            void redistribute_keys(Page* left, Page* right, uint64_t transaction_id);

            // Message operations (core fractal tree functionality)
            void add_message_to_node(Page* node, MessageType type, int64_t key, uint64_t value_offset, uint32_t value_length, uint64_t transaction_id);
            void flush_messages_in_node(Page* node, uint32_t page_id, uint64_t transaction_id);
            void compact_messages(Page* node);
            void apply_messages_to_key_values(Page* node);
            void apply_single_message(Page* node, const Message& message);

            // Key operations
            void insert_key_value_into_leaf(Page* leaf, int64_t key, uint64_t value_offset, uint32_t value_length);
            void insert_key_into_internal(Page* internal, int64_t key, uint32_t child_page_id);
            void remove_key_from_leaf(Page* leaf, int64_t key);
            void remove_key_from_internal(Page* internal, int64_t key);

            // Helper methods
            Page* get_page(uint32_t page_id, bool exclusive = false);
            Page* get_page(uint32_t page_id, bool exclusive = false) const;
            void release_page(uint32_t page_id, bool dirty = false);
            void release_page(uint32_t page_id, bool dirty = false) const;
            uint32_t allocate_new_page(PageType type, uint64_t transaction_id);
            void free_page(uint32_t page_id, uint64_t transaction_id);
            uint32_t find_child_index(const Page* node, int64_t key);
            uint32_t find_key_index(const Page* node, int64_t key);
            int64_t get_min_key(const Page* node) const;
            int64_t get_max_key(const Page* node) const;

            // Space management
            bool is_node_full(const Page* node) const;
            bool is_node_underfull(const Page* node) const;
            size_t get_used_space(const Page* node) const;
            size_t get_available_space(const Page* node) const;

            // Adaptive Flushing (Fractal tree optimisation)
            bool should_flush_messages(const Page* node) const;
            void adaptive_flush(Page* node, uint32_t page_id, uint64_t transaction_id);
            bool memory_usage_high() const;

            // Data block management
            uint64_t write_data_block(const std::string& data);
            std::string read_data_block(uint64_t offset, uint32_t length);

            // Tree traversal helpers
            uint32_t find_left_sibling(uint32_t page_id, uint32_t parent_page_id);
            uint32_t find_right_sibling(uint32_t page_id, uint32_t parent_page_id);
            void collect_all_leaf_pages(std::vector<uint32_t>& leaf_pages) const;

            // Internal node operations
            uint32_t* get_child_pointers(Page* node);
            const uint32_t* get_child_pointers(const Page* node) const;
            void set_child_pointers(Page* node, uint32_t index, uint32_t child_page_id);
            uint32_t get_child_pointer(const Page* node, uint32_t index) const;

            // Leaf node operations
            KeyValue* get_key_values(Page* node);
            const KeyValue* get_key_values(const Page* node) const;
            Message* get_messages(Page* node);
            const Message* get_messages(const Page* node) const;

            // Tree structure operation
            void create_new_root(uint32_t left_child, uint32_t right_child, int64_t split_key, uint64_t transaction_id);
            void update_parent_after_split(uint32_t parent_page_id, uint32_t old_child, uint32_t new_child, int64_t split_key, uint64_t transaction_id);
            void update_children_parent_references(Page* parent, uint32_t parent_page_id, uint64_t transaction_id);
            void build_internal_nodes_from_leaves(uint64_t transaction_id);

            // Node merging helpers
            void merge_leaf_nodes(Page* left, Page* right, Page* parent, uint64_t transaction_id);
            void merge_internal_nodes(Page* left, Page* right, Page* parent, uint64_t transaction_id);
            void remove_node_from_parent(Page* parent, uint32_t child_page_id, uint64_t transaction_id);
            int64_t find_separator_key(Page* parent, uint32_t left_child, uint32_t right_child);
            void update_parent_key_after_merge(Page* parent, uint32_t remaining_child, uint32_t removed_child, uint64_t transaction_id);

            // Redistribution helpers
            void redistribute_leaf_keys(Page* left, Page* right, uint64_t transaction_id);
            void redistribute_right_to_left(Page* left, Page* right, uint32_t target_keys, uint64_t transaction_id);
            void redistribute_left_to_right(Page* left, Page* right, uint32_t target_keys, uint64_t transaction_id);
            void redistribute_internal_keys(Page* left, Page* right, uint64_t transaction_id);
            void redistribute_internal_right_to_left(Page* left, Page* right, int64_t separator_key, uint32_t target_keys, uint64_t transaction_id);
            void redistribute_internal_left_to_right(Page* left, Page* right, int64_t separator_key, uint32_t target_keys, uint64_t transaction_id);
            void update_parent_separator_key(Page* left, Page* right, uint64_t transaction_id);
            void update_separator_in_parent(Page* parent, uint32_t left_child, uint32_t right_child, int64_t new_separator, uint64_t transaction_id);
            void update_parent_separator_after_internal_redistribute(Page* parent, Page* left, Page* right, uint64_t transaction_id);

            // Bulk load helpers
            uint32_t build_internal_node_for_children(const std::vector<uint32_t>& children, uint64_t transaction_id);
            int64_t get_min_key_from_subtree(uint32_t root_page_id);
            int64_t get_max_key_from_subtree(uint32_t root_page_id);

            // Debug and validation
            void print_subtree(uint32_t page_id, int depth, std::stringstream& ss) const;
    };
}

#endif
