#pragma once
#ifndef BUFFER_POOL_H
#define BUFFER_POOL_H

#include "common_types.h"
#include "database_file.h"
#include <unordered_map>
#include <list>
#include <vector>
#include <memory>

namespace fractal {

class BufferPool {
private:
    struct BufferFrame {
        Page page;
        uint32_t page_id;
        bool dirty;
        bool pinned;
        std::list<uint32_t>::iterator lru_iterator;
        
        BufferFrame() : page_id(0), dirty(false), pinned(false) {}
    };

    DatabaseFile* db_file;
    size_t capacity;
    std::vector<BufferFrame> frames;
    std::unordered_map<uint32_t, size_t> page_to_frame;
    std::list<uint32_t> lru_list;
    
    size_t hit_count{0};
    size_t miss_count{0};
    size_t write_count{0};
    size_t read_count{0};

public:
    BufferPool(DatabaseFile& db_file, size_t capacity);
    ~BufferPool();

    Page* get_page(uint32_t page_id, bool exclusive = false);
    void release_page(uint32_t page_id, bool dirty = false);
    void mark_dirty(uint32_t page_id);
    void unpin_page(uint32_t page_id);
    
    void flush_page(uint32_t page_id);
    void flush_all();
    
    size_t get_hit_count() const { return hit_count; }
    size_t get_miss_count() const { return miss_count; }
    size_t get_write_count() const { return write_count; }
    size_t get_read_count() const { return read_count; }
    double get_hit_ratio() const;
    
    size_t get_capacity() const { return capacity; }
    size_t get_size() const { return page_to_frame.size(); }
    
    void print_stats() const;

private:
    size_t allocate_frame();
    void evict_page();
    void load_page(uint32_t page_id, size_t frame_index);
    void store_page(size_t frame_index);
};

} // namespace fractal

#endif // BUFFER_POOL_H
