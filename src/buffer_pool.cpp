#include "buffer_pool.h"
#include <iostream>
#include <algorithm>

namespace fractal {

BufferPool::BufferPool(DatabaseFile& db_file, size_t capacity) 
    : db_file(&db_file), capacity(capacity) {
    frames.resize(capacity);
    std::cout << "BufferPool initialized with capacity: " << capacity << " pages" << std::endl;
}

BufferPool::~BufferPool() {
    std::cout << "BufferPool destructor called" << std::endl;
    try {
        flush_all();
        std::cout << "BufferPool destructor completed" << std::endl;
        std::cout << "BufferPool destroyed. Hit ratio: " << get_hit_ratio() * 100 << "%" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error in BufferPool destruction: " << e.what() << std::endl;
    }
}

Page* BufferPool::get_page(uint32_t page_id, bool exclusive) {
    //std::cout << "DEBUG BufferPool::get_page: Requesting page " << page_id << std::endl;
    auto it = page_to_frame.find(page_id);
    
    if (it != page_to_frame.end()) {
        //std::cout << "DEBUG BufferPool::get_page: Page " << page_id << " found in cache" << std::endl;
        size_t frame_index = it->second;
        BufferFrame& frame = frames[frame_index];
        
        hit_count++;
        
        if (!frame.pinned) {
            lru_list.erase(frame.lru_iterator);
            lru_list.push_front(page_id);
            frame.lru_iterator = lru_list.begin();
        }
        
        frame.pinned = true;
        return &frame.page;
    }
    
    //std::cout << "DEBUG BufferPool::get_page: Page " << page_id << " not in cache, loading from disk" << std::endl;
    miss_count++;
    read_count++;
    
    size_t frame_index = allocate_frame();
    BufferFrame& frame = frames[frame_index];
    
    if (frame.dirty) {
        std::cout << "DEBUG BufferPool::get_page: Storing dirty frame " << frame_index << std::endl;
        store_page(frame_index);
    }
    
    page_to_frame.erase(frame.page_id);
    page_to_frame[page_id] = frame_index;
    
    frame.page_id = page_id;
    frame.dirty = false;
    frame.pinned = true;
    
    //std::cout << "DEBUG BufferPool::get_page: Loading page " << page_id << " into frame " << frame_index << std::endl;
    db_file->read_page(page_id, &frame.page);
    
    lru_list.push_front(page_id);
    frame.lru_iterator = lru_list.begin();
    
    return &frame.page;
}

void BufferPool::release_page(uint32_t page_id, bool dirty) {
    auto it = page_to_frame.find(page_id);
    if (it == page_to_frame.end()) {
        return;
    }
    
    size_t frame_index = it->second;
    BufferFrame& frame = frames[frame_index];
    
    if (dirty) {
        frame.dirty = true;
    }
    
    frame.pinned = false;
    
    lru_list.erase(frame.lru_iterator);
    lru_list.push_front(page_id);
    frame.lru_iterator = lru_list.begin();
}

void BufferPool::mark_dirty(uint32_t page_id) {
    auto it = page_to_frame.find(page_id);
    if (it != page_to_frame.end()) {
        frames[it->second].dirty = true;
    }
}

void BufferPool::unpin_page(uint32_t page_id) {
    release_page(page_id, false);
}

void BufferPool::flush_page(uint32_t page_id) {
    auto it = page_to_frame.find(page_id);
    if (it != page_to_frame.end()) {
        store_page(it->second);
    }
}

void BufferPool::flush_all() {
    for (size_t i = 0; i < frames.size(); ++i) {
        if (frames[i].dirty && frames[i].page_id != 0) {
            store_page(i);
        }
    }
}

double BufferPool::get_hit_ratio() const {
    size_t total = hit_count + miss_count;
    if (total == 0) return 0.0;
    return static_cast<double>(hit_count) / total;
}

void BufferPool::print_stats() const {
    std::cout << "=== BufferPool Statistics ===" << std::endl;
    std::cout << "Capacity: " << capacity << " pages" << std::endl;
    std::cout << "Current size: " << page_to_frame.size() << " pages" << std::endl;
    std::cout << "Hit count: " << hit_count << std::endl;
    std::cout << "Miss count: " << miss_count << std::endl;
    std::cout << "Read count: " << read_count << std::endl;
    std::cout << "Write count: " << write_count << std::endl;
    std::cout << "Hit ratio: " << get_hit_ratio() * 100 << "%" << std::endl;
    std::cout << "=============================" << std::endl;
}

size_t BufferPool::allocate_frame() {
    if (page_to_frame.size() < capacity) {
        for (size_t i = 0; i < frames.size(); ++i) {
            if (frames[i].page_id == 0) {
                return i;
            }
        }
    }
    
    evict_page();
    
    for (size_t i = 0; i < frames.size(); ++i) {
        if (frames[i].page_id == 0) {
            return i;
        }
    }
    
    throw std::runtime_error("BufferPool: Failed to allocate frame");
}

void BufferPool::evict_page() {
    for (auto it = lru_list.rbegin(); it != lru_list.rend(); ++it) {
        uint32_t page_id = *it;
        auto frame_it = page_to_frame.find(page_id);
        if (frame_it != page_to_frame.end()) {
            BufferFrame& frame = frames[frame_it->second];
            
            if (!frame.pinned) {
                if (frame.dirty) {
                    store_page(frame_it->second);
                }
                
                page_to_frame.erase(page_id);
                lru_list.erase(frame.lru_iterator);
                frame.page_id = 0;
                frame.dirty = false;
                frame.pinned = false;
                return;
            }
        }
    }
    
    throw std::runtime_error("BufferPool: All pages are pinned, cannot evict");
}

void BufferPool::load_page(uint32_t page_id, size_t frame_index) {
    db_file->read_page(page_id, &frames[frame_index].page);
}

void BufferPool::store_page(size_t frame_index) {
    BufferFrame& frame = frames[frame_index];
    if (frame.dirty && frame.page_id != 0) {
        db_file->write_page(frame.page_id, &frame.page);
        frame.dirty = false;
        write_count++;
    }
}

} // namespace fractal
