#include "common_types.h"
#include <cstring>

namespace fractal {

    void Page::initialize(uint32_t page_id, PageType type, uint32_t db_id) {
        std::memset(this, 0, sizeof(Page));
        header.page_id = page_id;
        header.type = type;
        header.database_id = db_id;
        update_checksum();
    }

    void Page::update_checksum() {
        // Simplified version. Will come back later
        header.checksum = 0;
        uint32_t sum = 0;
        const char* bytes = reinterpret_cast<const char*>(this);
        for (size_t i = 0; i < sizeof(Page); i++) {
            sum += bytes[i];
        }
        header.checksum = sum;
    }

    bool Page::validate_checksum() const {
        uint32_t saved_checksum = header.checksum;
        
        // Calculate current checksum
        Page* non_const_this = const_cast<Page*>(this);
        non_const_this->header.checksum = 0;
        
        uint32_t calculated_checksum = 0;
        const char* bytes = reinterpret_cast<const char*>(this);
        for (size_t i = 0; i < sizeof(Page); i++) {
            calculated_checksum += bytes[i];
        }
        
        // Restore original checksum
        non_const_this->header.checksum = saved_checksum;
        
        return calculated_checksum == saved_checksum;
    }

    bool Page::can_add_message() const {
        return header.message_count < MAX_MESSAGES && get_available_space() >= sizeof(Message);
    }
    
    bool Page::can_add_key_value() const {
        return get_available_space() >= (sizeof(int64_t) + sizeof(KeyValue));
    }
    
    bool Page::is_full() const {
        return get_available_space() < (sizeof(int64_t) + sizeof(KeyValue));
    }
    
    bool Page::is_underfull() const {
        return get_used_space() < (PAGE_SIZE / 2);
    }

    size_t Page::get_used_space() const {
        size_t space = PageHeader::SIZE;
        // Add space used by keys, values, messages based on page type
        return space;
    }
    
    size_t Page::get_available_space() const {
        return PAGE_SIZE - get_used_space();
    }
}
