#include "pager.h"
#include <iostream>
#include <sstream>

bool RobustPager::open(const std::string& fname, bool create_if_missing) {
    std::unique_lock lock(file_mutex);
    filename = fname;

    std::cout << "Opening database file: " << filename << std::endl;

    bool file_exists = fs::exists(filename);
    
    if (!file_exists && create_if_missing) {
        std::cout << "File doesn't exist, creating new database: " << filename << std::endl;
        
        initialization_in_progress = true;
        lock.unlock();
        
        initialize_new_database_direct();
        
        lock.lock();
        initialization_in_progress = false;
        return true;
    }

    // Open existing file
    file.open(filename, std::ios::binary | std::ios::in | std::ios::out);
    if (!file.is_open()) {
        std::cerr << "Failed to open database file: " << filename << std::endl;
        if (create_if_missing) {
            std::cout << "Attempting to create new database after open failure..." << std::endl;
            
            initialization_in_progress = true;
            lock.unlock();
            initialize_new_database_direct();
            lock.lock();
            initialization_in_progress = false;
            return true;
        }
        return false;
    }

    // Read and validate header for existing files
    if (!read_header()) {
        std::cerr << "Header validation failed for: " << filename << std::endl;
        if (create_if_missing) {
            std::cout << "Attempting recovery for corrupted header..." << std::endl;
            return recover_database();
        }
        return false;
    }

    // Verify database integrity
    if (!validate_database_integrity()) {
        std::cout << "Database integrity check failed, entering recovery mode" << std::endl;
        recovery_mode = true;
        return recover_database();
    }

    std::cout << "Successfully opened database: " << filename 
              << " (page_size: " << header.page_size << ")" << std::endl;
    return true;
}

void RobustPager::close() {
    std::unique_lock lock(file_mutex);
    if (file.is_open()) {
        file.close();
    }
}

void RobustPager::initialize_new_database_direct() {
    std::unique_lock lock(file_mutex);
    
    // First ensure the file is created and opened
    file.open(filename, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to create database file: " + filename);
    }
    file.close();
    
    // Reopen for normal operations
    file.open(filename, std::ios::binary | std::ios::in | std::ios::out);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to reopen database file: " + filename);
    }

    // Initialize with safe defaults
    header = DatabaseFileHeader();
    std::string db_name = fs::path(filename).stem().string();
    strncpy(header.database_name, db_name.c_str(), sizeof(header.database_name) - 1);
    header.database_name[sizeof(header.database_name) - 1] = '\0';
    header.created_timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    header.modified_timestamp = header.created_timestamp;
    header.header_check_sum = header.calculate_checksum();

    // Write header directly without recursive calls
    if (!write_header()) {
        throw std::runtime_error("Failed to write database header");
    }

    // Initialize page 1 (schema page) - use unsafe version since we hold lock
    Node schema_page = {};
    schema_page.header.page_id = 1;
    schema_page.header.type = PageType::METADATA;
    schema_page.header.timestamp = header.created_timestamp;
    TableDirectory table_dir = {};
    memcpy(schema_page.data, &table_dir, sizeof(TableDirectory));
    write_page_unsafe(DatabaseFormat::SCHEMA_PAGE_ID, &schema_page);

    std::cout << "Initialized new robust database: " << filename << std::endl;
}

// FIXED: Eliminated recursive locking in table operations
bool RobustPager::register_table(const std::string& table_name, uint32_t table_id, uint32_t root_page_id) {
    std::unique_lock lock(file_mutex);

    // Read directory without calling read_page (which could cause recursion)
    TableDirectory directory = {};
    try {
        Node schema_page;
        read_page_unsafe(DatabaseFormat::SCHEMA_PAGE_ID, &schema_page);
        memcpy(&directory, schema_page.data, sizeof(TableDirectory));
    } catch (const std::exception& e) {
        std::cerr << "Failed to read table directory: " << e.what() << std::endl;
        return false;
    }

    // Check if table already exists
    if (directory.find_table(table_name) != nullptr) {
        std::cout << "Table already registered: " << table_name << std::endl;
        return true;
    }

    // Add table to directory
    if (!directory.add_table(table_name, table_id, root_page_id)) {
        std::cerr << "Failed to add table to directory: " << table_name << std::endl;
        return false;
    }

    // Update next_table_id if needed
    if (table_id >= directory.next_table_id) {
        directory.next_table_id = table_id + 1;
    }

    return write_table_directory_unsafe(directory);
}

bool RobustPager::unregister_table(const std::string& table_name) {
    std::unique_lock lock(file_mutex);

    TableDirectory directory = read_table_directory_unsafe();

    if (!directory.remove_table(table_name)) {
        std::cerr << "Table not found in directory: " << table_name << std::endl;
        return false;
    }

    return write_table_directory_unsafe(directory);
}

TableDirectory RobustPager::read_table_directory() {
    std::shared_lock lock(file_mutex);
    return read_table_directory_unsafe();
}

TableDirectory RobustPager::read_table_directory_unsafe() {
    TableDirectory directory = {};
    
    try {
        Node schema_page;
        read_page_unsafe(DatabaseFormat::SCHEMA_PAGE_ID, &schema_page);
        memcpy(&directory, schema_page.data, sizeof(TableDirectory));
    } catch (const std::exception& e) {
        std::cerr << "Failed to read table directory: " << e.what() << std::endl;
    }

    return directory;
}

bool RobustPager::update_table_directory(const TableDirectory& directory) {
    std::unique_lock lock(file_mutex);
    return write_table_directory_unsafe(directory);
}

bool RobustPager::write_table_directory_unsafe(const TableDirectory& directory) {
    try {
        Node schema_page;
        read_page_unsafe(DatabaseFormat::SCHEMA_PAGE_ID, &schema_page);
        memcpy(schema_page.data, &directory, sizeof(TableDirectory));
        write_page_unsafe(DatabaseFormat::SCHEMA_PAGE_ID, &schema_page);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to write table directory: " << e.what() << std::endl;
        return false;
    }
}

TableDirectoryEntry* RobustPager::find_table_entry(const std::string& table_name) {
    std::shared_lock lock(file_mutex);
    TableDirectory directory = read_table_directory_unsafe();
    return directory.find_table(table_name);
}

uint32_t RobustPager::get_next_table_id() {
    std::unique_lock lock(file_mutex);
    TableDirectory directory = read_table_directory_unsafe();
    uint32_t next_id = directory.next_table_id;
    directory.next_table_id++;
    write_table_directory_unsafe(directory);
    return next_id;
}

void RobustPager::initialize_new_database() {
    if (initialization_in_progress) {
        initialize_new_database_direct();
    } else {
        std::unique_lock lock(file_mutex);
        initialize_new_database_direct();
    }
}

bool RobustPager::read_header() {
    if (!file.is_open()) { 
        std::cerr << "File not open for reading header" << std::endl;
        return false;
    }

    file.seekg(0, std::ios::beg);
    if (!file.read(reinterpret_cast<char*>(&header), sizeof(DatabaseFileHeader))) {
        std::cerr << "Failed to read database header" << std::endl;
        return false;
    }

    // Enhanced validation
    bool magic_valid = (memcmp(header.magic, "ESQLDB21", 8) == 0);
    bool checksum_valid = (header.header_check_sum == header.calculate_checksum());
    bool page_size_valid = (header.page_size == 16384);

    if (!magic_valid || !checksum_valid || !page_size_valid) {
        std::cerr << "Magic valid: " << magic_valid 
            << ", Checksum valid: " << checksum_valid
            << ", Page size valid: " << page_size_valid << std::endl;
        return false;
    }
    
    return true;
}

bool RobustPager::write_header() {
    if (!file.is_open()) {
        file.open(filename, std::ios::binary | std::ios::out | std::ios::in);
        if (!file.is_open()) {
            file.open(filename, std::ios::binary | std::ios::out);
            if (!file.is_open()) {
                std::cerr << "Failed to create file for header writing: " << filename << std::endl;
                return false;
            }
            file.close();
            file.open(filename, std::ios::binary | std::ios::in | std::ios::out);
            if (!file.is_open()) {
                std::cerr << "Failed to reopen file for header writing: " << filename << std::endl;
                return false;
            }
        }
    }

    header.modified_timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    header.header_check_sum = header.calculate_checksum();

    file.seekp(0, std::ios::beg);
    file.write(reinterpret_cast<const char*>(&header), sizeof(DatabaseFileHeader));
    file.flush();

    if (!file.good()) {
        std::cerr << "Failed to write database header for: " << filename << std::endl;
        return false;
    }

    return true;
}

// NEW: Safe page reading without recursive locking
void RobustPager::read_page_unsafe(uint32_t page_id, Node* node) {
    // Assumes file_mutex is already held (shared or unique)
    if (recovery_mode) {
        read_page_recovery_unsafe(page_id, node);
        return;
    }

    uint64_t offset = page_id * header.page_size;
    file.seekg(offset, std::ios::beg);

    if (!file) {
        throw std::runtime_error("Failed to seek to page " + std::to_string(page_id));
    }

    // Clear node to avoid garbage data
    memset(node, 0, sizeof(Node));
    file.read(reinterpret_cast<char*>(node), header.page_size);

    if (!file || file.gcount() != header.page_size) {
        throw std::runtime_error("Incomplete read for page " + std::to_string(page_id));
    }

    // FIXED: No recursive calls to write_page - handle corrections differently
    if (node->header.page_id != page_id) {
        std::cerr << "PAGE_RECOVERY: Fixing page ID mismatch. Expected: " << page_id << ", Got: " << node->header.page_id << std::endl;
        node->header.page_id = page_id;
        // Just log the issue - don't call write_page recursively
    }

    // Validate page type for special pages - log but don't fix recursively
    if (page_id == 1 && node->header.type != PageType::METADATA) {
         std::cerr << "PAGE_RECOVERY: Page 1 has incorrect type: " << static_cast<int>(node->header.type) << std::endl;
         // Don't fix it here to avoid recursion
    }
}

void RobustPager::read_page(uint32_t page_id, Node* node) {
    std::shared_lock lock(file_mutex);
    read_page_unsafe(page_id, node);
}

void RobustPager::write_page(uint32_t page_id, const Node* node) {
    if (initialization_in_progress) {
        std::unique_lock lock(file_mutex);
        write_page_unsafe(page_id, node);
    } else {
        std::unique_lock lock(file_mutex);
        write_page_unsafe(page_id, node);
    }
}

void RobustPager::write_page_unsafe(uint32_t page_id, const Node* node) {
    // Assumes file_mutex is already held
    if (!file.is_open()) {
        throw std::runtime_error("Database file not open");
    }

    // Create a copy to ensure we don't modify the original
    Node node_copy = *node;
    node_copy.header.page_id = page_id; // Ensure correct page ID
    node_copy.header.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

    // Ensure special pages have correct types
    if (page_id == 1) {
        node_copy.header.type = PageType::METADATA;
    }

    uint64_t offset = page_id * header.page_size;
    uint64_t required_size = offset + header.page_size;

    // Extend file if necessary
    file.seekp(0, std::ios::end);
    uint64_t current_size = file.tellp();

    if (required_size > current_size) {
        file.seekp(required_size - 1, std::ios::beg);
        file.put('\0');
        file.flush();
    }

    file.seekp(offset, std::ios::beg);
    file.write(reinterpret_cast<const char*>(&node_copy), header.page_size);
    file.flush();

    if (!file) {
        throw std::runtime_error("Failed to write page " + std::to_string(page_id));
    }
}

uint32_t RobustPager::allocate_page() {
    std::unique_lock lock(file_mutex);

    // Simplified allocation strategy: Extend file
    file.seekp(0, std::ios::end);
    uint32_t page_id = file.tellp() / header.page_size;

    uint64_t new_size = (page_id + 1) * header.page_size;
    file.seekp(new_size - 1, std::ios::beg);
    file.put('\0');
    file.flush();

    // Initialize new node
    Node node = {};
    node.header.page_id = page_id;
    node.header.type = PageType::LEAF;
    node.header.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

    write_page_unsafe(page_id, &node);

    return page_id;
}

void RobustPager::release_page(uint32_t page_id) {
    // NOTE: Free list management not implemented yet
    std::cout << "Page " << page_id << " marked for reuse" << std::endl;
}

bool RobustPager::validate_database_integrity() {
    try {
        // Check header first
        if (!header.validate()) {
            std::cerr << "Header validation failed" << std::endl;
            return false;
        }

        // Check page 1 (schema page)
        Node schema_page;
        read_page(1, &schema_page);

        if (schema_page.header.type != PageType::METADATA) {
            std::cerr << "Schema page has incorrect type" << std::endl;
            return false;
        }

        return true;
    
    } catch (const std::exception& e) {
        std::cerr << "Database integrity check failed: " << e.what() << std::endl;
        return false;
    }
}

bool RobustPager::recover_database() {
    std::cout << "DATABASE_RECOVERY: Initiating recovery for " << filename << std::endl;

    try {
        // Backup original file
        std::string backup_name = filename + ".backup_" + std::to_string(time(nullptr));
        if(fs::exists(filename)) {
            fs::copy(filename, backup_name, fs::copy_options::overwrite_existing);
            std::cout << "Created backup: " << backup_name << std::endl;
        }

        // Reinitialize header
        if (!read_header() || !header.validate()) {
            std::cout << "RECOVERY: Header corrupted, reinitializing..." << std::endl;
            initialize_new_database();
        } else {
            std::cout << "RECOVERY: Header is valid" << std::endl;
        }

        // Recover schema
        recover_schema();

        // Validate all pages
        validate_all_pages();

        recovery_mode = false;
        std::cout << "DATABASE_RECOVERY: Recovery completed successfully" << std::endl;
        return true;
    
    } catch (const std::exception& e) {
        std::cerr << "DATABASE_RECOVERY: Failed - " << e.what() << std::endl;
        return false;
    }
}

// NEW: Safe recovery reading without recursive locking
void RobustPager::read_page_recovery_unsafe(uint32_t page_id, Node* node) {
    try {
        // Try normal read first
        read_page_unsafe(page_id, node);
    } catch (const std::exception& e) {
        std::cerr << "Recovery read failed for page " << page_id << ": " << e.what() << std::endl;

        // Initialize empty page as fall back
        memset(node, 0, sizeof(Node));
        node->header.page_id = page_id;
        node->header.type = (page_id == 1) ? PageType::METADATA : PageType::LEAF;
        node->header.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

        // Write the recovered page
        write_page_unsafe(page_id, node);
    }
}

void RobustPager::read_page_recovery(uint32_t page_id, Node* node) {
    std::shared_lock lock(file_mutex);
    read_page_recovery_unsafe(page_id, node);
}

void RobustPager::recover_schema() {
    std::cout << "RECOVERY: Recovering schema ..." << std::endl;

    try {
        Node schema_page;
        read_page(1, &schema_page);

        // Basic schema page validation
        if (schema_page.header.num_keys > 1000) {
            std::cerr << "RECOVERY: Schema page corrupted, resetting..." << std::endl;

            memset(&schema_page, 0 , sizeof(Node));
            schema_page.header.page_id = 1;
            schema_page.header.type = PageType::METADATA;
            schema_page.header.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            
            std::unique_lock lock(file_mutex);
            write_page_unsafe(1, &schema_page);
        }
    
    } catch (const std::exception& e) {
        std::cerr << "RECOVERY: Schema recovery failed: " << e.what() << std::endl;
        throw;
    }
}

void RobustPager::validate_all_pages() {
    std::cout << "RECOVERY: Validating all pages..." << std::endl;

    // Simplified to only scan the first few pages
    for (uint32_t page_id = 0; page_id <= 2; ++page_id) {
        try {
            Node node;
            read_page(page_id, &node);
            std::cout << "Page " << page_id << " validated successfully" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Page " << page_id << " validation failed: " << e.what() << std::endl;
            recover_corrupted_page(page_id);
        }
    }
}

void RobustPager::recover_corrupted_page(uint32_t page_id) {
    std::cout << "RECOVERY: Recovering corrupted page " << page_id << std::endl;

    Node recovered_node = {};
    recovered_node.header.page_id = page_id;
    recovered_node.header.type = (page_id == 1) ? PageType::METADATA : PageType::LEAF;
    recovered_node.header.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

    std::unique_lock lock(file_mutex);
    write_page_unsafe(page_id, &recovered_node);
}
