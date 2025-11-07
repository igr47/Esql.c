#include "database_file.h"
#include <iostream>
#include <filesystem>
#include <cstring>
#include <algorithm>

namespace fractal {

DatabaseFile::DatabaseFile(const std::string& filename) 
    : filename(filename), next_available_page(0) {
    
    db_header.magic = DATABASE_MAGIC;
    db_header.version = 1;
    db_header.page_size = PAGE_SIZE;
    db_header.total_pages = 0;
    db_header.first_free_page = 0;
    db_header.table_count = 0;
    db_header.created_timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    db_header.last_checkpoint = 0;
    std::memset(db_header.database_name, 0, sizeof(db_header.database_name));
    std::memset(db_header.reserved, 0, sizeof(db_header.reserved));
    
    table_directory.num_tables = 0;
    table_directory.next_table_id = 1;
}

DatabaseFile::~DatabaseFile() {
    std::cout << "DatabaseFile destructor started for: " << filename << std::endl;
    destroyed = true;
    if (file.is_open()) {
        std::cout << "Closing database file: " << filename << std::endl;
        sync();
        file.close();
        std::cout << "Database file closed: " << filename << std::endl;
    }
    std::cout << "DatabaseFile destructor completed for: " << filename << std::endl;
}

void DatabaseFile::create() {
    if (exists()) {
        throw std::runtime_error("Database file already exists: " + filename);
    }

    std::filesystem::path path(filename);
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }

    file.open(filename, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!file) {
        throw std::runtime_error("Failed to create database file: " + filename);
    }

    db_header.total_pages = 10000;
    next_available_page = 1000;

    write_header();
    
    table_directory.num_tables = 0;
    table_directory.next_table_id = 1;
    write_table_directory();
    
    for (uint32_t i = 2; i < 1000; ++i) {
        free_pages.push_back(i);
    }
    save_free_page_list();
    
    extend_file(db_header.total_pages);
    
    std::cout << "Database file created: " << filename << " with " << db_header.total_pages << " pages" << std::endl;
}

void DatabaseFile::open() {
    if (!exists()) {
        throw std::runtime_error("Database file does not exist: " + filename);
    }
    
    file.open(filename, std::ios::binary | std::ios::in | std::ios::out);
    if (!file) {
        throw std::runtime_error("Failed to open database file: " + filename);
    }

    read_header();
    
    if (db_header.magic != DATABASE_MAGIC) {
        file.close();
        throw std::runtime_error("Invalid database file format: " + filename);
    }
    
    if (db_header.page_size != PAGE_SIZE) {
        file.close();
        throw std::runtime_error("Page size mismatch in database file: " + filename);
    }

    table_directory.num_tables = 0;
    table_directory.next_table_id = 1;

    read_table_directory();
    
    load_free_page_list();
    
    initialize_table_data_blocks();
    
    table_name_to_id.clear();
    for (uint32_t i = 0; i < table_directory.num_tables; ++i) {
        const auto& entry = table_directory.entries[i];
        table_name_to_id[std::string(entry.table_name)] = entry.table_id;
    }
    
    std::cout << "Database file opened: " << filename << " with " << table_directory.num_tables << " tables" << std::endl;
}

void DatabaseFile::close() {
    if (file.is_open()) {
        sync();
        file.close();
    }
}

void DatabaseFile::sync() {
    if (file.is_open()) {
        file.flush();
        sync_count++;
    }
}

void DatabaseFile::remove() {
    close();
    std::filesystem::remove(filename);
}

void DatabaseFile::read_page(uint32_t page_id, Page* page) {
    validate_page_id(page_id);
    ensure_file_open();
    
    off_t offset = calculate_file_offset(page_id);
    file.seekg(offset);
    
    if (!file.read(reinterpret_cast<char*>(page), sizeof(Page))) {
        throw std::runtime_error("Failed to read page " + std::to_string(page_id));
    }
    
    read_count++;
}


void DatabaseFile::write_page(uint32_t page_id, const Page* page) {
    validate_page_id(page_id);
    ensure_file_open();
    
    off_t offset = calculate_file_offset(page_id);
    
    std::cout << "DEBUG DatabaseFile::write_page: Writing page " << page_id 
              << " at offset " << offset << std::endl;
    
    file.seekp(offset);
    
    if (!file) {
        std::cout << "DEBUG DatabaseFile::write_page: Seek failed, clearing errors" << std::endl;
        file.clear();
        file.seekp(offset);
    }
    
    if (!file.write(reinterpret_cast<const char*>(page), sizeof(Page))) {
        std::cout << "DEBUG DatabaseFile::write_page: Write failed for page " << page_id << std::endl;
        file.clear();
        throw std::runtime_error("Failed to write page " + std::to_string(page_id));
    }
    
    // Force flush
    file.flush();
    
    std::cout << "DEBUG DatabaseFile::write_page: Page " << page_id << " written successfully" << std::endl;
    write_count++;
}

uint32_t DatabaseFile::allocate_page(PageType type) {
    uint32_t page_id;
    
    if (!free_pages.empty()) {
        page_id = free_pages.back();
        free_pages.pop_back();
    } else {
        page_id = next_available_page++;
        if (page_id >= db_header.total_pages) {
            extend_file(1000);
        }
    }
    
    Page new_page;
    new_page.initialize(page_id, type, 0);
    
    write_page(page_id, &new_page);
    save_free_page_list();
    
    return page_id;
}

void DatabaseFile::deallocate_page(uint32_t page_id) {
    validate_page_id(page_id);
    
    if (std::find(free_pages.begin(), free_pages.end(), page_id) == free_pages.end()) {
        free_pages.push_back(page_id);
        save_free_page_list();
    }
}

void DatabaseFile::free_page(uint32_t page_id) {
    deallocate_page(page_id);
}

uint64_t DatabaseFile::allocate_data_block(uint32_t table_id, uint32_t size) {
    auto it = table_data_blocks.find(table_id);
    if (it == table_data_blocks.end()) {
        throw std::runtime_error("Table not found for data block allocation: " + std::to_string(table_id));
    }
    
    return it->second.allocate_block(size);
}

void DatabaseFile::read_data_block(uint64_t offset, char* data, uint32_t size) {
    ensure_file_open();
    
    file.seekg(offset);
    if (!file.read(data, size)) {
        throw std::runtime_error("Failed to read data block at offset " + std::to_string(offset));
    }
    
    read_count++;
}

void DatabaseFile::write_data_block(uint64_t offset, const char* data, uint32_t size) {
    ensure_file_open();
    
    file.seekp(offset);
    if (!file.write(data, size)) {
        throw std::runtime_error("Failed to write data block at offset " + std::to_string(offset));
    }
    
    write_count++;
}

void DatabaseFile::extend_data_region(uint32_t table_id, uint64_t additional_size) {
    auto it = table_data_blocks.find(table_id);
    if (it == table_data_blocks.end()) {
        throw std::runtime_error("Table not found for data region extension: " + std::to_string(table_id));
    }

    it->second.end_offset += additional_size;

    for (uint32_t i = 0; i < table_directory.num_tables; ++i) {
        if (table_directory.entries[i].table_id == table_id) {
            table_directory.entries[i].data_end_offset = it->second.end_offset;
            break;
        }
    }

    write_table_directory();
}

uint32_t DatabaseFile::create_table(const std::string& table_name) {
    if (table_name_to_id.find(table_name) != table_name_to_id.end()) {
        throw std::runtime_error("Table already exists: " + table_name);
    }
    
    if (table_directory.num_tables >= 100) {
        throw std::runtime_error("Maximum table limit reached (100 tables)");
    }
    
    uint32_t table_id = table_directory.next_table_id++;
    uint32_t root_page_id = get_table_start_page(table_id);
    
    auto& entry = table_directory.entries[table_directory.num_tables++];
    std::memset(entry.table_name, 0, sizeof(entry.table_name));
    std::strncpy(entry.table_name, table_name.c_str(), sizeof(entry.table_name) - 1);
    entry.table_id = table_id;
    entry.root_page_id = root_page_id;
    entry.data_start_offset = calculate_data_block_offset(table_id);
    entry.data_end_offset = entry.data_start_offset + (1024 * 1024);
    entry.record_count = 0;
    entry.created_timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    entry.last_modified = entry.created_timestamp;
    
    uint64_t data_start = calculate_data_block_offset(table_id);
    uint64_t data_end = data_start + (1024 * 1024);
    table_data_blocks[table_id] = TableDataBlockInfo(data_start, data_end);
    
    table_name_to_id[table_name] = table_id;
    
    write_table_directory();
    
    std::cout << "Created table '" << table_name << "' with ID " << table_id 
              << ", root page " << root_page_id << std::endl;
    
    return table_id;
}

void DatabaseFile::drop_table(const std::string& table_name) {
    auto it = table_name_to_id.find(table_name);
    if (it == table_name_to_id.end()) {
        throw std::runtime_error("Table not found: " + table_name);
    }
    
    uint32_t table_id = it->second;
    
    bool found = false;
    for (uint32_t i = 0; i < table_directory.num_tables; ++i) {
        if (table_directory.entries[i].table_id == table_id) {
            for (uint32_t j = i; j < table_directory.num_tables - 1; ++j) {
                table_directory.entries[j] = table_directory.entries[j + 1];
            }
            table_directory.num_tables--;
            found = true;
            break;
        }
    }
    
    if (!found) {
        throw std::runtime_error("Table directory corrupted");
    }
    
    table_data_blocks.erase(table_id);
    table_name_to_id.erase(table_name);
    
    write_table_directory();
    
    std::cout << "Dropped table '" << table_name << "' with ID " << table_id << std::endl;
}

uint32_t DatabaseFile::get_table_id(const std::string& table_name) const {
    auto it = table_name_to_id.find(table_name);
    if (it == table_name_to_id.end()) {
        throw std::runtime_error("Table not found: " + table_name);
    }
    
    return it->second;
}

uint32_t DatabaseFile::get_table_root_page(const std::string& table_name) const {
    uint32_t table_id = get_table_id(table_name);
    
    for (uint32_t i = 0; i < table_directory.num_tables; ++i) {
        if (table_directory.entries[i].table_id == table_id) {
            return table_directory.entries[i].root_page_id;
        }
    }
    
    throw std::runtime_error("Table directory entry not found for: " + table_name);
}

void DatabaseFile::set_table_root_page(const std::string& table_name, uint32_t root_page_id) {
    uint32_t table_id = get_table_id(table_name);
    
    for (uint32_t i = 0; i < table_directory.num_tables; ++i) {
        if (table_directory.entries[i].table_id == table_id) {
            table_directory.entries[i].root_page_id = root_page_id;
            table_directory.entries[i].last_modified = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            write_table_directory();
            return;
        }
    }
    
    throw std::runtime_error("Table directory entry not found for: " + table_name);
}

std::vector<std::string> DatabaseFile::get_table_names() const {
    std::vector<std::string> names;
    for (const auto& pair : table_name_to_id) {
        names.push_back(pair.first);
    }
    return names;
}


std::vector<DatabaseSchema::Table> DatabaseFile::get_all_tables() const {
    if (destroyed) {
        std::cerr << "ERROR: Accessing destroyed DatabaseFile object!" << std::endl;
        return std::vector<DatabaseSchema::Table>();
    }

    std::vector<DatabaseSchema::Table> tables;
    
    // First, validate the table directory state
    if (table_directory.num_tables == 0) {
        std::cout << "DEBUG: No tables in database." << std::endl;
        return tables; // Return empty vector
    }
    
    // Additional safety check
    if (table_directory.num_tables > 100) {
        return tables;
    }
    
    for (uint32_t i = 0; i < table_directory.num_tables; ++i) {
        DatabaseSchema::Table table;
        
        // Safe table name extraction
        const char* name_ptr = table_directory.entries[i].table_name;
        table.name = std::string(name_ptr, strnlen(name_ptr, 64));
        table.root_page = table_directory.entries[i].root_page_id;
        table.table_id = table_directory.entries[i].table_id;
        
        tables.push_back(table);
        std::cout << "DEBUG: Loaded table: " << table.name << std::endl;
    }
    
    std::cout << "DEBUG: Finished getting table data. Loaded " 
              << tables.size() << " tables." << std::endl;
    return tables;
}

void DatabaseFile::read_header() {
    ensure_file_open();
    
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(&db_header), sizeof(DatabaseHeader))) {
        throw std::runtime_error("Failed to read database header");
    }
}

void DatabaseFile::write_header() {
    ensure_file_open();
    
    file.seekp(0);
    if (!file.write(reinterpret_cast<const char*>(&db_header), sizeof(DatabaseHeader))) {
        throw std::runtime_error("Failed to write database header");
    }
    
    write_count++;
}

void DatabaseFile::read_table_directory() {
    ensure_file_open();

    auto original_pos = file.tellg();

    std::cout << "DEBUG: Reading table directory from offset " << PAGE_SIZE << std::endl;

    
    file.seekg(PAGE_SIZE);
    if (!file.read(reinterpret_cast<char*>(&table_directory), sizeof(TableDirectoryPage))) {
        //throw std::runtime_error("Failed to read table directory, Initializing tble directory");
        std::cerr << "WARNING: Failed to read table directory, initializing empty directory" << std::endl;
        table_directory.num_tables = 0;
        table_directory.next_table_id = 1;
        return;
    }

    std::cout << "DEBUG: Successfully read table directory" << std::endl;
    std::cout << "DEBUG: Read num_tables = " << table_directory.num_tables << std::endl;
    std::cout << "DEBUG: Read next_table_id = " << table_directory.next_table_id << std::endl;

    file.seekg(original_pos);
}

void DatabaseFile::write_table_directory() {
    ensure_file_open();

    if (!file.is_open()) {
        throw std::runtime_error("Database file not open for writing table directory");
    }

        // Check file state before writing
    if (!file.good()) {
        file.clear(); // Clear error flags
    }
    
    file.seekp(PAGE_SIZE);
        if (!file) {
        throw std::runtime_error("Failed to seek to table directory position");
    }

    if (!file.write(reinterpret_cast<const char*>(&table_directory), sizeof(TableDirectoryPage))) {
        throw std::runtime_error("Failed to write table directory - stream error: " + std::to_string(file.rdstate()));
    }
    
    write_count++;
}

uint64_t DatabaseFile::get_file_size() const {
    ensure_file_open();

    // Save current position
    auto current_pos = file.tellg();

    if (current_pos == -1) {
        std::cout << "DEBUG DatabaseFile::get_file_size: tellg() failed, clearing error flags" << std::endl;
        file.clear();
    }
    
    // Seek to end
    file.seekg(0, std::ios::end);
        if (!file) {
        std::cout << "DEBUG DatabaseFile::get_file_size: seek to end failed" << std::endl;
        file.clear();
        return 0; // Return 0 instead of garbage value
    }

    uint64_t size = file.tellg();
    if (size == static_cast<uint64_t>(-1)) {
        std::cout << "DEBUG DatabaseFile::get_file_size: tellg() returned -1" << std::endl;
        file.clear();
        size = 0;
    }

    // Restore original position
    if (current_pos != -1) {
        file.seekg(current_pos);
    }

    std::cout << "DEBUG DatabaseFile::get_file_size: returning " << size << std::endl;
    return size;
}

void DatabaseFile::print_stats() const {
    std::cout << "=== Database File Statistics ===" << std::endl;
    std::cout << "File: " << filename << std::endl;
    std::cout << "Size: " << get_file_size() << " bytes" << std::endl;
    std::cout << "Total pages: " << db_header.total_pages << std::endl;
    std::cout << "Free pages: " << free_pages.size() << std::endl;
    std::cout << "Tables: " << table_directory.num_tables << std::endl;
    std::cout << "Read operations: " << read_count << std::endl;
    std::cout << "Write operations: " << write_count << std::endl;
    std::cout << "Sync operations: " << sync_count << std::endl;
    std::cout << "================================" << std::endl;
}

bool DatabaseFile::exists() const {
    return std::filesystem::exists(filename);
}

uint32_t DatabaseFile::get_table_start_page(uint32_t table_id) const {
    return 1000 + (table_id * 1000);
}

uint32_t DatabaseFile::get_table_end_page(uint32_t table_id) const {
    return get_table_start_page(table_id) + 999;
}

bool DatabaseFile::is_page_in_table_range(uint32_t page_id, uint32_t table_id) const {
    uint32_t start = get_table_start_page(table_id);
    uint32_t end = get_table_end_page(table_id);
    return page_id >= start && page_id <= end;
}

// Private implementation methods

void DatabaseFile::initialize_free_pages() {
    free_pages.clear();
    for (uint32_t i = 2; i < 1000; ++i) {
        free_pages.push_back(i);
    }
}

void DatabaseFile::load_free_page_list() {
    free_pages.clear();
    
    for (uint32_t page_id = 1000; page_id < db_header.total_pages; ++page_id) {
        Page page;
        try {
            read_page(page_id, &page);
            if (page.header.type == PageType::FREE_PAGE) {
                free_pages.push_back(page_id);
            }
        } catch (...) {
            // Page might not exist yet, skip
        }
    }
}

void DatabaseFile::save_free_page_list() {
    for (uint32_t page_id : free_pages) {
        Page page;
        page.initialize(page_id, PageType::FREE_PAGE, 0);
        write_page(page_id, &page);
    }
}

/*void DatabaseFile::extend_file(uint32_t additional_pages) {
    uint64_t current_size = get_file_size();
    uint64_t new_size = current_size + (additional_pages * PAGE_SIZE);
    
    file.close();
    file.open(filename, std::ios::binary | std::ios::in | std::ios::out);
    
    file.seekp(new_size - 1);
    file.put(0);
    file.flush();
    
    db_header.total_pages += additional_pages;
    write_header();
    
    std::cout << "Extended database file by " << additional_pages << " pages, new total: " 
              << db_header.total_pages << " pages" << std::endl;
}*/

void DatabaseFile::extend_file(uint32_t additional_pages) {
    // Close and reopen the file to ensure clean state
    if (file.is_open()) {
        file.close();
    }
    
    // Reopen in append mode to extend the file
    file.open(filename, std::ios::binary | std::ios::in | std::ios::out | std::ios::app);
    if (!file) {
        throw std::runtime_error("Failed to reopen database file for extension: " + filename);
    }
    
    // Get current size properly
    file.seekg(0, std::ios::end);
    uint64_t current_size = file.tellg();
    if (current_size == static_cast<uint64_t>(-1)) {
        current_size = 0;
        file.clear();
    }
    
    uint64_t new_size = current_size + (additional_pages * PAGE_SIZE);
    std::cout << "DEBUG DatabaseFile::extend_file: Extending from " << current_size 
              << " to " << new_size << " bytes" << std::endl;
    
    // Extend the file by writing zeros
    file.seekp(current_size);
    std::vector<char> zeros(additional_pages * PAGE_SIZE, 0);
    if (!file.write(zeros.data(), zeros.size())) {
        throw std::runtime_error("Failed to extend database file");
    }
    
    file.flush();
    
    // Update header
    db_header.total_pages += additional_pages;
    
    // Reopen in normal mode
    file.close();
    file.open(filename, std::ios::binary | std::ios::in | std::ios::out);
    if (!file) {
        throw std::runtime_error("Failed to reopen database file after extension: " + filename);
    }
    
    write_header();
    
    std::cout << "DEBUG DatabaseFile::extend_file: Extended to " << new_size 
              << " bytes, " << db_header.total_pages << " pages" << std::endl;
}

void DatabaseFile::validate_page_id(uint32_t page_id) const {
    if (page_id >= db_header.total_pages) {
        throw std::runtime_error("Page ID out of range: " + std::to_string(page_id) + 
                                " (max: " + std::to_string(db_header.total_pages - 1) + ")");
    }
}

void DatabaseFile::ensure_file_open() const {
    if (!file.is_open()) {
        throw std::runtime_error("Database file is not open: " + filename);
    }
}

off_t DatabaseFile::calculate_file_offset(uint32_t page_id) const {
    return static_cast<off_t>(page_id) * PAGE_SIZE;
}

void DatabaseFile::initialize_table_data_blocks() {
    table_data_blocks.clear();
    
    for (uint32_t i = 0; i < table_directory.num_tables; ++i) {
        const auto& entry = table_directory.entries[i];
        table_data_blocks[entry.table_id] = TableDataBlockInfo(
            entry.data_start_offset, entry.data_end_offset);
    }
}

uint64_t DatabaseFile::calculate_data_block_offset(uint32_t table_id) const {
    uint64_t base_offset = db_header.total_pages * PAGE_SIZE;
    return base_offset + (table_id * 1024 * 1024);
}
void DatabaseFile::debug_page_access(uint32_t page_id) {
    std::cout << "DEBUG Page Access Analysis for page " << page_id << ":\n";
    std::cout << "  - File: " << filename << "\n";
    std::cout << "  - File open: " << (file.is_open() ? "YES" : "NO") << "\n";
    std::cout << "  - File good: " << (file.good() ? "YES" : "NO") << "\n";
    
    if (file.is_open()) {
        // Check if page exists by trying to read it
        try {
            Page test_page;
            read_page(page_id, &test_page);
            std::cout << "  - Page exists: YES\n";
            std::cout << "  - Page type: " << static_cast<int>(test_page.header.type) << "\n";
            std::cout << "  - Page initialized: " << (test_page.header.page_id == page_id ? "YES" : "NO") << "\n";
        } catch (const std::exception& e) {
            std::cout << "  - Page exists: NO - " << e.what() << "\n";
        }
    }
    std::cout << "  - Total pages in header: " << db_header.total_pages << "\n";
    std::cout << "  - Page offset: " << calculate_file_offset(page_id) << "\n";
    std::cout << "  - File size: " << get_file_size() << "\n";
}
} // namespace fractal
