#include "multifilepager.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <cstdio>

// DatabaseFile implementation
DatabaseFile::DatabaseFile(const std::string& fname) : filename(fname) {}

DatabaseFile::~DatabaseFile() {
    close();
}

bool DatabaseFile::open(bool create_if_missing) {
    if (file.is_open()) {
        return true; // Already open
    }

    // Try to open existing file
    file.open(filename, std::ios::binary | std::ios::in | std::ios::out);
    
    if (!file.is_open() && create_if_missing) {
        // Create new file
        std::ofstream create_file(filename, std::ios::binary);
        if (!create_file) {
            std::cerr << "Failed to create database file: " << filename << std::endl;
            return false;
        }
        create_file.close();
        
        // Reopen for read/write
        file.open(filename, std::ios::binary | std::ios::in | std::ios::out);
        if (!file) {
            std::cerr << "Failed to open database file: " << filename << std::endl;
            return false;
        }
        
        // Initialize with metadata page
        Node metadata_page = {};
        metadata_page.header.type = PageType::METADATA;
        metadata_page.header.page_id = 0;
        metadata_page.header.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        write_page(0, &metadata_page);
        initialize_free_page_management();
        std::cout << "Created new database file: " << filename << std::endl;
    } else if (!file.is_open()) {
        return false;
    } else {
        // Read existing file and initialize free page management
        file.seekg(0, std::ios::end);
        uint64_t file_size = file.tellg();
        next_page_id.store(file_size / BPTREE_PAGE_SIZE);
        
        // Initialize free page management for existing files
        if (file_size > 0) {
            initialize_free_page_management();
        }
    }
    
    return file.is_open();
}

void DatabaseFile::close() {
    if (file.is_open()) {
        file.close();
    }
}


void DatabaseFile::read_page(uint32_t page_id, Node* node) const {
    if (!file.is_open()) {
        throw std::runtime_error("Database file not open: " + filename);
    }
    
    file.seekg(page_id * BPTREE_PAGE_SIZE, std::ios::beg);
    file.read(reinterpret_cast<char*>(node), BPTREE_PAGE_SIZE);
    if (!file) {
        throw std::runtime_error("Failed to read page " + std::to_string(page_id) + 
                               " from " + filename);
    }
}

void DatabaseFile::write_page(uint32_t page_id, const Node* node) {
    if (!file.is_open()) {
        throw std::runtime_error("Database file not open: " + filename);
    }
    
    // Extend file if needed
    uint64_t required_size = (page_id + 1) * BPTREE_PAGE_SIZE;
    file.seekp(0, std::ios::end);
    uint64_t current_size = file.tellp();
    
    if (required_size > current_size) {
        file.seekp(required_size - 1, std::ios::beg);
        file.put('\0');
        file.flush();
    }
    
    std::cout << "DEBUG: Seeking to whre to write data. "<< std::endl;
    file.seekp(page_id * BPTREE_PAGE_SIZE, std::ios::beg);
    std::cout << "DEBUG: Writting data to disk. " << std::endl;
    file.write(reinterpret_cast<const char*>(node), BPTREE_PAGE_SIZE);
    std::cout << "DEBUG: Data written to disk" << std::endl;
    if (!file) {
        throw std::runtime_error("Failed to write page " + std::to_string(page_id) + 
                               " to " + filename);
    }
    file.flush();
}

uint32_t DatabaseFile::allocate_page() {
    if (!file.is_open()) {
        throw std::runtime_error("Database file not open: " + filename);
    }
    
    // First try free list
    uint32_t page_id = allocate_from_free_list();
    if (page_id != 0) {
        return page_id;
    }
    
    // Allocate new page at end of file
    file.seekp(0, std::ios::end);
    page_id = file.tellp() / BPTREE_PAGE_SIZE;
    
    // Extend file
    uint64_t new_size = (page_id + 1) * BPTREE_PAGE_SIZE;
    file.seekp(new_size - 1, std::ios::beg);
    file.put('\0');
    file.flush();
    
    // Initialize page
    Node node = {};
    node.header.page_id = page_id;
    node.header.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    write_page(page_id, &node);
    
    next_page_id.store(page_id + 1);
    return page_id;
}

void DatabaseFile::flush() {
    if (file.is_open()) {
        file.flush();
    }
}

size_t DatabaseFile::get_file_size() const {
    if (!file.is_open()) return 0;
    
    file.seekg(0, std::ios::end);
    return file.tellg();
}

void DatabaseFile::initialize_free_page_management() {
    // For new files, start free list from page 1
    // For existing files, we'd need to scan for free pages, but for simplicity
    // we'll just initialize the free list structure
    if (free_list_head.load() == 0) {
        free_list_head = 1;
        write_free_page_header(1, 0);
    }
}

uint32_t DatabaseFile::allocate_from_free_list() {
    uint32_t current_head = free_list_head.load();
    if (current_head == 0) return 0;
    
    FreePageHeader header = read_free_page_header(current_head);
    
    // Try to update free_list_head atomically
    while (!free_list_head.compare_exchange_weak(current_head, header.next_free_page)) {
        if (current_head == 0) return 0;
        header = read_free_page_header(current_head);
    }
    
    return current_head;
}

void DatabaseFile::release_page(uint32_t page_id) {
    uint32_t current_head = free_list_head.load();
    
    do {
        write_free_page_header(page_id, current_head);
    } while (!free_list_head.compare_exchange_weak(current_head, page_id));
}

void DatabaseFile::write_free_page_header(uint32_t page_id, uint32_t next_free) {
    Node node = {};
    node.header.page_id = page_id;
    node.header.type = PageType::METADATA;
    
    FreePageHeader free_header;
    free_header.magic_number = FREE_PAGE_MAGIC;
    free_header.next_free_page = next_free;
    free_header.prev_free_page = 0;
    free_header.free_region_size = 1;
    
    memcpy(node.data, &free_header, sizeof(FreePageHeader));
    write_page(page_id, &node);
}

FreePageHeader DatabaseFile::read_free_page_header(uint32_t page_id) {
    Node node;
    read_page(page_id, &node);
    
    FreePageHeader header;
    memcpy(&header, node.data, sizeof(FreePageHeader));
    
    if (header.magic_number != FREE_PAGE_MAGIC) {
        throw std::runtime_error("Invalid free page header");
    }
    
    return header;
}

// MultiFilePager implementation
MultiFilePager::MultiFilePager(const std::string& base_path) 
    : base_path(base_path), registry_file(base_path + "meta.esqlregistry") {
    
    try {
        fs::create_directories(base_path);
        std::cout << "Database directory: " << fs::absolute(base_path) << std::endl;
        std::string test_file = base_path + "write_test.tmp";
        std::ofstream test(test_file);

        if (!test) {
            throw std::runtime_error("Cannot write to database directory: " + base_path);
        }

        test << "test";
        test.close();
        fs::remove(test_file);
    } catch (const std::exception& e) {
        std::cerr << "Failed to create base directory: " << e.what() << std::endl;
        throw;
    }

    // Load registry without locks
    read_registry_internal();
}

MultiFilePager::~MultiFilePager() {
    flush_all();
}

/*DatabaseFile* MultiFilePager::get_or_create_database_file(const std::string& db_name) {
    // Try to get existing file first
    auto it = database_files.find(db_name);
    if (it != database_files.end()) {
        return it->second.get();
    }
    
    // File doesn't exist in memory, try to open/create it
    std::string filename = base_path + db_name + ".esqldb";
    auto db_file = std::make_unique<DatabaseFile>(filename);
    
    // Try to open existing file first, create if it doesn't exist
    if (!db_file->open(true)) {
        std::cerr << "Failed to open/create database file: " << filename << std::endl;
        return nullptr;
    }
    
    DatabaseFile* result = db_file.get();
    database_files[db_name] = std::move(db_file);

    // Update registry
    write_registry_internal();

    return result;
}*/

DatabaseFile* MultiFilePager::get_or_create_database_file(const std::string& db_name) {
    // Try to get existing file first
    auto it = database_files.find(db_name);
    if (it != database_files.end()) {
        return it->second.get();
    }
    
    std::string filename = base_path + db_name + ".esqldb";
    
    // Check file permissions and existence
    if (fs::exists(filename)) {
        // Check if we can read the file
        std::ifstream test_file(filename, std::ios::binary);
        if (!test_file) {
            std::cerr << "Permission denied or corrupted file: " << filename << std::endl;
            return nullptr;
        }
        test_file.close();
    }
    
    auto db_file = std::make_unique<DatabaseFile>(filename);
    
    // Try to open existing file first, create if it doesn't exist
    if (!db_file->open(true)) {
        std::cerr << "Failed to open/create database file: " << filename << std::endl;
        
        // If file exists but can't be opened, try with different permissions
        if (fs::exists(filename)) {
            try {
                fs::permissions(filename, 
                    fs::perms::owner_read | fs::perms::owner_write |
                    fs::perms::group_read | fs::perms::others_read);
                
                // Try again
                db_file = std::make_unique<DatabaseFile>(filename);
                if (db_file->open(false)) {
                    DatabaseFile* result = db_file.get();
                    database_files[db_name] = std::move(db_file);
                    return result;
                }
            } catch (const std::exception& e) {
                std::cerr << "Failed to fix permissions for " << filename << ": " << e.what() << std::endl;
            }
        }
        return nullptr;
    }
    
    DatabaseFile* result = db_file.get();
    database_files[db_name] = std::move(db_file);

    // Update registry
    write_registry_internal();

    return result;
}

bool MultiFilePager::validate_page_type(const std::string& db_name, uint32_t page_id, PageType expected_type) {
       try {
            Node node;
            read_page(db_name, page_id, &node);
            
            if (page_id == 0) {
                return node.header.type == PageType::METADATA;
            } else {
                // Data pages should never be METADATA type
                if (node.header.type == PageType::METADATA) {
                    std::cout << "VALIDATION: Fixing page " << page_id 
                              << " from METADATA to " << static_cast<int>(expected_type) << std::endl;
                    node.header.type = expected_type;
                    write_page(db_name, page_id, &node);
                    return false;
                }
                return node.header.type == expected_type;
            }
        } catch (...) {
            return false;
        }
}

DatabaseFile* MultiFilePager::get_database_file(const std::string& db_name) {
    auto it = database_files.find(db_name);
    if (it != database_files.end()) {
        return it->second.get();
    }
    
    // Try to load the database file if it exists on disk but not in memory
    std::string filename = base_path + db_name + ".esqldb";
    if (fs::exists(filename)) {
        auto db_file = std::make_unique<DatabaseFile>(filename);
        if (db_file->open(false)) { // Open existing, don't create
            DatabaseFile* result = db_file.get();
            database_files[db_name] = std::move(db_file);
            return result;
        }
    }
    
    return nullptr;
}

bool MultiFilePager::create_database_file_internal(const std::string& db_name) {
    std::string filename = base_path + db_name + ".esqldb";

    // Check if file already exists
    if (fs::exists(filename)) {
        return false;
    }

    auto db_file = std::make_unique<DatabaseFile>(filename);

    if (!db_file->open(true)) { // true = create if missing
        return false;
    }

    database_files[db_name] = std::move(db_file);
    return true;
}

bool MultiFilePager::create_database_file(const std::string& db_name) {
    // Check if already exists in memory
    if (database_files.find(db_name) != database_files.end()) {
        return false;
    }
    
    std::string filename = base_path + db_name + ".esqldb";
    
    // Check if file already exists on disk
    if (fs::exists(filename)) {
        // File exists but not in memory, load it
        auto db_file = std::make_unique<DatabaseFile>(filename);
        if (db_file->open(false)) { // false = don't create, just open
            database_files[db_name] = std::move(db_file);
            write_registry_internal();
            return true;
        }
        return false;
    }
    
    // Create new file
    auto db_file = std::make_unique<DatabaseFile>(filename);
    if (!db_file->open(true)) { // true = create if missing
        std::cerr << "Failed to create database file: " << filename << std::endl;
        return false;
    }
    
    database_files[db_name] = std::move(db_file);
    write_registry_internal();
    return true;
}

void MultiFilePager::close_database_file(const std::string& db_name) {
    auto it = database_files.find(db_name);
    if (it != database_files.end()) {
        it->second->flush();
        it->second->close();
        database_files.erase(it);
    }
}

bool MultiFilePager::database_exists(const std::string& db_name) const {
    return fs::exists(base_path + db_name + ".esqldb");
}

std::vector<std::string> MultiFilePager::list_databases() const {
    std::vector<std::string> databases;
    
    // Add in-memory databases
    for (const auto& [db_name, db_file] : database_files) {
        databases.push_back(db_name);
    }

    // Add on-disk databases not yet in memory
    try {
        for (const auto& entry : fs::directory_iterator(base_path)) {
            if (entry.is_regular_file() && entry.path().extension() == ".esqldb") {
                std::string filename = entry.path().stem().string();

                if (std::find(databases.begin(), databases.end(), filename) == databases.end()) {
                    databases.push_back(filename);
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error scanning database directory: " << e.what() << std::endl;
    }
    
    return databases;
}

void MultiFilePager::read_page(const std::string& db_name, uint32_t page_id, Node* node) {
    DatabaseFile* db_file = get_or_create_database_file(db_name);
    if (!db_file) {
        throw std::runtime_error("Database not found: " + db_name);
    }
    db_file->read_page(page_id, node);
}

void MultiFilePager::write_page(const std::string& db_name, uint32_t page_id, const Node* node) {
    if (page_id != 0 && node->header.type == PageType::METADATA) {
        std::cerr << "WARNING: Attempting to write data page " << page_id << " as METADATA type - this indicates a bug" << std::endl;
    }
    DatabaseFile* db_file = get_or_create_database_file(db_name);
    if (!db_file) {
        throw std::runtime_error("Database not found: " + db_name);
    }
    db_file->write_page(page_id, node);
}

uint32_t MultiFilePager::allocate_page(const std::string& db_name) {
    DatabaseFile* db_file = get_or_create_database_file(db_name);
    if (!db_file) {
        throw std::runtime_error("Database not found: " + db_name);
    }
    //return db_file->allocate_page();
    uint32_t page_id = db_file->allocate_page();
    if (page_id <= 1) {
        // Skip pages o and 1 for data
        db_file->release_page(page_id);
        page_id = db_file->allocate_page();
        if (page_id <= 1) {
            page_id = 2;
            Node dummy_page = {};
            db_file->write_page(page_id, &dummy_page);
        }
    }

    Node new_page;
    db_file->read_page(page_id, &new_page);
    if (new_page.header.type == PageType::METADATA) {
        new_page.header.type = PageType::LEAF;
        db_file->write_page(page_id, &new_page);
    }

    return page_id;
}

void MultiFilePager::release_page(const std::string& db_name, uint32_t page_id) {
    DatabaseFile* db_file = get_database_file(db_name);
    if (!db_file) {
        throw std::runtime_error("Database not found: " + db_name);
    }
    db_file->release_page(page_id);
}

void MultiFilePager::flush_all() {
    for (auto& [db_name, db_file] : database_files) {
        db_file->flush();

        std::cout << "DEBUG: Forcefully flushed and synced database: " << db_name << std::endl;
    }
}

void MultiFilePager::checkpoint_all() {
    flush_all();
}

void MultiFilePager::write_registry_internal() {
    std::vector<std::string> dbs;
    for (const auto& [db_name, db_file] : database_files) {
        dbs.push_back(db_name);
    }
    
    std::ofstream registry(registry_file, std::ios::binary | std::ios::trunc);
    if (!registry) {
        std::cerr << "Failed to open registry file for writing: " << registry_file << std::endl;
        return;
    }
    
    size_t count = dbs.size();
    registry.write(reinterpret_cast<const char*>(&count), sizeof(count));
    
    for (const auto& db_name : dbs) {
        size_t name_len = db_name.size();
        registry.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
        registry.write(db_name.c_str(), name_len);
    }
    
    registry.flush();
}

void MultiFilePager::read_registry_internal() {
    if (!fs::exists(registry_file)) {
        std::cout << "No registry file found, starting fresh" << std::endl;
        return;
    }
    
    std::ifstream registry(registry_file, std::ios::binary);
    if (!registry) {
        std::cerr << "Failed to open registry file: " << registry_file << std::endl;
        return;
    }
    
    size_t count;
    registry.read(reinterpret_cast<char*>(&count), sizeof(count));
    
    if (!registry) {
        std::cerr << "Failed to read database count from registry" << std::endl;
        return;
    }
    
    std::cout << "Loading " << count << " databases from registry" << std::endl;
    
    for (size_t i = 0; i < count; i++) {
        size_t name_len;
        registry.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
        
        if (!registry) {
            std::cerr << "Failed to read database name length from registry" << std::endl;
            break;
        }
        
        std::string db_name(name_len, '\0');
        registry.read(&db_name[0], name_len);
        
        if (!registry) {
            std::cerr << "Failed to read database name from registry" << std::endl;
            break;
        }
        
        // Pre-open database files
        std::cout << "Loading database: " << db_name << std::endl;
        if (!get_or_create_database_file(db_name)) {
            std::cerr << "Failed to load database: " << db_name << std::endl;
        }
    }
    
    registry_loaded.store(true);
}

uint64_t MultiFilePager::write_data_block(const std::string& db_name, const std::string& data) {
    DatabaseFile* db_file = get_database_file(db_name);
    if (!db_file) {
        throw std::runtime_error("Database not found: " + db_name);
    }
    
    std::fstream& file = db_file->get_file_stream();
    
    file.seekp(0, std::ios::end);
    uint64_t offset = file.tellp();

    // Handle empty data
    if (data.empty()) {
        uint32_t original_size = 0;
        uint32_t compressed_size = 0;
        uint8_t compression_flag = 0; // 0 = uncompressed

        file.write(reinterpret_cast<const char*>(&original_size), sizeof(original_size));
        file.write(reinterpret_cast<const char*>(&compressed_size), sizeof(compressed_size));
        file.write(reinterpret_cast<const char*>(&compression_flag), sizeof(compression_flag));
        file.flush();
        return offset;
    }

    uint32_t original_size = static_cast<uint32_t>(data.size());
    uint32_t compressed_size;
    uint8_t compression_flag;

    // For small data or debugging, skip compression
    if (data.size() <= 64) { // Don't compress small data
        compression_flag = 0;
        compressed_size = original_size;

        file.write(reinterpret_cast<const char*>(&original_size), sizeof(original_size));
        file.write(reinterpret_cast<const char*>(&compressed_size), sizeof(compressed_size));
        file.write(reinterpret_cast<const char*>(&compression_flag), sizeof(compression_flag));
        file.write(data.data(), data.size());
    } else {
        // Use Zstd compression
        size_t max_compressed_size = ZSTD_compressBound(data.size());
        std::vector<char> compressed(max_compressed_size);

        size_t zstd_result = ZSTD_compress(
            compressed.data(), max_compressed_size,
            data.data(), data.size(),
            3  // Medium compression level
        );

        if (ZSTD_isError(zstd_result)) {
            // Fallback to uncompressed storage
            compression_flag = 0;
            compressed_size = original_size;

            file.write(reinterpret_cast<const char*>(&original_size), sizeof(original_size));
            file.write(reinterpret_cast<const char*>(&compressed_size), sizeof(compressed_size));
            file.write(reinterpret_cast<const char*>(&compression_flag), sizeof(compression_flag));
            file.write(data.data(), data.size());
        } else {
            // Successfully compressed
            compression_flag = 1;
            compressed_size = static_cast<uint32_t>(zstd_result);

            file.write(reinterpret_cast<const char*>(&original_size), sizeof(original_size));
            file.write(reinterpret_cast<const char*>(&compressed_size), sizeof(compressed_size));
            file.write(reinterpret_cast<const char*>(&compression_flag), sizeof(compression_flag));
            file.write(compressed.data(), compressed_size);
        }
    }

    file.flush();
    return offset;
}

std::string MultiFilePager::read_data_block(const std::string& db_name, uint64_t offset, uint32_t expected_length) {
    DatabaseFile* db_file = get_database_file(db_name);
    if (!db_file) {
        throw std::runtime_error("Database not found: " + db_name);
    }
    
    std::fstream& file = db_file->get_file_stream();
    
    if (offset == 0) {
        return ""; // No data stored at offset 0
    }

    file.seekg(offset, std::ios::beg);
    if (!file) {
        throw std::runtime_error("Failed to seek to offset " + std::to_string(offset));
    }

    // Read the metadata we stored
    uint32_t original_size, compressed_size;
    uint8_t compression_flag;

    file.read(reinterpret_cast<char*>(&original_size), sizeof(original_size));
    file.read(reinterpret_cast<char*>(&compressed_size), sizeof(compressed_size));
    file.read(reinterpret_cast<char*>(&compression_flag), sizeof(compression_flag));

    if (!file) {
        throw std::runtime_error("Failed to read data block metadata at offset " +
                                std::to_string(offset));
    }

    // Handle empty data block
    if (original_size == 0 && compressed_size == 0) {
        return "";
    }

    // Validate sizes
    if (original_size > 100 * 1024 * 1024) { // 100MB sanity check
        throw std::runtime_error("Suspiciously large original size: " +
                               std::to_string(original_size) + " at offset " +
                               std::to_string(offset));
    }

    if (compressed_size > 100 * 1024 * 1024) { // 100MB sanity check
        throw std::runtime_error("Suspiciously large compressed size: " +
                               std::to_string(compressed_size) + " at offset " +
                               std::to_string(offset));
    }

    // Check if data is compressed
    if (compression_flag == 0) {
        // Data is uncompressed
        std::string result(original_size, '\0');
        file.read(&result[0], original_size);

        if (!file || file.gcount() != static_cast<std::streamsize>(original_size)) {
            throw std::runtime_error("Failed to read uncompressed data at offset " +
                                    std::to_string(offset) + ", expected " +
                                    std::to_string(original_size) + " bytes, got " +
                                    std::to_string(file.gcount()) + " bytes");
        }

        return result;
    } else {
        // Data is compressed with Zstd
        std::vector<char> compressed(compressed_size);
        file.read(compressed.data(), compressed_size);

        if (!file || file.gcount() != static_cast<std::streamsize>(compressed_size)) {
            throw std::runtime_error("Failed to read compressed data at offset " +
                                    std::to_string(offset) + ", expected " +
                                    std::to_string(compressed_size) + " bytes, got " +
                                    std::to_string(file.gcount()) + " bytes");
        }

        std::vector<char> decompressed(original_size);
        size_t actual_decompressed_size = ZSTD_decompress(
            decompressed.data(), original_size,
            compressed.data(), compressed_size
        );

        if (ZSTD_isError(actual_decompressed_size)) {
            std::string error_msg = "Zstd decompression failed: " +
                std::string(ZSTD_getErrorName(actual_decompressed_size)) +
                " at offset " + std::to_string(offset) +
                ", original_size: " + std::to_string(original_size) +
                ", compressed_size: " + std::to_string(compressed_size) +
                ", Zstd version: " + ZSTD_versionString();

            // Try to dump some debug info
            std::cerr << error_msg << std::endl;

            // Try to read as uncompressed data as fallback
            try {
                file.seekg(offset + sizeof(original_size) + sizeof(compressed_size) + sizeof(compression_flag), std::ios::beg);
                std::string fallback_result(original_size, '\0');
                file.read(&fallback_result[0], original_size);
                if (file && file.gcount() == static_cast<std::streamsize>(original_size)) {
                    std::cerr << "Fallback to uncompressed read successful" << std::endl;
                    return fallback_result;
                }
            } catch (...) {
                // Ignore fallback errors
            }

            throw std::runtime_error(error_msg);
        }

        if (actual_decompressed_size != original_size) {
            throw std::runtime_error("Decompressed size mismatch: expected " +
                std::to_string(original_size) + ", got " + std::to_string(actual_decompressed_size));
        }

        return std::string(decompressed.data(), actual_decompressed_size);
    }
}
