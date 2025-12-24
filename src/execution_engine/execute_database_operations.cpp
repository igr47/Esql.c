#include "execution_engine_includes/executionengine_main.h"
//#include "execution_engine_includes/structs.h"
#include "database.h"
#include <iostream>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <cstdlib>
#include <sstream>
#include <set>
#include <chrono>
#include <cmath>

// Database operations
ExecutionEngine::ResultSet ExecutionEngine::executeCreateDatabase(AST::CreateDatabaseStatement& stmt) {
    storage.createDatabase(stmt.dbName);
    return ResultSet({"Status", {{"Database '" + stmt.dbName + "' created successfully"}}});
}

ExecutionEngine::ResultSet ExecutionEngine::executeUse(AST::UseDatabaseStatement& stmt) {
    storage.useDatabase(stmt.dbName);
    db.setCurrentDatabase(stmt.dbName);
    return ResultSet({"Status", {{"Using database '" + stmt.dbName + "'"}}});
}

ExecutionEngine::ResultSet ExecutionEngine::executeShow(AST::ShowDatabaseStatement& stmt) {
    auto databases = storage.listDatabases();
    ResultSet result({"Database"});
    for (const auto& name : databases) {
        result.rows.push_back({name});
    }
    return result;
}

ExecutionEngine::ResultSet ExecutionEngine::executeShowTables(AST::ShowTableStatement& stmt) {
    std::vector<std::string> tables;

    try {
        tables = storage.getTableNames(db.currentDatabase());
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to get tables.");
    }

    ResultSet result({"Table Name","Rows","Size","Created"});

    for (const auto& table : tables) {
        try {
            auto tableData = storage.getTableData(db.currentDatabase(), table);
            std::string rowCount = std::to_string(tableData.size());
            std::string size = "~" + std::to_string(tableData.size() * 100) + " KB"; // Estimate
            std::string created = "N/A"; // Should check cretaed at timestamp which is not immplemented

            result.rows.push_back({table, rowCount, size, created});
        } catch (const std::exception& e) {
            result.rows.push_back({table, "Error", "N/A", "N/A"});
        }
    }
    return result;
}

ExecutionEngine::ResultSet ExecutionEngine::executeShowTableStructure(AST::ShowTableStructureStatement& stmt) {
    const auto* table = storage.getTable(db.currentDatabase(), stmt.tableName);
    if (!table) {
        throw std::runtime_error("Table not found: " + stmt.tableName);
    }

    ResultSet result;

    // Table overview section
    result.columns = {"Property", "Value"};

    // Basic table info
    auto tableData = storage.getTableData(db.currentDatabase(), stmt.tableName);
    result.rows.push_back({"Table Name", stmt.tableName});
    result.rows.push_back({"Database", db.currentDatabase()});
    result.rows.push_back({"Total Rows", std::to_string(tableData.size())});
    result.rows.push_back({"Total Columns", std::to_string(table->columns.size())});

    // Find primary keys
    std::vector<std::string> primaryKeys;
    for (const auto& col : table->columns) {
        if (col.isPrimaryKey) {
            primaryKeys.push_back(col.name);
        }
    }

    if (!primaryKeys.empty()) {
        std::string pkStr;
        for (size_t i = 0; i < primaryKeys.size(); ++i) {
            if (i > 0) pkStr += ", ";
            pkStr += primaryKeys[i];
        }
        result.rows.push_back({"Primary Key", pkStr});
    } else {
        result.rows.push_back({"Primary Key", "None"});
    }

    // Separator
    result.rows.push_back({"---","---"});

    // Column section header
    result.rows.push_back({"Columns", ""});
    result.rows.push_back({"Name", "Type", "Nullable", "Pk", "Unique", "AUTOInc", "Default", "GEN_DATE", "GEN_DATE_TIME","GEN_UUID"});
    result.rows.push_back({"----", "----", "--------", "---", "-----", "-------", "-------", "-------","--------------", "--------"});

    // Column details
    for (const auto& column : table->columns) {
        std::vector<std::string> colInfo;
        colInfo.push_back(column.name);
        colInfo.push_back(getTypeString(column.type));
        colInfo.push_back(column.isNullable ? "YES" : "NO");
        colInfo.push_back(column.isPrimaryKey ? "YES" : "NO");
        colInfo.push_back(column.isUnique ? "YES" : "NO");
        colInfo.push_back(column.autoIncreament ? "YES" : "NO");
        colInfo.push_back(column.defaultValue.empty() ? "NULL" : column.defaultValue);
        colInfo.push_back(column.generateDate ? "YES" : "NO");
        colInfo.push_back(column.generateDateTime ? "YES" : "NO");
        colInfo.push_back(column.generateUUID ? "YES" : "NO");

        // Convert to the two-column format for display
        std::string colDetails = colInfo[0] + " | " + colInfo[1] + " | " + colInfo[2] + " | " + colInfo[3] + " | " + colInfo[4] + " | " + colInfo[5] + " | " + colInfo[6];
        result.rows.push_back({"", colDetails});
    }

    // Constarint section
    std::vector<std::string> constraints;
    for (const auto& column : table->columns) {
        for (const auto& constraint : column.constraints) {
            constraints.push_back(constraint.name + ": " + constraint.value + " (" + column.name + ")");
        }
    }

    if (!constraints.empty()) {
        result.rows.push_back({"---", "---"});
        result.rows.push_back({"CONSTRAINTS", ""});
        for (const auto& constraint : constraints) {
            result.rows.push_back({"", constraint});
        }
    }

    return result;
}

ExecutionEngine::ResultSet ExecutionEngine::executeShowDatabaseStats(AST::ShowDatabaseStats& stmt) {
    DatabaseStats stats = getDatabaseStatistics();

    ResultSet result({"Metric", "Value", "Describtion"});

    result.rows.push_back({"Database Name", db.currentDatabase(), "Current active database"});
    result.rows.push_back({"File Size", std::to_string(stats.fileSize) + " bytes", "Total database file size"});
    result.rows.push_back({"Total Pages", std::to_string(stats.totalPages), "Total allocated pages"});
    result.rows.push_back({"Used pages", std::to_string(stats.usedPages), "Pages currently in use"});
    result.rows.push_back({"Free pages", std::to_string(stats.freePages), "Available free pages"});
    result.rows.push_back({"Page Utilisation", std::to_string((stats.usedPages * 100) / std::max(1.0, (double)stats.totalPages)) + "%", "Percentage of pages used"});
    result.rows.push_back({"Read Operations", std::to_string(stats.readOperations), "Total page reads"});
    result.rows.push_back({"Write Operations", std::to_string(stats.writeOperations), "Total page writes"});
    result.rows.push_back({"Sync Operations", std::to_string(stats.syncOperations), "File sync operations"});
    result.rows.push_back({"Buffer Pool Size", std::to_string(stats.bufferPoolSize) + " pages", "Buffer pool size"});
    result.rows.push_back({"Buffer Pool Used", std::to_string(stats.bufferPoolUsed) + " pages", "Pages currently used"});
    result.rows.push_back({"Buffer Pool Hits", std::to_string(stats.bufferPoolHits), "Cache hits"});
    result.rows.push_back({"Buffer Pool Misses", std::to_string(stats.bufferPoolMisses), "Cache Misses"});
    result.rows.push_back({"Buffer Pool Hit Ratio", std::to_string(stats.bufferPoolHitRatio * 100) + "%", "Cache effectiveness"});

    return result;
}

ExecutionEngine::ResultSet ExecutionEngine::executeShowTableStats(AST::ShowTableStats& stmt) {
    TableStats tableStats = getTableStatistics(stmt.tableName);
    TreeStats treeStats = getTreeStatistics(stmt.tableName);

    ResultSet result({"Property", "Value", "Details"});

    //Basic table info
    result.rows.push_back({"Table Name", tableStats.tableName, "Name of the table"});
    result.rows.push_back({"Table ID", std::to_string(tableStats.tableId), "Internal table identifier"});
    result.rows.push_back({"Root page", std::to_string(tableStats.rootPageId), "B+Tree root page ID"});
    result.rows.push_back({"Record Count", std::to_string(tableStats.recordCount), "Total rows in table"});

    // Tree structure
    result.rows.push_back({"Tree Height", std::to_string(tableStats.treeHeight), "B+Tree height (levels)"});
    result.rows.push_back({"Total Nodes", std::to_string(tableStats.totalNodes), "All tree nodes"});
    result.rows.push_back({"Leaf Nodes", std::to_string(tableStats.leafNodes), "Leaf nodes containing data"});
    result.rows.push_back({"Internal Nodes", std::to_string(tableStats.internalNodes), "Index nodes"});

    // Page management
    result.rows.push_back({"Page count", std::to_string(tableStats.pageCount), "Pages allocated to table"});
    result.rows.push_back({"Data Block Offset", std::to_string(tableStats.dataBlockOffset), "Data region starting point"});
    result.rows.push_back({"Data Block Size", std::to_string(tableStats.dataBlockSize) + "bytes", "Data region size"});

    result.rows.push_back({"Buffered Messages", std::to_string(tableStats.bufferedMessages), "Unflashed messages"});
    result.rows.push_back({"Memory Usage", std::to_string(tableStats.memoryUsage) + " bytes", "In-memory buffer usage"});
    result.rows.push_back({"Space Utilization", std::to_string(treeStats.spaceUtilization * 100) + "%", "Page space usage efficiency"});
    result.rows.push_back({"Avg Keys/Node", std::to_string(treeStats.avgKeysPerNode), "Average keys per node"});
    result.rows.push_back({"Min Keys/Node", std::to_string(treeStats.minKeysPerNode), "Minimum keys in any node"});
    result.rows.push_back({"Max Keys/Node", std::to_string(treeStats.maxKeysPerNode), "Maximum keys in any node"});

    // Timestamps
    std::string createdTime = std::to_string(tableStats.createdTimestamp);
    std::string modifiedTime = std::to_string(tableStats.lastModified);
    result.rows.push_back({"Created", createdTime, "Creation timestamp"});
    result.rows.push_back({"Last Modified", modifiedTime, "Last modification timestamp"});

    return result;

}

ExecutionEngine::ResultSet ExecutionEngine::executeShowIndexStats(AST::ShowIndexStats& stmt) {
    TreeStats treeStats = getTreeStatistics(stmt.tableName);

    ResultSet result({"Index Property", "Value", "Analysis"});

    // Tree structure analysis
    double avgFanout = treeStats.totalNodes > 0 ? static_cast<double>(treeStats.totalNodes) / std::max(1.0, (double)treeStats.internalNodes) : 0.0;

    result.rows.push_back({"Index Type", "Fractal B+Tree", "Index implementation"});
    result.rows.push_back({"Tree Height", std::to_string(treeStats.treeHeight), "Number of levels"});
    result.rows.push_back({"Total Nodes", std::to_string(treeStats.totalNodes), "All index nodes"});
    result.rows.push_back({"Leaf Nodes", std::to_string(treeStats.leafNodes), "Data nodes"});
    result.rows.push_back({"Internal Nodes", std::to_string(treeStats.internalNodes), "Routing nodes"});
    result.rows.push_back({"Average Fanout", std::to_string(avgFanout), "Average children per internal node"});

    // Key distribution
    result.rows.push_back({"Total Keys", std::to_string(treeStats.avgKeysPerNode * treeStats.leafNodes), "Estimated total keys"});
    result.rows.push_back({"Keys/Node (Avg)", std::to_string(treeStats.avgKeysPerNode), "Average keys per node"});
    result.rows.push_back({"Keys/Node (Min)", std::to_string(treeStats.minKeysPerNode), "Minimum keys per node"});
    result.rows.push_back({"Keys/Node (Max)", std::to_string(treeStats.maxKeysPerNode), "Maximum keys per node"});

    // Space efficiency
    result.rows.push_back({"Space Utilization", std::to_string(treeStats.spaceUtilization * 100) + "%", "Page space usage"});
    result.rows.push_back({"Used Space", std::to_string(treeStats.usedSpace) + " bytes", "Total used space"});
    result.rows.push_back({"Free Space", std::to_string(treeStats.freeSpace) + " bytes", "Total free space"});

    // Fractal Tree specific metrics
    result.rows.push_back({"Buffered Messages", std::to_string(treeStats.totalMessages), "Unapplied fractal messages"});
    result.rows.push_back({"Split Operations", std::to_string(treeStats.splitCount), "Node splits"});
    result.rows.push_back({"Merge Operations", std::to_string(treeStats.mergeCount), "Node merges"});

    // Performance indicators
    double balanceFactor = treeStats.maxKeysPerNode > 0 ? static_cast<double>(treeStats.minKeysPerNode) / treeStats.maxKeysPerNode : 0.0;

    std::string balanceStatus = "Good";
    if (balanceFactor < 0.3) balanceStatus = "Poor (needs rebalancing)";
    else if (balanceFactor < 0.6) balanceStatus = "Fair";

    result.rows.push_back({"Balance Factor", std::to_string(balanceFactor), balanceStatus});

    return result;
}

ExecutionEngine::ResultSet ExecutionEngine::executeShowBufferPoolStats(AST::ShowBufferPoolStats& stmt) {
    BufferPoolStats stats = getBufferPoolStatistics();

    ResultSet result({"Buffer Pool Metric", "Value", "Description"});

    // Capacity and usage
    result.rows.push_back({"Capacity", std::to_string(stats.capacity) + " pages", "Maximum pages in buffer"});
    result.rows.push_back({"Current Size", std::to_string(stats.currentSize) + " pages", "Pages currently cached"});
    result.rows.push_back({"Utilization", std::to_string((stats.currentSize * 100) / std::max<size_t>(1, stats.capacity)) + "%", "Buffer usage percentage"});

    // Performance metrics
    result.rows.push_back({"Hit Count", std::to_string(stats.hitCount), "Successful cache accesses"});
    result.rows.push_back({"Miss Count", std::to_string(stats.missCount), "Cache misses (disk reads)"});
    result.rows.push_back({"Total Reads", std::to_string(stats.totalReads), "Total page read operations"});
    result.rows.push_back({"Total Writes", std::to_string(stats.totalWrites), "Total page write operations"});

    // Hit ratio analysis
    double hitRatio = stats.hitCount + stats.missCount > 0 ? static_cast<double>(stats.hitCount) / (stats.hitCount + stats.missCount) : 0.0;

    std::string hitRatioStatus = "Poor";
    if (hitRatio > 0.9) hitRatioStatus = "Excellent";
    else if (hitRatio > 0.7) hitRatioStatus = "Good";
    else if (hitRatio > 0.5) hitRatioStatus = "Fair";

    result.rows.push_back({"Hit Ratio", std::to_string(hitRatio * 100) + "%", hitRatioStatus});

    // Page statistics
    result.rows.push_back({"Dirty Pages", std::to_string(stats.dirtyPages), "Modified pages not yet written"});
    result.rows.push_back({"Pinned Pages", std::to_string(stats.pinnedPages), "Pages currently in use"});

    // Cache efficiency analysis
    double cacheEfficiency = stats.currentSize > 0 ?
    static_cast<double>(stats.hitCount) / std::max(1.0, (double)stats.currentSize) : 0.0;

    std::string efficiencyStatus = "Low";
    if (cacheEfficiency > 100) efficiencyStatus = "Excellent";
    else if (cacheEfficiency > 50) efficiencyStatus = "Good";
    else if (cacheEfficiency > 20) efficiencyStatus = "Fair";

    result.rows.push_back({"Cache Efficiency", std::to_string(cacheEfficiency), efficiencyStatus + " hits/page"});

    return result;
}

ExecutionEngine::ResultSet ExecutionEngine::executeShowStorageInfo(AST::ShowStorageInfo& stmt) {
    DatabaseStats dbStats = getDatabaseStatistics();

    ResultSet result({"Storage Component", "Details", "Statistics"});

    // File system information
    //result.rows.push_back(std::vector<std::string>{"Database File", storage.getDatabaseFile(db.currentDatabase()), "Size: " + std::to_string(dbStats.fileSize) + " bytes"});

    result.rows.push_back({"Database File", db.currentDatabase() + ".db", "Size: " + std::to_string(dbStats.fileSize) + " bytes"});

    // Page allocation
    result.rows.push_back({"Page Allocation", "Page size: " + std::to_string(fractal::PAGE_SIZE) + " bytes", "Total: " + std::to_string(dbStats.totalPages) + " pages"});

    // Table ranges
    try {
        auto tables = storage.getTableNames(db.currentDatabase());
        result.rows.push_back({"Table Ranges", "Allocated page ranges per table", std::to_string(tables.size()) + " tables"});

        for (const auto& tableName : tables) {
            TableStats tableStats = getTableStatistics(tableName);
            std::string rangeInfo = tableName + ": Pages " + std::to_string(tableStats.pageCount) + ", Records " + std::to_string(tableStats.recordCount);
            result.rows.push_back({"", rangeInfo, ""});
       }
    } catch (...) {
        result.rows.push_back({"Table Ranges", "Error retrieving table information", ""});
    }

    // Data blocks
    result.rows.push_back({"Data Blocks", "Out-of-band data storage", "For large values and text data"});

    // Fractal Tree features
    result.rows.push_back({"Fractal Features", "Message buffering, Adaptive flushing", "Optimized for write-heavy workloads"});

    // Storage engine
    result.rows.push_back({"Storage Engine", "Fractal B+Tree with Buffer Pool", "Page-oriented storage with LRU caching"});

    // Performance tuning
    std::string tuningAdvice = "Buffer pool size: " + std::to_string(dbStats.bufferPoolSize) + " pages";
    if (dbStats.bufferPoolHitRatio < 0.7) {
        tuningAdvice += " (Consider increasing buffer pool)";
    }

    result.rows.push_back({"Performance Tuning", "Current configuration", tuningAdvice});

    return result;
}

DatabaseStats ExecutionEngine::getDatabaseStatistics() {
DatabaseStats stats;

    try {
        // Get basic file statistics
        auto& dbFile = storage.getDatabaseFile(db.currentDatabase());
        stats.totalPages = dbFile.get_total_pages();
        stats.freePages = dbFile.get_free_page_count();
        stats.usedPages = stats.totalPages - stats.freePages;
        stats.fileSize = dbFile.get_file_size();
        stats.readOperations = dbFile.get_read_count();
        stats.writeOperations = dbFile.get_write_count();
        stats.syncOperations = dbFile.get_sync_count();

        // Get buffer pool statistics
        auto bufferPool = storage.getBufferPool();
        if (bufferPool) {
            stats.bufferPoolSize = bufferPool->get_capacity();
            stats.bufferPoolUsed = bufferPool->get_size();
            stats.bufferPoolHits = bufferPool->get_hit_count();
            stats.bufferPoolMisses = bufferPool->get_miss_count();
            stats.bufferPoolHitRatio = bufferPool->get_hit_ratio();
        }
    } catch (const std::exception& e) {
       std::cerr << "Error getting database statistics: " << e.what() << std::endl;
    }

    return stats;
}

TableStats ExecutionEngine::getTableStatistics(const std::string& tableName) {
    TableStats stats;

    try {
        const auto* tableSchema = storage.getTable(db.currentDatabase(), tableName);
        if (!tableSchema) {
            throw std::runtime_error("Table not found: " + tableName);
        }

        stats.tableName = tableName;
        stats.tableId = tableSchema->table_id;
        stats.rootPageId = tableSchema->root_page;

        // Get tree instance for this table
        auto tree = storage.getTree(db.currentDatabase(), tableName);
        if (tree) {
            stats.treeHeight = tree->get_tree_height();
            stats.pageCount = tree->get_total_pages();

            // Get record count from tree scan (expensive but accurate)
            auto allRecords = tree->scan_all(0);
            stats.recordCount = allRecords.size();

            // Get buffered messages count
           auto treeStatsStr = tree->get_tree_stats();
            // Parse the string to extract messages count  Siplified
            size_t msgPos = treeStatsStr.find("Total Messages Buffered: ");
            if (msgPos != std::string::npos) {
                std::string msgStr = treeStatsStr.substr(msgPos + 25);
                size_t endPos = msgStr.find('\n');
                if (endPos != std::string::npos) {
                    msgStr = msgStr.substr(0, endPos);
                    stats.bufferedMessages = std::stoul(msgStr);
                }
            }
        }

        // Get additional metadata from database file
        auto& dbFile = storage.getDatabaseFile(db.currentDatabase());

        // Try to get table metadata from directory
        auto tables = dbFile.get_all_tables();
        for (const auto& table : tables) {
            if (table.name == tableName) {
                // We already have root_page from tableSchema
                break;
            }
        }

    } catch (const std::exception& e) {
       std::cerr << "Error getting table statistics for " << tableName << ": " << e.what() << std::endl;
    }

    return stats;
}

TreeStats ExecutionEngine::getTreeStatistics(const std::string& tableName) {
    TreeStats stats;

    try {
        auto tree = storage.getTree(db.currentDatabase(), tableName);
        if (!tree) {
            return stats;
        }

        // Analyze tree structure by traversing it
        std::vector<uint32_t> leafPages;
        tree->collect_all_leaf_pages_wrapper(leafPages);

        stats.leafNodes = leafPages.size();
        stats.totalNodes = tree->get_total_pages();
        stats.internalNodes = stats.totalNodes - stats.leafNodes;
        stats.treeHeight = tree->get_tree_height();

       // Analyze key distribution in leaf nodes
        size_t totalKeys = 0;
        size_t minKeys = std::numeric_limits<size_t>::max();
        size_t maxKeys = 0;
        size_t totalUsedSpace = 0;
        size_t totalFreeSpace = 0;

        for (uint32_t pageId : leafPages) {
            fractal::Page* page = tree->get_page_wrapper(pageId, false);

            size_t keysInNode = page->header.key_count;
            totalKeys += keysInNode;
            minKeys = std::min(minKeys, keysInNode);
            maxKeys = std::max(maxKeys, keysInNode);

            // Calculate space usage
            size_t usedSpace = sizeof(fractal::PageHeader) +
                              keysInNode * (sizeof(int64_t) + sizeof(fractal::KeyValue)) +
                              page->header.message_count * sizeof(fractal::Message);

            size_t freeSpace = fractal::PAGE_SIZE - usedSpace;
            totalUsedSpace += usedSpace;
            totalFreeSpace += freeSpace;
          tree->release_page_wrapper(pageId, false);
        }

        if (stats.leafNodes > 0) {
            stats.avgKeysPerNode = totalKeys / stats.leafNodes;
            stats.minKeysPerNode = minKeys;
            stats.maxKeysPerNode = maxKeys;
            stats.usedSpace = totalUsedSpace;
            stats.freeSpace = totalFreeSpace;
            stats.spaceUtilization = static_cast<double>(totalUsedSpace) /
                                    (stats.leafNodes * fractal::PAGE_SIZE);
        }

        // Get fractal message count
        auto treeStatsStr = tree->get_tree_stats();
        size_t msgPos = treeStatsStr.find("Total Messages Buffered: ");
        if (msgPos != std::string::npos) {
            std::string msgStr = treeStatsStr.substr(msgPos + 25);
            size_t endPos = msgStr.find('\n');
            if (endPos != std::string::npos) {
                msgStr = msgStr.substr(0, endPos);
                stats.totalMessages = std::stoul(msgStr);
            }
        }

       // Note: splitCount and mergeCount would need to be tracked in FractalBPlusTree
        // For now, we'll set them to 0
        stats.splitCount = 0;
        stats.mergeCount = 0;

    } catch (const std::exception& e) {
        std::cerr << "Error getting tree statistics for " << tableName << ": " << e.what() << std::endl;
    }

    return stats;
}

BufferPoolStats ExecutionEngine::getBufferPoolStatistics() {
    BufferPoolStats stats;

    try {
        auto bufferPool = storage.getBufferPool();
        if (bufferPool) {
            stats.capacity = bufferPool->get_capacity();
            stats.currentSize = bufferPool->get_size();
            stats.hitCount = bufferPool->get_hit_count();
            stats.missCount = bufferPool->get_miss_count();
            stats.totalReads = bufferPool->get_read_count();
            stats.totalWrites = bufferPool->get_write_count();

            // We need to track dirty and pinned pages separately
            // For now, we'll estimate or leave as 0
            stats.dirtyPages = 0;
            stats.pinnedPages = 0;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error getting buffer pool statistics: " << e.what() << std::endl;
    }

    return stats;
}

void ExecutionEngine::analyzeTableFragmentation(const std::string& tableName) {
    try {
        auto tree = storage.getTree(db.currentDatabase(), tableName);
        if (!tree) {
            std::cout << "Table not found or no tree available: " << tableName << std::endl;
            return;
        }

        std::vector<uint32_t> leafPages;
        tree->collect_all_leaf_pages_wrapper(leafPages);

        size_t underfilledNodes = 0;
        size_t severelyUnderfilled = 0;
        size_t totalKeys = 0;

        for (uint32_t pageId : leafPages) {
            fractal::Page* page = tree->get_page_wrapper(pageId, false);

            size_t keysInNode = page->header.key_count;
            totalKeys += keysInNode;

            // B+Tree order is typically 50-100, check for underfilled nodes
            double fillRatio = static_cast<double>(keysInNode) / fractal::BPTREE_ORDER;

            if (fillRatio < 0.5) underfilledNodes++;
          if (fillRatio < 0.3) severelyUnderfilled++;

            tree->release_page_wrapper(pageId, false);
        }

        std::cout << "=== Table Fragmentation Analysis: " << tableName << " ===" << std::endl;
        std::cout << "Total leaf nodes: " << leafPages.size() << std::endl;
        std::cout << "Total records: " << totalKeys << std::endl;
        std::cout << "Average records per node: " << (leafPages.size() > 0 ? totalKeys / leafPages.size() : 0) << std::endl;
        std::cout << "Underfilled nodes (<50%): " << underfilledNodes << " ("
                  << (leafPages.size() > 0 ? (underfilledNodes * 100) / leafPages.size() : 0) << "%)" << std::endl;
        std::cout << "Severely underfilled (<30%): " << severelyUnderfilled << " ("
                  << (leafPages.size() > 0 ? (severelyUnderfilled * 100) / leafPages.size() : 0) << "%)" << std::endl;

        if (severelyUnderfilled > leafPages.size() * 0.2) {
            std::cout << "RECOMMENDATION: Consider running OPTIMIZE TABLE to reduce fragmentation" << std::endl;
        }

       std::cout << "========================================" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error analyzing table fragmentation: " << e.what() << std::endl;
    }
}

void ExecutionEngine::analyzeIndexEfficiency(const std::string& tableName) {
    try {
        TreeStats stats = getTreeStatistics(tableName);

        std::cout << "=== Index Efficiency Analysis: " << tableName << " ===" << std::endl;

        // Tree height analysis
        std::cout << "Tree Height: " << stats.treeHeight << std::endl;
        if (stats.treeHeight > 4) {
            std::cout << "WARNING: Tree height is high. Consider increasing BPTREE_ORDER." << std::endl;
        }

        // Space utilization
        std::cout << "Space Utilization: " << (stats.spaceUtilization * 100) << "%" << std::endl;
        if (stats.spaceUtilization < 0.6) {
            std::cout << "WARNING: Low space utilization. Consider smaller BPTREE_ORDER." << std::endl;
        }

        // Key distribution
        if (stats.leafNodes > 0) {
            std::cout << "Key Distribution: Min=" << stats.minKeysPerNode
                      << ", Avg=" << stats.avgKeysPerNode
                      << ", Max=" << stats.maxKeysPerNode << std::endl;

            double distributionRatio = stats.minKeysPerNode > 0 ?
                static_cast<double>(stats.maxKeysPerNode) / stats.minKeysPerNode : 0.0;

            if (distributionRatio > 3.0) {
                std::cout << "WARNING: Uneven key distribution. Tree may need rebalancing." << std::endl;
            }
        }

        // Fractal message analysis
        if (stats.totalMessages > 1000) {
           std::cout << "WARNING: High message buffer (" << stats.totalMessages
                      << "). Consider manual flush or checkpoint." << std::endl;
        }

        std::cout << "==========================================" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error analyzing index efficiency: " << e.what() << std::endl;
    }
}


ExecutionEngine::ResultSet ExecutionEngine::executeShowDatabaseStructure(AST::ShowDatabaseStructure& stmt) {
    ResultSet result({"DatabaseStructure", "Value"});

    try {
        // Database Overview
        auto tables = storage.getTableNames(db.currentDatabase());
        size_t totalRows = 0;
        size_t totalColumns = 0;

        for (const auto& table : tables) {
            try {
                auto tableData = storage.getTableData(db.currentDatabase(), table);
                totalRows += tableData.size();

                const auto* tableInfo = storage.getTable(db.currentDatabase(), table);
                if (tableInfo) {
                    totalColumns += tableInfo->columns.size();
                    DatabaseStats stats = getDatabaseStatistics();
                }
            } catch (...) {
                // Skip tables that can't be accessed
            }
        }

        result.rows.push_back({"Database Name", db.currentDatabase()});
        result.rows.push_back({"Total Tables", std::to_string(tables.size())});
        result.rows.push_back({"Total Rows", std::to_string(totalRows)});
        result.rows.push_back({"Total Columns", std::to_string(totalColumns)});
        result.rows.push_back({"Storage Engine", "Fractal B+Tree"});
        result.rows.push_back({"Page Size", std::to_string(fractal::PAGE_SIZE) + " bytes"});

        // Separator
        result.rows.push_back({"---", "---"});

        // Table details
        result.rows.push_back({"TABLE DETAILS", ""});

        for (const auto& table : tables) {
            try {
                auto tableData = storage.getTableData(db.currentDatabase(),table);
                const auto* tableInfo = storage.getTable(db.currentDatabase(), table);

                if (tableInfo) {
                    std::string tableSummary = table + " (" + std::to_string(tableData.size()) + " rows, " + std::to_string(tableInfo->columns.size()) + " cols)";
                    result.rows.push_back({"", tableSummary});

                    // Show primary key if exists
                    std::vector<std::string> primaryKeys;
                    for (const auto& col : tableInfo->columns) {
                        if (col.isPrimaryKey) {
                            primaryKeys.push_back(col.name);
                        }
                    }


                    if (!primaryKeys.empty()) {
                        std::string pkStr = "  PK: ";
                        for (size_t i = 0; i < primaryKeys.size(); ++i) {
                            if (i > 0) pkStr += ", ";
                            pkStr += primaryKeys[i];
                        }
                        result.rows.push_back({"", pkStr});
                    }
                }
            } catch (const std::exception& e) {
                result.rows.push_back({"", table + " - ERROR: " + e.what()});
            }
        }

                // Storage statistics
        result.rows.push_back({"---", "---"});
        result.rows.push_back({"STORAGE INFO", ""});

          try {
            // This would require exposing tree statistics
            // auto storageStats = storage.getDatabaseStats(dbName);
            // for (const auto& [key, value] : storageStats) {
            //     result.rows.push_back({key, value});
            // }
            result.rows.push_back({"", "Fractal Tree Optimization: Enabled"});
            result.rows.push_back({"", "Message Buffering: Active"});
            result.rows.push_back({"", "Adaptive Flushing: Enabled"});
        } catch (...) {
            // Ignore if stats not available
        }

    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to get database structure: " + std::string(e.what()));
    }

    return result;
}

std::string ExecutionEngine::getTypeString(DatabaseSchema::Column::Type type) {
    switch (type) {
        case DatabaseSchema::Column::INTEGER: return "INTEGER";
        case DatabaseSchema::Column::FLOAT: return "FLOAT";
        case DatabaseSchema::Column::STRING: return "STRING";
        case DatabaseSchema::Column::BOOLEAN: return "BOOLEAN";
        case DatabaseSchema::Column::TEXT: return "TEXT";
        case DatabaseSchema::Column::VARCHAR: return "VARCHAR";
        case DatabaseSchema::Column::DATETIME: return "DATETIME";
        case DatabaseSchema::Column::DATE: return "DATE";
        case DatabaseSchema::Column::UUID: return "UUID";
        default: return "UNKNOWN";
    }
}
