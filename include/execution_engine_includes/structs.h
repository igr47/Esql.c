#pragma once
#ifndef STRUCTS_H
#define STRUCTS_H

#include <string>
#include <cstdint>


struct DatabaseStats {
    size_t totalPages = 0;
    size_t freePages = 0;
    size_t usedPages = 0;
    size_t bufferPoolHits = 0;
    size_t bufferPoolMisses = 0;
    double bufferPoolHitRatio = 0.0;
    size_t bufferPoolSize = 0;
    size_t bufferPoolUsed = 0;
    size_t readOperations = 0;
    size_t writeOperations = 0;
    size_t syncOperations = 0;
    uint64_t fileSize = 0;
};

struct TableStats {
    std::string tableName;
    uint32_t tableId = 0;
    uint32_t rootPageId = 0;
    size_t recordCount = 0;
    size_t pageCount = 0;

    size_t totalNodes = 0;
    size_t leafNodes = 0;
    size_t internalNodes = 0;
    size_t treeHeight = 0;

    size_t bufferedMessages = 0;
    size_t memoryUsage = 0;
    uint64_t dataBlockOffset = 0;
    uint64_t dataBlockSize = 0;
    uint64_t createdTimestamp = 0;
    uint64_t lastModified = 0;
};

struct TreeStats {
    size_t totalNodes = 0;
    size_t leafNodes = 0;
    size_t internalNodes = 0;
    size_t treeHeight = 0;

    size_t avgKeysPerNode = 0;
    size_t minKeysPerNode = 0;
    size_t maxKeysPerNode = 0;
    size_t totalMessages = 0;
    size_t freeSpace = 0;
    size_t usedSpace = 0;
    double spaceUtilization = 0.0;
    size_t splitCount = 0;
    size_t mergeCount = 0;
};

struct BufferPoolStats {
    size_t capacity = 0;
    size_t currentSize = 0;
    size_t hitCount = 0;
    size_t missCount = 0;
    size_t totalReads = 0;
    size_t totalWrites = 0;
    size_t dirtyPages = 0;
    size_t pinnedPages = 0;
};

#endif
