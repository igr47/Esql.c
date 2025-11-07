#pragma once
#ifndef LOCKING_POLICY_H
#define LOCKING_POLICY_H

#include <shared_mutex>
#include <mutex>
#include <atomic>
#include <thread>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <stdexcept>

namespace fractal {

// Lock hierarchy to prevent deadlocks - from highest to lowest priority
enum class LockLevel : int {
    STORAGE = 0,
    WAL_LOG = 1,      // Highest priority
    TREE = 2, 
    BUFFER_POOL = 3,
    PAGE_FRAME = 4,   // Lowest priority
    MAX_LEVEL = 5
};

class LockManager {
private:
    static thread_local std::vector<LockLevel> acquired_locks;
    
public:
    static void validate_lock_order(LockLevel level) {
        // Check if we're trying to acquire a lower-priority lock after higher ones
        for (auto acquired : acquired_locks) {
            if (acquired <= level) {
                throw std::runtime_error("Lock order violation: trying to acquire " + 
                    std::to_string(static_cast<int>(level)) + " after " + 
                    std::to_string(static_cast<int>(acquired)));
            }
        }
        acquired_locks.push_back(level);
    }
    
    static void release_lock(LockLevel level) {
        auto it = std::find(acquired_locks.begin(), acquired_locks.end(), level);
        if (it != acquired_locks.end()) {
            acquired_locks.erase(it);
        }
    }
    
    static void clear_thread_locks() {
        acquired_locks.clear();
    }
    
    // For debugging
    static std::vector<LockLevel> get_current_locks() {
        return acquired_locks;
    }
};

// Hierarchical mutex wrapper with deadlock prevention
template<LockLevel Level>
class HierarchicalMutex {
private:
    std::shared_mutex mutex;
    
public:
    void lock() {
        LockManager::validate_lock_order(Level);
        mutex.lock();
    }
    
    void unlock() {
        mutex.unlock();
        LockManager::release_lock(Level);
    }
    
    bool try_lock() {
        if (mutex.try_lock()) {
            LockManager::validate_lock_order(Level);
            return true;
        }
        return false;
    }
    
    void lock_shared() {
        LockManager::validate_lock_order(Level);
        mutex.lock_shared();
    }
    
    void unlock_shared() {
        mutex.unlock_shared();
        LockManager::release_lock(Level);
    }
    
    bool try_lock_shared() {
        if (mutex.try_lock_shared()) {
            LockManager::validate_lock_order(Level);
            return true;
        }
        return false;
    }
};

// Scoped lock helpers
template<LockLevel Level>
using HierarchicalUniqueLock = std::unique_lock<HierarchicalMutex<Level>>;

template<LockLevel Level>
using HierarchicalSharedLock = std::shared_lock<HierarchicalMutex<Level>>;

} // namespace fractal

#endif
