#pragma once
#ifndef DEADLOCK_DETECTOR_H
#define DEADLOCK_DETECTOR_H

#include "locking_policy.h"
#include <unordered_map>
#include <shared_mutex>
#include <thread>
#include <vector>
#include <algorithm>
#include <chrono>
#include <iostream>

namespace fractal {

class DeadlockDetector {
private:
    static std::unordered_map<std::thread::id, std::vector<LockLevel>> thread_locks;
    static std::shared_mutex detector_mutex;
    static constexpr int LOCK_TIMEOUT_MS = 5000;
    static std::atomic<bool> enabled;

public:
    static void enable() { enabled = true; }
    static void disable() { enabled = false; }
    
    static bool would_cause_deadlock(LockLevel requested, std::thread::id thread_id) {
        if (!enabled) return false;
        
        std::shared_lock lock(detector_mutex);
        auto it = thread_locks.find(thread_id);
        if (it == thread_locks.end()) return false;
        
        // Check if any other thread holds locks that would conflict with our desired order
        for (const auto& [other_thread, other_locks] : thread_locks) {
            if (other_thread == thread_id) continue;
            
            if (!other_locks.empty()) {
                // Check if the other thread holds a lock that we want, but in wrong order
                for (LockLevel other_lock : other_locks) {
                    if (other_lock <= requested) {
                        // This could cause a deadlock if the other thread wants our locks
                        for (LockLevel my_lock : it->second) {
                            if (my_lock >= other_lock) {
                                std::cerr << "DEADLOCK DETECTED: Thread " << thread_id 
                                          << " wants " << static_cast<int>(requested)
                                          << " but thread " << other_thread 
                                          << " holds " << static_cast<int>(other_lock) << std::endl;
                                return true;
                            }
                        }
                    }
                }
            }
        }
        return false;
    }
    
    static void register_lock_attempt(LockLevel level, std::thread::id thread_id) {
        if (!enabled) return;
        
        std::unique_lock lock(detector_mutex);
        
        // Check for timeout - if we've been waiting too long, assume deadlock
        static std::unordered_map<std::thread::id, std::chrono::steady_clock::time_point> attempt_times;
        auto now = std::chrono::steady_clock::now();
        auto last_attempt = attempt_times[thread_id];
        
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_attempt).count() > LOCK_TIMEOUT_MS) {
            std::cerr << "LOCK TIMEOUT: Thread " << thread_id << " timed out waiting for lock level " 
                      << static_cast<int>(level) << std::endl;
            throw std::runtime_error("Lock acquisition timeout - possible deadlock");
        }
        
        attempt_times[thread_id] = now;
        thread_locks[thread_id].push_back(level);
    }
    
    static void register_lock_acquired(LockLevel level, std::thread::id thread_id) {
        if (!enabled) return;
        
        std::unique_lock lock(detector_mutex);
        // Lock is already in the list from register_lock_attempt
    }
    
    static void register_lock_release(LockLevel level, std::thread::id thread_id) {
        if (!enabled) return;
        
        std::unique_lock lock(detector_mutex);
        auto& locks = thread_locks[thread_id];
        auto it = std::find(locks.begin(), locks.end(), level);
        if (it != locks.end()) {
            locks.erase(it);
        }
        if (locks.empty()) {
            thread_locks.erase(thread_id);
        }
    }
    
    static void clear_thread_locks(std::thread::id thread_id) {
        std::unique_lock lock(detector_mutex);
        thread_locks.erase(thread_id);
    }
    
    // For debugging
    static void print_lock_state() {
        std::shared_lock lock(detector_mutex);
        std::cout << "=== Deadlock Detector State ===" << std::endl;
        for (const auto& [thread_id, locks] : thread_locks) {
            std::cout << "Thread " << thread_id << " holds locks: ";
            for (LockLevel level : locks) {
                std::cout << static_cast<int>(level) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "===============================" << std::endl;
    }
};

// Initialize static members
//std::unordered_map<std::thread::id, std::vector<LockLevel>> DeadlockDetector::thread_locks;
//std::shared_mutex DeadlockDetector::detector_mutex;
//std::atomic<bool> DeadlockDetector::enabled{true};

} // namespace fractal

#endif
