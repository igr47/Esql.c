#include "deadlock_detector.h"

namespace fractal {
    std::unordered_map<std::thread::id, std::vector<LockLevel>> DeadlockDetector::thread_locks;
    std::shared_mutex DeadlockDetector::detector_mutex;
    std::atomic<bool> DeadlockDetector::enabled{true};
}
