#include "locking_policy.h"

namespace fractal {
    thread_local std::vector<LockLevel> LockManager::acquired_locks;
}
