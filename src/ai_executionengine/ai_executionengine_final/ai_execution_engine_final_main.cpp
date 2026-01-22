// ============================================
// ai_execution_engine_final_main.cpp
// ============================================
#include "ai_execution_engine_final.h"
#include "ai_execution_engine.h"
#include "execution_engine_includes/executionengine_main.h"
#include "database.h"
#include "model_registry.h"
#include "data_extractor.h"
#include "algorithm_registry.h"
#include "ai_grammer.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <random>
#include <future>
#include <queue>
#include <thread>
#include <unordered_set>

AIExecutionEngineFinal::AIExecutionEngineFinal(ExecutionEngine& base_engine, Database& db, fractal::DiskStorage& storage)
    : base_engine_(base_engine), db_(db), storage_(storage) {

    // Initialize base AI engine
    ai_engine_ = std::make_unique<AIExecutionEngine>(db, storage);

    // Initialize data extractor
    data_extractor_ = std::make_unique<esql::DataExtractor>(&storage);

    // Initialize schema discoverer
    schema_discoverer_ = std::make_unique<esql::ai::SchemaDiscoverer>();

    // Initialize thread pool for parallel operations
    initializeWorkerThreads();

    // Warm up model cache
    warmupModelCache();

    std::cout << "[AIExecutionEngineFinal] Initialized with thread pool and model caching" << std::endl;
}

AIExecutionEngineFinal::~AIExecutionEngineFinal() {
    stopWorkerThreads();
}

ExecutionEngine::ResultSet AIExecutionEngineFinal::execute(std::unique_ptr<AST::Statement> stmt) {
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    // Update statistics
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        ai_stats_.total_ai_queries++;
    }

    try {
        if (auto create_table = dynamic_cast<AST::CreateTableStatement*>(stmt.get())) {
            // Check if this is a CREATE TABLE AS SELECT with AI functions
            if (create_table->query && dynamic_cast<AST::SelectStatement*>(create_table->query.get())) {
                auto select_stmt = dynamic_cast<AST::SelectStatement*>(create_table->query.get());
                if (base_engine_.hasWindowFunctionWrapper(*select_stmt) || extractAIFunctionsFromSelect(*select_stmt).size() > 0) {
                    return executeCreateTableAsAIPrediction(*create_table);
                }
            }
        }
        // Check for SELECT with AI functions
        else if (auto select_stmt = dynamic_cast<AST::SelectStatement*>(stmt.get())) {
            auto ai_functions = extractAIFunctionsFromSelect(*select_stmt);
            if (!ai_functions.empty()) {
                return executeSelectWithAIFunctions(*select_stmt);
            }
        }
        // Check for specific AI statement types
        else if (auto create_model = dynamic_cast<AST::CreateModelStatement*>(stmt.get())) {
            return executeCreateModel(*create_model);
        } else if (auto create_or_replace = dynamic_cast<AST::CreateOrReplaceModelStatement*>(stmt.get())) {
            return executeCreateOrReplaceModel(*create_or_replace);
        } else if (auto describe_model = dynamic_cast<AST::DescribeModelStatement*>(stmt.get())) {
            return executeDescribeModel(*describe_model);
        } else if (auto analyze_data = dynamic_cast<AST::AnalyzeDataStatement*>(stmt.get())) {
            return executeAnalyzeData(*analyze_data);
        } else if (auto create_pipeline = dynamic_cast<AST::CreatePipelineStatement*>(stmt.get())) {
            return executeCreatePipeline(*create_pipeline);
        } else if (auto batch_ai = dynamic_cast<AST::BatchAIStatement*>(stmt.get())) {
            return executeBatchAI(*batch_ai);
        } else if (auto* ai_stmt = dynamic_cast<AST::AIStatement*>(stmt.get())) {
            return ai_engine_->executeAIStatement(std::move(stmt));
        }

        // Fall back to base execution engine
        return base_engine_.execute(std::move(stmt));

    } catch (const std::exception& e) {
        // Update error statistics
        logAIOperation("execute", "unknown", "FAILED", e.what());

        // Re-throw for higher level handling
        throw;
    }
    // Update timing statistics
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time
    );

    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        ai_stats_.total_execution_time += duration;
    }
}

void AIExecutionEngineFinal::initializeWorkerThreads(size_t num_threads) {
    for (size_t i = 0; i < num_threads; ++i) {
        worker_threads_.emplace_back([this]() {
            while (true) {
                std::function<void()> task;

                {
                    std::unique_lock<std::mutex> lock(queue_mutex_);
                    cv_.wait(lock, [this]() {
                        return !task_queue_.empty() || stop_workers_;
                    });

                    if (stop_workers_) {
                        return;
                    }

                    task = std::move(task_queue_.front());
                    task_queue_.pop();
                }

                task();
            }
        });
    }
}

void AIExecutionEngineFinal::stopWorkerThreads() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        stop_workers_ = true;
    }

    cv_.notify_all();

    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    worker_threads_.clear();
}

void AIExecutionEngineFinal::addTask(std::function<void()> task) {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        task_queue_.push(std::move(task));
    }
    cv_.notify_one();
}

std::shared_ptr<esql::ai::AdaptiveLightGBMModel>
AIExecutionEngineFinal::getOrLoadModel(const std::string& model_name) {
    {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        auto it = model_cache_.find(model_name);
        if (it != model_cache_.end()) {
            return it->second;
        }
    }

    // Load model
    auto& registry = esql::ai::ModelRegistry::instance();
    auto model = registry.load_model(model_name);

    if (!model) {
        throw std::runtime_error("Failed to load model: " + model_name);
    }

    // Cache the model
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);

    // Check cache size and evict if necessary
    if (model_cache_.size() >= max_cache_size_) {
        // Simple LRU eviction - remove first element
        model_cache_.erase(model_cache_.begin());
    }

    auto shared_model = std::shared_ptr<esql::ai::AdaptiveLightGBMModel>(model.release());
    model_cache_[model_name] = shared_model;

    return shared_model;
}

bool AIExecutionEngineFinal::ensureModelLoaded(const std::string& model_name) {
    try {
        getOrLoadModel(model_name);
        return true;
    } catch (...) {
        return false;
    }
}

void AIExecutionEngineFinal::warmupModelCache() {
    std::cout << "[AIExecutionEngineFinal] Warming up model cache..." << std::endl;

    try {
        auto& registry = esql::ai::ModelRegistry::instance();
        auto models = registry.list_models();

        // Load top 5 most recently used models
        size_t count = 0;
        for (const auto& model_name : models) {
            if (count >= 5) break;

            try {
                getOrLoadModel(model_name);
                count++;
                std::cout << "  Loaded model: " << model_name << std::endl;
            } catch (...) {
                // Skip models that can't be loaded
            }
        }

        std::cout << "[AIExecutionEngineFinal] Model cache warmup complete. Loaded "
                  << count << " models." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[AIExecutionEngineFinal] Model cache warmup failed: "
                  << e.what() << std::endl;
    }
}

void AIExecutionEngineFinal::clearModelCache() {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    model_cache_.clear();
    std::cout << "[AIExecutionEngineFinal] Model cache cleared." << std::endl;
}

AIExecutionEngineFinal::AIStats AIExecutionEngineFinal::getAIStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return ai_stats_;
}

void AIExecutionEngineFinal::logAIOperation(const std::string& operation,
                                          const std::string& model_name,
                                          const std::string& status,
                                          const std::string& details) {

    std::string log_entry = "[AI] " + operation + " - " + model_name +
                           " - " + status + " - " + details;

    std::cout << log_entry << std::endl;

    // You could also write to a log file here
}

bool AIExecutionEngineFinal::isAIFunctionExpression(const AST::Expression* expr) {
    return dynamic_cast<const AST::AIFunctionCall*>(expr) != nullptr ||
           dynamic_cast<const AST::AIScalarExpression*>(expr) != nullptr ||
           dynamic_cast<const AST::ModelFunctionCall*>(expr) != nullptr;
}

std::vector<std::string> AIExecutionEngineFinal::extractAIFunctionsFromSelect(
    AST::SelectStatement& stmt) {

    std::vector<std::string> ai_functions;

    std::function<void(const AST::Expression*)> check_expr =
    [&](const AST::Expression* expr) {
        if (auto ai_func = dynamic_cast<const AST::AIFunctionCall*>(expr)) {
            ai_functions.push_back("AI_" + ai_func->model_name);
        } else if (auto ai_scalar = dynamic_cast<const AST::AIScalarExpression*>(expr)) {
            ai_functions.push_back("SCALAR_" + ai_scalar->model_name);
        } else if (auto model_func = dynamic_cast<const AST::ModelFunctionCall*>(expr)) {
            ai_functions.push_back("MODEL_" + model_func->model_name);
        }

        // Recursively check nested expressions
        if (auto func_call = dynamic_cast<const AST::FunctionCall*>(expr)) {
            for (const auto& arg : func_call->arguments) {
                check_expr(arg.get());
            }
        }
    };

    for (const auto& expr : stmt.columns) {
        check_expr(expr.get());
    }

    return ai_functions;
}

bool AIExecutionEngineFinal::isValidModelName(const std::string& name) const {
    // Simple validation - can be enhanced
    return !name.empty() && name.length() <= 64 &&
           name.find_first_of(" \t\n\r") == std::string::npos;
}
