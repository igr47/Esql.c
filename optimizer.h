#ifndef OPTIMIZER_H
#define OPTIMIZER_H
#include "executionplan.h"
#include "analyzer.h"
#include <memory>
#include <string>
#include <vector>

class QueryOptimizer{
	public:
		explicit QueryOptimizer(DatabaseSchema& schema);
		ExecutionPlan optimize(std::unique_ptr<AST::Statement> stmt);
	private:
		DatabaseSchema schema;
		//parent for SELECT statement optimization
		std::unique_ptr<ExecutionPlan::Node> optimizeSelect(AST::SelectStatement& select);
		std::unique_ptr<ExecutionPlan::Node> generateInitialSelectPlan(AST::SelectStatement& select);
		void applySelectOptimization(std::unique_ptr<ExecutionPlan::Node>& plan);
		void pushDownFilters(std::unique_ptr<ExecutionPlan::Node>& node);
		void removeRedundantProjections(std::unique_ptr<ExecutionPlan::Node>& node);
		std::unique_ptr<ExecutionPlan::Node> optimizeInsert(AST::InsertStatement& insert);
		std::unique_ptr<ExecutionPlan::Node> optimizeCreate(AST::CreateTableStatement& create);
		std::unique_ptr<ExecutionPlan::Node> optimizeAlter(AST::AlterTableStatement& alter);
		std::unique_ptr<ExecutionPlan::Node> optimizeDelete(AST::DeleteStatement& del);
		std::unique_ptr<ExecutionPlan::Node> optimizeDrop(AST::DropStatement& drop);
		std::unique_ptr<ExecutionPlan::Node> optimizeUpdate(AST::UpdateStatement& update);
		void estimateNode(ExecutionPlan& plam);
		void estimateNodeCost(std::unique_ptr<ExecutionPlan::Node>& node);
		size_t estimateTableSize(const std::string& tableName);
};
#endif

