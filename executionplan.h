#ifndef AXECUTION_PLAN_H
#define EXECUTION_PLAN_H
#include <string>
#include <vector>
#include <memory>

class ExecutionPlan{
	public:
		enum class NodeType{SCAN,FILTER,PROJECT,NESTED_LOOP_JOIN,HASH_JOIN,AGGREGATE,SORT,LIMIT,INSERT,CREATE TABLE,DELETE,DROP,ALTER TABLE,UPDATE};
		struct Node{
			NodeType type;
			std::unique_ptr<Node> left;
			std::unique_ptr<Node> right;
			//OPERATION SPECIFIC DATA
			std::string tableName;//For SCAN,INSERT
			std::vector<std::string> columns;//for PROJECT
			std::unique_ptr<AST::Expression> condition;//For FILTER
			std::unique_ptr<AST::Expression> conditions;//for  DELETE and UPDATE
			std::vector<std::string> insertColumns;//For INSERT
			std::vector<std::pair<std::unique_ptr<AST::Identifier>,std::unique_ptr<AST::Expression>>> setClauses;
			std ::vector<std::unique_ptr<AST::Expression>> insertValues//For INSERT
			std::vector<AST::ColumnDefination> createColumns//For CREATE TABLES
			struct alterInfo{
				AST::AlterTableStatement::Action::action;
				std::string columnNmae;
				std::string columnType;
				std::vector<std::string> constraints;
			}

			//statistics for estimation
			size_t estimateCardinality=0;
			double estimateCost=0.0;
		};

		std::unique_ptr<Node> root;
		void visualize() const;
	private:
		void visualizeNode(const Node* node,int ident);
		std::string nodeToString(NodeType type) const;

}
#endif
