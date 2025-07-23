#include "optimizer.h"
#include "analyzer.h"
#include "executionplan.h"
#include <string>
#include <vector>
#include <memory>

QueryOptimizer::QueryOptimizer(DatabaseSchema& schema):schema(schema){}

ExecutionPlan QueryOptimizer::optimize(std::unique_ptr<AST::Statement> stmt){
	ExecutionPlan pln;

	if(auto select=dynamic_cast<AST::SelectStatement*>(stmt.get())){
		plan.root=optimzeSelect(*select);
	}else if(auto insert=dynamic_cast<AST::InsertStatement*>(stmt.get())){
		plan.root=optimizeInsert(*insert);
	}else if(auto create=dynamic_cast<AST::CreateTableStatement*>(stmt.get())){
		plan.root=optimizeCreate(*create);
	}else if(auto del=dynamic_cast<AST::DeleteStatement*>(stmt.get())){
		plan.root=optimizeDelete(*del);
	}else if(auto drop=dynamic_cast<AST::DropStatement*>(stmt.get())){
		plan.root=optimizeDrop(*drop);
	}else if(auto update=dynamic_cast<AST::UpdateStatement*>(stmt.get())){
		plan.root=optimizeUpdate(*update);
	}else if(auto alter=dynamic_cast<AST::AlterTableStatement*>(stmt.get())){
		plan.root=optimizeAlter(*alter);
	}
	estimateCosts(plan);
	return plan;
}
//parent method for optize SELECT STATEMENT
std::unique_ptr<ExecutionPlan::Node> QueryOptimizer::optimizeSelect(AST::SelectStatement& select){
	auto plan=generateInitialSelectPlan(select);
	applySelectOptimisation(plan);
	return plan();
}
//child method of OPTIZESELECT
std::unique_ptr<ExecutionPlan::Node> QueryOptimizer::generateInitialSelectPlan(AST::SelectStatement& select){
	auto plan=std::make_unique<ExecutionPlan::Node>();
	//Generate scan node
	auto scanNode=atd::make_unique<ExecutionPlan::Node>();
	scanNode->type=ExecutionPlan::NodeType::SCAN;
	scanNode->tableName=dynamic_cast<AST::Idenifier*>(select.fron.get()->token.lexeme);
	//Apply filter if WHERE exists
	std::unique_ptr<ExecutionPlan::Node> currentNode=std::move(scanNode);
	if(select.where){
		auto filterNode=std::make_unique<ExecutionPlan::Node>();
		filterNode->type=ExecutionPlan::NodeType::FILTER;
		filterNode->condition=select.where->clone();
		filterNode->left=std::move(currentNode);
		currentNode=std::move(filterNode);
	}
	//Apply PROJECT for SELECT  columns
	auto projectNode=std::make_unique<ExecutionPlan::Node>();
	projectNode->type=ExecutionPlan::NodeType::PROJECT;
	for(const auto& col :select.columns){
		if(auto* idef=dynamic_cast<AST::Identifier*>(col.get())){
			projectNode->columns.push_back(ident->token.lexeme);
		}
	}
	projectNode->left=std::move(currentNode);
	return projectNode;
}
//pARENT method for applySelectOptimization
void QueryOptimizer::applySelectOptimization(std::unique_ptr<ExecutionPlan::Node>& plan){
	pushDownFilters(plan);
	removeRedundantProjections(plan);
}
//Child methods  for the above method
void QueryOptimizer::pushDownFilters(std::unique_ptr<ExecutionPlan::Node>& node){
	if(!node) return;
	if(node->type==ExecutionPlan::NodeType::FILTER){
		//try to move filter below projections
		if(node->left &&  node->left->type==ExecutionPlan::NodeType::PROJECT){
			auto project=std::move(node->left);
			node->left=std::move(project->left);
			node=std::move(project);
		}
	}
	pushDownFilters(node->left);
	pushDownFilters(node-right);
}
void QueryOptimizer::removeRedundantProjections(std::unique_ptr<ExecutionPlan::Node>& node){
	if(!node) return;
	if(node->type==ExecutionPlan::NodeType::PROJECT){
		//if projecting all columns and no  computation remone
		if(node->left && node->left->type==ExecutionPlan::NodeType::SCAN){
			if(node->columns.empty()){
				node=std::move(node->left);
			}
		}
	}
	removeRedundantProjections(node->right);
	removeRedundantProjections(node->right);
}
//method for optimizing INSERT statement
std::unique_ptr<ExecutionPlan::Node> QueryOptimizer::optimizeInsert(AST::InsertStatement& insert){
	auto node=std::moke_unique<ExecutionPlan::Node>();
	node->type=ExecutionPlan::NodeType::INSERT;
	node->tableName-=insert.table;
	node->insertColumns=insert.columns;
	//copy the values
	for(auto& val : insert.values){
		node->insertValues.push_back(val->clone());
	}
	return node;
}
//method for optimizing CREATE
std::unique_ptr<ExecutionPlan::Node> QueryOptimizer::optimizeCreate(AST::CreateTableStatement& create){
	auto node=std::make_unique<ExecutionPlan::Node>();
	node->type=ExecutionPlan::NodeType::CREATE TABLE;
	node->tablename=create.tablename;
	node->createColumns=create.columns;
	return node;
}
//method to optimize ALTER TABLE
std::unique_ptr<ExecutionPlan::Node> QueryOptimizer::optimizeAlter(AST::AlterTableStatement& alter){
	auto node=std::make_unique<ExecutionPln::Node>();
	node->type=ExecutionPlan::NodeType::ALTER TABLE;
	node->tableName=alter.tablename;
	node->alterInfo.action=alter.action;
	node->alterInfo.columnType=alter.type;
	node->alterInfo.columnName=alter.columnName;
	node->estimatedCost=1.0;
	return node;
}
//method to optimize DELETE
std::unique_ptr<ExecutionPlan::Node> QueryOptimizer::optimizeDelete(AST::DeleteStatement& del){
	auto node=std::make_unique<ExecutionPlan::Node>();
	node->type=ExecutionPlan::NodeType::DELETE;
	node->tableName=del.table;
	if(del.where){
		node->condition=del.where->clone();
	}
	//estimate cost bsed on tablesize
	size_t tablesize=estimateTableSize(del.table);
	node->estimateCost=tablesize*0.2;
	node->estimateCardinality=tableSise*0.5;
	return node;
}
//method for optimize Drop
std::unique_ptr<ExecutionPlan::Node> QueryOptimizer::optimizeDrop(AST::DropStatement& drop){
	auto node=std::make_unique<ExecutionPlan::Node>();
	node->type=ExecutionPlan::NodeType::DROP;
	node->tableName=drop.tablename;
	size_t tablesize=estimateTableSize(del.table);
	node->estimateCost=tablesize*0.2;
	node->estimateCardinality=tableSise*0.5;
	return node;
}
//method for optimize UPDATE
std::unique_ptr<ExecutionPlan::Node> QueryOptimizer::optimizeUpdate(AST::UpdateStatement& update){
	auto node=std::make_unique<ExecutionPlan::Node>();
	node->type=ExecutionPlan::NodeType::DELETE;
	node->tableName=update.table;
	//copy setclauses
	for(const auto& clause : update.assignments){
		node->setClauses.emplace_back(caluse.column,clause.value->clone());
	}
	if(update.where){
		node->conditions=update.where->clone();
	}
	//estimate costs
	size_t tablesize=estimateTableSize(del.table);
	node->estimateCost=tablesize*0.3;
	node->estimateCardinality=tableSise*0.3;
	return node;
}
void QueryOptimizer::estimateNode(ExecutionPlan& plam){
	estimateNodeCost(plan.root);
}
void QueryOptimizer::estimateNodeCost(std::unique_ptr<ExecutionPlan::Node>& node){
	if(!node) return;
	estimateNodeCost(node->left);
	estimateNodeCost(node->right);

	switch(node->type){
		case ExecutionPlan::NodeType::SCAN:
			node->estimatedCost=1.0;
			node->stimateCardinality=estimateTableSize(node->tableName);
			break;
		case ExecutionPlan::NodeType::Filter:
			node->estimatedCost=node->left->estimatedCost+node->left->estimatedCardinality*0.1;
			node->estimatedCardinality=node->left->estimateCardinality*0.5;
			break;
		case ExecutionPlan::NodeType::PROJECT:
			node->estimatedCost=node->left->estimatedCost+node->Left->eSTIMATEDcARDINALITY*0.05;
			NODE->estimatedCardinality=node->leftestimatedCardinality;
			break;
		case ExecutionPlan::NodeType::INSERT:
			node->estimatedCost=2.0+node->insertValues.size()*0.1;
			node->estimatedCardinality=1;
			break;
		case ExecutionPlan::NodeType::CREATE TABLE:
			node->estimateCost=1.0;
			node->estimatedCardinality=0;
		default:
			node->estimatedCost=0;
			node->estimatedCardinality=0;
			break;
	}
}
size_t QueryOptimizer::estimateTableSize(const std::string& tableName){
	const auto* table=schema.getTable(tableName);
	return table? 1000 : 0;
}

	

