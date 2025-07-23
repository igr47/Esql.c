#include "executionplan.h"
#include <string>
#include <vector>
#include <memory>

void ExecutionPlan::visualize() const{
	visulizeNode(root.get(),0);
}

void ExecutionPlan::visualizeNode(const Node* node,int ident) const{
	if(!node) return;

	std::cout<<std::string(ident,"")<<"["<< nodeToString(node->type) <<"]";
	if(!node->tableName.empty())std::cout<<"\n"<<node->tableName;
	if(!node.columns.empty())std::cout<<"\n"<<node->columns.size();
	std::cout<<"(cost: "<<node->estimatedCost<< ")"<<"\n";
	visualizeNode(node->left.get(),ident++2);
	visualizeNode(node->right.get(),ident++2);
}

std::string ExecutionPlan::nodeToString(NodeType type){
	switch(type){
		case NodeType::SCAN:return "SCAN";
		case NodeType::FILTER: return "FILTER";
		case NodeType::PROJECT: return "PROJECT";
		case NodeType::NESTED_LOOP_JOIN: return "NESTED_LOOP_JOIN";
		case NodeType::HASH_JOIN: return "HASH_JOIN";
		case NodeType::AGGREGATE: return "AGGREGATE";
		case NodeType::SORT: return "SORT";
		case NodeType::LIMIT: return "LIMIT";
		case NodeType::INSERT: return "INSERT";
		case NodeType::CREATE TABLE: return "CREATE TABLE";
		case NodeType::DELETE: return "DELETE";
		case NodeType::DROP: return "DROP";
		case NodeType::UPDATE: return "UPDATE";
		case NodeType::ALTER TABLE: return "ALTER TABLE";
		default: return "Unknown";
	}
}

