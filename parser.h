#ifndef PARSER_H
#define PARSER_H
#include "scanner.h"
#include <stdexcept>
#include <memory>
#include <vector>
#include <string>

namespace AST{
	class Node{
		public:
			virtual ~Node()=default;
	};
	class Statement:public Node{
		public:
			virtual ~Statement()=default;
	};
	class Expression:public Node{
		public:
			virtual std::unique_ptr<Expression> clone() const =0;
			virtual ~Expression()=default;
			virtual std::string toString() const=0;
	};

	class BetweenOp : public Expression{
		public:
			std::unique_ptr<Expression> column;
			std::unique_ptr<Expression> lower;
			std::unique_ptr<Expression> upper;
			std::unique_ptr<Expression> clone() const override{
				return std::make_unique<BetweenOp>(column->clone(),lower->clone(),upper->clone());
			}
			BetweenOp(std::unique_ptr<Expression> col, std::unique_ptr<Expression> low,std::unique_ptr<Expression> up) : column(std::move(col)),lower(std::move(low)),upper(std::move(up)){}
			std::string toString() const override{
				return column->toString() + "BETWEEN" +lower->toString()+ "AND" +upper->toString();
			}
	};

	class HavingClause : public Expression{
		public:
			std::unique_ptr<Expression> condition;
			std::unique_ptr<Expression> clone() const override{
				return std::make_unique<HavingClause>(condition->clone());
			}

			explicit HavingClause(std::unique_ptr<Expression> cond) : condition(std::move(cond)) {}
			std::string toString() const override {
				return "HAVING "+ condition->toString();
			}
	};

	class OrderByClause : public Expression{
		public:
			std::vector<std::pair<std::unique_ptr<Expression>,bool>> columns;
			std::unique_ptr<Expression> clone() const override{
				std::vector<std::pair<std::unique_ptr<Expression>, bool>> clonedColumns;
				for(const auto& col : columns){
					clonedColumns.emplace_back(col.first->clone(),col.second);
				}
				return std::make_unique<OrderByClause>(std::move(clonedColumns));
			}
			explicit OrderByClause(std::vector<std::pair<std::unique_ptr<Expression>, bool>> cols) : columns(std::move(cols)) {}
			std::string toString() const override{
				std::string result = "ORDER BY ";
				for(size_t i = 0 ; i<columns.size(); i++){
					result += columns[i].first->toString();
					if(!columns[i].second) result += "DESC";
					if(i<columns.size()-1) result += ",";
				}

				return result;
			}
	};

	class GroupByClause : public Expression{
		public:
			std::vector<std::unique_ptr<Expression>> columns;
			std::unique_ptr<Expression> clone() const override{
				std::vector<std::unique_ptr<Expression>> clonedColumns;
				for (const auto& col : columns){
					clonedColumns.push_back(col->clone());
				}
				return std::make_unique<GroupByClause>(std::move(clonedColumns));
			}

			explicit GroupByClause(std::vector<std::unique_ptr<Expression>> cols) : columns(std::move(cols)) {}
			std::string toString() const override{
				std::string result = "GROUP BY " ;
				for(size_t i =0; i<columns.size();++i){
					result += columns[i]->toString();
					if(i<columns.size() - 1) result += "'";
				}
				return result;
			}
	};


	class InOp : public Expression{
		public:
			std::unique_ptr<Expression> column;
			std::vector<std::unique_ptr<Expression>> values;
			std::unique_ptr<Expression> clone() const override{
				std::vector<std::unique_ptr<Expression>> clonedValues;
				for(const auto& val : values){
					clonedValues.push_back(val->clone());
				}
				return std::make_unique<InOp> (column->clone(), std::move(clonedValues));
			}
			InOp(std::unique_ptr<Expression> col, std::vector<std::unique_ptr<Expression>> vals) : column(std::move(col)), values(std::move(vals)){}

			std::string toString() const override {
				std::string result = column->toString() + "IN(" ;
				for(size_t i=0; i<values.size(); ++i){
					result += values[i]->toString();
					if(i<values.size()-1) result += ",";
				}
				return result += ")";
			}
	};

	class NotOp : public Expression{
		public:
			std::unique_ptr<Expression> expr;
			std::unique_ptr<Expression> clone() const override{
				return std::make_unique<NotOp> (expr->clone());
			}

			 explicit NotOp(std::unique_ptr<Expression> e) : expr(std::move(e)) {}
			 std::string toString() const override{
				 return "NOT " +expr->toString();
			}
	};

	class Literal:public Expression{
		public:
			Token token;
			std::unique_ptr<Expression> clone() const override{
				return std::make_unique<Literal>(token);
			}
			//Literal()=default;
			explicit Literal(const Token& token);
			std::string toString() const override{
				return token.lexeme;
			}
	};

	class Identifier:public Expression{
		public:
			Token token;
			std::unique_ptr<Expression> clone() const override {
				return std::make_unique<Identifier>(token);
			}
			explicit Identifier(const Token& token);
			std::string toString() const override{
				return token.lexeme;
			}
	};

	class BinaryOp:public Expression{
		public:
			Token op;
			std::unique_ptr<Expression> clone() const override {
				return std::make_unique<BinaryOp>(op, left->clone(),right->clone());
			}
			std::unique_ptr<Expression> left;
			std::unique_ptr<Expression> right;

			BinaryOp(Token op,std::unique_ptr<Expression> left,std::unique_ptr<Expression> right);
			std::string toString() const override{
				return left->toString()+" "+op.lexeme+" "+right->toString();
			}
	};
	class ColumnDefination:public Node{
		public:
		        std::string name;
		        std::string type;
		        std::vector<std::string> constraints;
	};
	class CreateDatabaseStatement:public Statement{
		public:
			std::string dbName;
	};
	class UseDatabaseStatement:public Statement{
		public:
			std::string dbName;
	};
	class ShowDatabaseStatement:public Statement{
		public:
	};
	class ShowTableStatement:public Statement{
		public:
	};
	class SelectStatement:public Statement{
		public:
			std::vector<std::unique_ptr<Expression>> columns;
			std::vector<std::pair<std::unique_ptr<Expression>,std::string>> newCols;
			std::unique_ptr<Expression> from;
			std::unique_ptr<Expression> where;
			std::unique_ptr<GroupByClause> groupBy;
			std::unique_ptr<HavingClause> having;
			std::unique_ptr<OrderByClause> orderBy;
			std::unique_ptr<Expression> limit;
			std::unique_ptr<Expression> offset;
			bool distinct = false;
	};
	class AggregateExpression : public Expression{
		public:
			Token function;
			std::unique_ptr<Expression> argument;
			std::unique_ptr<Expression> argument2;
			bool isCountAll = false;

			
			std::unique_ptr<Expression> clone() const override {
				return std::make_unique<AggregateExpression>(function,argument ? argument->clone() : nullptr , argument2 ? argument2->clone() : nullptr ,isCountAll);
			}

			AggregateExpression(Token func,std::unique_ptr<Expression> arg,std::unique_ptr<Expression> arg2 = nullptr ,bool countAll = false) : function(func) , argument(std::move(arg)),argument2(std::move(arg2)),isCountAll(countAll){}
			std::string toString() const override {
				if(isCountAll){
					return function.lexeme + "(*)";
				}
				return function.lexeme + "(" + (argument ? argument-> toString() : "") + ")";
			}
	};

	class UpdateStatement:public Statement{
		public:
			std::string table;
			std::vector<std::pair<std::string,std::unique_ptr<Expression>>> setClauses;
			std::unique_ptr<Expression> where;

	};
	//class to hanle delete statements
	class DeleteStatement:public Statement{
		public:
			std::string table;
			std::unique_ptr<Expression> where;
	};
	class DropStatement:public Statement{
		public:
			std::string tablename;
			bool ifNotExists=false;
	};
	class InsertStatement:public Statement{
		public:
			std::string table;
			std::vector<std::string> columns;
			std::vector<std::vector<std::unique_ptr<Expression>>> values;
	};
	class CreateTableStatement:public Statement{
		public:
			std::string tablename;
			std::vector<ColumnDefination> columns;
			bool ifNotExists=false;
	};
	class AlterTableStatement:public Statement{
		public:
			std::string tablename;
			enum Action{ADD,DROP,RENAME}action;
			std::string columnName;
			std::string type;
			std::string newColumnName;
	};
        class BulkInsertStatement : public Statement {
                public:
                        std::string table;
                        std::vector<std::string> columns;
                        std::vector<std::vector<std::unique_ptr<Expression>>> rows;
        };

        class BulkUpdateStatement : public Statement {
                public:
                        struct UpdateSpec {
                                uint32_t row_id;
                                std::vector<std::pair<std::string, std::unique_ptr<Expression>>> setClauses;
                        };

                        std::string table;
                        std::vector<UpdateSpec> updates;
        };

        class BulkDeleteStatement : public Statement {
                public:
                        std::string table;
                        std::vector<uint32_t> row_ids;
        };

};

class ParseError : public std::runtime_error {
public:
    size_t line;
    size_t column;
    
    ParseError(size_t line, size_t column, const std::string& message)
        : std::runtime_error(message), line(line), column(column) {}
    
    // Helper function to format the error message
    std::string fullMessage() const {
        return "Parse error at line " + std::to_string(line) + ", column " + std::to_string(column) + ": " + what();
    }
};
class Parse{
	public:
		explicit Parse(Lexer& lexer);
		std::unique_ptr<AST::Statement> parse();
	private:
		Lexer& lexer;
		Token currentToken;
		Token previousToken_;
		bool inValueContext=false;
		
		const Token& previousToken() const;
		std::unique_ptr<AST::Expression> parseValue();
		void consume(Token::Type expected);
		Token peekToken();
		void advance();
		bool match(Token::Type type) const;
		bool matchAny(const std::vector<Token::Type>& types) const;
		std::unique_ptr<AST::Statement> parseStatement();
		std::unique_ptr<AST::CreateDatabaseStatement> parseCreateDatabaseStatement();
		std::unique_ptr<AST::UseDatabaseStatement> parseUseStatement();
		std::unique_ptr<AST::ShowDatabaseStatement> parseShowDatabaseStatement();
		std::unique_ptr<AST::ShowTableStatement> parseShowTableStatement();
		std::unique_ptr<AST::GroupByClause>  parseGroupByClause();
		std::unique_ptr<AST::HavingClause> parseHavingClause();
		std::unique_ptr<AST::OrderByClause> parseOrderByClause();
		std::unique_ptr<AST::SelectStatement> parseSelectStatement();
		std::unique_ptr<AST::UpdateStatement> parseUpdateStatement();
		std::unique_ptr<AST::DeleteStatement> parseDeleteStatement();
		std::unique_ptr<AST::DropStatement> parseDropStatement();
		std::unique_ptr<AST::InsertStatement> parseInsertStatement();
		std::unique_ptr<AST::CreateTableStatement> parseCreateTableStatement();
		std::unique_ptr<AST::BulkInsertStatement> parseBulkInsertStatement();
		std::unique_ptr<AST::BulkUpdateStatement> parseBulkUpdateStatement();
		std::unique_ptr<AST::BulkDeleteStatement> parseBulkDeleteStatement();
		void parseColumnDefinition(AST::CreateTableStatement& stmt);
		std::unique_ptr<AST::AlterTableStatement> parseAlterTableStatement();
		std::vector<std::unique_ptr<AST::Expression>> parseColumnList();
		std::vector<std::pair<std::unique_ptr<AST::Expression>,std::string>> parseColumnListAs();
		std::unique_ptr<AST::Expression> parseFromClause();
		std::unique_ptr<AST::Expression> parseExpression();
		std::unique_ptr<AST::Expression> parseBinaryExpression(int minPrecedence);
		std::unique_ptr<AST::Expression> parsePrimaryExpression();
		std::unique_ptr<AST::Expression> parseIdentifier();
		std::unique_ptr<AST::Expression> parseLiteral();
		int getPrecedence(Token::Type type);
		bool isBinaryOperator(Token::Type type);
};
#endif
