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
			virtual ~Expression()=default;
			virtual std::string toString() const=0;
	};

	class Literal:public Expression{
		public:
			Token token;
			//Literal()=default;
			explicit Literal(const Token& token);
			std::string toString() const override{
				return token.lexeme;
			}
	};

	class Identifier:public Expression{
		public:
			Token token;
			explicit Identifier(const Token& token);
			std::string toString() const override{
				return token.lexeme;
			}
	};

	class BinaryOp:public Expression{
		public:
			Token op;
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
			std::unique_ptr<Expression> from;
			std::unique_ptr<Expression> where;
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
			std::vector<std::unique_ptr<Expression>> values;
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
		void advance();
		bool match(Token::Type type) const;
		bool matchAny(const std::vector<Token::Type>& types) const;
		std::unique_ptr<AST::Statement> parseStatement();
		std::unique_ptr<AST::CreateDatabaseStatement> parseCreateDatabaseStatement();
		std::unique_ptr<AST::UseDatabaseStatement> parseUseStatement();
		std::unique_ptr<AST::ShowDatabaseStatement> parseShowDatabaseStatement();
		std::unique_ptr<AST::ShowTableStatement> parseShowTableStatement();
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
