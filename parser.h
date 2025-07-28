#ifndef PARSER_H
#define PARSER_H
#include "scanner.h"
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
	};

	class Literal:public Expression{
		public:
			Token token;
			explicit Literal(const Token& token);
	};

	class Identifier:public Expression{
		public:
			Token token;
			explicit Identifier(const Token& token);
	};

	class BinaryOp:public Expression{
		public:
			Token op;
			std::unique_ptr<Expression> left;
			std::unique_ptr<Expression> right;

			BinaryOp(Token op,std::unique_ptr<Expression> left,std::unique_ptr<Expression> right);
	};
	class ColumnDefination:public Node{
		public:
		        std::string name;
		        std::string type;
		        std::vector<std::string> constraints;
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
	};

};

class Parse{
	public:
		explicit Parse(Lexer& lexer);
		std::unique_ptr<AST::Statement> parse();
	private:
		Lexer& lexer;
		Token currentToken;

		void consume(Token::Type expected);
		void advance();
		bool match(Token::Type type) const;
		bool matchAny(const std::vector<Token::Type>& types) const;
		std::unique_ptr<AST::Statement> parseStatement();
		std::unique_ptr<AST::SelectStatement> parseSelectStatement();
		std::unique_ptr<AST::UpdateStatement> parseUpdateStatement();
		std::unique_ptr<AST::DeleteStatement> parseDeleteStatement();
		std::unique_ptr<AST::DropStatement> parseDropStatement();
		std::unique_ptr<AST::InsertStatement> parseInsertStatement();
		std::unique_ptr<AST::CreateTableStatement> parseCreateTableStatement();
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
