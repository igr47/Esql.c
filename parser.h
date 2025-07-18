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
	class SelectStatement:public Node{
		public:
			std::vector<std::unique_ptr<Expression>> columns;
			std::unique_ptr<Expression> from;
			std::unique_ptr<Expression> where;
	};
	class UpdateStatement:public Node{
		public:
			std::unique_ptr<Expression> table;
			struct Assignment{
				std::unique_ptr<Identifier> column;
				std::unique_ptr<Expression> value;
			};

			std::vector<Assignment> assignments;
			std::unique_ptr<Expression> where;

	};
	//class to hanle delete statements
	class DeleteStatement:public Node{
		public:
			std::unique_ptr<Expression> table;
			std::unique_ptr<Expression> where;
	};
	class DropStatement:public Node{
		public:
			std::string tablename;
	};
	class InsertStatement:public Node{
		public:
			std::unique_ptr<Expression> table;
			std::vector<std::unique_ptr<Identifier>> columns;
			std::vector<std::unique_ptr<Expression>> values;
	};
	class CreateTableStatement:public Node{
		public:
			std::unique_ptr<Expression> tablename;
			struct ColumnDef{
				std::string name;
				std::string type;
			};
			std::vector<ColumnDef> columns;
	};
	class AlterTableStatement:public Node{
		public:
			std::unique_ptr<Expression> tablename;
			enum Action{ADD,DROP,RENAME}action;
			std::unique_ptr<Expression> columnName;
			std::unique_ptr<Expression> type;
	};

};

class Parse{
	public:
		explicit Parse(Lexer& lexer);
		std::unique_ptr<AST::Node> parse();
	private:
		Lexer& lexer;
		Token currentToken;

		void consume(Token::Type expected);
		bool match(Token::Type type) const;
		bool matchAny(const std::vector<Token::Type>& types) const;
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
