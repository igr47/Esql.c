#ifndef PARSER_H
#define PARSER_H
#include "scanner.h"
#include <memory>
#include <vector>
#include <string>

namesapce AST{
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
			explicit Identifier(Const Token& token);
	};

	class BinaryOp::public Expression{
		public:
			Token op;
			std::unique_ptr<Expression> left;
			std::unique_ptr<Expression> right;

			BinareOp(Token op,std::unique_ptr<Expression> left,std::unique_ptr<Expression> right);
	};
	class SelectStatement:public Node{
		public:
			std::vector<Std::unique_ptr<Expression> colums;
			std::unique_ptr<Expression> from;
			std::unique_ptr<Expression> where;
	}
};

class Parse{
	public:
		explicit Parser(Lexer& lexer);
		std::unique_ptr<AST::SelectStatement> parse();
	private:
		Lexer& lexer;
		Token currentToken;

		void consume(Token::Type expected);
		bool match(Token::Type type) const;
		bool matchAny(const std::vector<Token::Type>& types) const;
		std::unique_ptr<AST::SelectStatement> parseSelectStatement();
		std::vector<std::unique_ptr<AST::Expression>> parseColumnList();
		std::unique_ptr<AST::Expression> parseFromClause();
		std::unique_ptr<AST::Expression> parseExpression();
		std::unique_ptr<AST::Expression> parseBinaryExpression(int minPrecedence);
		std::unique_ptr<AST::Expression> parsePrimaryExpression();
		std::unique_ptr<AST::Expression> parseIdentifier();
		std::unique_ptr<AST::Expression> parseLiteral();
		int getPrecedence(Token::Type type);
		bool isBinaryOperator(Token::Type type);
