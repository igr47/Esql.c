#include "parser.h"
#include <string>
#include <stdexcept>
#include <vector>

namespace AST{
	Identifier::Identifier(const Token& token):token(token){}

	BinaryOp::BinaryOp(Token op,std::unique_ptr<Expression> left,std::unique_ptr<Expression> right): op(op),left(std::move(left)),right(std::move(right)){}
};

Parse::Parse(Lexer& lexer) : lexer(lexer){
	currentToken=lexer.nextToken();
}

std::unique_ptr<AST::SelectStatement> Parse::parse(){
	return  parseSelectStatement();
}

void Parse::consume(Token::Type expected){
	if(currentToken.type==expected){
		advance();
	}else{
		throw std::runtime_error("Unexpected token at line "+ std::to_string(currentToken.li    ne) +",column ," +std::to_string(currentToken.column));
	}
}

bool match(Token::Type type) const{
	return currentToken.type==type;
}

bool Parse::matchAny(const std::vector<Token::Type>& types) const{
	for(auto type : types){
		if(match(type)) return true;
	}
	return false;
}

std::unique_ptr<AST::SelectStatement> Parse::parseSelectStatement(){
	auto stmt=std::make_unique<AST::SelectStatement>();
	//parse select clause
	consume(Token::SELECT);
	stmt->columns=parseColumnList();
	//parse From clause
	consume(Token::FROM);
	stmt->from=parseFromClause();
	//parse optional WHERE clause
	if(match(Token::WHERE)){
		consume(Token::WHERE);
		stmt->where=parseExpression();
	}
	return stmt;
}

std::vector<std::unique_ptr<AST::Expression>> Parse::parseColumnList(){
	std::vector<std::unique_ptr<AST::Expression>> columns;
	do{
		if(match(Token::COMMA)){
			consume(Token::COMMA);
		}
		columns.push_back(parseExpression());
	}while(match(Tokem::COMMA));
	return columns;
}

std::unique_ptr<AST::Expression> Parse::parseFromClause(){
	return parseIdentifier();
}

std::unique_ptr<AST::Expression> Parse::parseExpression(){
	return parseBinaryExpression(0);
}

std::unique_ptr<AST::Expression> Parse::parseBinaryExpression(int minPrecedence){
	auto left=parsePrimaryExpression();

	while(true){
		Token op=currentToken;
		int precedence=getPrecedence(op.type);
		if(precedence<minPrecedence){
			break;
		}
		if(!isBinareOperator(op.type)){
			break;
		}
		advance();
		auto right=parseBinareExpression(precedence+1);
		left=std::make_unique<AST::BinaryOp>(op,,std::move(left),std::move(right));
	}
	return left;
}

std::unique_ptr<AST::Expression> Parse::parsePrimaryExpression(){
	if(match(Token::IDENTIFIER)){
		return parseIdentifier();
	}else if(match(Token::NUMBER_LITERAL) || match(Token::STRING_LITERAL)){
		return parseLiteral();
	}else if(match(Token::L_PAREN)){
		consume(Token::L_PAREN);
		auto expr=parseExpression();
		consume(Token::R_PAREN);
		return expr;
	}else{
		throw std::runtime("Expected expression at line " + std::to_string(currentToken.line) + ",column, " + std::to_string(currentToken.column));
	}
}

std::unique_ptr<AST::Expression> Parse::parseIdentifier(){
	auto identifier=std::make_unique<AST::Identifier>(currentToken);
	consume(Token::IDENTIFIER);
	return identifier;
}
std::unique_ptr<AST::Expression> Parse::parseLiteral(){
	auto literal=std::male_unique<AST::Literal>(currentToken);
	if(match(Token::NUMBER_LITERAL)){
		consume(Token::NUMBER_LITERAL);
	}else{
		consume(Token::STRING_LITERAL);
	}
}

int Parse::getPrecedence(Token::Type type){
	switch(type){
		case TOKEN::OR: return 1;
		case Token::AND: return 2;
		case Token::EQUAL:
		case Token::NOT_EQUAL:
		case Token::LESS:
		case Token::LESS_EQUAL:
		case Token::GREATER:
		case Token::GREATER_EQUAL: return 3;
		default: return 0;
	}
}
bool Parse::isBinaryOperator(Token::Type type){
	return getPrecedence(type) >0;
}
