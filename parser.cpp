#include "parser.h"
#include "scanner.h"
#include <string>
#include <stdexcept>
#include <vector>

namespace AST{
	Literal::Literal(const Token& token):token(token){}
	Identifier::Identifier(const Token& token):token(token){}

	BinaryOp::BinaryOp(Token op,std::unique_ptr<Expression> left,std::unique_ptr<Expression> right): op(op),left(std::move(left)),right(std::move(right)){}
};

Parse::Parse(Lexer& lexer) : lexer(lexer),currentToken(lexer.nextToken()){}
std::unique_ptr<AST::Statement> Parse::parse(){
	return parseStatement();
}
std::unique_ptr<AST::Statement> Parse::parseStatement(){
	if(match(Token::Type::SELECT)){
	        return  parseSelectStatement();
	}else if(match(Token::Type::UPDATE)){
		return parseUpdateStatement();
	}else if(match(Token::Type::DELETE)){
		return parseDeleteStatement();
	}else if(match(Token::Type::DROP)){
		return parseDropStatement();
	}else if(match(Token::Type::INSERT)){
		return parseInsertStatement();
	}else if(match(Token::Type::CREATE)){
		return parseCreateTableStatement();
	}else if(match(Token::Type::ALTER)){
		return parseAlterTableStatement();
	}else{
		throw std::runtime_error("unexpected token at start of statement");
	}
}

void Parse::consume(Token::Type expected){
	if(currentToken.type==expected){
		advance();
	}else{
		throw std::runtime_error("Unexpected token at line "+ std::to_string(currentToken.line) +",column ," +std::to_string(currentToken.column));
	}
}
void Parse::advance() {
    currentToken = lexer.nextToken();
}

bool Parse::match(Token::Type type) const{
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
	consume(Token::Type::SELECT);
	stmt->columns=parseColumnList();
	//parse From clause
	consume(Token::Type::FROM);
	stmt->from=parseFromClause();
	//parse optional WHERE clause
	if(match(Token::Type::WHERE)){
		consume(Token::Type::WHERE);
		stmt->where=parseExpression();
	}
	return stmt;
}
std::unique_ptr<AST::UpdateStatement> Parse::parseUpdateStatement(){
	auto stmt=std::make_unique<AST::UpdateStatement>();
	//parse update clause
	consume(Token::Type::UPDATE);
	stmt->table=currentToken.lexeme;
	consume(Token::Type::IDENTIFIER);
	//parse SET clause
	consume(Token::Type::SET);
	do{
		if(match(Token::Type::COMMA)){
			consume(Token::Type::COMMA);
		}
		auto column=currentToken.lexeme;
		consume(Token::Type::IDENTIFIER);
		consume(Token::Type::EQUAL);
		auto value=parseExpression();
		stmt->setClauses.emplace_back(column,parseExpression());
	}while(match(Token::Type::COMMA));
	if(match(Token::Type::WHERE)){
		consume(Token::Type::WHERE);
		stmt->where=parseExpression();
	}
	return stmt;
}
//parser method for delete
std::unique_ptr<AST::DeleteStatement> Parse::parseDeleteStatement(){
	auto stmt=std::make_unique<AST::DeleteStatement>();
	//parse the DELETE clause
	consume(Token::Type::DELETE);
	consume(Token::Type::FROM);
	stmt->table=currentToken.lexeme;
	consume(Token::Type::IDENTIFIER);
	//parse the WHERE clause
	if(match(Token::Type::WHERE)){
	        consume(Token::Type::WHERE);
	        stmt->where=parseExpression();
	}
	return stmt;
}
//parse method for drop statement
std::unique_ptr<AST::DropStatement> Parse::parseDropStatement(){
	auto stmt=std::make_unique<AST::DropStatement>();
	//parse the Drop clause
	consume(Token::Type::DROP);
	consume(Token::Type::TABLE);
	stmt->tablename=currentToken.lexeme;
	consume(Token::Type::IDENTIFIER);
	return stmt;
}
//parse the insert statement
std::unique_ptr<AST::InsertStatement> Parse::parseInsertStatement(){
	auto stmt=std::make_unique<AST::InsertStatement>();
	//parse the INSERT statement
	consume(Token::Type::INSERT);
	consume(Token::Type::INTO);
	stmt->table=currentToken.lexeme;
	consume(Token::Type::IDENTIFIER);
	if(match(Token::Type::L_PAREN)){
	        consume(Token::Type::L_PAREN);
	        do{
		        if(match(Token::Type::COMMA)){
			        consume(Token::Type::COMMA);
		        }
	                stmt->columns.push_back(currentToken.lexeme);
		        consume(Token::Type::IDENTIFIER);
	        }while(match(Token::Type::COMMA));
	         consume(Token::Type::R_PAREN);
	}
	//parse the values clause
	consume(Token::Type::VALUES);
	consume(Token::Type::L_PAREN);
	do{
		if(match(Token::Type::COMMA)){
			consume(Token::Type::COMMA);
		}
	        stmt->values.push_back(parseExpression());
	}while(match(Token::Type::COMMA));
	consume(Token::Type::R_PAREN);
	return stmt;
}
//parse the create  statement
std::unique_ptr<AST::CreateTableStatement> Parse::parseCreateTableStatement(){
	auto stmt=std::make_unique<AST::CreateTableStatement>();
	//parse the CREATE clause
	consume(Token::Type::CREATE);
	consume(Token::Type::TABLE);
	if(match(Token::Type::IDENTIFIER)){
		stmt->tablename=currentToken.lexeme;
		consume(Token::Type::IDENTIFIER);
	}
	//parse the table describtions
	consume(Token::Type::L_PAREN);
	do{
		if(match(Token::Type::COMMA)){
			consume(Token::Type::COMMA);
		}
		AST::ColumnDefination col;
		col.name=currentToken.lexeme;
		consume(Token::Type::IDENTIFIER);
		consume(Token::Type::COLON);
		col.type=currentToken.lexeme;
		if(match(Token::Type::INT) || match(Token::Type::TEXT) || match(Token::Type::BOOL) || match(Token::Type::FLOAT)){
			consume(currentToken.type);
		}else{
			throw std::runtime_error("expected column type");
		}
		//parse constraints if any
		while(match(Token::Type::IDENTIFIER)){
			col.constraints.push_back(currentToken.lexeme);
			consume(Token::Type::IDENTIFIER);
		}
		stmt->columns.push_back(std::move(col));
	}while(match(Token::Type::COMMA));
	consume(Token::Type::R_PAREN);
	return stmt;
}
std::unique_ptr<AST::AlterTableStatement> Parse::parseAlterTableStatement(){
	auto stmt=std::make_unique<AST::AlterTableStatement>();
	//parse Alter caluse
	consume(Token::Type::ALTER);
	consume(Token::Type::TABLE);
	stmt->tablename=currentToken.lexeme;
	consume(Token::Type::IDENTIFIER);
	if(match(Token::Type::ADD)){
		stmt->action=AST::AlterTableStatement::ADD;
		stmt->columnName=currentToken.lexeme;
		consume(Token::Type::IDENTIFIER);
		stmt->type=currentToken.lexeme;
		consume(Token::Type::IDENTIFIER);
	}else if(match(Token::Type::DROP)){
		stmt->action=AST::AlterTableStatement::DROP;
		stmt->columnName=currentToken.lexeme;
		consume(Token::Type::IDENTIFIER);
	}else if(match(Token::Type::RENAME)){
		stmt->action=AST::AlterTableStatement::RENAME;
		stmt->columnName=currentToken.lexeme;
		consume(Token::Type::IDENTIFIER);
	}
	return stmt;
}
std::vector<std::unique_ptr<AST::Expression>> Parse::parseColumnList(){
	std::vector<std::unique_ptr<AST::Expression>> columns;
	do{
		if(match(Token::Type::COMMA)){
			consume(Token::Type::COMMA);
		}
		columns.push_back(parseExpression());
	}while(match(Token::Type::COMMA));
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
		if(!isBinaryOperator(op.type)){
			break;
		}
		advance();
		auto right=parseBinaryExpression(precedence+1);
		left=std::make_unique<AST::BinaryOp>(op,std::move(left),std::move(right));
	}
	return left;
}

std::unique_ptr<AST::Expression> Parse::parsePrimaryExpression(){
	if(match(Token::Type::IDENTIFIER)){
		return parseIdentifier();
	}else if(match(Token::Type::NUMBER_LITERAL) || match(Token::Type::STRING_LITERAL)){
		return parseLiteral();
	}else if(match(Token::Type::L_PAREN)){
		consume(Token::Type::L_PAREN);
		auto expr=parseExpression();
		consume(Token::Type::R_PAREN);
		return expr;
	}else{
		throw std::runtime_error("Expected expression at line " + std::to_string(currentToken.line) + ",column, " + std::to_string(currentToken.column));
	}
}

std::unique_ptr<AST::Expression> Parse::parseIdentifier(){
	auto identifier=std::make_unique<AST::Identifier>(currentToken);
	consume(Token::Type::IDENTIFIER);
	return identifier;
}
std::unique_ptr<AST::Expression> Parse::parseLiteral(){
	auto literal=std::make_unique<AST::Literal>(currentToken);
	if(match(Token::Type::NUMBER_LITERAL)){
		consume(Token::Type::NUMBER_LITERAL);
	}else{
		consume(Token::Type::STRING_LITERAL);
	}
	return literal;
}

int Parse::getPrecedence(Token::Type type){
	switch(type){
		case Token::Type::OR: return 1;
		case Token::Type::AND: return 2;
		case Token::Type::EQUAL:
		case Token::Type::NOT_EQUAL:
		case Token::Type::LESS:
		case Token::Type::LESS_EQUAL:
		case Token::Type::GREATER:
		case Token::Type::GREATER_EQUAL: return 3;
		default: return 0;
	}
}
bool Parse::isBinaryOperator(Token::Type type){
	return getPrecedence(type) >0;
}
