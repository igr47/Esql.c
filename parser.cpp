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

Parse::Parse(Lexer& lexer) : lexer(lexer),currentToken(lexer.nextToken()),previousToken_(Token(Token::Type::ERROR,"",0,0)){}
std::unique_ptr<AST::Statement> Parse::parse(){
	return parseStatement();
}
std::unique_ptr<AST::Statement> Parse::parseStatement(){
	if(match(Token::Type::SELECT)){
	        return  parseSelectStatement();
	}else if (match(Token::Type::UPDATE)) {
                return parseUpdateStatement();
        }else if (match(Token::Type::DELETE)) {
                return parseDeleteStatement();
        }else if (match(Token::Type::INSERT)) {
		return parseInsertStatement();
	}else if(match(Token::Type::BULK)){
		advance();
		if(match(Token::Type::INSERT)){
			advance();
			return parseBulkInsertStatement();
		}else if(match(Token::Type::UPDATE)){
			advance();
			return parseBulkUpdateStatement();
		}else if(match(Token::Type::DELETE)){
			advance();
			return parseBulkDeleteStatement();
		}

	}else if(match(Token::Type::CREATE)){
		advance();
		if(match(Token::Type::TABLE)){
			advance();
		        return parseCreateTableStatement();
		}else if(match(Token::Type::DATABASE)){
			advance();
			return parseCreateDatabaseStatement();
		}
	}else if(match(Token::Type::USE)){
		return parseUseStatement();
	}else if(match(Token::Type::SHOW)){
		advance();
		if(match(Token::Type::DATABASES)){
			return parseShowDatabaseStatement();
		}else if(match(Token::Type::TABLES)){
			advance();
			return parseShowTableStatement();
		}
	}else if(match(Token::Type::ALTER)){
		return parseAlterTableStatement();
	}
	throw std::runtime_error("unexpected token at start of statement");
}

std::unique_ptr<AST::Expression> Parse::parseValue() {
    if (match(Token::Type::STRING_LITERAL) ||
        match(Token::Type::DOUBLE_QUOTED_STRING)) {
        auto literal = std::make_unique<AST::Literal>(currentToken);
        advance();
        return literal;
    }
    else if (match(Token::Type::NUMBER_LITERAL)) {
        auto literal = std::make_unique<AST::Literal>(currentToken);
        advance();
        return literal;
    }
    else if (match(Token::Type::TRUE) || match(Token::Type::FALSE)) {
        auto literal = std::make_unique<AST::Literal>(currentToken);
        advance();
        return literal;
    }
    else if (inValueContext && match(Token::Type::IDENTIFIER)) {
        throw ParseError(
            currentToken.line,
            currentToken.column,
            "String value must be quoted. Did you mean '" + 
            currentToken.lexeme + "'?"
        );
    }
    throw ParseError(
        currentToken.line,
        currentToken.column,
        "Unexpected token in value position. Expected quoted string or number"
    );
}
void Parse::consume(Token::Type expected){
	if(currentToken.type==expected){
		advance();
	}else{
		throw std::runtime_error("Unexpected token at line "+ std::to_string(currentToken.line) +",column ," +std::to_string(currentToken.column));
	}
}
const Token& Parse::previousToken() const{
	return previousToken_;
}
void Parse::advance() {
    previousToken_=currentToken;
    if(currentToken.type!=Token::Type::END_OF_INPUT){
	    currentToken = lexer.nextToken();
    }
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
std::unique_ptr<AST::CreateDatabaseStatement> Parse::parseCreateDatabaseStatement(){
	auto stmt=std::make_unique<AST::CreateDatabaseStatement>();
	//consume(Token::Type::CREATE);
	//consume(Token::Type::DATABASE);
	stmt->dbName=currentToken.lexeme;
	consume(Token::Type::IDENTIFIER);
	return stmt;
}
std::unique_ptr<AST::UseDatabaseStatement> Parse::parseUseStatement(){
	auto stmt=std::make_unique<AST::UseDatabaseStatement>();
	consume(Token::Type::USE);
	consume(Token::Type::DATABASE);
	stmt->dbName=currentToken.lexeme;
	consume(Token::Type::IDENTIFIER);
	return stmt;
}
std::unique_ptr<AST::ShowDatabaseStatement> Parse::parseShowDatabaseStatement(){
	auto stmt=std::make_unique<AST::ShowDatabaseStatement>();
	//consume(Token::Type::SHOW);
	//consume(Token::Type::DATABASES);
	return stmt;
}
std::unique_ptr<AST::ShowTableStatement> Parse::parseShowTableStatement(){
	auto stmt=std::make_unique<AST::ShowTableStatement>();
	consume(Token::Type::SHOW);
	consume(Token::Type::TABLES);
	return stmt;
}
std::unique_ptr<AST::SelectStatement> Parse::parseSelectStatement(){
	auto stmt=std::make_unique<AST::SelectStatement>();
	//parse select clause
	consume(Token::Type::SELECT);
	if(match(Token::Type::ASTERIST)){
		consume(Token::Type::ASTERIST);
	}else{
	        stmt->columns=parseColumnList();
	}
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
std::unique_ptr<AST::UpdateStatement> Parse::parseUpdateStatement() {
    auto stmt = std::make_unique<AST::UpdateStatement>();

    // Parse UPDATE clause
    consume(Token::Type::UPDATE);
    stmt->table = currentToken.lexeme;
    consume(Token::Type::IDENTIFIER);

    // Parse SET clause
    consume(Token::Type::SET);

    //bool wasInValueContext = inValueContext;
    //inValueContext = true;  // We're now parsing values

    do {
        if (match(Token::Type::COMMA)) {
            consume(Token::Type::COMMA);
        }

        auto column = currentToken.lexeme;
        consume(Token::Type::IDENTIFIER);
        consume(Token::Type::EQUAL);

        // Use parseExpression() for strict value parsing
        stmt->setClauses.emplace_back(column, parseExpression());
    } while (match(Token::Type::COMMA));

    //inValueContext = wasInValueContext;  // Restore context

    // Parse WHERE clause
    if (match(Token::Type::WHERE)) {
        consume(Token::Type::WHERE);
        stmt->where = parseExpression();
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
	//set value context flag
	bool wasInValueContext=inValueContext;
	inValueContext=true;
	do{
		if(match(Token::Type::COMMA)){
			consume(Token::Type::COMMA);
		}
	        stmt->values.push_back(parseValue());
	}while(match(Token::Type::COMMA));
	inValueContext=wasInValueContext;
	consume(Token::Type::R_PAREN);
	return stmt;
}
//parse the create  statement

std::unique_ptr<AST::CreateTableStatement> Parse::parseCreateTableStatement() {
    auto stmt = std::make_unique<AST::CreateTableStatement>();

    // Get table name
    if (!match(Token::Type::IDENTIFIER)) {
        throw std::runtime_error("Expected table name after CREATE TABLE");
    }
    stmt->tablename = currentToken.lexeme;
    consume(Token::Type::IDENTIFIER);

    // Parse column definitions
    consume(Token::Type::L_PAREN);

    // Parse first column
    parseColumnDefinition(*stmt);

    // Parse additional columns
    do{
        consume(Token::Type::COMMA);
        parseColumnDefinition(*stmt);
    }while (match(Token::Type::COMMA));

    consume(Token::Type::R_PAREN);

    // Optional semicolon
    if (match(Token::Type::SEMICOLON)) {
        consume(Token::Type::SEMICOLON);
    }

    return stmt;
}

void Parse::parseColumnDefinition(AST::CreateTableStatement& stmt) {
    AST::ColumnDefination col;

    // Column name
    if (!match(Token::Type::IDENTIFIER)) {
        throw std::runtime_error("Expected column name");
    }
    col.name = currentToken.lexeme;
    consume(Token::Type::IDENTIFIER);

    // Colon separator
    consume(Token::Type::COLON);

    // Column type
    if (!matchAny({Token::Type::INT, Token::Type::TEXT, Token::Type::BOOL, Token::Type::FLOAT})) {
        throw std::runtime_error("Expected column type (INT, TEXT, BOOL, or FLOAT)");
    }
    col.type = currentToken.lexeme;
    consume(currentToken.type);

    // Column constraints
    while (match(Token::Type::IDENTIFIER)) {
        col.constraints.push_back(currentToken.lexeme);
        consume(Token::Type::IDENTIFIER);
    }

    stmt.columns.push_back(std::move(col));
}
std::unique_ptr<AST::AlterTableStatement> Parse::parseAlterTableStatement() {
    auto stmt = std::make_unique<AST::AlterTableStatement>();
    
    consume(Token::Type::ALTER);
    consume(Token::Type::TABLE);
    
    // Get table name
    stmt->tablename = currentToken.lexeme;
    consume(Token::Type::IDENTIFIER);
    
    if (match(Token::Type::ADD)) {
        consume(Token::Type::ADD);
        stmt->action = AST::AlterTableStatement::ADD;
        
        // Get column name
        stmt->columnName = currentToken.lexeme;
        consume(Token::Type::IDENTIFIER);
        
        // Get column type
        if (matchAny({Token::Type::INT, Token::Type::TEXT, Token::Type::BOOL, Token::Type::FLOAT})) {
            stmt->type = currentToken.lexeme;
            consume(currentToken.type);
        } else {
            throw std::runtime_error("Expected column type after ADD COLUMN");
        }
    } 
    else if (match(Token::Type::DROP)) {
        consume(Token::Type::DROP);
        stmt->action = AST::AlterTableStatement::DROP;
        
        // Get column name
        stmt->columnName = currentToken.lexeme;
        consume(Token::Type::IDENTIFIER);
    } 
    else if (match(Token::Type::RENAME)) {
        consume(Token::Type::RENAME);
        stmt->action = AST::AlterTableStatement::RENAME;
        
        // Get old column name
        stmt->columnName = currentToken.lexeme;
        consume(Token::Type::IDENTIFIER);
        
        // Consume TO keyword
        consume(Token::Type::TO);
        
        // Get new column name
        stmt->newColumnName = currentToken.lexeme;
        consume(Token::Type::IDENTIFIER);
    }
    else {
        throw std::runtime_error("Expected ADD, DROP, or RENAME after ALTER TABLE");
    }
    
    return stmt;
}


/*std::unique_ptr<AST::BulkInsertStatement> Parse::parseBulkInsertStatement() {
    auto stmt = std::make_unique<AST::BulkInsertStatement>();

    //consume(Token::Type::BULK);
    //consume(Token::Type::INSERT);
    consume(Token::Type::INTO);

    stmt->table = currentToken.lexeme;
    consume(Token::Type::IDENTIFIER);

    if (match(Token::Type::L_PAREN)) {
        consume(Token::Type::L_PAREN);
        do {
            if (match(Token::Type::COMMA)) {
                consume(Token::Type::COMMA);
            }
            stmt->columns.push_back(currentToken.lexeme);
            consume(Token::Type::IDENTIFIER);
        } while (match(Token::Type::COMMA));
        consume(Token::Type::R_PAREN);
    }

    consume(Token::Type::VALUES);

    bool wasInValueContext = inValueContext;
    inValueContext = true;

    do {
        consume(Token::Type::L_PAREN);
        std::vector<std::unique_ptr<AST::Expression>> rowValues;

        do {
            if (match(Token::Type::COMMA)) {
                consume(Token::Type::COMMA);
            }
            rowValues.push_back(parseValue());
        } while (match(Token::Type::COMMA));

        stmt->rows.push_back(std::move(rowValues));
        consume(Token::Type::R_PAREN);

    } while (match(Token::Type::COMMA));

    inValueContext = wasInValueContext;
    return stmt;
}*/

std::unique_ptr<AST::BulkInsertStatement> Parse::parseBulkInsertStatement() {
    auto stmt = std::make_unique<AST::BulkInsertStatement>();

    consume(Token::Type::INTO);

    stmt->table = currentToken.lexeme;
    consume(Token::Type::IDENTIFIER);

    // Parse optional column list
    if (match(Token::Type::L_PAREN)) {
        consume(Token::Type::L_PAREN);
        do {
            if (match(Token::Type::COMMA)) {
                consume(Token::Type::COMMA);
            }
            stmt->columns.push_back(currentToken.lexeme);
            consume(Token::Type::IDENTIFIER);
        } while (match(Token::Type::COMMA));
        consume(Token::Type::R_PAREN);
    }

    consume(Token::Type::VALUES);

    bool wasInValueContext = inValueContext;
    inValueContext = true;

    // Parse first row
    consume(Token::Type::L_PAREN);
    std::vector<std::unique_ptr<AST::Expression>> firstRowValues;

    do {
        if (match(Token::Type::COMMA)) {
            consume(Token::Type::COMMA);
        }
        firstRowValues.push_back(parseValue());
    } while (match(Token::Type::COMMA));

    stmt->rows.push_back(std::move(firstRowValues));
    consume(Token::Type::R_PAREN);

    // Parse additional rows
    while (match(Token::Type::COMMA)) {
        consume(Token::Type::COMMA);
        consume(Token::Type::L_PAREN);

        std::vector<std::unique_ptr<AST::Expression>> rowValues;

        do {
            if (match(Token::Type::COMMA)) {
                consume(Token::Type::COMMA);
            }
            rowValues.push_back(parseValue());
        } while (match(Token::Type::COMMA));

        stmt->rows.push_back(std::move(rowValues));
        consume(Token::Type::R_PAREN);
    }

    inValueContext = wasInValueContext;
    return stmt;
}

std::unique_ptr<AST::BulkUpdateStatement> Parse::parseBulkUpdateStatement() {
    auto stmt = std::make_unique<AST::BulkUpdateStatement>();

    //consume(Token::Type::BULK);
    //consume(Token::Type::UPDATE);

    stmt->table = currentToken.lexeme;
    consume(Token::Type::IDENTIFIER);

    consume(Token::Type::SET);

    bool wasInValueContext = inValueContext;
    inValueContext = true;

    do {
        AST::BulkUpdateStatement::UpdateSpec updateSpec;

        consume(Token::Type::ROW);
        consume(Token::Type::NUMBER_LITERAL);
        updateSpec.row_id = std::stoul(previousToken().lexeme);

        consume(Token::Type::SET);

        do {
            if (match(Token::Type::COMMA)) {
                consume(Token::Type::COMMA);
            }

            std::string column = currentToken.lexeme;
            consume(Token::Type::IDENTIFIER);
            consume(Token::Type::EQUAL);

            updateSpec.setClauses.emplace_back(column, parseExpression());

        } while (match(Token::Type::COMMA));

        stmt->updates.push_back(std::move(updateSpec));

    } while (match(Token::Type::COMMA));

    inValueContext = wasInValueContext;
    return stmt;
}

std::unique_ptr<AST::BulkDeleteStatement> Parse::parseBulkDeleteStatement() {
    auto stmt = std::make_unique<AST::BulkDeleteStatement>();

    //consume(Token::Type::BULK);
    //consume(Token::Type::DELETE);
    consume(Token::Type::FROM);

    stmt->table = currentToken.lexeme;
    consume(Token::Type::IDENTIFIER);

    consume(Token::Type::WHERE);
    consume(Token::Type::ROW);
    consume(Token::Type::IN);
    consume(Token::Type::L_PAREN);

    do {
        if (match(Token::Type::COMMA)) {
            consume(Token::Type::COMMA);
        }

        if (match(Token::Type::NUMBER_LITERAL)) {
            stmt->row_ids.push_back(std::stoul(currentToken.lexeme));
            consume(Token::Type::NUMBER_LITERAL);
        }

    } while (match(Token::Type::COMMA));

    consume(Token::Type::R_PAREN);
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
		if(op.type == Token::Type::BETWEEN){
			advance(); //consume between
		        
			auto lower = parseExpression();
			consume(Token::Type::AND);
			auto upper = parseExpression();

			//Create a compound expression: value >= lower AND value <=upper
			auto ge = std::make_unique<AST::BinaryOp>(Token(Token::Type::GREATER_EQUAL, ">=" , op.line, op.column),left->clone(),std::move(lower));
			auto le = std::make_unique<AST::BinaryOp>(Token(Token::Type::LESS_EQUAL, "<=" ,op.line,op.column),left->clone(),std::move(upper));
			left = std::make_unique<AST::BinaryOp>(Token(Token::Type::AND, "AND" , op.line,op.column),std::move(ge),std::move(le));
			continue;
		}
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
	}else if(match(Token::Type::NUMBER_LITERAL) || match(Token::Type::STRING_LITERAL) || match(Token::Type::DOUBLE_QUOTED_STRING) || match(Token::Type::FALSE) || match(Token::Type::TRUE)){
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
std::unique_ptr<AST::Expression> Parse::parseLiteral() {
    auto literal = std::make_unique<AST::Literal>(currentToken);
    if (match(Token::Type::NUMBER_LITERAL)) {
        advance();
    }
    else if (match(Token::Type::STRING_LITERAL) || 
             match(Token::Type::DOUBLE_QUOTED_STRING)) {
        advance();
    }
    else if (match(Token::Type::TRUE) || match(Token::Type::FALSE)) {
        advance();
    }
    else {
        throw ParseError(
            currentToken.line,
            currentToken.column,
            "Expected literal value"
        );
    }
    return literal;
}

int Parse::getPrecedence(Token::Type type){
	switch(type){
		case Token::Type::OR: return 1;
		case Token::Type::AND: return 2;
		case Token::Type::BETWEEN: return 3;
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
	return getPrecedence(type) >0 || type == Token::Type::BETWEEN;
}
