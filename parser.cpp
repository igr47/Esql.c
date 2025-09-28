#include "parser.h"
#include "scanner.h"
#include <string>
#include <stdexcept>
#include <vector>
#include <iostream>

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
	}else if (match(Token::Type::DROP)) {
		return parseDropStatement();
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
	if(currentToken.lexeme == "true" || currentToken.lexeme == "false"){
		auto literal = std::make_unique<AST::Literal>(currentToken);
		advance();
		return literal;
	}
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

Token Parse::peekToken() {
    // Save current state
    //size_t savedPosition = lexer.position;
    //size_t savedLine = lexer.line;
    //size_t savedColumn = lexer.column;
    //Token savedCurrent = currentToken;
    size_t savedPos, savedLine , savedCol;
    lexer.saveState(savedPos,savedLine,savedCol);

    // Get next token
    Token nextToken = lexer.nextToken();

    // Restore state
    //lexer.position = savedPosition;
    //lexer.line = savedLine;
    //lexer.column = savedColumn;
    //currentToken = savedCurrent;
    lexer.restoreState(savedPos, savedLine , savedCol);

    return nextToken;
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
	if (match(Token::Type::DATABASE)){
	         consume(Token::Type::DATABASE);
        }
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

std::unique_ptr<AST::GroupByClause> Parse::parseGroupByClause() {
	consume(Token::Type::GROUP);
	consume(Token::Type::BY);
	std::vector<std::unique_ptr<AST::Expression>> columns;

	columns.push_back(parseExpression());

	std::cout <<"DEBUG: Parsing GROUP B clause" <<std::endl;

	while(match(Token::Type::COMMA)){
		std::cout<<"DEBUG : Found comma, parsing next expression" <<std::endl;
		consume(Token::Type::COMMA);
		columns.push_back(parseExpression());
		std::cout<<"DEBUG: Added expression to GROUP BY " <<std::endl;
	}

	return std::make_unique<AST::GroupByClause>(std::move(columns));
}

std::unique_ptr<AST::HavingClause> Parse::parseHavingClause(){
	consume(Token::Type::HAVING);
	return std::make_unique<AST::HavingClause>(parseExpression());
}

std::unique_ptr<AST::OrderByClause> Parse::parseOrderByClause() {
	consume(Token::Type::ORDER);
	consume(Token::Type::BY);

	std::vector<std::pair<std::unique_ptr<AST::Expression>, bool>> columns;
	do{
		if(match(Token::Type::COMMA)){
			consume(Token::Type::COMMA);
		}

		auto expr=parseExpression();
		bool ascending = true;

		if(match(Token::Type::ASC)){
			consume(Token::Type::ASC);
			ascending = true;
		}else if(match(Token::Type::DESC)) {
			consume(Token::Type::DESC);
			ascending = false;
		}

		columns.emplace_back(std::move(expr),ascending);
	}while(match(Token::Type::COMMA));
	return std::make_unique<AST::OrderByClause>(std::move(columns));
}

std::unique_ptr<AST::SelectStatement> Parse::parseSelectStatement(){
	auto stmt=std::make_unique<AST::SelectStatement>();
	//parse select clause
	consume(Token::Type::SELECT);
	if(match(Token::Type::DISTINCT)){
		consume(Token::Type::DISTINCT);
		stmt->distinct = true;
	}

	//bool hasAggregates = false;
	//size_t savePos,saveLine,saveCol;
	
	//Save state and peek ahead
	//lexer.saveState(savePos,saveLine,saveCol);
	//Token peeked = peekToken();

	//if(peeked.type == Token::Type::COUNT || peeked.type == Token::Type::SUM || peeked.type == Token::Type::AVG || peeked.type == Token::Type::MIN || peeked.type == Token::Type::MAX){
		//hasAggregates = true;
	//}
	if(match(Token::Type::ASTERIST)){
		consume(Token::Type::ASTERIST);
	//} else if(hasAggregates) {
		//stmt->newCols = parseColumnListAs();
	}else{
		//lexer.restoreState(savePos, saveLine, saveCol);
		//parse the first expression to see if it followed by AS
		//auto firstExpr = parseExpression();
		Token nextToken = peekToken();
		//lexer.restoreState(savePos,saveLine,saveCol);

		if(nextToken.type == Token::Type::AS){
			//lexer.restoreState(savePos, saveLine, saveCol);
			stmt->newCols = parseColumnListAs();

		//if(peeked.type == Token::Type::AS){
			//stmt->newCols = parseColumnListAs();
		//}
		}
		else{
			//lexer.restoreState(savePos, saveLine, saveCol);
	                stmt->columns=parseColumnList();
		}

	}

	//parse From clause
	consume(Token::Type::FROM);
	stmt->from=parseFromClause();
	//parse optional WHERE clause
	if(match(Token::Type::WHERE)){
		consume(Token::Type::WHERE);
		stmt->where=parseExpression();
	}
	//parse Group By clause
	if(match(Token::Type::GROUP)){
		stmt->groupBy=parseGroupByClause();
	}
	//parse Having clause
	if(match(Token::Type::HAVING)){
			stmt->having = parseHavingClause();
	}
	//parse orderBy clause
	if(match(Token::Type::ORDER)){
		stmt->orderBy = parseOrderByClause();
	}
	if(match(Token::Type::LIMIT)){
		consume(Token::Type::LIMIT);
		stmt->limit = parseExpression();
		if(match(Token::Type::OFFSET)){
			consume(Token::Type::OFFSET);
			stmt->offset = parseExpression();
		}
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

    bool wasInValueContext = inValueContext;
    inValueContext = true;  // We're now parsing values

    do {
        if (match(Token::Type::COMMA)) {
            consume(Token::Type::COMMA);
        }

        auto column = currentToken.lexeme;
        consume(Token::Type::IDENTIFIER);
        consume(Token::Type::EQUAL);

	//auto expr = parseExpression();

	//std::cout<<"DEBUG: Parsed expression:" << expr->toString()<<std::endl;

        // Use parseExpression() for strict value parsing
        stmt->setClauses.emplace_back(column, /*std::move(expr)*/parseExpression());
    } while (match(Token::Type::COMMA));

    inValueContext = wasInValueContext;  // Restore context

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
std::unique_ptr<AST::InsertStatement> Parse::parseInsertStatement() {
	auto stmt = std::make_unique<AST::InsertStatement>();

	//parse the INSERT statement
	consume(Token::Type::INSERT);
	consume(Token::Type::INTO);
	
	stmt->table = currentToken.lexeme;
	consume(Token::Type::IDENTIFIER);

	//Parse optional column list
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
	//Parse VALUES clause
	consume(Token::Type::VALUES);
	bool wasInValueContext = inValueContext;
	inValueContext = true;

	//Parse first row
	consume(Token::Type::L_PAREN);
	std::vector<std::unique_ptr<AST::Expression>> firstRowValues;

	do{
		if(match(Token::Token::Type::COMMA)){
			consume(Token::Type::COMMA);
		}
		firstRowValues.push_back(parseValue());
	}while(match(Token::Type::COMMA));

	stmt->values.push_back(std::move(firstRowValues));
	consume(Token::Type::R_PAREN);

	//Parse additional rows(For multi_row INSERT)
	while(match(Token::Type::COMMA)){
		consume(Token::Type::COMMA);
		consume(Token::Type::L_PAREN);
		std::vector<std::unique_ptr<AST::Expression>> rowValues;

		do{
			if(match(Token::Type::COMMA)){
				consume(Token::Type::COMMA);
			}
			rowValues.push_back(parseValue());
		}while(match(Token::Type::COMMA));

		stmt->values.push_back(std::move(rowValues));
		consume(Token::Type::R_PAREN);
	}
	inValueContext = wasInValueContext;
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
    //consume(Token::Type::COLON);

    // Column type
    if (!matchAny({Token::Type::INT, Token::Type::TEXT, Token::Type::BOOL, Token::Type::FLOAT})) {
        throw std::runtime_error("Expected column type (INT, TEXT, BOOL, or FLOAT)");
    }
    col.type = currentToken.lexeme;
    consume(currentToken.type);

    // Column constraints
    while(matchAny({Token::Type::PRIMARY_KEY , Token::Type::NOT_NULL,Token::Type::UNIQUE,Token::Type::CHECK,Token::Type::DEFAULT,Token::Type::AUTO_INCREAMENT})){
	     if(match(Token::Type::PRIMARY_KEY)){
                   col.constraints.push_back("PRIMARY_KEY");
		   consume(Token::Type::PRIMARY_KEY);

		   //Handle PRIMARY KEY AUTOINCREAMENT combination
		   if(match(Token::Type::AUTO_INCREAMENT)){
			   col.autoIncreament = true;
			   col.constraints.push_back("AUTO_INCREAMENT");
			   consume(Token::Type::AUTO_INCREAMENT);
		   }
	     }else if(match(Token::Type::NOT_NULL)) {
		    col.constraints.push_back("NOT_NULL");
		    consume(Token::Type::NOT_NULL);
	     } else if (match(Token::Type::UNIQUE)){
		     col.constraints.push_back("UNIQUE");
		     consume(Token::Type::UNIQUE);
	     } else if(match(Token::Type::AUTO_INCREAMENT)){
		     col.autoIncreament = true;
		     col.constraints.push_back("AUTO_INCREAMENT");
		     consume(Token::Type::AUTO_INCREAMENT);
	     }else if (match(Token::Type::DEFAULT)){
		     consume(Token::Type::DEFAULT);

		     //Save the current state for proper default value parsing
		     bool wasInValueContext = inValueContext;
		     inValueContext = true;

		     try {
			     if(matchAny({Token::Type::STRING_LITERAL,Token::Type::NUMBER_LITERAL,Token::Type::TRUE,Token::Type::FALSE})){
				     col.defaultValue = currentToken.lexeme;
				     col.constraints.push_back("DEFAULT");
				     consume(currentToken.type);
			     }else if(match(Token::Type::NULL_TOKEN)) {
				     col.defaultValue = "NULL";
				     col.constraints.push_back("DEFAULT");
				     consume(Token::Type::NULL_TOKEN);
			     } else {
				     throw std::runtime_error("Expected default value after DEFULT keword");
			     }
		     } catch (...){
			     inValueContext = wasInValueContext;
			     throw;
		     }
		     inValueContext = wasInValueContext;
	     } else if (match(Token::Type::CHECK)){
		     consume(Token::Type::CHECK);
		     consume(Token::Type::L_PAREN);

		     //Parse the expression using the existing expression Parser
		     auto checkExpr = parseExpression();

		     //Convert the expression into a string for storage in constrints
		     std::string checkConstraint = "CHECK" + checkExpr->toString() + ")";
		     col.constraints.push_back(checkConstraint);

		     consume (Token::Type::R_PAREN);
	     }
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



void Parse::parseSingleRowUpdate(AST::BulkUpdateStatement& stmt) {
    AST::BulkUpdateStatement::UpdateSpec updateSpec;

    // Parse ROW <number>
    consume(Token::Type::ROW);

    if (!match(Token::Type::NUMBER_LITERAL)) {
        throw ParseError(currentToken.line, currentToken.column, "Expected row number after ROW keyword");
    }

    try {
        updateSpec.row_id = std::stoul(currentToken.lexeme);
        consume(Token::Type::NUMBER_LITERAL);
    } catch (const std::exception& e) {
        throw ParseError(currentToken.line, currentToken.column, "Invalid row number: " + currentToken.lexeme);
    }

    // Parse SET clause
    consume(Token::Type::SET);

    // Parse first SET clause
    parseSingleSetClause(updateSpec);

    // Parse additional SET clauses for the same row
    while (match(Token::Type::COMMA)) {
        // Use peekToken to look ahead and check if the next token is ROW
        Token nextToken = peekToken();
        
        // If the next token after comma is ROW, this comma separates row updates, not set clauses
        if (nextToken.type == Token::Type::ROW) {
            break; // Exit the loop to let the outer loop handle the next ROW
        }
        
        // Otherwise, this comma separates set clauses within the same row
        consume(Token::Type::COMMA);
        parseSingleSetClause(updateSpec);
    }

    stmt.updates.push_back(std::move(updateSpec));
}

std::unique_ptr<AST::BulkUpdateStatement> Parse::parseBulkUpdateStatement() {
    auto stmt = std::make_unique<AST::BulkUpdateStatement>();
    
    stmt->table = currentToken.lexeme;
    consume(Token::Type::IDENTIFIER);
    consume(Token::Type::SET);

    bool wasInValueContext = inValueContext;
    inValueContext = true;

    // Parse first row update
    parseSingleRowUpdate(*stmt);

    // Parse additional row updates
    while (match(Token::Type::COMMA)) {
        // Use peekToken to confirm this comma is followed by ROW
        Token nextToken = peekToken();
        if (nextToken.type != Token::Type::ROW) {
            throw ParseError(currentToken.line, currentToken.column, 
                           "Expected ROW after comma in BULK UPDATE");
        }
        
        consume(Token::Type::COMMA);
        parseSingleRowUpdate(*stmt);
    }

    inValueContext = wasInValueContext;
    return stmt;
}

void Parse::parseSingleSetClause(AST::BulkUpdateStatement::UpdateSpec& updateSpec) {
    std::string column = currentToken.lexeme;
    consume(Token::Type::IDENTIFIER);
    consume(Token::Type::EQUAL);
    
    bool wasInValueContext = inValueContext;
    inValueContext = true;
    
    updateSpec.setClauses.emplace_back(column, parseExpression());
    
    inValueContext = wasInValueContext;
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
/*std::vector<std::pair<std::unique_ptr<AST::Expression>,std::string>> Parse::parseColumnListAs(){
	std::vector<std::pair<std::unique_ptr<AST::Expression>,std::string>> newColumns;
	
	do{
		if(match(Token::Type::COMMA)){
			consume(Token::Type::COMMA);
		}
		std::unique_ptr<AST::Expression> expr;
		if(matchAny({Token::Type::COUNT,Token::Type::SUM,Token::Type::AVG,Token::Type::MIN,Token::Type::MAX})){
			Token funcToken = currentToken;
			consume(currentToken.type);

			consume(Token::Type::L_PAREN);

			std::unique_ptr<AST::Expression> arg = nullptr;
			bool isCountAll = false;
			if(funcToken.type == Token::Type::COUNT && match(Token::Type::ASTERIST)){
				consume(Token::Type::ASTERIST);
				isCountAll = true;
			}else if(match(Token::Type::IDENTIFIER)){
				arg = parseIdentifier();
			}else if(match(Token::Type::ASTERIST)){
				if(funcToken.type == Token::Type::COUNT){
					consume(Token::Type::ASTERIST);
					isCountAll = true;
				} else {
					throw ParseError(currentToken.line,currentToken.column,"Onl COUNT suports * argument");
				}
			}else {
                                arg = parseExpression();
                        }
                        consume(Token::Type::R_PAREN);
                        expr = std::make_unique<AST::AggregateExpression>(funcToken,std::move(arg),isCountAll);
                }else{			
			expr = parseExpression();
		}
		std::string alias;
		if(match(Token::Type::AS)){
			consume(Token::Type::AS);
			if(match(Token::Type::STRING_LITERAL) || match(Token::Type::DOUBLE_QUOTED_STRING)){
				//remove quotes from alises
				alias = currentToken.lexeme;
				if(alias.size()>2 && ((alias[0] == '\'' && alias[alias.size()-1] == '\'' || (alias[0] == '"' && alias[alias.size() -1] == '"')))){
					alias = alias.substr(1,alias.size() - 2);
				}
				consume(currentToken.type);
			}else if(match(Token::Type::IDENTIFIER)){
				alias= currentToken.lexeme;
				consume(Token::Type::IDENTIFIER);
			}
		}
		newColumns.emplace_back(std::move(expr),alias);
	}while(match(Token::Type::COMMA));
	return newColumns;
}*/

std::vector<std::pair<std::unique_ptr<AST::Expression>, std::string>> Parse::parseColumnListAs() {
    std::vector<std::pair<std::unique_ptr<AST::Expression>, std::string>> newColumns;
    
    do {
        if (match(Token::Type::COMMA)) {
            consume(Token::Type::COMMA);
        }

        std::unique_ptr<AST::Expression> expr;
        
        // Check for aggregate functions first
        if (matchAny({Token::Type::COUNT, Token::Type::SUM, Token::Type::AVG, 
                     Token::Type::MIN, Token::Type::MAX})) {
            Token funcToken = currentToken;
            advance();  // Consume the function name

            consume(Token::Type::L_PAREN);

            std::unique_ptr<AST::Expression> arg = nullptr;
            bool isCountAll = false;

            if (funcToken.type == Token::Type::COUNT && match(Token::Type::ASTERIST)) {
                consume(Token::Type::ASTERIST);
                isCountAll = true;
            } else if (match(Token::Type::IDENTIFIER)) {
                arg = parseIdentifier();
            } else {
                arg = parseExpression();  // Handle complex expressions
            }

            consume(Token::Type::R_PAREN);
            expr = std::make_unique<AST::AggregateExpression>(funcToken, std::move(arg),nullptr, isCountAll);
        } else {
            // Regular expression
            expr = parseExpression();
        }

        std::string alias;
        if (match(Token::Type::AS)) {
            consume(Token::Type::AS);
            if (match(Token::Type::STRING_LITERAL) || match(Token::Type::DOUBLE_QUOTED_STRING)) {
                alias = currentToken.lexeme;
                if (alias.size() > 2 && 
                    ((alias[0] == '\'' && alias.back() == '\'') || 
                     (alias[0] == '"' && alias.back() == '"'))) {
                    alias = alias.substr(1, alias.size() - 2);
                }
                consume(currentToken.type);
            } else if (match(Token::Type::IDENTIFIER)) {
                alias = currentToken.lexeme;
                consume(Token::Type::IDENTIFIER);
            }
        }

        newColumns.emplace_back(std::move(expr), alias);
    } while (match(Token::Type::COMMA));
    
    return newColumns;
}

std::vector<std::unique_ptr<AST::Expression>> Parse::parseColumnList() {
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
	return parseBinaryExpression(1);
}


std::unique_ptr<AST::Expression> Parse::parseBinaryExpression(int minPrecedence) {
    // Parse left side 
    std::unique_ptr<AST::Expression> left;
    
    if (match(Token::Type::NOT)) {
        consume(Token::Type::NOT);
        auto expr = parseBinaryExpression(getPrecedence(Token::Type::NOT) + 1);
        left = std::make_unique<AST::NotOp>(std::move(expr));
    } else {
        left = parsePrimaryExpression();
    }

    // Now process binary operators
    while (true) {
        Token op = currentToken;
        int precedence = getPrecedence(op.type);
        
        // Stop if precedence is too low or not a binary operator
        if (precedence < minPrecedence || !isBinaryOperator(op.type)) {
            break;
        }

        // Handle special operators first
        if (op.type == Token::Type::BETWEEN) {
            consume(Token::Type::BETWEEN);
            auto lower = parseBinaryExpression(precedence+1);
            if (!match(Token::Type::AND)) {
                throw ParseError(currentToken.line, currentToken.column, 
                               "Expected AND after BETWEEN lower bound");
            }
            consume(Token::Type::AND);
            auto upper = parseBinaryExpression(precedence+1);
            left = std::make_unique<AST::BetweenOp>(std::move(left), std::move(lower), std::move(upper));
            continue;
        } 
        else if (op.type == Token::Type::IN) {
            consume(Token::Type::IN);
            consume(Token::Type::L_PAREN);
            std::vector<std::unique_ptr<AST::Expression>> values;
            
            // Parse values inside IN clause
            do {
                values.push_back(parseExpression());
            } while (match(Token::Type::COMMA) && (consume(Token::Type::COMMA), true));
            
            consume(Token::Type::R_PAREN);
            left = std::make_unique<AST::InOp>(std::move(left), std::move(values));
            continue;
        }
	else if (op.type == Token::Type::IS){
		consume(Token::Type::IS);

		if(match(Token::Type::NOT)){
			consume(Token::Type::NOT);
			auto right = parseBinaryExpression(precedence+1);
			Token isNotToken = Token(Token::Type::IS_NOT, "IS NOT", op.line, op.column);
			left = std::make_unique<AST::BinaryOp>(isNotToken, std::move(left), std::move(right));
		}else {
			auto right = parseBinaryExpression(precedence +1 );
			left = std::make_unique<AST::BinaryOp>(op, std::move(left),std::move(right));
		}
		continue;
	}

        // Handle regular binary operators
        if (isBinaryOperator(op.type)) {
            advance(); // Consume the operator
            auto right = parseBinaryExpression(precedence + 1);
            left = std::make_unique<AST::BinaryOp>(op, std::move(left), std::move(right));
        } else {
            break;
        }
    }
    
    return left;
}

std::unique_ptr<AST::Expression> Parse::parsePrimaryExpression(){
	std::cout<< "DEBUG: parsePrimarExpression() - current Token:" << static_cast<int>(currentToken.type) <<"'" <<currentToken.lexeme <<"'"<<std::endl;

	std::unique_ptr<AST::Expression> left;
	if(matchAny({Token::Type::COUNT,Token::Type::SUM,Token::Type::AVG,Token::Type::MIN,Token::Type::MAX})){
		std::cout<< "DEBUG: Foung aggregate function: " <<currentToken.lexeme<<std::endl;
		Token funcToken = currentToken;
		consume(currentToken.type);

		consume(Token::Type::L_PAREN);

		std::unique_ptr<AST::Expression> expre;
		std::unique_ptr<AST::Expression> arg = nullptr;
		bool isCountAll = false;
		//Handle COUNT(*) specifically
		if(funcToken.type == Token::Type::COUNT && match(Token::Type::ASTERIST)){
			consume(Token::Type::ASTERIST);
			isCountAll = true;
		}else if(match(Token::Type::IDENTIFIER)){
			std::cout<<"DEBUG: Handling identifier argument"<<std::endl;
			arg = parseIdentifier();
		}else if(match(Token::Type::ASTERIST)){
			if(funcToken.type == Token::Type::COUNT){
				consume(Token::Type::ASTERIST);
				isCountAll = true;
			} else {
				throw ParseError(currentToken.line,currentToken.column,"Only COUNT function supports * argument");
			}
			//continue;
		}else{
			std::cout<< "DEBUG: Handling expression argument"<<std::endl;
			arg=parseExpression();
		}

		consume(Token::Type::R_PAREN);

		if(match(Token::Type::AS)){
			consume(Token::Type::AS);
			expre = parseIdentifier();
		}


		return std::make_unique<AST::AggregateExpression>(funcToken, std::move(arg),std::move(expre), isCountAll);
	}else if(match(Token::Type::IDENTIFIER)){
		//return parseIdentifier();
		left = parseIdentifier();
	}else if(match(Token::Type::NUMBER_LITERAL) || match(Token::Type::STRING_LITERAL) || match(Token::Type::DOUBLE_QUOTED_STRING) || match(Token::Type::FALSE) || match(Token::Type::TRUE)){
		//return parseLiteral();
		if(!inValueContext) {
			left = std::make_unique<AST::ColumnReference>(currentToken.lexeme);
			consume(Token::Type::IDENTIFIER);
		} else {
		        left = parseLiteral();
		}
	}else if(match(Token::Type::L_PAREN)){
		consume(Token::Type::L_PAREN);
		//auto expr=parseExpression();
		left = parseExpression();
		consume(Token::Type::R_PAREN);
		//return expr;
	}else if (match(Token::Type::NULL_TOKEN)){
		auto nullLiteral = std::make_unique<AST::Literal>(currentToken);
		consume(Token::Type::NULL_TOKEN);
		left = std::move(nullLiteral);
	}else{
		throw std::runtime_error("Expected expression at line " + std::to_string(currentToken.line) + ",column, " + std::to_string(currentToken.column));
	}

	Token next = peekToken();
	if(isBinaryOperator(next.type)){
		return parseBinaryExpression(0);
	}
	return left;
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

/*int Parse::getPrecedence(Token::Type type){
	switch(type){
		case Token::Type::OR: return 1;
		case Token::Type::AND: return 2;
	        case Token::Type::IN:
		ase Token::Type::BETWEEN: return 3;
	        case Token::Type::NOT: return 4;
		case Token::Type::EQUAL:
		case Token::Type::NOT_EQUAL:
		case Token::Type::LESS:
		case Token::Type::LESS_EQUAL:
		case Token::Type::GREATER:
		case Token::Type::GREATER_EQUAL: return 3;
		default: return 0;
	}
}*/

int Parse::getPrecedence(Token::Type type) {
    switch(type) {
        case Token::Type::OR: return 1;
        case Token::Type::AND: return 2;
        case Token::Type::NOT: return 3;
        case Token::Type::EQUAL:
        case Token::Type::NOT_EQUAL:
	case Token::Type::IS: 
	case Token::Type::IS_NOT: return 4;
        case Token::Type::LESS:
        case Token::Type::LESS_EQUAL:
        case Token::Type::GREATER:
        case Token::Type::GREATER_EQUAL: return 5;
        case Token::Type::BETWEEN:
	case Token::Type::IN:
        case Token::Type::LIKE: return 6;
        case Token::Type::PLUS:
        case Token::Type::MINUS: return 7;
        case Token::Type::ASTERIST:
	case Token::Type::SLASH:
        case Token::Type::MOD: return 8;
        default: return 0;
    }
}
/*bool Parse::isBinaryOperator(Token::Type type){
	return getPrecedence(type) >0; //|| type == Token::Type::BETWEEN;
}*/

bool Parse::isBinaryOperator(Token::Type type) {
    switch(type) {
        case Token::Type::EQUAL:
        case Token::Type::NOT_EQUAL:
        case Token::Type::LESS:
        case Token::Type::LESS_EQUAL:
        case Token::Type::GREATER:
        case Token::Type::GREATER_EQUAL:
        case Token::Type::PLUS:
        case Token::Type::MINUS:
        case Token::Type::ASTERIST:
        case Token::Type::SLASH:
        case Token::Type::AND:
        case Token::Type::OR:
	case Token::Type::LIKE:
	case Token::Type::IS:
	case Token::Type::IS_NOT:
	case Token::Type::MOD:
            return true;
        case Token::Type::BETWEEN:
        case Token::Type::IN:
            return true; // These are handled specially but are still binary operators
        default:
            return false;
    }
}
