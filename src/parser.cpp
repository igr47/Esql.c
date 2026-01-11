#include "parser.h"
#include "ai_parser.h"
#include "scanner.h"
#include "plotter_includes/plotter.h"
#include <sstream>
#include <string>
#include <stdexcept>
#include <vector>
#include <iostream>

namespace AST{
	Literal::Literal(const Token& token):token(token){}
	Identifier::Identifier(const Token& token):token(token){}

	BinaryOp::BinaryOp(Token op,std::unique_ptr<Expression> left,std::unique_ptr<Expression> right, std::unique_ptr<Expression> al): op(op),left(std::move(left)),right(std::move(right)), alias(std::move(al)){}
};

Parse::Parse(Lexer& lexer) : lexer(lexer),currentToken(lexer.nextToken()),previousToken_(Token(Token::Type::ERROR,"",0,0)){}
std::unique_ptr<AST::Statement> Parse::parse(){
    try {
        return parseStatement();
    } catch (const ParseError& e) {
        // Add context about which statement was being parsed
        std::string context = "While parsing " + (previousToken_.lexeme.empty() ? "query" : previousToken_.lexeme) +
                             " statement";
        throw ParseError(e.line, e.column, std::string(e.what()) + "\n  " + context);
    }
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
       } else if(match(Token::Type::PLOT)) {
	       return parsePlotStatement();
    } else if (match(Token::Type::LOAD)) {
        return parseLoadDataStatement();
	}else if(match(Token::Type::SHOW)){
		advance();
		if(match(Token::Type::DATABASES)){
			return parseShowDatabaseStatement();
		}else if(match(Token::Type::TABLES)){
			advance();
			return parseShowTableStatement();
        } else if (match(Token::Type::TABLE)) {
            advance();
            if (match(Token::Type::STATS)) {
                advance();
                return parseShowTableStats();
            }
           return parseShowTableStructureStatement();
        } else if (match(Token::Type::DATABASE)) {
            advance();
            return parseShowDatabaseStructureStatement();
        }
	}else if(match(Token::Type::ALTER)){
		return parseAlterTableStatement();
	} else if (match(Token::Type::CREATE_MODEL) || match(Token::Type::CREATE_OR_REPLACE) || match(Token::Type::TRAIN) ||
            match(Token::Type::AI_TRAIN) ||match(Token::Type::DROP_MODEL) ||match(Token::Type::SHOW_MODELS) ||match(Token::Type::DESCRIBE_MODEL)) {

            AIParser ai_parser(lexer,*this);
            return ai_parser.parseAIStatement();
    }
	   throw createSyntaxError(currentToken, "Expected statement (SELECT, UPDATE, DELETE, INSERT, CREATE, etc.)","Start of query");
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

    throw createExpectedOneOfError(currentToken,{Token::Type::STRING_LITERAL, Token::Type::DOUBLE_QUOTED_STRING,
         Token::Type::NUMBER_LITERAL, Token::Type::TRUE, Token::Type::FALSE},"In value position");
}

// Error helper functions
std::string Parse::formatExpectedTokens(const std::vector<Token::Type>& types) const {
    std::stringstream ss;
    for (size_t i = 0; i < types.size(); ++i) {
        if (i > 0) {
            if (i == types.size() - 1) {
                ss << " or ";
            } else {
                ss << ", ";
            }
        }
        ss << TokenUtils::typeToString(types[i]);
    }
    return ss.str();
}

ParseError Parse::createUnexpectedTokenError(const Token& token, const std::string& context) const{
    return ParseError(token.line, token.column, TokenUtils::getTokenDescription(token),token.lexeme,context);
}

ParseError Parse::createExpectedTokenError(const Token& token, Token::Type expected, const std::string& context) const {
    return ParseError(token.line, token.column,TokenUtils::typeToString(expected),TokenUtils::getTokenDescription(token),context);
}


ParseError Parse::createExpectedOneOfError(const Token& token, const std::vector<Token::Type>& expected,const std::string& context) const {
    return ParseError(token.line, token.column,
                     formatExpectedTokens(expected),
                     TokenUtils::getTokenDescription(token),
                     context);
}

ParseError Parse::createSyntaxError(const Token& token, const std::string& message,const std::string& context) const {
    return ParseError(token.line, token.column, message + ", got " +
                     TokenUtils::getTokenDescription(token), context);
}

void Parse::consume(Token::Type expected){
	if(currentToken.type==expected){
		advance();
	}else{
		throw createExpectedTokenError(currentToken, expected, "Syntax error");
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
    size_t savedPos, savedLine , savedCol;
    lexer.saveState(savedPos,savedLine,savedCol);

    // Get next token
    Token nextToken = lexer.nextToken();

    // Restore state
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

// CREATE AND DATABASE tokens are not consumed since they wre consumed in the main method
std::unique_ptr<AST::CreateDatabaseStatement> Parse::parseCreateDatabaseStatement(){
	auto stmt=std::make_unique<AST::CreateDatabaseStatement>();
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
	return stmt;
}

std::unique_ptr<AST::GroupByClause> Parse::parseGroupByClause() {
	consume(Token::Type::GROUP);
	consume(Token::Type::BY);
	std::vector<std::unique_ptr<AST::Expression>> columns;

	columns.push_back(parseExpression());

	//std::cout <<"DEBUG: Parsing GROUP B clause" <<std::endl;

	while(match(Token::Type::COMMA)){
		//std::cout<<"DEBUG : Found comma, parsing next expression" <<std::endl;
		consume(Token::Type::COMMA);
		columns.push_back(parseExpression());
		//std::cout<<"DEBUG: Added expression to GROUP BY " <<std::endl;
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

    // Parse WITH clause if present
    if (match(Token::Type::WITH)) {
        stmt->withClause = parseWithClause();
    }

    if (!match(Token::Type::SELECT)) {
        throw createExpectedTokenError(currentToken, Token::Type::SELECT, "Start of SELECT statement");
    }

	//parse select clause
	consume(Token::Type::SELECT);
	if(match(Token::Type::DISTINCT)){
		consume(Token::Type::DISTINCT);
		stmt->distinct = true;
	}
	if(match(Token::Type::ASTERIST)){
		consume(Token::Type::ASTERIST);

        // Check if there's a comma after * (invalid syntax)
        if (match(Token::Type::COMMA)) {
            throw ParseError(currentToken.line, currentToken.column,"Cannot mix * with other columns in SELECT list");
        }
	}else{
		//lexer.restoreState(savePos, saveLine, saveCol);
		//parse the first expression to see if it followed by AS
		Token nextToken = peekToken();

		if(nextToken.type == Token::Type::AS){
            //std::cout << "DEBUG: Entered AS statement evaluation." << std::endl;
			stmt->newCols = parseColumnListAs();
		}
		else{
            //std::cout << "DEBUG: Entered Column list parsing." << std::endl;
	        stmt->columns=parseColumnList();
            if (match(Token::Type::AS)) {
                stmt->newCols = parseColumnListAs();
            }
            //std::cout << "DEBUG: Finished column list parsing." << std::endl;
		}



	}

	//parse From clause
	consume(Token::Type::FROM);
	stmt->from=parseFromClause();

        // Parse JOIN clauses
    while (matchAny({Token::Type::INNER, Token::Type::LEFT,
                     Token::Type::RIGHT, Token::Type::FULL,
                     Token::Type::JOIN})) {
        stmt->joins.push_back(std::move(parseJoinClause()));
    }

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

std::unique_ptr<AST::LoadDataStatement> Parse::parseLoadDataStatement() {
    auto stmt = std::make_unique<AST::LoadDataStatement>();

    // Parse LOAD DATA
    consume(Token::Type::LOAD);
    consume(Token::Type::DATA);

    // Optional LOCAL keyword
    if (match(Token::Type::LOCAL)) {
        consume(Token::Type::LOCAL);
    }

    // Parse INFILE
    consume(Token::Type::INFILE);

    // Parse filename
    stmt->filename = currentToken.lexeme;
    // Remove quotes if present
    if (stmt->filename.size() >= 2 &&
        ((stmt->filename[0] == '\'' && stmt->filename.back() == '\'') ||
         (stmt->filename[0] == '"' && stmt->filename.back() == '"'))) {
        stmt->filename = stmt->filename.substr(1, stmt->filename.size() - 2);
    }
    consume(currentToken.type);

    // Parse INTO TABLE
    consume(Token::Type::INTO);
    consume(Token::Type::TABLE);

    // Parse table name
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

        // Parse optional file options
    while (matchAny({Token::Type::HEADER, Token::Type::DELIMITER, Token::Type::TYPE})) {
        if (match(Token::Type::HEADER)) {
            consume(Token::Type::HEADER);
            stmt->hasHeader = true;
        } else if (match(Token::Type::DELIMITER)) {
            consume(Token::Type::DELIMITER);
            if (match(Token::Type::EQUAL)) {
                consume(Token::Type::EQUAL);
            }
            if (match(Token::Type::STRING_LITERAL) || match(Token::Type::DOUBLE_QUOTED_STRING)) {
                std::string delim = currentToken.lexeme;
                if (delim.size() >= 2 &&
                    ((delim[0] == '\'' && delim.back() == '\'') ||
                     (delim[0] == '"' && delim.back() == '"'))) {
                    delim = delim.substr(1, delim.size() - 2);
                }
                if (!delim.empty()) {
                    stmt->delimiter = delim[0];
                }
                consume(currentToken.type);
            }
        } else if (match(Token::Type::TYPE)) {
            consume(Token::Type::TYPE);
            if (match(Token::Type::EQUAL)) {
                consume(Token::Type::EQUAL);
            }
            if (match(Token::Type::STRING_LITERAL) || match(Token::Type::DOUBLE_QUOTED_STRING)) {
                stmt->fileType = currentToken.lexeme;
                if (stmt->fileType.size() >= 2 &&
                    ((stmt->fileType[0] == '\'' && stmt->fileType.back() == '\'') ||
                     (stmt->fileType[0] == '"' && stmt->fileType.back() == '"'))) {
                    stmt->fileType = stmt->fileType.substr(1, stmt->fileType.size() - 2);
                }
                consume(currentToken.type);
            }
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


	//std::cout<<"DEBUG: Parsed expression:" << expr->toString()<<std::endl;

        // Use parseExpression() for strict value parsing
        stmt->setClauses.emplace_back(column, parseExpression());
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

    if (match(Token::Type::L_PAREN)) {
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
    }
	inValueContext = wasInValueContext;

    if (match(Token::Type::IN)) {
        consume(Token::Type::IN);
        stmt->filename = currentToken.lexeme;
        // Remove quotes if present
        if (stmt->filename.size() >= 2 && ((stmt->filename[0] == '\'' && stmt->filename.back() == '\'') || (stmt->filename[0] == '"' && stmt->filename.back() == '"'))) {
            stmt->filename = stmt->filename.substr(1, stmt->filename.size() - 2);
        }
        consume(currentToken.type);
    }
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
    if (!matchAny({Token::Type::INT, Token::Type::TEXT, Token::Type::BOOL, Token::Type::FLOAT,Token::Type::DATE,Token::Type::DATETIME, Token::Type::UUID})) {
        throw std::runtime_error("Expected column type (INT, TEXT, BOOL, DATE, DATETIME, UUID or FLOAT)");
    }
    col.type = currentToken.lexeme;
    consume(currentToken.type);

    // Column constraints
    while(matchAny({Token::Type::PRIMARY_KEY , Token::Type::NOT_NULL,Token::Type::UNIQUE,Token::Type::CHECK,Token::Type::DEFAULT,Token::Type::AUTO_INCREAMENT, Token::Type::GENERATE_DATE, Token::Type::GENERATE_DATE_TIME,Token::Type::GENERATE_UUID})){
	     if(match(Token::Type::PRIMARY_KEY)){
                   col.constraints.push_back("PRIMARY_KEY");
		   consume(Token::Type::PRIMARY_KEY);

		   //Handle PRIMARY KEY AUTOINCREAMENT combination
		   if(match(Token::Type::AUTO_INCREAMENT)){
			   if (col.type != "INT" && col.type != "INTEGER") {
				   throw std::runtime_error("AUTO_INCREAMENT can only be  applied to INT columns");
			   }
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
         } else if (match(Token::Type::GENERATE_DATE)) {
             col.constraints.push_back("GENERATE_DATE");
             //col.generateDate = true;
             consume(Token::Type::GENERATE_DATE);

             if (col.type != "DATE") {
                 throw std::runtime_error("GENERATE_DATE can only be applied to DATE columns");
             }
         } else if (match(Token::Type::GENERATE_DATE_TIME)) {
             col.constraints.push_back("GENERATE_DATE_TIME");
             //col.generateDateTime = true;
             consume(Token::Type::GENERATE_DATE_TIME);

             // Validate that column type is DATETIME
             if (col.type != "DATETIME" && col.type != "TIMESTAMP") {
                 throw std::runtime_error("GENERATE_DATE_TIME can only be applied toDATETIME columns");
             }
         } else if (match(Token::Type::GENERATE_UUID)) {
             col.constraints.push_back("GENERATE_UUID");
             //col.generateUUID = true;
             consume(Token::Type::GENERATE_UUID);

             // Valiadte taht columns is UUID or TEXT
             if (col.type != "UUID" && col.type != "TEXT") {
                 throw std::runtime_error("GENERATE_UUID can only be applied to UUID or TEXT columns");
             }
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
		     //std::cout << "DEBUG: Consumed Token (. Moving to expression. " << std::endl;

		     bool wasInValueContext = inValueContext;
		     inValueContext = false;

		     try {
			     //Parse the expression using the existing expression Parser
			     auto checkExpr = parseExpression();
			     //std::cout << "DEBUG: Parsed CHECK expression: " + checkExpr->toString() << std::endl;
			     consume(Token::Type::R_PAREN);
			     //Convert the expression into a string for storage in constrints
			     std::string checkConstraint = "CHECK(" + checkExpr->toString() + ")";
			     col.constraints.push_back(checkConstraint);
			     //col.checkExpression = "CHECK(" + checkExpr->toString() + ")";
			     col.checkExpression = checkExpr->toString();
			     //std::cout << "DEBUG: Succssefully copleted CHECK constraint: " + checkConstraint << std::endl;
		     } catch (...) {
			     inValueContext = wasInValueContext;
			     throw;
		     }

		     inValueContext = wasInValueContext;
	     }
    }

    stmt.columns.push_back(std::move(col));
}

std::unique_ptr<AST::ShowTableStatement> Parse::parseShowTableStatement() {
    auto stmt = std::make_unique<AST::ShowTableStatement>();

    return stmt;
}

std::unique_ptr<AST::ShowTableStructureStatement> Parse::parseShowTableStructureStatement() {
    auto stmt = std::make_unique<AST::ShowTableStructureStatement>();

    consume(Token::Type::STRUCTURE);
    stmt->tableName = currentToken.lexeme;
    consume(Token::Type::IDENTIFIER);

    return stmt;
}

std::unique_ptr<AST::ShowDatabaseStructure> Parse::parseShowDatabaseStructureStatement() {
    auto stmt = std::make_unique<AST::ShowDatabaseStructure>();

    consume(Token::Type::STRUCTURE);
    stmt->dbName = currentToken.lexeme;
    consume(Token::Type::IDENTIFIER);

    return stmt;
}

std::unique_ptr<AST::ShowTableStats> Parse::parseShowTableStats() {
    auto stmt = std::make_unique<AST::ShowTableStats>();
    stmt->tableName = currentToken.lexeme;
    consume(Token::Type::IDENTIFIER);

    return stmt;
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
        if (matchAny({Token::Type::INT, Token::Type::TEXT, Token::Type::BOOL, Token::Type::FLOAT,Token::Type::UUID,Token::Type::DATE,Token::Type::DATETIME})) {
            stmt->type = currentToken.lexeme;
            consume(currentToken.type);
        } else {
            throw std::runtime_error("Expected column type after ADD COLUMN");
        }

        // Parse constraints (same as in CREATE TABLE)
        while (matchAny({Token::Type::PRIMARY_KEY, Token::Type::NOT_NULL,
                        Token::Type::UNIQUE, Token::Type::CHECK,
                        Token::Type::DEFAULT, Token::Type::AUTO_INCREAMENT,Token::Type::GENERATE_UUID,Token::Type::GENERATE_DATE,Token::Type::GENERATE_DATE_TIME})) {

            if (match(Token::Type::PRIMARY_KEY)) {
                stmt->constraints.push_back("PRIMARY_KEY");
                consume(Token::Type::PRIMARY_KEY);

                // Handle PRIMARY KEY AUTOINCREMENT combination
                if (match(Token::Type::AUTO_INCREAMENT)) {
                    if (stmt->type != "INT" && stmt->type != "INTEGER") {
                        throw std::runtime_error("AUTO_INCREMENT can only be applied to INT columns");
                    }
                    stmt->autoIncreament = true;
                    stmt->constraints.push_back("AUTO_INCREAMENT");
                    consume(Token::Type::AUTO_INCREAMENT);
                }
            } else if (match(Token::Type::NOT_NULL)) {
                stmt->constraints.push_back("NOT_NULL");
                consume(Token::Type::NOT_NULL);
            } else if (match(Token::Type::UNIQUE)) {
                stmt->constraints.push_back("UNIQUE");
                consume(Token::Type::UNIQUE);
            } else if (match(Token::Type::AUTO_INCREAMENT)) {
                if (stmt->type != "INT" && stmt->type != "INTEGER") {
                    throw std::runtime_error("AUTO_INCREMENT can only be applied to INT columns");
                }
                stmt->autoIncreament = true;
                stmt->constraints.push_back("AUTO_INCREAMENT");
                consume(Token::Type::AUTO_INCREAMENT);
            } else if (match(Token::Type::GENERATE_UUID)) {
                stmt->constraints.push_back("GENERATE_UUID");
                consume(Token::Type::GENERATE_UUID);
            } else if (match(Token::Type::GENERATE_DATE)) {
                stmt->constraints.push_back("GENERATE_DATE");
                consume(Token::Type::GENERATE_DATE);
            } else if (match(Token::Type::GENERATE_DATE_TIME)) {
                stmt->constraints.push_back("GENERATE_DATE_TIME");
                consume(Token::Type::GENERATE_DATE_TIME);
            } else if (match(Token::Type::DEFAULT)) {
                consume(Token::Type::DEFAULT);

                bool wasInValueContext = inValueContext;
                inValueContext = true;

                try {
                    if (matchAny({Token::Type::STRING_LITERAL, Token::Type::NUMBER_LITERAL,
                                 Token::Type::TRUE, Token::Type::FALSE})) {
                        stmt->defaultValue = currentToken.lexeme;
                        stmt->constraints.push_back("DEFAULT");
                        consume(currentToken.type);
                    } else if (match(Token::Type::NULL_TOKEN)) {
                        stmt->defaultValue = "NULL";
                        stmt->constraints.push_back("DEFAULT");
                        consume(Token::Type::NULL_TOKEN);
                    } else {
                        throw std::runtime_error("Expected default value after DEFAULT keyword");
                    }
                } catch (...) {
                    inValueContext = wasInValueContext;
                    throw;
                }
                inValueContext = wasInValueContext;
            } else if (match(Token::Type::CHECK)) {
                consume(Token::Type::CHECK);
                consume(Token::Type::L_PAREN);

                bool wasInValueContext = inValueContext;
                inValueContext = false;

                try {
                    // Parse the expression using the existing expression parser
                    auto checkExpr = parseExpression();
                    //std::cout << "DEBUG: Parsed CHECK expression: " + checkExpr->toString() << std::endl;
                    consume(Token::Type::R_PAREN);

                    // Convert the expression into a string for storage
                    std::string checkConstraint = "CHECK(" + checkExpr->toString() + ")";
                    stmt->constraints.push_back(checkConstraint);
                    stmt->checkExpression = checkExpr->toString();

                    //std::cout << "DEBUG: Successfully completed CHECK constraint: " + checkConstraint << std::endl;
                } catch (...) {
                    inValueContext = wasInValueContext;
                    throw;
                }
                inValueContext = wasInValueContext;
            }
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

std::unique_ptr<AST::Expression> Parse::parseAIFunction() {
    // Create AI expression parser and try to parse
    AIParser ai_parser(lexer,*this);
    return ai_parser.parseAIFunctionCall();
}

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

    try {
        // Parse first row update
        parseSingleRowUpdate(*stmt);

        // Parse additional row updates
        while (match(Token::Type::COMMA)) {
            // Use peekToken to confirm this comma is followed by ROW
            Token nextToken = peekToken();
            if (nextToken.type != Token::Type::ROW) {
                throw ParseError(currentToken.line, currentToken.column, "Expected ROW after comma in BULK UPDATE");
            }

            consume(Token::Type::COMMA);
            parseSingleRowUpdate(*stmt);
        }

        inValueContext = wasInValueContext;
        return stmt;
    } catch (...) {
        inValueContext = wasInValueContext;
        throw;
    }
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


std::unique_ptr<AST::Statement> Parse::parsePlotStatement() {
    auto plotStmt = std::make_unique<Visualization::PlotStatement>();

    consume(Token::Type::PLOT);

    // Parse plot type
    if (match(Token::Type::LINE)) {
        plotStmt->config.type = Visualization::PlotType::LINE;
        consume(Token::Type::LINE);
    } else if (match(Token::Type::SCATTER)) {
        plotStmt->config.type = Visualization::PlotType::SCATTER;
        consume(Token::Type::SCATTER);
    } else if (match(Token::Type::BAR)) {
        plotStmt->config.type = Visualization::PlotType::BAR;
        consume(Token::Type::BAR);
    } else if (match(Token::Type::HISTOGRAM)) {
        plotStmt->config.type = Visualization::PlotType::HISTOGRAM;
        plotStmt->subType = Visualization::PlotStatement::PlotSubType::DISTRIBUTION;
        consume(Token::Type::HISTOGRAM);
    } else if (match(Token::Type::BOXPLOT)) {
        plotStmt->config.type = Visualization::PlotType::BOXPLOT;
        consume(Token::Type::BOXPLOT);
    } else if (match(Token::Type::CORRELATION)) {
        plotStmt->subType = Visualization::PlotStatement::PlotSubType::CORRELATION;
        consume(Token::Type::CORRELATION);
    } else if (match(Token::Type::PIE)) {
        plotStmt->config.type = Visualization::PlotType::PIE;
        consume(Token::Type::PIE);
    } else if (match(Token::Type::HEATMAP)) {
        plotStmt->config.type = Visualization::PlotType::HEATMAP;
        consume(Token::Type::HEATMAP);
    } else if (match(Token::Type::MULTI_LINE)) {
        plotStmt->config.type = Visualization::PlotType::MULTI_LINE;
        consume(Token::Type::MULTI_LINE);
    } else if (match(Token::Type::AREA)) {
        plotStmt->config.type = Visualization::PlotType::AREA;
        consume(Token::Type::AREA);
    } else if (match(Token::Type::STACKED_BAR)) {
        plotStmt->config.type = Visualization::PlotType::STACKED_BAR;
        consume(Token::Type::STACKED_BAR);
    } else if (match(Token::Type::VIOLIN)) {
        plotStmt->config.type = Visualization::PlotType::VIOLIN;
        consume(Token::Type::VIOLIN);
    } else if (match(Token::Type::CONTOUR)) {
        plotStmt->config.type = Visualization::PlotType::CONTOUR;
        consume(Token::Type::CONTOUR);
    } else if (match(Token::Type::SURFACE)) {
        plotStmt->config.type = Visualization::PlotType::SURFACE;
        consume(Token::Type::SURFACE);
    } else if (match(Token::Type::WIREFRAME)) {
        plotStmt->config.type = Visualization::PlotType::WIREFRAME;
        consume(Token::Type::WIREFRAME);
    } else if (match(Token::Type::HISTOGRAM_2D)) {
        plotStmt->config.type = Visualization::PlotType::HISTOGRAM_2D;
        consume(Token::Type::HISTOGRAM_2D);
    } else {
        // Default to scatter plot
        plotStmt->config.type = Visualization::PlotType::SCATTER;
    }

    // Parse comprehensive style options in parentheses
    std::map<std::string, std::string> styleOptions;
    if (match(Token::Type::L_PAREN)) {
        consume(Token::Type::L_PAREN);

        while (!match(Token::Type::R_PAREN) && !match(Token::Type::END_OF_INPUT)) {
            if (match(Token::Type::IDENTIFIER)) {
                std::string key = currentToken.lexeme;
                consume(Token::Type::IDENTIFIER);

                consume(Token::Type::EQUAL);

                // Parse value (can be string, number, boolean, or identifier)
                std::string value;
                if (match(Token::Type::STRING_LITERAL) || match(Token::Type::DOUBLE_QUOTED_STRING)) {
                    value = currentToken.lexeme;
                    // Remove quotes
                    if (value.size() >= 2 &&
                        ((value[0] == '\'' && value.back() == '\'') ||
                         (value[0] == '"' && value.back() == '"'))) {
                        value = value.substr(1, value.size() - 2);
                    }
                    consume(currentToken.type);
                } else if (match(Token::Type::NUMBER_LITERAL)) {
                    value = currentToken.lexeme;
                    consume(Token::Type::NUMBER_LITERAL);
                } else if (match(Token::Type::TRUE) || match(Token::Type::FALSE)) {
                    value = currentToken.lexeme;
                    consume(currentToken.type);
                } else if (match(Token::Type::IDENTIFIER)) {
                    value = currentToken.lexeme;
                    consume(Token::Type::IDENTIFIER);
                } else if (match(Token::Type::NULL_TOKEN)) {
                    value = "NULL";
                    consume(Token::Type::NULL_TOKEN);
                }


		styleOptions[key] = value;

                // Check for comma or end
                if (match(Token::Type::COMMA)) {
                    consume(Token::Type::COMMA);
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        consume(Token::Type::R_PAREN);
    }

    // Parse optional WITH clause for additional features
    if (match(Token::Type::WITH)) {
        consume(Token::Type::WITH);
        if (match(Token::Type::TREND)) {
            plotStmt->subType = Visualization::PlotStatement::PlotSubType::TREND;
            consume(Token::Type::TREND);
        } else if (match(Token::Type::DISTRIBUTION)) {
            plotStmt->subType = Visualization::PlotStatement::PlotSubType::DISTRIBUTION;
            consume(Token::Type::DISTRIBUTION);
        } else if (match(Token::Type::TIME_SERIES)) {
            plotStmt->subType = Visualization::PlotStatement::PlotSubType::TIME_SERIES;
            consume(Token::Type::TIME_SERIES);
        } else if (match(Token::Type::CORRELATION)) {
            plotStmt->subType = Visualization::PlotStatement::PlotSubType::CORRELATION;
            consume(Token::Type::CORRELATION);
        } else if (match(Token::Type::QQ_PLOT)) {
            plotStmt->subType = Visualization::PlotStatement::PlotSubType::QQ_PLOT;
            consume(Token::Type::QQ_PLOT);
        } else if (match(Token::Type::RESIDUALS)) {
            plotStmt->subType = Visualization::PlotStatement::PlotSubType::RESIDUALS;
            consume(Token::Type::RESIDUALS);
        } else if (match(Token::Type::ANIMATION)) {
            plotStmt->subType = Visualization::PlotStatement::PlotSubType::ANIMATION;
            consume(Token::Type::ANIMATION);
        } else if (match(Token::Type::INTERACTIVE)) {
            plotStmt->subType = Visualization::PlotStatement::PlotSubType::INTERACTIVE;
            consume(Token::Type::INTERACTIVE);
        } else if (match(Token::Type::DASHBOARD)) {
            plotStmt->subType = Visualization::PlotStatement::PlotSubType::DASHBOARD;
            consume(Token::Type::DASHBOARD);
        }
    }

    // Parse optional title
    if (match(Token::Type::TITLE)) {
        consume(Token::Type::TITLE);
        if (match(Token::Type::STRING_LITERAL) || match(Token::Type::DOUBLE_QUOTED_STRING)) {
            plotStmt->config.title = currentToken.lexeme;
            // Remove quotes
            if (plotStmt->config.title.size() >= 2 && ((plotStmt->config.title[0] == '\'' && plotStmt->config.title.back() == '\'') && (plotStmt->config.title[0] == '"' && plotStmt->config.title.back() == '"'))) {
                plotStmt->config.title = plotStmt->config.title.substr(1,
                    plotStmt->config.title.size() - 2);
            }
            consume(currentToken.type);
        }
    }

    // Parse optional xLabel
    if (match(Token::Type::XLABEL) || match(Token::Type::X_LABEL)) {
        consume(currentToken.type);
        if (match(Token::Type::STRING_LITERAL) || match(Token::Type::DOUBLE_QUOTED_STRING)) {
            plotStmt->config.xLabel = currentToken.lexeme;
            if (plotStmt->config.xLabel.size() >= 2) {
                plotStmt->config.xLabel = plotStmt->config.xLabel.substr(1,
                    plotStmt->config.xLabel.size() - 2);
            }
            consume(currentToken.type);
        }
    }

    // Parse optional yLabel
    if (match(Token::Type::YLABEL) || match(Token::Type::Y_LABEL)) {
        consume(currentToken.type);
        if (match(Token::Type::STRING_LITERAL) || match(Token::Type::DOUBLE_QUOTED_STRING)) {
            plotStmt->config.yLabel = currentToken.lexeme;
            if (plotStmt->config.yLabel.size() >= 2) {
                plotStmt->config.yLabel = plotStmt->config.yLabel.substr(1,
                    plotStmt->config.yLabel.size() - 2);
            }
            consume(currentToken.type);
        }
    }

    // Parse optional zLabel for 3D plots
    if (match(Token::Type::ZLABEL) || match(Token::Type::Z_LABEL)) {
        consume(currentToken.type);
        if (match(Token::Type::STRING_LITERAL) || match(Token::Type::DOUBLE_QUOTED_STRING)) {
            plotStmt->config.zLabel = currentToken.lexeme;
            if (plotStmt->config.zLabel.size() >= 2) {
                plotStmt->config.zLabel = plotStmt->config.zLabel.substr(1,
                    plotStmt->config.zLabel.size() - 2);
            }
            consume(currentToken.type);
        }
    }

    // Parse optional series names
    if (match(Token::Type::SERIES) || match(Token::Type::SERIES_NAMES)) {
        consume(currentToken.type);
        if (match(Token::Type::L_PAREN)) {
            consume(Token::Type::L_PAREN);
            do {
                if (match(Token::Type::COMMA)) {
                    consume(Token::Type::COMMA);
                }
                if (match(Token::Type::STRING_LITERAL) || match(Token::Type::DOUBLE_QUOTED_STRING)) {
                    std::string seriesName = currentToken.lexeme;
                    if (seriesName.size() >= 2) {
                        seriesName = seriesName.substr(1, seriesName.size() - 2);
                    }
                    plotStmt->config.seriesNames.push_back(seriesName);
                    consume(currentToken.type);
                }
            } while (match(Token::Type::COMMA));
            consume(Token::Type::R_PAREN);
        }
    }

    // Parse FOR clause with SELECT query
    consume(Token::Type::FOR);
    plotStmt->query = parseSelectStatement();

    // Parse optional GROUP BY clause
    if (match(Token::Type::GROUP)) {
        plotStmt->query->groupBy = parseGroupByClause();
    }

    // Parse optional HAVING clause
    if (match(Token::Type::HAVING)) {
        plotStmt->query->having = parseHavingClause();
    }

    // Parse optional ORDER BY clause
    if (match(Token::Type::ORDER)) {
        plotStmt->query->orderBy = parseOrderByClause();
    }

    // Parse optional LIMIT clause
    if (match(Token::Type::LIMIT)) {
        consume(Token::Type::LIMIT);
        plotStmt->query->limit = parseExpression();
        if (match(Token::Type::OFFSET)) {
            consume(Token::Type::OFFSET);
            plotStmt->query->offset = parseExpression();
        }
    }

    // Parse optional USING clause for columns
    if (match(Token::Type::USING)) {
        consume(Token::Type::USING);
        if (match(Token::Type::L_PAREN)) {
            consume(Token::Type::L_PAREN);
            do {
                if (match(Token::Type::COMMA)) {
                    consume(Token::Type::COMMA);
                }
                if (match(Token::Type::IDENTIFIER)) {
                    plotStmt->xColumns.push_back(currentToken.lexeme);
                    consume(Token::Type::IDENTIFIER);
                }
            } while (match(Token::Type::COMMA));
            consume(Token::Type::R_PAREN);
        }
    }

    // Parse optional FOR COLUMNS clause
    if (match(Token::Type::FOR_COLUMNS) || match(Token::Type::COLUMN)) {
        consume(currentToken.type);
        if (match(Token::Type::L_PAREN)) {
            consume(Token::Type::L_PAREN);
            do {
                if (match(Token::Type::COMMA)) {
                    consume(Token::Type::COMMA);
                }
                if (match(Token::Type::IDENTIFIER)) {
                    plotStmt->xColumns.push_back(currentToken.lexeme);
                    consume(Token::Type::IDENTIFIER);
                }
            } while (match(Token::Type::COMMA));
            consume(Token::Type::R_PAREN);
        }
    }

    // Parse optional FOR X Y columns
    if (match(Token::Type::FOR_X) || match(Token::Type::X)) {
        consume(currentToken.type);
        if (match(Token::Type::IDENTIFIER)) {
            plotStmt->xColumns.push_back(currentToken.lexeme);
            consume(Token::Type::IDENTIFIER);
        }
        if (match(Token::Type::COMMA)) {
            consume(Token::Type::COMMA);
            if (match(Token::Type::Y)) {
                consume(Token::Type::Y);
                if (match(Token::Type::IDENTIFIER)) {
                    plotStmt->yColumns.push_back(currentToken.lexeme);
                    consume(Token::Type::IDENTIFIER);
                }
            }
        }
    }

    // Parse optional TARGET column for distribution plots
    if (match(Token::Type::TARGET) || match(Token::Type::FOR_TARGET)) {
        consume(currentToken.type);
        if (match(Token::Type::IDENTIFIER)) {
            plotStmt->targetColumn = currentToken.lexeme;
            consume(Token::Type::IDENTIFIER);
        }
    }

    // Parse optional TIME column for time series
    if (match(Token::Type::TIME) || match(Token::Type::TIME_COLUMN)) {
        consume(currentToken.type);
        if (match(Token::Type::IDENTIFIER)) {
            plotStmt->timeColumn = currentToken.lexeme;
            consume(Token::Type::IDENTIFIER);
        }
    }

    // Parse optional GROUP column
    if (match(Token::Type::GROUP_COLUMN) || match(Token::Type::BY_GROUP)) {
        consume(currentToken.type);
        if (match(Token::Type::IDENTIFIER)) {
            plotStmt->groupColumn = currentToken.lexeme;
            consume(Token::Type::IDENTIFIER);
        }
    }

    // Parse optional ANIMATION column
    if (match(Token::Type::ANIMATE) || match(Token::Type::ANIMATION_COLUMN)) {
        consume(currentToken.type);
        if (match(Token::Type::IDENTIFIER)) {
            plotStmt->animationColumn = currentToken.lexeme;
            consume(Token::Type::IDENTIFIER);
        }
    }

    // Parse optional FPS for animation
    if (match(Token::Type::FPS) || match(Token::Type::FRAMES_PER_SECOND)) {
        consume(currentToken.type);
        if (match(Token::Type::NUMBER_LITERAL)) {
            try {
                plotStmt->animationFPS = std::stoi(currentToken.lexeme);
                consume(Token::Type::NUMBER_LITERAL);
            } catch (...) {
                throw ParseError(currentToken.line, currentToken.column,
                               "Invalid FPS value");
            }
        }
    }

    // Parse optional DASHBOARD layout
    if (match(Token::Type::DASHBOARD_LAYOUT) || match(Token::Type::LAYOUT)) {
        consume(currentToken.type);
        if (match(Token::Type::NUMBER_LITERAL)) {
            try {
                plotStmt->dashboardRows = std::stoi(currentToken.lexeme);
                consume(Token::Type::NUMBER_LITERAL);
                if (match(Token::Type::BY) || match(Token::Type::X)) {
                    consume(currentToken.type);
                    if (match(Token::Type::NUMBER_LITERAL)) {
                        plotStmt->dashboardCols = std::stoi(currentToken.lexeme);
                        consume(Token::Type::NUMBER_LITERAL);
                    }
                }
            } catch (...) {
                throw ParseError(currentToken.line, currentToken.column,
                               "Invalid dashboard layout");
            }
        }
    }

    // Parse optional output format
    if (match(Token::Type::FORMAT) || match(Token::Type::OUTPUT_FORMAT)) {
        consume(currentToken.type);
        if (match(Token::Type::PNG)) {
            plotStmt->outputFormat = Visualization::PlotStatement::OutputFormat::PNG;
            consume(Token::Type::PNG);
        } else if (match(Token::Type::PDF)) {
            plotStmt->outputFormat = Visualization::PlotStatement::OutputFormat::PDF;
            consume(Token::Type::PDF);
        } else if (match(Token::Type::SVG)) {
            plotStmt->outputFormat = Visualization::PlotStatement::OutputFormat::SVG;
            consume(Token::Type::SVG);
        } else if (match(Token::Type::JPG) || match(Token::Type::JPEG)) {
            plotStmt->outputFormat = Visualization::PlotStatement::OutputFormat::JPG;
            consume(currentToken.type);
        } else if (match(Token::Type::GIF)) {
            plotStmt->outputFormat = Visualization::PlotStatement::OutputFormat::GIF;
            consume(Token::Type::GIF);
        } else if (match(Token::Type::MP4)) {
            plotStmt->outputFormat = Visualization::PlotStatement::OutputFormat::MP4;
            consume(Token::Type::MP4);
        } else if (match(Token::Type::HTML)) {
            plotStmt->outputFormat = Visualization::PlotStatement::OutputFormat::HTML;
            consume(Token::Type::HTML);
        }
    }

    // Parse optional output file
    if (match(Token::Type::SAVE) || match(Token::Type::TO_FILE)) {
        consume(currentToken.type);
        if (match(Token::Type::STRING_LITERAL) || match(Token::Type::DOUBLE_QUOTED_STRING)) {
            plotStmt->config.outputFile = currentToken.lexeme;
            if (plotStmt->config.outputFile.size() >= 2) {
                plotStmt->config.outputFile = plotStmt->config.outputFile.substr(1,
                    plotStmt->config.outputFile.size() - 2);
            }
            consume(currentToken.type);
        }
    }

    // Parse interactive controls
    if (match(Token::Type::CONTROLS) || match(Token::Type::WIDGETS)) {
        consume(currentToken.type);
        if (match(Token::Type::L_PAREN)) {
            consume(Token::Type::L_PAREN);
            while (!match(Token::Type::R_PAREN) && !match(Token::Type::END_OF_INPUT)) {
                Visualization::PlotStatement::Control control;

                // Parse control type
                if (match(Token::Type::SLIDER) || match(Token::Type::DROPDOWN) ||
                    match(Token::Type::CHECKBOX) || match(Token::Type::BUTTON)) {
                    control.type = currentToken.lexeme;
                    consume(currentToken.type);
                }

                // Parse control name
                if (match(Token::Type::IDENTIFIER)) {
                    control.name = currentToken.lexeme;
                    consume(Token::Type::IDENTIFIER);
                }

                // Parse label if present
                if (match(Token::Type::LABEL)) {
                    consume(Token::Type::LABEL);
                    if (match(Token::Type::STRING_LITERAL) || match(Token::Type::DOUBLE_QUOTED_STRING)) {
                        control.label = currentToken.lexeme;
                        if (control.label.size() >= 2) {
                            control.label = control.label.substr(1, control.label.size() - 2);
                        }
                        consume(currentToken.type);
                    }
                }

                // Parse slider-specific parameters
                if (control.type == "slider") {
                    if (match(Token::Type::MIN)) {
                        consume(Token::Type::MIN);
                        if (match(Token::Type::NUMBER_LITERAL)) {
                            control.minValue = std::stod(currentToken.lexeme);
                            consume(Token::Type::NUMBER_LITERAL);
                        }
                    }
                    if (match(Token::Type::MAX)) {
                        consume(Token::Type::MAX);
                        if (match(Token::Type::NUMBER_LITERAL)) {
                            control.maxValue = std::stod(currentToken.lexeme);
                            consume(Token::Type::NUMBER_LITERAL);
                        }
                    }
                    if (match(Token::Type::STEP)) {
                        consume(Token::Type::STEP);
                        if (match(Token::Type::NUMBER_LITERAL)) {
                            control.step = std::stod(currentToken.lexeme);
                            consume(Token::Type::NUMBER_LITERAL);
                        }
                    }
                    if (match(Token::Type::DEFAULT)) {
                        consume(Token::Type::DEFAULT);
                        if (match(Token::Type::NUMBER_LITERAL)) {
                            control.defaultValue = std::stod(currentToken.lexeme);
                            consume(Token::Type::NUMBER_LITERAL);
                        }
                    }
                }

                // Parse dropdown options
                if (control.type == "dropdown" && match(Token::Type::OPTIONS)) {
                    consume(Token::Type::OPTIONS);
                    if (match(Token::Type::L_PAREN)) {
                        consume(Token::Type::L_PAREN);
                        while (!match(Token::Type::R_PAREN)) {
                            if (match(Token::Type::STRING_LITERAL) || match(Token::Type::DOUBLE_QUOTED_STRING)) {
                                std::string option = currentToken.lexeme;
                                if (option.size() >= 2) {
                                    option = option.substr(1, option.size() - 2);
                                }
                                control.options.push_back(option);
                                consume(currentToken.type);
                            }
                            if (match(Token::Type::COMMA)) {
                                consume(Token::Type::COMMA);
                            }
                        }
                        consume(Token::Type::R_PAREN);
                    }
                }

                plotStmt->controls.push_back(control);

                if (match(Token::Type::COMMA)) {
                    consume(Token::Type::COMMA);
                } else {
                    break;
                }
            }
            consume(Token::Type::R_PAREN);
        }
    }

    // Now parse style options into config
    plotStmt->config.style.parseFromMap(styleOptions);

    return plotStmt;
}

/*std::unique_ptr<AST::Statement> Parse::parsePlotStatement() {
    auto plotStmt = std::make_unique<Visualization::PlotStatement>();

    consume(Token::Type::PLOT);

    // Parse plot type
    if (match(Token::Type::LINE)) {
        plotStmt->config.type = Visualization::PlotType::LINE;
        consume(Token::Type::LINE);
    } else if (match(Token::Type::SCATTER)) {
        plotStmt->config.type = Visualization::PlotType::SCATTER;
        consume(Token::Type::SCATTER);
    } else if (match(Token::Type::BAR)) {
        plotStmt->config.type = Visualization::PlotType::BAR;
        consume(Token::Type::BAR);
    } else if (match(Token::Type::HISTOGRAM)) {
        plotStmt->config.type = Visualization::PlotType::HISTOGRAM;
        plotStmt->subType = Visualization::PlotStatement::PlotSubType::DISTRIBUTION;
        consume(Token::Type::HISTOGRAM);
    } else if (match(Token::Type::BOXPLOT)) {
        plotStmt->config.type = Visualization::PlotType::BOXPLOT;
        consume(Token::Type::BOXPLOT);
    } else if (match(Token::Type::CORRELATION)) {
        plotStmt->subType = Visualization::PlotStatement::PlotSubType::CORRELATION;
        consume(Token::Type::CORRELATION);
    } else if (match(Token::Type::PIE)) {
        plotStmt->config.type = Visualization::PlotType::PIE;
        consume(Token::Type::PIE);
    } else if (match(Token::Type::HEATMAP)) {
        plotStmt->config.type = Visualization::PlotType::HEATMAP;
        consume(Token::Type::HEATMAP);
    } else if (match(Token::Type::MULTI_LINE)) {
        plotStmt->config.type = Visualization::PlotType::MULTI_LINE;
        consume(Token::Type::MULTI_LINE);
    } else if (match(Token::Type::AREA)) {
	    plotStmt->config.type = Visualization::PlotType::AREA;
	    consume(Token::Type::AREA);
    } else if (match(Token::Type::STACKED_BAR)) {
	    plotStmt->config.type = Visualization::PlotType::STACKED_BAR;
	    consume(Token::Type::STACKED_BAR);
    } else {
        // Default to scatter plot
        plotStmt->config.type = Visualization::PlotType::SCATTER;
    }

    // Parse plot options/configuration in parentheses
    if (match(Token::Type::L_PAREN)) {
        consume(Token::Type::L_PAREN);

        while (!match(Token::Type::R_PAREN) && !match(Token::Type::END_OF_INPUT)) {
            // Parse key-value pairs
            if (match(Token::Type::IDENTIFIER)) {
                std::string key = currentToken.lexeme;
                consume(Token::Type::IDENTIFIER);

                consume(Token::Type::EQUAL);

                if (match(Token::Type::STRING_LITERAL) || match(Token::Type::DOUBLE_QUOTED_STRING)) {
                    std::string value = currentToken.lexeme;
                    // Remove quotes
                    if (value.size() >= 2) {
                        value = value.substr(1, value.size() - 2);
                    }
                    plotStmt->config.style[key] = value;
                    consume(currentToken.type);
                } else if (match(Token::Type::NUMBER_LITERAL)) {
                    plotStmt->config.style[key] = currentToken.lexeme;
                    consume(Token::Type::NUMBER_LITERAL);
                } else if (match(Token::Type::TRUE) || match(Token::Type::FALSE)) {
                    plotStmt->config.style[key] = currentToken.lexeme;
                    consume(currentToken.type);
                } else if (match(Token::Type::IDENTIFIER)) {
                    plotStmt->config.style[key] = currentToken.lexeme;
                    consume(Token::Type::IDENTIFIER);
                }

                // Check for comma or end
                if (match(Token::Type::COMMA)) {
                    consume(Token::Type::COMMA);
                }
            } else {
                break;
            }
        }

        consume(Token::Type::R_PAREN);
    }

    // Parse optional WITH clause for additional features
    if (match(Token::Type::WITH)) {
        consume(Token::Type::WITH);
        if (match(Token::Type::TREND)) {
            plotStmt->subType = Visualization::PlotStatement::PlotSubType::TREND;
            consume(Token::Type::TREND);
        }
    }

    // Parse optional title
    if (match(Token::Type::TITLE)) {
        consume(Token::Type::TITLE);
        if (match(Token::Type::STRING_LITERAL) || match(Token::Type::DOUBLE_QUOTED_STRING)) {
            plotStmt->config.title = currentToken.lexeme;
            // Remove quotes
            if (plotStmt->config.title.size() >= 2) {
                plotStmt->config.title = plotStmt->config.title.substr(1,
                    plotStmt->config.title.size() - 2);
            }
            consume(currentToken.type);
        }
    }

    // Parse FOR clause with SELECT query
    consume(Token::Type::FOR);
    plotStmt->query = parseSelectStatement();

    // Parse GROUP BY clause if present
    if (match(Token::Type::GROUP)) {
        // This needs to be integrated into the SelectStatement
        plotStmt->query->groupBy = parseGroupByClause();
    }

    // Parse optional output file
    if (match(Token::Type::SAVE)) {
        consume(Token::Type::SAVE);
        if (match(Token::Type::STRING_LITERAL) || match(Token::Type::DOUBLE_QUOTED_STRING)) {
            plotStmt->config.outputFile = currentToken.lexeme;
            if (plotStmt->config.outputFile.size() >= 2) {
                plotStmt->config.outputFile = plotStmt->config.outputFile.substr(1,
                    plotStmt->config.outputFile.size() - 2);
            }
            consume(currentToken.type);
        }
    }

    return plotStmt;
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
    //std::cout << "DEBUG: Entered the parseColumnList() method" << std::endl;
	std::vector<std::unique_ptr<AST::Expression>> columns;

	do{
		if(match(Token::Type::COMMA)){
			consume(Token::Type::COMMA);
		}
        //std::cout << "DEBUG: Starting prseExpression() method." << std::endl;
		columns.push_back(parseExpression());
        //std::cout << "DEBUG: Finished parseExpression() method." << std::endl;
	}while(match(Token::Type::COMMA));
	return columns;
}


std::unique_ptr<AST::Expression> Parse::parseFromClause(){
	return parseIdentifier();
}

std::unique_ptr<AST::Expression> Parse::parseLikePattern() {
    if (match(Token::Type::L_PAREN)) {
        // Handle complex expressions in like
        return parseExpression();
    } else if (match(Token::Type::STRING_LITERAL) || match(Token::Type::DOUBLE_QUOTED_STRING)) {
        // Parse string literals that may contain character classes
        std::string pattern = currentToken.lexeme;

        // Remove quotes
        if (pattern.size() >= 2 && ((pattern[0] == '\'' && pattern.back() == '\'') || (pattern[0] == '"' && pattern.back() == '"'))) {
            pattern = pattern.substr(1, pattern.size() - 2);
        }

        advance();

        // Check if this string contains character classes that nee parsing
        if (pattern.find('[') != std::string::npos) {
            return parseCharacterClassPattern(pattern);
        }

        return std::make_unique<AST::Literal>(Token(Token::Type::STRING_LITERAL,pattern, 0, 0));
    } else {
        throw ParseError(currentToken.line, currentToken.column,"Expected string literal in LIKE pattern");
    }
}

std::unique_ptr<AST::Expression> Parse::parseCharacterClassPattern(const std::string& pattern) {
    // Charachter class expansion will be taken care off in the execution enging
    return std::make_unique<AST::CharClass>(pattern);
}

std::unique_ptr<AST::Expression> Parse::parseWindowFunction() {
    Token funcToken = currentToken;
    consume(currentToken.type);

    std::unique_ptr<AST::Expression> arg = nullptr;
    if (match(Token::Type::L_PAREN)) {
        consume(Token::Type::L_PAREN);
        if (!match(Token::Type::R_PAREN)) {
            arg = parseExpression();
        }
        consume(Token::Type::R_PAREN);
    }

    consume(Token::Type::OVER);
    consume(Token::Type::L_PAREN);

    std::vector<std::unique_ptr<AST::Expression>> partitionBy;
    std::vector<std::pair<std::unique_ptr<AST::Expression>, bool>> orderBy;

    // Parse PARTITION BY
    if (match(Token::Type::PARTITION)) {
        consume(Token::Type::PARTITION);
        consume(Token::Type::BY);

        do {
            if (match(Token::Type::COMMA)) consume(Token::Type::COMMA);
            partitionBy.push_back(parseExpression());
        } while (match(Token::Type::COMMA));
    }

    // Parse ORDER BY
    if (match(Token::Type::ORDER)) {
        consume(Token::Type::ORDER);
        consume(Token::Type::BY);

        do {
            if (match(Token::Type::COMMA)) consume(Token::Type::COMMA);
            auto expr = parseExpression();
            bool ascending = true;

            if (match(Token::Type::ASC)) {
                consume(Token::Type::ASC);
                ascending = true;
            } else if (match(Token::Type::DESC)) {
                consume(Token::Type::DESC);
                ascending = false;
            }

            orderBy.emplace_back(std::move(expr), ascending);
        } while (match(Token::Type::COMMA));
    }

    consume(Token::Type::R_PAREN);

    std::unique_ptr<AST::Expression> aliasExpr = nullptr;
    if (match(Token::Type::AS)) {
        consume(Token::Type::AS);
        aliasExpr = parseIdentifier();
    }

    return std::make_unique<AST::WindowFunction>(funcToken, std::move(arg), std::move(partitionBy), std::move(orderBy),nullptr,std::move(aliasExpr));
}

std::unique_ptr<AST::WithClause> Parse::parseWithClause() {
    consume(Token::Type::WITH);

    std::vector<AST::CommonTableExpression> ctes;

    do {
        if (match(Token::Type::COMMA)) {
            consume(Token::Type::COMMA);
        }

        AST::CommonTableExpression cte;
        cte.name = currentToken.lexeme;
        consume(Token::Type::IDENTIFIER);

        // Optional column list
        if (match(Token::Type::L_PAREN)) {
            consume(Token::Type::L_PAREN);
            do {
                cte.columns.push_back(currentToken.lexeme);
                consume(Token::Type::IDENTIFIER);
            } while (match(Token::Type::COMMA) && (consume(Token::Type::COMMA), true));
            consume(Token::Type::R_PAREN);
        }

        consume(Token::Type::AS);
        consume(Token::Type::L_PAREN);
        cte.query = parseSelectStatement();
        consume(Token::Type::R_PAREN);

        ctes.push_back(std::move(cte));
    } while (match(Token::Type::COMMA));
    return std::make_unique<AST::WithClause>(std::move(ctes));
}

// Parse join clause
std::unique_ptr<AST::JoinClause> Parse::parseJoinClause() {
    auto joinClause = std::make_unique<AST::JoinClause>();

    // Parse Join type
    if (match(Token::Type::INNER)) {
        joinClause->type = AST::JoinClause::INNER;
        consume(Token::Type::INNER);
    } else if (match(Token::Type::LEFT)) {
        consume(Token::Type::LEFT);
        if (match(Token::Type::OUTER)) {
            consume(Token::Type::OUTER);
        }
        joinClause->type = AST::JoinClause::LEFT;
    } else if (match(Token::Type::RIGHT)) {
        consume(Token::Type::RIGHT);
        if (match(Token::Type::OUTER)) {
            consume(Token::Type::OUTER);
        }
        joinClause->type = AST::JoinClause::RIGHT;
    } else if (match(Token::Type::FULL)) {
        consume(Token::Type::FULL);
        if (match(Token::Type::OUTER)) {
            consume(Token::Type::OUTER);
        }
        joinClause->type = AST::JoinClause::FULL;
    }

    consume(Token::Type::JOIN);

    // Parse table
    joinClause->table = parseExpression();

        // Parse ON Condition
    consume(Token::Type::ON);
    joinClause->condition = parseExpression();

    return joinClause;
}

std::vector<std::unique_ptr<AST::Expression>> Parse::parseFunctionArguments() {
    std::vector<std::unique_ptr<AST::Expression>> args;

    if (!match(Token::Type::R_PAREN)) {
        do {
            args.push_back(parseExpression());
        } while (match(Token::Type::COMMA) && (consume(Token::Type::COMMA), true));
    }
    return args;
}

std::unique_ptr<AST::Expression> Parse::parseStatisticalFunction() {
    //std::cout << "DEBUG: Reached Statistical function parsing";
    Token funcToken = currentToken;
    AST::StatisticalExpression::StatType statType;

    switch(funcToken.type) {
        case Token::Type::STDDEV: statType = AST::StatisticalExpression::StatType::STDDEV; break;
        case Token::Type::VARIANCE: statType = AST::StatisticalExpression::StatType::VARIANCE; break;
        case Token::Type::PERCENTILE_CONT: statType= AST::StatisticalExpression::StatType::PERCENTILE; break;
        case Token::Type::CORR: statType= AST::StatisticalExpression::StatType::CORRELATION; break;
        case Token::Type::REGR_SLOPE: statType = AST::StatisticalExpression::StatType::REGRESSION; break;
        default: throw ParseError(currentToken.line, currentToken.column, "Unkown statistical function");
    }

    std::cout << "DEBUG: Consumed Token: currentToken.type "<< std::endl;
    consume(currentToken.type);
    consume(Token::Type::L_PAREN);

    std::cout << "DEBUG: Consumed ( Token : " << std::endl;
    std::cout << "DUBUG: Starting parsing parseExpression()" <<std::endl;
    auto arg1 = parseExpression();
    std::cout << "DEBUG: Parsed parseExpression()" << std::endl;
    std::unique_ptr<AST::Expression> arg2= nullptr;
    double percentile = 0.5;

    if (statType == AST::StatisticalExpression::StatType::PERCENTILE) {
        if (match(Token::Type::COMMA)) {
            consume(Token::Type::COMMA);
        }
        if (match(Token::Type::NUMBER_LITERAL)) {
            try {
                percentile = std::stod(currentToken.lexeme);
                consume(Token::Type::NUMBER_LITERAL);
            } catch (...) {
                throw ParseError(currentToken.line, currentToken.column, "Invalid percentile value");
            }
        }

        Token nextToken = peekToken();

        consume(Token::Type::R_PAREN);
        //std::cout << "DEBUG: Consumed R_PAREN" << std::endl;

        // Parse WITHIN GROUP clause
        if (match(Token::Type::WITHIN)) {
            //consume(Token::Type::R_PAREN);

            consume(Token::Type::WITHIN);
            std::cout << "DEBUG: Found and consumed WITHIN" << std::endl;
            consume(Token::Type::GROUP);
            consume(Token::Type::L_PAREN);
            std::cout << "DEBUG: Found and consumes L_paren" << std::endl;

            // Parse ORDER BY clause
            consume(Token::Type::ORDER);
            consume(Token::Type::BY);
            std::cout << "DEBUG: Consumed ORDER BY" << std::endl;

            // Parse the ordering expression (salary in your example)
            arg2 = parseExpression();

            // Parse ASC/DESC if present
            if (match(Token::Type::ASC) || match(Token::Type::DESC)) {
                std::cout << "DEBUG: Found ASC/DEC";
                advance();  // Consume ASC/DESC
            }

            consume(Token::Type::R_PAREN);
            std::cout << "DEBUG: Consumed R_PAREN" << std::endl;
        }
    } else if (statType == AST::StatisticalExpression::StatType::CORRELATION || statType == AST::StatisticalExpression::StatType::REGRESSION) {
        consume(Token::Type::COMMA);
        arg2 = parseExpression();
        consume(Token::Type::R_PAREN);
    }

    if (match(Token::Type::R_PAREN)){
        consume(Token::Type::R_PAREN);
    }

    std::cout << "DEBUG: Reached the end and returning the constructor." << std::endl;

    std::unique_ptr<AST::Expression> aliasExpr = nullptr;
    if(match(Token::Type::AS)){
        consume(Token::Type::AS);
        std::cout << "DEBUG: Found and consumed AS expression" << std::endl;
        aliasExpr = parseIdentifier();
    }

    return std::make_unique<AST::StatisticalExpression>(statType, std::move(arg1), std::move(arg2),std::move(aliasExpr), percentile);
}


std::unique_ptr<AST::Expression> Parse::parseExpression(){
    try {
        //return parseBinaryExpression(1);
        auto expr = parseBinaryExpression(1);

        if(pendingAlias) {
            if (auto* binaryOp = dynamic_cast<AST::BinaryOp*>(expr.get())) {
                binaryOp->alias = std::move(pendingAlias);
            }
        }
        return expr;

        // Check for alias after the full expression
        /*if (match(Token::Type::AS)) {
            consume(Token::Type::AS);
            auto aliasExpr = parseIdentifier();

            // If the expression is a BinaryOp, add alias to it
            if (auto* binaryOp = dynamic_cast<AST::BinaryOp*>(expr.get())) {
                binaryOp->alias = std::move(aliasExpr);
            }
            // You might want to handle other expression types too
        }*/

        //return expr;
    } catch(const ParseError&) {
        throw; // Re-throw ParseError
    } catch (const std::exception& e) {
        // Convert any other exception to ParseError
        throw ParseError(currentToken.line, currentToken.column,std::string("Error parsing expression: ") + e.what());
    }
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
        else if (op.type == Token::Type::LIKE) {
           consume(Token::Type::LIKE);
           auto right = parseLikePattern();
           left = std::make_unique<AST::LikeOp>(std::move(left), std::move(right));
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

        bool isNot = false;
        if (match(Token::Type::NOT)) {
            consume(Token::Type::NOT);
            isNot = true;
        }

        // Handle IS NULL, IS TRUE, IS FALSE
        if (match(Token::Type::NULL_TOKEN)) {
            consume(Token::Type::NULL_TOKEN);
            // Create a special IS NULL/IS NOT NULL operation
            auto nullLiteral = std::make_unique<AST::Literal>(Token(Token::Type::NULL_TOKEN, "NULL",op.line, op.column));
            Token isOpToken = isNot ? Token(Token::Type::IS_NOT_NULL, "IS NOT NULL", op.line, op.column) : Token(Token::Type::IS_NULL, "IS NULL", op.line, op.column);
            left = std::make_unique<AST::BinaryOp>(isOpToken, std::move(left), std::move(nullLiteral));
        } else if (match(Token::Type::TRUE)) {
                consume (Token::Type::TRUE);
                auto trueLiteral = std::make_unique<AST::Literal>(Token(Token::Type::TRUE, "TRUE", op.line, op.column));
                Token isOpToken = isNot ? Token(Token::Type::IS_NOT_TRUE, "IS NOT TRUE", op.line, op.column) : Token(Token::Type::IS_TRUE, "IS TRUE", op.line, op.column);
                left = std::make_unique<AST::BinaryOp>(isOpToken, std::move(left), std::move(trueLiteral));
        } else if (match(Token::Type::FALSE)) {
                consume(Token::Type::FALSE);
                auto falseLiteral = std::make_unique<AST::Literal>(Token(Token::Type::FALSE, "FALSE", op.line, op.column));
                Token isOpToken = isNot ? Token(Token::Type::IS_NOT_FALSE, "IS NOT FALSE", op.line, op.column) : Token(Token::Type::IS_FALSE, "IS FALSE", op.line, op.column);
                left = std::make_unique<AST::BinaryOp>(isOpToken, std::move(left), std::move(falseLiteral));
        } else {
                // Regular IS comparison with another expression
                auto right = parseBinaryExpression(precedence + 1);
                Token isOpToken = isNot ? Token(Token::Type::IS_NOT, "IS NOT", op.line, op.column) : Token(Token::Type::IS, "IS", op.line, op.column);
                left = std::make_unique<AST::BinaryOp>(isOpToken, std::move(left), std::move(right));
       }
       continue;
    }

		/*if(match(Token::Type::NOT)){
			consume(Token::Type::NOT);
			auto right = parseBinaryExpression(precedence+1);
			Token isNotToken = Token(Token::Type::IS_NOT, "IS NOT", op.line, op.column);
			left = std::make_unique<AST::BinaryOp>(isNotToken, std::move(left), std::move(right));
		}else {
			auto right = parseBinaryExpression(precedence +1 );
			left = std::make_unique<AST::BinaryOp>(op, std::move(left),std::move(right));
		}
		continue;
	}*/

    /*std::unique_ptr<AST::Expression> aliasExpr = nullptr;
    if (match(Token::Type::AS)) {
        consume(Token::Type::AS);
        aliasExpr = parseIdentifier();
    }*/

        // Handle regular binary operators
        if (isBinaryOperator(op.type)) {
            advance(); // Consume the operator
            auto right = parseBinaryExpression(precedence + 1);
            left = std::make_unique<AST::BinaryOp>(op, std::move(left), std::move(right));
            //auto binaryOp = std::make_unique<AST::BinaryOp>(op, std::move(left), std::move(right), std::move(aliasExpr));
            //left = std::move(binaryOp);
        } else {
            break;
        }
    }

    return left;
}

std::unique_ptr<AST::Expression> Parse::parsePrimaryExpression(){
	//std::cout<< "DEBUG: parsePrimarExpression() - current Token:" << static_cast<int>(currentToken.type) <<"'" <<currentToken.lexeme <<"'"<<std::endl;

	std::unique_ptr<AST::Expression> left;

    if (match(Token::Type::CASE)) {
            return parseCaseExpression();
    } else if (auto ai_func = parseAIFunction()) {
        return ai_func;
    } else if (matchAny({Token::Type::COUNT,Token::Type::SUM,Token::Type::AVG,Token::Type::MIN,Token::Type::MAX})) {
		//std::cout<< "DEBUG: Foung aggregate function: " <<currentToken.lexeme<<std::endl;
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
			//std::cout<<"DEBUG: Handling identifier argument"<<std::endl;
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
			//std::cout<< "DEBUG: Handling expression argument"<<std::endl;
			arg=parseExpression();
		}

		consume(Token::Type::R_PAREN);

		if(match(Token::Type::AS)){
			consume(Token::Type::AS);
			expre = parseIdentifier();
		}


		//return std::make_unique<AST::AggregateExpression>(funcToken, std::move(arg),std::move(expre), isCountAll);
        left = std::make_unique<AST::AggregateExpression>(funcToken, std::move(arg),std::move(expre), isCountAll);
	}else if(match(Token::Type::IDENTIFIER)){
		//return parseIdentifier();
		left = parseIdentifier();
	}else if(match(Token::Type::NUMBER_LITERAL) || match(Token::Type::STRING_LITERAL) || match(Token::Type::DOUBLE_QUOTED_STRING) || match(Token::Type::FALSE) || match(Token::Type::TRUE)){
		//return parseLiteral();
		left = parseLiteral();
	}else if(match(Token::Type::L_PAREN)){
		consume(Token::Type::L_PAREN);
        if (!match(Token::Type::R_PAREN)) {
            throw createExpectedTokenError(currentToken, Token::Type::R_PAREN,"To close parenthesized expression");
        }
		//auto expr=parseExpression();
		left = parseExpression();
		consume(Token::Type::R_PAREN);
		//return expr;
	}else if (match(Token::Type::NULL_TOKEN)){
		auto nullLiteral = std::make_unique<AST::Literal>(currentToken);
		consume(Token::Type::NULL_TOKEN);
		left = std::move(nullLiteral);
    } else if (match(Token::Type::ROUND) /*|| match(Token::Type::LOWER) || match(Token::Type::UPPER) || match(Token::Type::SUBSTRING)*/) {
        Token funcToken = currentToken;
        consume(currentToken.type);
        consume(Token::Type::L_PAREN);

        std::vector<std::unique_ptr<AST::Expression>> args;
        if (!match(Token::Type::R_PAREN)) {
            do {
                args.push_back(parseExpression());
            } while (match(Token::Type::COMMA) && (consume(Token::Type::COMMA), true));
        }
        consume(Token::Type::R_PAREN);

        //return std::make_unique<AST::FunctionCall>(funcToken, std::move(args));
        left = std::make_unique<AST::FunctionCall>(funcToken, std::move(args));
    } else if (matchAny({Token::Type::ROW_NUMBER, Token::Type::RANK, Token::Type::DENSE_RANK, Token::Type::LAG, Token::Type::LEAD, Token::Type::FIRST_VALUE, Token::Type::LAST_VALUE,Token::Type::NTILE})) {
        //return parseWindowFunction();
         left = parseWindowFunction();
    } else if (matchAny({Token::Type::STDDEV, Token::Type::VARIANCE, Token::Type::PERCENTILE_CONT, Token::Type::CORR, Token::Type::REGR_SLOPE})) {
        //return parseStatisticalFunction();
          left = parseStatisticalFunction();
    } else if (match(Token::Type::JULIANDAY)) {
        Token funcToken = currentToken;
        consume(Token::Type::JULIANDAY);
        consume(Token::Type::L_PAREN);
        auto arg = parseExpression();
        consume(Token::Type::R_PAREN);

        std::unique_ptr<AST::Expression> aliasExpr = nullptr;
        if (match(Token::Type::AS)) {
            consume(Token::Type::AS);
            aliasExpr = parseIdentifier();
            pendingAlias = aliasExpr->clone();
        }
        //return std::make_unique<AST::DateFunction>(funcToken, std::move(arg),std::move(aliasExpr));
        left = std::make_unique<AST::DateFunction>(funcToken, std::move(arg),std::move(aliasExpr));
    } else if (matchAny({Token::Type::YEAR, Token::Type::MONTH, Token::Type::DAY,Token::Type::SUBSTR, Token::Type::CONCAT, Token::Type::LENGTH,Token::Type::LOWER,Token::Type::UPPER})) {
        Token funcToken = currentToken;
        consume(currentToken.type);
        std::vector<std::unique_ptr<AST::Expression>> args;
        consume(Token::Type::L_PAREN);
        args = parseFunctionArguments();
        consume(Token::Type::R_PAREN);
        //auto arg = parseExpression();
        //consume(Token::Type::R_PAREN);

        std::unique_ptr<AST::Expression> aliasExpr = nullptr;
        if (match(Token::Type::AS)) {
            consume(Token::Type::AS);
            aliasExpr = parseIdentifier();
        }

        //std::vector<std::unique_ptr<AST::Expression>> args;
        //args.push_back(std::move(arg));
        //return std::make_unique<AST::FunctionCall>(funcToken,std::move(args), std::move(aliasExpr));
        left = std::make_unique<AST::FunctionCall>(funcToken, std::move(args), std::move(aliasExpr));
        /*left = std::make_unique<AST::FunctionCall>(
            funcToken,
            std::vector<std::unique_ptr<AST::Expression>>{},
            std::move(aliasExpr));*/
    } else if (match(Token::Type::NOW)) {
        Token funcToken = currentToken;
        consume(Token::Type::NOW);

        // Handle optional parentheses
        if (match(Token::Type::L_PAREN)) {
            consume(Token::Type::L_PAREN);
            consume(Token::Type::R_PAREN);
        }
        std::unique_ptr<AST::Expression> aliasExpr = nullptr;
        if (match(Token::Type::AS)) {
            consume(Token::Type::AS);
            aliasExpr = parseIdentifier();
        }

        return std::make_unique<AST::FunctionCall>(
            funcToken,
            std::vector<std::unique_ptr<AST::Expression>>{},
            std::move(aliasExpr)
        );
	}else {
        throw createSyntaxError(currentToken, "Expected expression (identifier, literal, function, etc.)","In expression");
	}

    Token next = peekToken();
    if(isBinaryOperator(next.type)){
        // Don't consume the operator here - let parseBinaryExpression handle it
        // Just return what we've parsed so far
        return left;
    }
	/*if(isBinaryOperator(next.type)){

        auto currentLeft = std::move(left);

        std::cout << " DEBUG: Reached the operator." << std::endl;
        Token op = currentToken;
        std::cout << "DEBUG: Starting to consume operator" << std::endl;
        advance(); // Consume the operator
        std::cout << "DEBUG: Completed consuming operator" << std::endl;

        // Parse the right side with appropriate precedence
        int precedence = getPrecedence(op.type);
        auto right = parseBinaryExpression(precedence + 1);

        // Create the binary expression
        return std::make_unique<AST::BinaryOp>(op, std::move(currentLeft), std::move(right));
		//return parseBinaryExpression(0);
		//return left;
	}*/
	return left;
}

// Method for parsing Conditionals
std::unique_ptr<AST::Expression> Parse::parseCaseExpression() {
    //std::cout << "DEBUG: Reached parseCaseExpression() method and consuming CASE." << std::endl;
    consume(Token::Type::CASE);
    //std::cout << "DEBUG: Consumed CASE clause." << std::endl;

    std::vector<std::pair<std::unique_ptr<AST::Expression>,std::unique_ptr<AST::Expression>>> whenClauses;
    std::unique_ptr<AST::Expression> elseClause = nullptr;

    //Parse optional CASE expressions (for simple CASE)
    std::unique_ptr<AST::Expression> caseExpression = nullptr;
    if (!match(Token::Type::WHEN) /*&& !match(Token::Type::END)*/) {
        caseExpression = parseExpression();
    }

    // Parse WHEN clauses
    //std::cout << "DEBUG: Consuming WHEN clause." << std::endl;
    while (match(Token::Type::WHEN)) {
        consume(Token::Type::WHEN);
        //std::cout << "DEBUG: Consumed WHEN. eNTERING EXPRESSION PARSING" << std::endl;
        auto condition = parseExpression();
        //std::cout << "DEBUG: Parsed parseExpression()." << std::endl;
        //std::cout << "DEBUG: Consuming THEN." << std::endl;
        consume (Token::Type::THEN);
        //std::cout << "DEBUG: Consumed THEN. Starting parseExpression()." << std::endl;
        auto result = parseExpression();
        //std::cout << "DEBUG: Completed ParseExpression()." << std::endl;
        whenClauses.emplace_back(std::move(condition), std::move(result));
    }

    // Parse ELSE clause
     //std::cout << "DEBUG: Consuming ELSE clause." << std::endl;
    if (match(Token::Type::ELSE)) {
        consume(Token::Type::ELSE);
        //std::cout << "DEBUG: Consumed ELSE. eNTERING EXPRESSION PARSING" << std::endl;
        elseClause = parseExpression();
        //std::cout << "DEBUG: Parsed parseExpression()." << std::endl;
    }


    // Must have END
    if (!match(Token::Type::END)) {
        throw ParseError(currentToken.line, currentToken.column,"Expected END to close CASE expression");
    }

     //std::cout << "DEBUG: Consuming END clause." << std::endl;
    consume(Token::Type::END);
    //std::cout << "DEBUG: Completed END expression" << std::endl;

        // Validate we have at least one WHEN clause or an ELSE
    if (whenClauses.empty() && !elseClause) {
        throw ParseError(currentToken.line, currentToken.column,"CASE expression must have at least one WHEN clause or an ELSE clause");
    }

    std::string alias;
    if (match(Token::Type::AS)) {
    //std::cout << "DEBUG: Found AS clause after CASE expression." << std::endl;
    consume(Token::Type::AS);

    if (match(Token::Type::STRING_LITERAL) || match(Token::Type::DOUBLE_QUOTED_STRING)) {
        alias = currentToken.lexeme;
        // Remove quotes
        if (alias.size() >= 2 &&
            ((alias[0] == '\'' && alias.back() == '\'') ||
             (alias[0] == '"' && alias.back() == '"'))) {
            alias = alias.substr(1, alias.size() - 2);
        }
        consume(currentToken.type);
        //std::cout << "DEBUG: Found string literal alias: " << alias << std::endl;
    } else if (match(Token::Type::IDENTIFIER)) {
        alias = currentToken.lexeme;
        consume(Token::Type::IDENTIFIER);
        //std::cout << "DEBUG: Found identifier alias: " << alias << std::endl;
    } else {
        throw ParseError(currentToken.line, currentToken.column,
                       "Expected column alias after AS");
    }
    }

    //std::cout << "DEBUG: Returning parsed Expressions." << std::endl;
    if (caseExpression) {
        if (!alias.empty()) {
            // Simple CASE with no alias: CASE expr WHEN value THEN result...
            return std::make_unique<AST::CaseExpression>(std::move(caseExpression),std::move(whenClauses),std::move(elseClause),alias);
        } else {
            return std::make_unique<AST::CaseExpression>(std::move(caseExpression),std::move(whenClauses),std::move(elseClause));
        }
    } else {
        if (!alias.empty()) {
            // SErced CASE: CASE WHEN condition THEN result...
            return std::make_unique<AST::CaseExpression>(std::move(whenClauses), std::move(elseClause),alias);
        } else {
            return std::make_unique<AST::CaseExpression>(std::move(whenClauses), std::move(elseClause));
        }
    }


    //std::cout << "DEBUG: Finished parsing CASE expression." << std::endl;
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


int Parse::getPrecedence(Token::Type type) {
    switch(type) {
        case Token::Type::OR: return 1;
        case Token::Type::AND: return 2;
        case Token::Type::NOT: return 3;
        case Token::Type::EQUAL:
        case Token::Type::NOT_EQUAL:
	    case Token::Type::IS:
        case Token::Type::IS_NULL:
        case Token::Type::IS_NOT_NULL:
        case Token::Type::IS_TRUE:
        case Token::Type::IS_NOT_TRUE:
        case Token::Type::IS_FALSE:
        case Token::Type::IS_NOT_FALSE:
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
        case Token::Type::IS_NULL:
        case Token::Type::IS_NOT_NULL:
        case Token::Type::IS_TRUE:
        case Token::Type::IS_NOT_TRUE:
        case Token::Type::IS_FALSE:
        case Token::Type::IS_NOT_FALSE:
	    case Token::Type::MOD:
            return true;
        case Token::Type::BETWEEN:
        case Token::Type::IN:
            return true; // These are handled specially but are still binary operators
        default:
            return false;
    }
}
