#include "scanner.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <cctype>
#include <stdexcept>


Token::Token(Type type,const std::string& lexeme,size_t line,size_t column):type(type),lexeme(lexeme),line(line),column(column){}
Lexer::Lexer(const std::string& input) : input(input),position(0),line(1),column(1){
	 initializeKeyWords();
}
Token Lexer::nextToken(){
	skipWhitespace();
	if(position>=input.length()){
		return Token(Token::Type::END_OF_INPUT,"",line,column);
	}
	char current=input[position];
	size_t start=position;
	size_t tokenline=line;
	size_t tokencolumn=column;
	//Handle identifiers and keywords
	if(isalpha(current) || current=='_'){
		return readIdentifierOrKeyword(tokenline,tokencolumn);
	}else if(isdigit(current)){
		return readNumber(tokenline,tokencolumn);
	}else if(current=='\'' || current=='"'){
		return readString(tokenline,tokencolumn);
	}else{
		return readOperatorOrPanctuation(tokenline,tokencolumn);
	}
}
void Lexer::initializeKeyWords(){
	keywords={
		{"SELECT",Token::Type::SELECT},
		{"FROM",Token::Type::FROM},
		{"WHERE",Token::Type::WHERE},
		{"AND",Token::Type::AND},
		{"OR",Token::Type::OR},
		{"NOT",Token::Type::NOT},
		{"UPDATE",Token::Type::UPDATE},
		{"SET",Token::Type::SET},
		{"DELETE",Token::Type::DELETE},
		{"TABLE",Token::Type::TABLE},
		{"DROP",Token::Type::DROP},
		{"INSERT",Token::Type::INSERT},
		{"INTO",Token::Type::INTO},
		{"CREATE",Token::Type::CREATE},
		{"ALTER",Token::Type::ALTER},
		{"ADD",Token::Type::ADD},
		{"RENAME",Token::Type::RENAME},
		{"VALUES",Token::Type::VALUES},
		{"INT",Token::Type::INT},
		{"BOOL",Token::Type::BOOL},
		{"TEXT",Token::Type::TEXT},
		{"FLOAT",Token::Type::FLOAT},
        {"DATE",Token::Type::DATE},
        {"DATETIME",Token::Type::DATETIME},
        {"UUID",Token::Type::UUID},
		{"COLON",Token::Type::COLON},
		{"TRUE",Token::Type::TRUE},
		{"FALSE",Token::Type::FALSE},
		{"DATABASE",Token::Type::DATABASE},
		{"DATABASES",Token::Type::DATABASES},
		{"SHOW",Token::Type::SHOW},
		{"USE",Token::Type::USE},
		{"TABLES",Token::Type::TABLES},
		{"COMMA",Token::Type::COMMA},
		{"TO",Token::Type::TO},
		{"BULK",Token::Type::BULK},
		{"IN",Token::Type::IN},
		{"ROW",Token::Type::ROW},
		{"COLUMN",Token::Type::COLUMN},
		{"BETWEEN",Token::Type::BETWEEN},
		{"SLASH",Token::Type::SLASH},
		{"PLUS",Token::Type::PLUS},
		{"MINUS",Token::Type::MINUS},
		{"GROUP",Token::Type::GROUP},
		{"BY",Token::Type::BY},
		{"ORDER",Token::Type::ORDER},
		{"ASC",Token::Type::ASC},
		{"HAVING",Token::Type::HAVING},
		{"DESC",Token::Type::DESC},
		{"LIMIT",Token::Type::LIMIT},
		{"OFFSET",Token::Type::OFFSET},
		{"PRIMARY_KEY",Token::Type::PRIMARY_KEY},
		{"NOT_NULL",Token::Type::NOT_NULL},
		{"AS" ,Token::Type::AS},
		{"DISTINCT",Token::Type::DISTINCT},
		{"COUNT", Token::Type::COUNT},
		{"SUM",Token::Type::SUM},
		{"AVG",Token::Type::AVG},
		{"MIN",Token::Type::MIN},
		{"MAX",Token::Type::MAX},
		{"UNIQUE",Token::Type::UNIQUE},
		{"DEFAULT",Token::Type::DEFAULT},
		{"AUTO_INCREAMENT",Token::Type::AUTO_INCREAMENT},
		{"IS",Token::Type::IS},
		{"CHECK",Token::Type::CHECK},
        {"CASE",Token::Type::CASE},
        {"WHEN",Token::Type::WHEN},
        {"THEN",Token::Type::THEN},
        {"ELSE",Token::Type::ELSE},
        {"END",Token::Type::END},
        {"ROUND",Token::Type::ROUND},
        {"LOWER",Token::Type::LOWER},
        {"UPPER",Token::Type::UPPER},
        {"SUBSTRING",Token::Type::SUBSTRING},
        {"LIKE",Token::Type::LIKE},
        {"GENERATE_DATE",Token::Type::GENERATE_DATE},
        {"GENERATE_DATE_TIME",Token::Type::GENERATE_DATE_TIME},
        {"GENERATE_UUID",Token::Type::GENERATE_UUID},
        {"STRUCTURE",Token::Type::STRUCTURE}
	};
}

void Lexer::skipWhitespace(){
	while(position<input.length()){
	        char current=input[position];
		if(current==' ' || current=='\t'){
			position ++;
			column ++;
		}else if(current=='\n'){
			position ++;
			line ++;
			column=1;
		}else{
			break;
		}
	}
}

Token Lexer::readIdentifierOrKeyword(size_t tokenline,size_t tokencolumn){
	size_t start=position;
	while(position<input.length() && (isalnum(input[position]) || input[position]=='_')){
		position ++;
		column ++;
	}
	std::string lexeme=input.substr(start,position-start);
	auto it=keywords.find(lexeme);
	Token::Type type=(it!=keywords.end()) ? it->second : Token::Type::IDENTIFIER;
	return Token(type,lexeme,tokenline,tokencolumn);
}
Token Lexer::readNumber(size_t tokenline,size_t tokencolumn){
	size_t start=position;
	bool hasDecimal=false;
	while(position<input.length()){
		char current=input[position];
		if(isdigit(current)){
			position ++;
			column ++;
		}else if(current=='.' && !hasDecimal){
			hasDecimal=true;
			position ++;
			column ++;
		}else{
			break;
		}
	}
	std::string lexeme=input.substr(start,position-start);
	return Token(Token::Type::NUMBER_LITERAL,lexeme,tokenline,tokencolumn);
}

Token Lexer::readString(size_t tokenline, size_t tokencolumn) {
    char quote_char = input[position];
    position++;
    column++;
    
    std::string value;
    bool escape_next = false;
    
    while (position < input.length()) {
        char current = input[position];
        
        if (!escape_next && current == quote_char) {
            // Found closing quote
            position++;
            column++;
            Token::Type type = (quote_char == '\'') ? 
                Token::Type::STRING_LITERAL : 
                Token::Type::DOUBLE_QUOTED_STRING;
            return Token(type, value, tokenline, tokencolumn);
        }
        
        if (escape_next) {
            switch (current) {
                case 'n': value += '\n'; break;
                case 't': value += '\t'; break;
                case 'r': value += '\r'; break;
                case '\\': value += '\\'; break;
                case '\'': value += '\''; break;
                case '"': value += '"'; break;
                default:
                    value += '\\';
                    value += current;
                    break;
            }
            escape_next = false;
        } 
        else if (current == '\\') {
            escape_next = true;
        } 
        else {
            value += current;
        }
        
        position++;
        if (current == '\n') {
            line++;
            column = 1;
        } else {
            column++;
        }
    }
    
    // If we get here, the string was unterminated
    return Token(Token::Type::ERROR, "Unterminated string", tokenline, tokencolumn);
}

/*Token Lexer::readOperatorOrPanctuation(size_t tokenline,size_t tokencolumn){
	char current=input[position];
	char next=(position+1<input.length() ? input[position] : '\n');
	//Handle multi charachter operators
	switch(current){
		case '=':
			position++; column++;
			return Token(Token::Type::EQUAL,"=",tokenline,tokencolumn);
		case '<':
			if(next=='='){
				position+=2; column+=2;
				return Token(Token::Type::LESS_EQUAL,"<=",tokenline,tokencolumn);
			}
			position ++; column++;
			return Token(Token::Type::LESS,"<",tokenline,tokencolumn);
		case '>':
			if(next=='='){
				position+=2; column+=2;
				return Token(Token::Type::GREATER_EQUAL,">=",tokenline,tokencolumn);
			}
			position++; column++;
			return Token(Token::Type::GREATER,">",tokenline,tokencolumn);
		case '!':
			if(next=='='){
				position+=2; column+=2;
				return Token(Token::Type::NOT_EQUAL,"!=",tokenline,tokencolumn);
			}
			break;
		case '.':
			position++; column++;
			return Token(Token::Type::DOT,".",tokenline,tokencolumn);
		case ',':
			position++; column++;
			return Token(Token::Type::COMMA,",",tokenline,tokencolumn);
		case '(':
			position++; column++;                                                                   
			return Token(Token::Type::L_PAREN,"(",tokenline,tokencolumn);
		case ')':
			position++; column++;                                                                   
			return Token(Token::Type::R_PAREN,")",tokenline,tokencolumn);
		case ';':
			position++; column++; 
			return Token(Token::Type::SEMICOLON,";",tokenline,tokencolumn);
		case ':':
			position++; column++;
			return Token(Token::Type::COLON,":",tokenline,tokencolumn);
		case '*':
			position++; column++;
			return Token(Token::Type::ASTERIST,"*",tokenline,tokencolumn);
	}

	position++; column++;
	return Token(Token::Type::ERROR,std::string(1,current),tokenline,tokencolumn);
}*/

Token Lexer::readOperatorOrPanctuation(size_t tokenline,size_t tokencolumn){
    char current=input[position];
    char next=(position+1<input.length() ? input[position+1] : '\0'); // Fixed: position+1

    //Handle multi character operators
    switch(current){
        case '=':
            position++; column++;
            return Token(Token::Type::EQUAL,"=",tokenline,tokencolumn);
        case '<':
            if(next=='='){
                position+=2; column+=2;
                return Token(Token::Type::LESS_EQUAL,"<=",tokenline,tokencolumn);
            }
            position ++; column++;
            return Token(Token::Type::LESS,"<",tokenline,tokencolumn);
        case '>':
            if(next=='='){
                position+=2; column+=2;
                return Token(Token::Type::GREATER_EQUAL,">=",tokenline,tokencolumn);
            }
            position++; column++;
            return Token(Token::Type::GREATER,">",tokenline,tokencolumn);
        case '!':
            if(next=='='){
                position+=2; column+=2;
                return Token(Token::Type::NOT_EQUAL,"!=",tokenline,tokencolumn);
            }
            break;
        case '.':
            position++; column++;
            return Token(Token::Type::DOT,".",tokenline,tokencolumn);
        case ',':
            position++; column++;
            return Token(Token::Type::COMMA,",",tokenline,tokencolumn);
        case '(':
            position++; column++;
            return Token(Token::Type::L_PAREN,"(",tokenline,tokencolumn);
        case ')':
            position++; column++;
            return Token(Token::Type::R_PAREN,")",tokenline,tokencolumn);
        case ';':
            position++; column++;
            return Token(Token::Type::SEMICOLON,";",tokenline,tokencolumn);
        case ':':
            position++; column++;
            return Token(Token::Type::COLON,":",tokenline,tokencolumn);
        case '*':
            position++; column++;
            return Token(Token::Type::ASTERIST,"*",tokenline,tokencolumn);
	case '+':
	    position++; column++;
	    return Token(Token::Type::PLUS,"+",tokenline,tokencolumn);
        case '-':
	    position++; column++;
	    return Token(Token::Type::MINUS,"-",tokenline,tokencolumn);
	case '/':
	    position++; column++;
	    return Token(Token::Type::SLASH,"/",tokenline,tokencolumn);
    }

    position++; column++;
    return Token(Token::Type::ERROR,std::string(1,current),tokenline,tokencolumn);
}
