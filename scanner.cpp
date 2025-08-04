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
	}else if(current=='\"'){
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
		{"COLON",Token::Type::COLON},
		{"TRUE",Token::Type::TRUE},
		{"FALSE",Token::Type::FALSE},
		{"DATABASE",Token::Type::DATABASE},
		{"DATABASES",Token::Type::DATABASES},
		{"SHOW",Token::Type::SHOW},
		{"USE",Token::Type::USE},
		{"TABLES",Token::Type::TABLES},
		{"COMMA",Token::Type::COMMA}

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

Token Lexer::readString(size_t tokenline,size_t tokencolumn){
	position ++;
	column ++;
	size_t start=position;
	while(position<input.length() && input[position]!='\''){
		if(input[position]=='\n'){
			line++;
			column=1;
		}
		position++;
		column++;
	}
	if(position>=input.length()){
		return Token(Token::Type::ERROR,"'Unterminated string",tokenline,tokencolumn);
	}
	std::string lexeme=input.substr(start,position-start);
	position++;
	column++;
	return Token(Token::Type::STRING_LITERAL,lexeme,tokenline,tokencolumn);
}

Token Lexer::readOperatorOrPanctuation(size_t tokenline,size_t tokencolumn){
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
	}

	position++; column++;
	return Token(Token::Type::ERROR,std::string(1,current),tokenline,tokencolumn);
}

