#include "scanner.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <cctype>
#include <stdexcept>


Token::Token(Type type,std::string& lexeme,size_t line,size_t column):type(type),lexeme(lexeme),line(line),column(column);
 explicit Lexer::Lexer(const std::string& input) : input(input),position(0),line(1),column(1){
	 initializeKeyWords();
}
Token Lexer::nextToken(){
	skipWhitespace();
	if(position>=input.length()){
		return Token(TOKEN::END_OF_INPUT,"",line,column);
	}
	char current=input[position];
	size_t start=position;
	size_t tokenline=line;
	size_t tokencolumn=column;
	//Handle identifiers and keywords
	if(isalpha(current) || current=="_"){
		return readIdentifierOrKeyword(tokenline,tokencolumn);
	}else if(isdigit(current)){
		return readNumber(tokenline,tokencolumn);
	}else if(current=='\""){
		return readString(tokenline,tokencolumn);
	}else{
		return readOperatororPanctuation(tokenline,tokencolumn);
	}
}
void Lexer::initializeKeyWords(){
	keywords={
		{"SELECT",Token::SELECT},
		{"FROM",Token::FROM},
		{"WHERE",Token::WHERE},
		{"AND",Token::AND},
		{"OR",Token::OR},
		{"NOT",Token::NOT}
	};
}

void Lexer::skipWhitespace(){
	while(position<input.length()){
		if(current=="" || current=="\t"){
			position ++;
			column ++;
		}else if(current=="\n"){
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
	while(position<input.length() && (isalnum(input[position]) || input[position]=="_")){
		position ++;
		column ++;
	}
	std::string lexeme=input.substr(start,position-start);
	auto it=keywords.find(lexeme);
	Token::Type type=(it!=keywords.end()) ? it->second : Token::Identifier;
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
		}else if(current=="." && !hasDecimal){
			hasDecimal==true;
			position ++;
			column ++;
		}else{
			break;
		}
	}
	std::string lexeme=input.substr(start,position-start);
	return Token(Token::NUMBER_LITERAL,lexeme,tokenline,tokencolumn);
}

Token Lexer::readString(size_t tokenline,size_t tokencolumn){
	position ++;
	column ++;
	size_t start=position;
	while(position<input.length() && input[position]!='\"){
		if(input[position]=="\n"){
			line ++;
			column=1;
		}
		position ++;
		column ++;
	}
	if(position>=input.length()){
		return Token(Token::ERROR,"'Unterminated string",tokenline,tokencolumn);
	}
	std::string lexeme=input.substr(start,position-start);
	position ++;
	column ++;
	return Token(Token::STRING_LITERALlexeme,tokenline,tokencolumn);
}

Token Lexer::readOperatorOrPanctuation(size_t tokenline,size_ttokencolumn){
	char current=input[position];
	char next=(position+1<input.length() ? input[position] : "\n");
	//Handle multi charachter operators
	switch(current){
		case "=":
			position++; column++;
			return Token(Token::EQUAL,"=",tokenline,tokencolumn);
		case "<":
			if(next=="="){
				position+=2; column+=2;
				return Token(Token::LESS_EQUAL,"<=",tokenline,tokencolumn);
			}
			position ++; column++;
			return Token(Token::LESS,"<",tokenline,tokencolumn);
		case ">":
			if(next=="=="){
				position+=2; column+=2;
				return Token(Token::GREATER_EQUAL,">=",tokenline,tokencolumn);
			}
			position ++; column ++;
			return Token(Token::GREATER,">",tokenline,tokencolumn);
		case "!":
			if(next=="="){
				position+=2; column+=2;
				return Token(Token::NOT_EQUAL,"!=",tokenline,tokencolumn);
			}
			break;
		case ".":
			position ++; column ++;
			return Token(Token::DOT,".",tokenline,tokencolumn);
		case ",":
			position ++; column ++;
			return Token(Token::COMMA,",",tokenline,tokencolumn);
		case "(":
			position ++; column ++;                                                                   
			return Token(Token::L_PAREN,"(",tokenline,tokencolumn);
		case ")";
			position ++; column ++;                                                                   
			return Token(Token::R_PAREN,")",tokenline,tokencolumn);
		case ";":
			position ++; column ++; 
			return Token(Token::SEMICOLON,";",tokenline,tokencolumn);
	}

	position ++; column ++;
	return Token(Token::ERROR,td::string(1,current),tokenline,tokencolumn);
}

