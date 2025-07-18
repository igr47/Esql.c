#ifndef SCANNER_H
#define SCANNER_H
#include <string>
#include <unordered_map>
#include <vector>

class Token{
	public:
		enum class Type{
			//Keywords
			SELECT,FROM,WHERE,AND,OR,NOT,UPDATE,SET,DROP,TABLE,DELETE,INSERT,INTO,ALTER,CREATE,
			//Identifier & Literals
			IDENTIFIER,STRING_LITERAL,NUMBER_LITERAL,
			//OPERATORS
			EQUAL,NOT_EQUAL,LESS,LESS_EQUAL,GREATER,GREATER_EQUAL,
			//Panctuation
			COMMA,DOT,SEMICOLON,L_PAREN,R_PAREN,
			//Special
			END_OF_INPUT,ERROR
		};
		Type type;
		std::string lexeme;
		size_t line;
		size_t column;
		Token(Type type,const std::string& lexeme,size_t line,size_t column);
};
class Lexer{
	public:
		explicit Lexer(const std::string& input);
		Token nextToken();
	private:
		const std::string input;
		size_t position;
		size_t line;
		size_t column;
		std::unordered_map<std::string,Token::Type> keywords;

		//I initialize the keywords
		void initializeKeyWords();
		void skipWhitespace();
		Token readIdentifierOrKeyword(size_t tokenline,size_t tokencolumn);
		Token readNumber(size_t tokenline,size_t tokencolumn);
		Token readString(size_t tokenline,size_t tokencolumn);
		Token readOperatorOrPanctuation(size_t tokenline,size_t tokencolumn);
};
#endif

