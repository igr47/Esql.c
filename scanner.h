#ifndef SCANNER_H
#define SCANNER_H
#include <string>
#include <unordered_map>
#include <vector>

class Token{
	public:
		enum class Type{
			//Keywords
			SELECT,FROM,WHERE,AND,OR,NOT,UPDATE,SET,DROP,TABLE,DELETE,INSERT,INTO,ALTER,CREATE,ADD,RENAME,VALUES,BOOL,TEXT,INT,FLOAT,DATABASE,DATABASES,SHOW,USE,TABLES,TO,ROW,BULK,IN,COLUMN,BETWEEN,GROUP,BY,HAVING,ORDER,ASC,DESC,LIMIT,OFFSET,PRIMARY_KEY,NOT_NULL,
			//Identifier & Literals
			IDENTIFIER,STRING_LITERAL,NUMBER_LITERAL,DOUBLE_QUOTED_STRING,
			//conditiinals
			TRUE,FALSE,
			//OPERATORS
			EQUAL,NOT_EQUAL,LESS,LESS_EQUAL,GREATER,GREATER_EQUAL,ASTERIST,PLUS,MINUS,
			//Panctuation
			COMMA,DOT,SEMICOLON,L_PAREN,R_PAREN,COLON,SLASH,
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

