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
		{"SELECT",Token::Type::SELECT}, {"ANALYZE", Token::Type::ANALYZE},
		{"FROM",Token::Type::FROM},     {"TYPE", Token::Type::TYPE},
		{"WHERE",Token::Type::WHERE},   {"TARGET", Token::Type::TARGET},
		{"AND",Token::Type::AND},       {"SUMMARY", Token::Type::SUMMARY},
		{"OR",Token::Type::OR},         {"CORRELATION", Token::Type::CORRELATION},
		{"NOT",Token::Type::NOT},       {"FEATURES", Token::Type::FEATURES},
		{"UPDATE",Token::Type::UPDATE}, {"QUALITY", Token::Type::QUALITY},
		{"SET",Token::Type::SET},       {"OUTPUT", Token::Type::OUTPUT},
		{"DELETE",Token::Type::DELETE}, {"JSON", Token::Type::JSON},
		{"TABLE",Token::Type::TABLE},   {"WITH", Token::Type::WITH},
		{"DROP",Token::Type::DROP},     {"COMPREHENSIVE", Token::Type::COMPREHENSIVE},
		{"INSERT",Token::Type::INSERT}, {"MARKDOWN", Token::Type::MARKDOWN},
		{"INTO",Token::Type::INTO},     {"MODEL", Token::Type::MODEL},
		{"CREATE",Token::Type::CREATE}, {"NUMERIC", Token::Type::NUMERIC},
		{"ALTER",Token::Type::ALTER},   {"CATEGORICAL", Token::Type::CATEGORICAL},
		{"ADD",Token::Type::ADD},       {"LIGHTGBM", Token::Type::LIGHTGBM},
		{"RENAME",Token::Type::RENAME}, {"REGRESSION", Token::Type::REGRESSION},
		{"VALUES",Token::Type::VALUES}, {"USING", Token::Type::USING},
		{"INT",Token::Type::INT},       {"TRAIN", Token::Type::TRAIN},
		{"BOOL",Token::Type::BOOL},     {"ON", Token::Type::ON},
		{"TEXT",Token::Type::TEXT},     {"SAVE", Token::Type::SAVE},
		{"FLOAT",Token::Type::FLOAT},   {"MODELS", Token::Type::MODELS},
        {"DATE",Token::Type::DATE},     {"DETAILED", Token::Type::DETAILED},
        {"DATETIME",Token::Type::DATETIME}, {"DESCRIBE", Token::Type::DESCRIBE},
        {"UUID",Token::Type::UUID},     {"EXTENDED", Token::Type::EXTENDED},
		{"COLON",Token::Type::COLON},   {"IF", Token::Type::IF},
		{"TRUE",Token::Type::TRUE},     {"EXISTS", Token::Type::EXISTS},
		{"FALSE",Token::Type::FALSE},   {"PREDICT", Token::Type::PREDICT},
		{"DATABASE",Token::Type::DATABASE}, {"CREATE_MODEL", Token::Type::CREATE_MODEL},
		{"DATABASES",Token::Type::DATABASES}, {"PROBABILITIES", Token::Type::PROBABILITIES},
		{"SHOW",Token::Type::SHOW},     {"AI_PREDICT", Token::Type::AI_PREDICT},
		{"USE",Token::Type::USE},       {"METRICS", Token::Type::METRICS},
		{"TABLES",Token::Type::TABLES}, {"FOR", Token::Type::FOR},
		{"COMMA",Token::Type::COMMA},   {"IMPORTANCE", Token::Type::IMPORTANCE},
		{"TO",Token::Type::TO},         {"TOP", Token::Type::TOP},
		{"BULK",Token::Type::BULK},     {"EXPLAIN", Token::Type::EXPLAIN},
		{"IN",Token::Type::IN},         {"SHAP_VALUES", Token::Type::SHAP_VALUES},
		{"ROW",Token::Type::ROW},       {"AI_PREDICT_PROBA", Token::Type::AI_PREDICT_PROBA},
		{"COLUMN",Token::Type::COLUMN}, {"AI_PREDICT_CLASS", Token::Type::AI_PREDICT_CLASS},
		{"BETWEEN",Token::Type::BETWEEN}, {"AI_PREDICT_VALUE", Token::Type::AI_PREDICT_VALUE},
		{"SLASH",Token::Type::SLASH},    {"AI_PREDICT_CLUSTER", Token::Type::AI_PREDICT_CLUSTER},
		{"PLUS",Token::Type::PLUS},     {"AI_PREDICT_ANOMALY", Token::Type::AI_PREDICT_ANOMALY},
		{"MINUS",Token::Type::MINUS},   {"AI_EXPLAIN", Token::Type::AI_EXPLAIN},
		{"GROUP",Token::Type::GROUP},   {"AI_TRAIN", Token::Type::AI_TRAIN},
		{"BY",Token::Type::BY},         {"AI_MODEL_METRICS", Token::Type::AI_MODEL_METRICS},
		{"ORDER",Token::Type::ORDER},   {"AI_FEATURE_IMPORTANCE", Token::Type::AI_FEATURE_IMPORTANCE},
		{"ASC",Token::Type::ASC},       {"AI_ANALYZE", Token::Type::AI_ANALYZE},
		{"HAVING",Token::Type::HAVING}, {"HYPERPARAMETERS", Token::Type::HYPERPARAMETERS},
		{"DESC",Token::Type::DESC},     {"SHOW_MODELS", Token::Type::SHOW_MODELS},
		{"LIMIT",Token::Type::LIMIT},   {"STATS", Token::Type::STATS},
		{"OFFSET",Token::Type::OFFSET}, {"LINE", Token::Type::LINE},
		{"PRIMARY_KEY",Token::Type::PRIMARY_KEY}, {"PLOT", Token::Type::PLOT},
		{"NOT_NULL",Token::Type::NOT_NULL},       {"TITLE", Token::Type::TITLE},
		{"AS" ,Token::Type::AS},        {"SCATTER", Token::Type::SCATTER},
		{"DISTINCT",Token::Type::DISTINCT}, {"BAR", Token::Type::BAR},
		{"COUNT", Token::Type::COUNT},      {"HISTOGRAM", Token::Type::HISTOGRAM},
		{"SUM",Token::Type::SUM},           {"BOXPLOT", Token::Type::BOXPLOT},
		{"AVG",Token::Type::AVG},       {"PIE", Token::Type::PIE},
		{"MIN",Token::Type::MIN},       {"HEATMAP", Token::Type::HEATMAP},
		{"MAX",Token::Type::MAX},       {"AREA", Token::Type::AREA},
		{"UNIQUE",Token::Type::UNIQUE}, {"STACKED_BAR", Token::Type::STACKED_BAR},
		{"DEFAULT",Token::Type::DEFAULT}, {"MULTI_LINE", Token::Type::MULTI_LINE},
		{"AUTO_INCREAMENT",Token::Type::AUTO_INCREAMENT}, {"VIOLIN", Token::Type::VIOLIN},
		{"IS",Token::Type::IS},         {"CONTOUR", Token::Type::CONTOUR},
		{"CHECK",Token::Type::CHECK},   {"SURFACE", Token::Type::SURFACE},
        {"CASE",Token::Type::CASE},     {"WIREFRAME", Token::Type::WIREFRAME},
        {"WHEN",Token::Type::WHEN},     {"HISTOGRAM_2D", Token::Type::HISTOGRAM_2D},
        {"THEN",Token::Type::THEN},     {"PARALLEL_COORDINATES", Token::Type::PARALLEL_COORDINATES},
        {"ELSE",Token::Type::ELSE},     {"RADAR", Token::Type::RADAR},
        {"END",Token::Type::END},       {"QUIVER", Token::Type::QUIVER},
        {"ROUND",Token::Type::ROUND},   {"STREAMPLOT", Token::Type::STREAMPLOT},
        {"LOWER",Token::Type::LOWER},   {"TREND", Token::Type::TREND},
        {"UPPER",Token::Type::UPPER},   {"TIME_SERIES", Token::Type::TIME_SERIES},
        {"SUBSTRING",Token::Type::SUBSTRING},  {"QQ_PLOT", Token::Type::QQ_PLOT},
        {"LIKE",Token::Type::LIKE},                   {"TUNE_HYPERPARAMETERS", Token::Type::TUNE_HYPERPARAMETERS},
        {"GENERATE_DATE",Token::Type::GENERATE_DATE}, {"RANDOM", Token::Type::RANDOM},
        {"GENERATE_DATE_TIME",Token::Type::GENERATE_DATE_TIME},  {"ITERATIONS", Token::Type::ITERATIONS},
        {"GENERATE_UUID",Token::Type::GENERATE_UUID},   {"FOLDS", Token::Type::FOLDS},
        {"STRUCTURE",Token::Type::STRUCTURE},         {"WITH_TUNING_GRID", Token::Type::WITH_TUNING_GRID},
        {"STDDEV", Token::Type::STDDEV},              {"CLUSTERING", Token::Type::CLUSTERING},
        {"VARIANCE", Token::Type::VARIANCE},          {"OUTLIER", Token::Type::OUTLIER},
        {"PERCENTILE_CONT", Token::Type::PERCENTILE_CONT}, {"DISTRIBUTION", Token::Type::DISTRIBUTION}, 
        {"CORR", Token::Type::CORR},                 {"QUALITY", Token::Type::QUALITY}, 
        {"REGR_SLOPE", Token::Type::REGR_SLOPE},     {"FORECAST", Token::Type::FORECAST},
        {"YEAR", Token::Type::YEAR},             {"COLUMNS", Token::Type::COLUMNS},
        {"MONTH", Token::Type::MONTH},           {"HORIZON", Token::Type::HORIZON},
        {"DAY", Token::Type::DAY},               {"CONFIDENCE", Token::Type::CONFIDENCE},
        {"NOW", Token::Type::NOW},               {"SCENARIOs", Token::Type::SCENARIOS},
        {"JULIANDAY", Token::Type::JULIANDAY},
        {"ROW_NUMBER", Token::Type::ROW_NUMBER},
        {"RANK", Token::Type::RANK},
        {"DENSE_RANK", Token::Type::DENSE_RANK},
        {"NTILE", Token::Type::NTILE},
        {"LAG", Token::Type::LAG},
        {"LEAD", Token::Type::LEAD},
        {"FIRST_VALUE", Token::Type::FIRST_VALUE},
        {"LAST_VALUE", Token::Type::LAST_VALUE},
        {"OVER", Token::Type::OVER},
        {"PARTITION", Token::Type::PARTITION},
        {"WITHIN", Token::Type::WITHIN},
        {"SUBSTR", Token::Type::SUBSTR},
        {"CONCAT", Token::Type::CONCAT},
        {"LENGTH", Token::Type::LENGTH},
        {"UPPER", Token::Type::UPPER},
        {"LOAD", Token::Type::LOAD},
        {"DATA", Token::Type::DATA},
        {"INFILE", Token::Type::INFILE},
        {"LOCAL", Token::Type::LOCAL},
        {"HEADER", Token::Type::HEADER},
        {"DELIMITER", Token::Type::DELIMITER},
        {"RESIDUALS", Token::Type::RESIDUALS},
        {"ANIMATION", Token::Type::ANIMATION},
        {"INTERACTIVE", Token::Type::INTERACTIVE},
        {"DASHBOARD", Token::Type::DASHBOARD},
        {"GEO_MAP", Token::Type::GEO_MAP},
        {"CANDLESTICK", Token::Type::CANDLESTICK},
        {"PARALLEL_COORDINATES", Token::Type::PARALLEL_COORDINATES},
        {"RADAR", Token::Type::RADAR},
        {"QUIVER", Token::Type::QUIVER},
        {"STREAMPLOT", Token::Type::STREAMPLOT},
        {"PNG", Token::Type::PNG},
        {"PDF", Token::Type::PDF},
        {"SVG", Token::Type::SVG},
        {"EPS", Token::Type::EPS},
        {"JPG", Token::Type::JPG},
        {"JPEG", Token::Type::JPEG},
        {"GIF", Token::Type::GIF},
        {"MP4", Token::Type::MP4},
        {"HTML", Token::Type::HTML},
        {"XLABEL", Token::Type::XLABEL},
        {"YLABEL", Token::Type::YLABEL},
        {"ZLABEL", Token::Type::ZLABEL},
        {"X_LABEL", Token::Type::X_LABEL},
        {"Y_LABEL", Token::Type::Y_LABEL},
        {"Z_LABEL", Token::Type::Z_LABEL},
        {"SERIES", Token::Type::SERIES},
        {"SERIES_NAMES", Token::Type::SERIES_NAMES},
        {"ANIMATE", Token::Type::ANIMATE},
        {"ANIMATION_COLUMN", Token::Type::ANIMATION_COLUMN},
        {"FPS", Token::Type::FPS},
        {"FRAMES_PER_SECOND", Token::Type::FRAMES_PER_SECOND},
        {"DASHBOARD_LAYOUT", Token::Type::DASHBOARD_LAYOUT},
        {"LAYOUT", Token::Type::LAYOUT},
        {"FOR_X", Token::Type::FOR_X},
        {"X", Token::Type::X},
        {"Y", Token::Type::Y},
	{"GEO_MAP", Token::Type::GEO_MAP},
	{"MAP", Token::Type::MAP},
	{"GEO_SCATTER",Token::Type::GEO_SCATTER},
	{"GEO_HEATMAP", Token::Type::GEO_HEATMAP},
	{"GEO_CHOROPLETH", Token::Type::GEO_CHOROPLETH},
	{"CHOROPLETH",  Token::Type::CHOROPLETH},
	{"GEO_BUBBLE", Token::Type::GEO_BUBBLE},
	{"GEO_LINE", Token::Type::GEO_LINE},
	{"GEO_CONTOUR", Token::Type::GEO_CONTOUR},
	{"GEO_POLYGON", Token::Type::GEO_POLYGON},
	{"GEO_GRID", Token::Type::GEO_GRID},
	{"GEO_FLOW", Token::Type::GEO_FLOW},
	{"LATITUDE", Token::Type::LATITUDE},
	{"LAT", Token::Type::LAT},
	{"LON", Token::Type::LON},
	{"LONGITUDE", Token::Type::LONGITUDE},
	{"REGION", Token::Type::REGION},
	{"VALUE",Token::Type::VALUE},
	{"VALUES",Token::Type::VALUES}
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
        case '[':
            position++; column++;
            return Token(Token::Type::R_BRACKET,"[",tokenline,tokencolumn);
        case ']':
            position++; column++;
            return Token(Token::Type::L_BRACKET,"]",tokenline,tokencolumn);
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
