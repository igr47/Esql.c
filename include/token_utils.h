#ifndef TOKEN_UTILS_H
#define TOKEN_UTILS_H

#include "scanner.h"
#include <string>
#include <unordered_map>

class TokenUtils {
public:
    static std::string typeToString(Token::Type type);
    static std::string getTokenDescription(const Token& token);

private:
    static const std::unordered_map<Token::Type, std::string> typeMap;

    static void initializeTypeMap();
};

#endif
