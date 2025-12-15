#ifndef PARSER_H
#define PARSER_H
#include "scanner.h"
#include "token_utils.h"
#include <stdexcept>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>
#include <cstdint>

namespace AST{
	class Node{
		public:
			virtual ~Node()=default;
	};
	class Statement:public Node{
		public:
			virtual ~Statement()=default;
	};
	class Expression:public Node{
		public:
			virtual std::unique_ptr<Expression> clone() const =0;
			virtual ~Expression()=default;
			virtual std::string toString() const=0;
	};

	class BetweenOp : public Expression{
		public:
			std::unique_ptr<Expression> column;
			std::unique_ptr<Expression> lower;
			std::unique_ptr<Expression> upper;
			std::unique_ptr<Expression> clone() const override{
				return std::make_unique<BetweenOp>(column->clone(),lower->clone(),upper->clone());
			}
			BetweenOp(std::unique_ptr<Expression> col, std::unique_ptr<Expression> low,std::unique_ptr<Expression> up) : column(std::move(col)),lower(std::move(low)),upper(std::move(up)){}
			std::string toString() const override{
				return column->toString() + "BETWEEN" +lower->toString()+ "AND" +upper->toString();
			}
	};

	class HavingClause : public Expression{
		public:
			std::unique_ptr<Expression> condition;
			//std::unique_ptr<Expression> operators;
			//std::unique_ptr<Expression> values;
			std::unique_ptr<Expression> clone() const override{
				return std::make_unique<HavingClause>(condition->clone());
			}

			//explicit HavingClause(std::unique_ptr<Expression> cond) : condition(std::move(cond)) {}
			HavingClause(std::unique_ptr<Expression> cond) : condition(std::move(cond)){}
			std::string toString() const override {
				return "HAVING "+ condition->toString();
			}
	};

	class OrderByClause : public Expression{
		public:
			std::vector<std::pair<std::unique_ptr<Expression>,bool>> columns;
			std::unique_ptr<Expression> clone() const override{
				std::vector<std::pair<std::unique_ptr<Expression>, bool>> clonedColumns;
				for(const auto& col : columns){
					clonedColumns.emplace_back(col.first->clone(),col.second);
				}
				return std::make_unique<OrderByClause>(std::move(clonedColumns));
			}
			explicit OrderByClause(std::vector<std::pair<std::unique_ptr<Expression>, bool>> cols) : columns(std::move(cols)) {}
			std::string toString() const override{
				std::string result = "ORDER BY ";
				for(size_t i = 0 ; i<columns.size(); i++){
					result += columns[i].first->toString();
					if(!columns[i].second) result += "DESC";
					if(i<columns.size()-1) result += ",";
				}

				return result;
			}
	};

	class GroupByClause : public Expression{
		public:
			std::vector<std::unique_ptr<Expression>> columns;
			std::unique_ptr<Expression> clone() const override{
				std::vector<std::unique_ptr<Expression>> clonedColumns;
				for (const auto& col : columns){
					clonedColumns.push_back(col->clone());
				}
				return std::make_unique<GroupByClause>(std::move(clonedColumns));
			}

			explicit GroupByClause(std::vector<std::unique_ptr<Expression>> cols) : columns(std::move(cols)) {}
			std::string toString() const override{
				std::string result = "GROUP BY " ;
				for(size_t i =0; i<columns.size();++i){
					result += columns[i]->toString();
					if(i<columns.size() - 1) result += "'";
				}
				return result;
			}
	};


	class InOp : public Expression{
		public:
			std::unique_ptr<Expression> column;
			std::vector<std::unique_ptr<Expression>> values;
			std::unique_ptr<Expression> clone() const override{
				std::vector<std::unique_ptr<Expression>> clonedValues;
				for(const auto& val : values){
					clonedValues.push_back(val->clone());
				}
				return std::make_unique<InOp> (column->clone(), std::move(clonedValues));
			}
			InOp(std::unique_ptr<Expression> col, std::vector<std::unique_ptr<Expression>> vals) : column(std::move(col)), values(std::move(vals)){}

			std::string toString() const override {
				std::string result = column->toString() + "IN(" ;
				for(size_t i=0; i<values.size(); ++i){
					result += values[i]->toString();
					if(i<values.size()-1) result += ",";
				}
				return result += ")";
			}
	};

	class NotOp : public Expression{
		public:
			std::unique_ptr<Expression> expr;
			std::unique_ptr<Expression> clone() const override{
				return std::make_unique<NotOp> (expr->clone());
			}

			 explicit NotOp(std::unique_ptr<Expression> e) : expr(std::move(e)) {}
			 std::string toString() const override{
				 return "NOT " +expr->toString();
			}
	};

    class CharClass : public Expression {
        public:
            std::string pattern;

            std::unique_ptr<Expression> clone() const override {
                return std::make_unique<CharClass>(pattern);
            }

            explicit CharClass(const std::string& p) : pattern(p) {}

            std::string toString() const override {
                return "[" + pattern + "]";
            }
    };

    class LikeOp : public Expression{
        public:
            std::unique_ptr<Expression> left;
            std::unique_ptr<Expression> right;

            std::unique_ptr<Expression> clone() const override {
                return std::make_unique<LikeOp>(left->clone(), right->clone());
            }

            LikeOp(std::unique_ptr<Expression> l, std::unique_ptr<Expression> r) : left(std::move(l)), right(std::move(r)) {}

            std::string toString() const override {
                return left->toString() + " LIKE " + right->toString();
            }
    };

    class CaseExpression : public Expression {
        public:
            std::unique_ptr<Expression> caseExpression; // For simple case
            std::vector<std::pair<std::unique_ptr<Expression>, std::unique_ptr<Expression>>> whenClauses;
            std::unique_ptr<Expression> elseClause;
            std::string alias;

            std::unique_ptr<Expression> clone() const override {
                std::vector<std::pair<std::unique_ptr<Expression>, std::unique_ptr<Expression>>> clonedWhens;
                for (const auto& when : whenClauses) {
                    clonedWhens.emplace_back(when.first->clone(), when.second->clone());
                }
                return std::make_unique<CaseExpression>(caseExpression ? caseExpression->clone() : nullptr, std::move(clonedWhens),elseClause ? elseClause->clone() : nullptr, alias);
            }

            CaseExpression(std::vector<std::pair<std::unique_ptr<Expression>, std::unique_ptr<Expression>>> whens,std::unique_ptr<Expression> elseExpr, const std::string& alias = "") : caseExpression(nullptr),whenClauses(std::move(whens)), elseClause(std::move(elseExpr)), alias(alias) {}

            // Construction for simple case
            CaseExpression(std::unique_ptr<Expression> caseExpr,std::vector<std::pair<std::unique_ptr<Expression>, std::unique_ptr<Expression>>> whens,std::unique_ptr<Expression> elseExpr, const std::string& alias = "") : caseExpression(std::move(caseExpr)), whenClauses(std::move(whens)), elseClause(std::move(elseExpr)), alias(alias) {}

            std::string toString() const override {
                std::string result = "CASE";
                if (caseExpression) {
                    result += " " + caseExpression->toString();
                }
                for (const auto& when : whenClauses) {
                    result += "WHEN" + when.first->toString() + "THEN" + when.second->toString() + " ";
                }
                if (elseClause) {
                    result += "ELSE" + elseClause->toString() + " ";
                }
                result += "END";
                if (!alias.empty()) {
                    result += " AS " + alias;
                }
                return result;
                //return result + "END";
            }
    };

    class FunctionCall : public Expression {
        public:
            Token function;
            std::vector<std::unique_ptr<Expression>> arguments;
            std::unique_ptr<Expression> alias;

            std::unique_ptr<Expression> clone() const override {
                std::vector<std::unique_ptr<Expression>> clonedArgs;
                for (const auto& arg : arguments) {
                    clonedArgs.push_back(arg->clone());
                }
                return std::make_unique<FunctionCall>(function,std::move(clonedArgs),alias ? alias->clone() : nullptr);
            }

            FunctionCall(Token func, std::vector<std::unique_ptr<Expression>> args, std::unique_ptr<Expression> al = nullptr) : function(func), arguments(std::move(args)),  alias(std::move(al)) {}

            std::string toString() const override {
                std::string result = function.lexeme + "(";
                for (size_t i = 0; i < arguments.size(); ++i) {
                    result += arguments[i]->toString();
                    if (i < arguments.size() - 1) result += ", ";
                }
                 result += ")";
                 if (alias) {
                     result += " AS " + alias->toString();
                 }
                 return result;
            }
    };


	class Literal:public Expression{
		public:
			Token token;
			std::unique_ptr<Expression> clone() const override{
				return std::make_unique<Literal>(token);
			}
			//Literal()=default;
			explicit Literal(const Token& token);
			std::string toString() const override{
				return token.lexeme;
			}
	};

	class Identifier:public Expression{
		public:
			Token token;
			std::unique_ptr<Expression> clone() const override {
				return std::make_unique<Identifier>(token);
			}
			explicit Identifier(const Token& token);
			std::string toString() const override{
				return token.lexeme;
			}
	};

	class BinaryOp:public Expression{
		public:
			Token op;
			std::unique_ptr<Expression> clone() const override {
				return std::make_unique<BinaryOp>(op, left->clone(),right->clone(),alias ? alias->clone() : nullptr);
			}
			std::unique_ptr<Expression> left;
			std::unique_ptr<Expression> right;
            std::unique_ptr<Expression> alias;

			BinaryOp(Token op,std::unique_ptr<Expression> left,std::unique_ptr<Expression> right,std::unique_ptr<Expression> al = nullptr);
			std::string toString() const override{
                std::string result = left->toString()+" "+op.lexeme+" "+right->toString();
            if (alias) {
                result += " AS " + alias->toString();
            }
            return result;
			}
	};
	class ColumnReference : public Expression {
		public:
			std::string columnName;

			std::unique_ptr<Expression> clone() const override{
				return std::make_unique<ColumnReference>(columnName);
			}
			explicit ColumnReference (const std::string& name) : columnName(name) {}

			std::string toString() const override{
				return columnName;
			}
	};
	class ColumnDefination:public Node{
		public:
		        std::string name;
		        std::string type;
		        std::vector<std::string> constraints;
			std::string defaultValue;
			bool autoIncreament;
			std::string checkExpression;

			bool hasConstraint(const std::string& constraint) const {
				return std::find(constraints.begin(),constraints.end(), constraint) != constraints.end();
			}
	};
	class CreateDatabaseStatement:public Statement{
		public:
			std::string dbName;
	};
	class UseDatabaseStatement:public Statement{
		public:
			std::string dbName;
	};
	class ShowDatabaseStatement:public Statement{
		public:
	};
	class ShowTableStatement:public Statement{
		public:
            std::string tableName;
	};
    class ShowTableStructureStatement : public Statement {
        public:
            std::string tableName;
    };
    class ShowDatabaseStructure : public Statement {
        public:
            std::string dbName;
    };

    class SelectStatement;

    class CommonTableExpression {
        public:
            std::string name;
            std::vector<std::string> columns;
            std::unique_ptr<Statement> query;
    };


    class WithClause {
        public:
            std::vector<CommonTableExpression> ctes;
            WithClause(std::vector<CommonTableExpression> c) : ctes(std::move(c)) {}
    };

    // Join Clause
    class JoinClause {
        public:
            enum Type { INNER, LEFT, RIGHT, FULL };

            Type type = INNER;
            std::unique_ptr<Expression> table;
            std::unique_ptr<Expression> condition;

            JoinClause() = default;
            JoinClause(Type t, std::unique_ptr<Expression> tab, std::unique_ptr<Expression> cond)
                : type(t), table(std::move(tab)), condition(std::move(cond)) {}
    };


	class SelectStatement:public Statement{
		public:
			std::vector<std::unique_ptr<Expression>> columns;
			std::vector<std::pair<std::unique_ptr<Expression>,std::string>> newCols;
			std::vector<std::string> check;
			std::unique_ptr<Expression> from;
			std::unique_ptr<Expression> where;
			std::unique_ptr<GroupByClause> groupBy;
			std::unique_ptr<HavingClause> having;
			std::unique_ptr<OrderByClause> orderBy;
			std::unique_ptr<Expression> limit;
			std::unique_ptr<Expression> offset;
			bool distinct = false;
            std::unique_ptr<WithClause> withClause;
            std::vector<std::unique_ptr<JoinClause>> joins;
	};
	class AggregateExpression : public Expression{
		public:
			Token function;
			std::unique_ptr<Expression> argument;
			std::unique_ptr<Expression> argument2;
			bool isCountAll = false;


			std::unique_ptr<Expression> clone() const override {
				return std::make_unique<AggregateExpression>(function,argument ? argument->clone() : nullptr , argument2 ? argument2->clone() : nullptr ,isCountAll);
			}

			AggregateExpression(Token func,std::unique_ptr<Expression> arg,std::unique_ptr<Expression> arg2 = nullptr ,bool countAll = false) : function(func) , argument(std::move(arg)),argument2(std::move(arg2)),isCountAll(countAll){}
			std::string toString() const override {
				std::string result=function.lexeme+ "(";

				if(isCountAll){
					//return function.lexeme + "(*)";
					result += "*";
				}else if(argument){
					result += argument->toString();
				}
				//return function.lexeme + "(" + (argument ? argument-> toString() : "") + ")";
				result += ")";
				//Add alias ifvargument 2 exists(Tpically an AS clause)
				if(argument2){
					result += "AS" + argument2->toString();
				}
				return result;
			}
	};

    class WindowFunction : public Expression {
        public:
            Token function;
            std::unique_ptr<Expression> argument;
            std::vector<std::unique_ptr<Expression>> partitionBy;
            std::vector<std::pair<std::unique_ptr<Expression>, bool>> orderBy;
            std::unique_ptr<Expression> nTileValue;
            std::unique_ptr<Expression> alias;

            std::unique_ptr<Expression> clone() const override {
                std::vector<std::unique_ptr<Expression>> clonedPartition;
                for (const auto& expr : partitionBy) {
                    clonedPartition.push_back(expr->clone());
                }
                std::vector<std::pair<std::unique_ptr<Expression>, bool>> clonedOrder;
                for (const auto& expr : orderBy) {
                    clonedOrder.emplace_back(expr.first->clone(), expr.second);
                }

                return std::make_unique<WindowFunction>(
                        function,
                        argument ? argument->clone() : nullptr,
                        std::move(clonedPartition),
                        std::move(clonedOrder),
                        nTileValue ? nTileValue->clone() : nullptr,
                        alias ? alias->clone() : nullptr
                        );
            }

            WindowFunction(Token func, std::unique_ptr<Expression> arg, std::vector<std::unique_ptr<Expression>> partition, std::vector<std::pair<std::unique_ptr<Expression>, bool>> order, std::unique_ptr<Expression> nTile = nullptr, std::unique_ptr<Expression> al = nullptr) : function(func), argument(std::move(arg)),partitionBy(std::move(partition)), orderBy(std::move(order)), nTileValue(std::move(nTile)),alias(std::move(al)) {}

            std::string toString() const override {
                std::string result = function.lexeme + "(";
                if (argument) result += argument->toString();
                result += ") OVER (";

                if (!partitionBy.empty()) {
                    result += "PARTITION BY";
                    for (size_t i = 0; i < partitionBy.size(); ++i) {
                        if(i > 0) result += ", ";
                        result += partitionBy[i]->toString();
                    }
                }

                if (!orderBy.empty()) {
                    if (!partitionBy.empty()) result += " ";
                    result += "ORDER BY ";
                    for (size_t i = 0; i < orderBy.size(); ++i) {
                        if (i > 0) result += ", ";
                        result += orderBy[i].first->toString();
                        result += orderBy[i].second ? " ASC" : " DESC";
                    }
                }

                result += ")";
                if (alias) {
                    result += " AS " + alias->toString();
                }
                return result;
            }
    };

    // Date Function
    class DateFunction : public Expression {
        public:
            Token function;
            std::unique_ptr<Expression> argument;
            std::unique_ptr<Expression> alias;

            std::unique_ptr<Expression> clone() const override {
                return std::make_unique<DateFunction>(function, argument->clone(), alias ? alias->clone() : nullptr);
            }

            DateFunction(Token func, std::unique_ptr<Expression> arg,std::unique_ptr<Expression> al = nullptr) : function(func), argument(std::move(arg)), alias(std::move(al)) {}

            std::string toString() const override {
                std::string result = function.lexeme + "(" + argument->toString() + ")";
                if (alias) {
                    result += " AS " + alias->toString();
                }
                return result;
            }
    };

    class StatisticalExpression : public Expression {
        public:
            enum class StatType {
                STDDEV, VARIANCE, PERCENTILE, CORRELATION, REGRESSION
            };

            StatType type;
            std::unique_ptr<Expression> argument;
            std::unique_ptr<Expression> argument2; //For correlation
            std::unique_ptr<Expression> alias;     // For AS alias
            double percentileValue; // For percentile

            std::unique_ptr<Expression> clone() const override {
                return std::make_unique<StatisticalExpression>(
                        type,
                        argument ? argument->clone() : nullptr,
                        argument2 ? argument2->clone() : nullptr,
                        alias ? alias->clone() : nullptr,
                        percentileValue
                        );
            }

            StatisticalExpression(StatType t, std::unique_ptr<Expression> arg1, std::unique_ptr<Expression> arg2 = nullptr, std::unique_ptr<Expression> al = nullptr,  double percentile = 0.5) : type(t), argument(std::move(arg1)), argument2(std::move(arg2)),alias(std::move(al)), percentileValue(percentile) {}

            std::string toString() const override {
                std::string typeStr;
                switch(type) {
                    case StatType::STDDEV: typeStr = "STDDEV"; break;
                    case StatType::VARIANCE: typeStr = "VARIANCE"; break;
                    case StatType::PERCENTILE:
                              typeStr = "PERCENTILE_CONT(" + std::to_string(percentileValue) + ")";
                              break;
                    case StatType::CORRELATION: typeStr = "CORR"; break;
                    case StatType::REGRESSION: typeStr = "REGR_SLOPE"; break;
                }
                std::string result = typeStr + "(" + (argument ? argument->toString() : "") + (argument2 ? ", " + argument2->toString() : "") + ")";
            if (alias) {
                result += " AS " + alias->toString();
            }
            return result;
            }
    };

	class UpdateStatement:public Statement{
		public:
			std::string table;
			std::vector<std::pair<std::string,std::unique_ptr<Expression>>> setClauses;
			std::unique_ptr<Expression> where;

	};
	//class to hanle delete statements
	class DeleteStatement:public Statement{
		public:
			std::string table;
			std::unique_ptr<Expression> where;
	};
	class DropStatement:public Statement{
		public:
			std::string tablename;
			bool ifNotExists=false;
	};

	class InsertStatement:public Statement{
		public:
			std::string table;
			std::vector<std::string> columns;
			std::vector<std::vector<std::unique_ptr<Expression>>> values;
            std::string filename;
            bool hasHeader = false;
            char delimiter;
	};
	class CreateTableStatement:public Statement{
		public:
			std::string tablename;
			std::vector<ColumnDefination> columns;
            std::unique_ptr<Statement> query;
			bool ifNotExists=false;
	};
	class AlterTableStatement:public Statement{
		public:
			std::string tablename;
			enum Action{ADD,DROP,RENAME}action;
			std::string columnName;
			std::string type;
			std::string newColumnName;
			//Support for constraints
			std::vector<std::string> constraints;
			std::string defaultValue;
			bool autoIncreament = false;
			std::string checkExpression;

			bool hasConstraint(const std::string& constraint) const{
				return std::find(constraints.begin(), constraints.end(), constraint) != constraints.end();
			}
	};
        class BulkInsertStatement : public Statement {
                public:
                        std::string table;
                        std::vector<std::string> columns;
                        std::vector<std::vector<std::unique_ptr<Expression>>> rows;
        };

        class BulkUpdateStatement : public Statement {
                public:
                        struct UpdateSpec {
                                uint32_t row_id;
                                std::vector<std::pair<std::string, std::unique_ptr<Expression>>> setClauses;
                        };

                        std::string table;
                        std::vector<UpdateSpec> updates;
        };

        class BulkDeleteStatement : public Statement {
                public:
                        std::string table;
                        std::vector<uint32_t> row_ids;
        };

};

class ParseError : public std::runtime_error {
public:
    size_t line;
    size_t column;
    std::string expected;
    std::string got;
    std::string context;

    ParseError(size_t line,size_t column,const std::string& message) : std::runtime_error(formatBasicMessage(line,column,message)),
        line(line),column(column),expected(""),got(""),context("") {}

    ParseError(size_t line, size_t column, const std::string& expected, const std::string& got, const std::string& context = "")
        : std::runtime_error(formatMessage(line, column, expected, got, context)),
          line(line), column(column), expected(expected), got(got), context(context) {}

    static std::string formatBasicMessage(size_t line, size_t column, const std::string& message) {

        return "Parse error at line " + std::to_string(line) +
               ", column " + std::to_string(column) + ": " + message;
    }

    static std::string formatMessage(size_t line, size_t column, const std::string& expected,const std::string& got,const std::string& context = "") {
        std::string message = "Parse error at line " + std::to_string(line) + ", column " + std::to_string(column) + ":\n";

        if (!context.empty()) {
            message += "  Context: " + context + "\n";
        }

        message += "  Expected: " + expected + "\n";
        message += "  Got: '" + got + "'";

        return message;
    }

    // Helper function to format the error message
    std::string fullMessage() const {
        return  what();
    }

    /*std::string formatExpectedTokens(const std::vector<Token::Type>& types);
    ParseError createUnexpectedTokenError(const Token& token, const std::string& context = "");
    ParseError createExpectedTokenError(const Token& token, Token::Type expected, const std::string& context = "");
    ParseError createExpectedOneOfError(const Token& token,const std::vector<Token::Type>& expected,const std::string& context = "");
    ParseError createSyntaxError(const Token& token, const std::string& message,const std::string& context = "");*/



};
class Parse{
	public:
		explicit Parse(Lexer& lexer);
		std::unique_ptr<AST::Statement> parse();
		std::unique_ptr<AST::Expression> parseExpression();

        // Getter methods to use in he AI parse
        Token getCurrentToken() const { return currentToken; }
        Token getPreviousToken() const { return previousToken_; }
        void setCurrentToken(const Token& token) { currentToken = token; }
        void setPreviousToken(const Token& token) { previousToken_ = token; }

        // Wrapper methods for private functions
        bool checkMatch(Token::Type type) const { return match(type); }
        bool checkMatchAny(const std::vector<Token::Type>& types) const { return matchAny(types); }
        Token checkPeekToken() { return peekToken(); }
        void consumeToken(Token::Type expected) { consume(expected); }
        void advanceToken() { advance(); }

        // Expression parsing wrapper
        std::unique_ptr<AST::Expression> parseExpressionWrapper() { return parseExpression(); }
        std::unique_ptr<AST::Expression> parseIdentifierWrapper() { return parseIdentifier(); }
        std::unique_ptr<AST::Expression> parseValueWrapper() { return parseValue(); }
	private:
		Lexer& lexer;
		Token currentToken;
		Token previousToken_;
		bool inValueContext=false;
        std::unique_ptr<AST::Expression> pendingAlias = nullptr;

		const Token& previousToken() const;
		std::unique_ptr<AST::Expression> parseValue();
        std::string formatExpectedTokens(const std::vector<Token::Type>& types) const;
        ParseError createUnexpectedTokenError(const Token& token, const std::string& context = "") const;
        ParseError createExpectedTokenError(const Token& token, Token::Type expected, const std::string& context = "") const;
        ParseError createExpectedOneOfError(const Token& token, const std::vector<Token::Type>& expected, const std::string& context = "") const;
        ParseError createSyntaxError(const Token& token, const std::string& message, const std::string& context = "") const;
		void consume(Token::Type expected);
		Token peekToken();
		void advance();
		bool match(Token::Type type) const;
		bool matchAny(const std::vector<Token::Type>& types) const;
		std::unique_ptr<AST::Statement> parseStatement();
		std::unique_ptr<AST::CreateDatabaseStatement> parseCreateDatabaseStatement();
		std::unique_ptr<AST::UseDatabaseStatement> parseUseStatement();
        // Queries to show the contents and structures of the database
        std::unique_ptr<AST::ShowTableStatement> parseShowTableStatement();
		std::unique_ptr<AST::ShowDatabaseStatement> parseShowDatabaseStatement();
        std::unique_ptr<AST::ShowTableStructureStatement> parseShowTableStructureStatement();
        std::unique_ptr<AST::ShowDatabaseStructure> parseShowDatabaseStructureStatement();
        // ****************** END *************************************************
		//std::unique_ptr<AST::ShowTableStatement> parseShowTableStatement();
		std::unique_ptr<AST::GroupByClause>  parseGroupByClause();
		std::unique_ptr<AST::HavingClause> parseHavingClause();
		std::unique_ptr<AST::OrderByClause> parseOrderByClause();
		std::unique_ptr<AST::SelectStatement> parseSelectStatement();
        std::vector<std::unique_ptr<AST::Expression>> parseFunctionArguments();
        //*****************HELPER METHODS FOR SELECT*****************************
        std::unique_ptr<AST::WithClause> parseWithClause();
        std::unique_ptr<AST::JoinClause> parseJoinClause();
		std::unique_ptr<AST::UpdateStatement> parseUpdateStatement();
		std::unique_ptr<AST::DeleteStatement> parseDeleteStatement();
		std::unique_ptr<AST::DropStatement> parseDropStatement();
		std::unique_ptr<AST::InsertStatement> parseInsertStatement();
		std::unique_ptr<AST::CreateTableStatement> parseCreateTableStatement();
		std::unique_ptr<AST::BulkInsertStatement> parseBulkInsertStatement();
		std::unique_ptr<AST::BulkUpdateStatement> parseBulkUpdateStatement();
		//Helper methods for ParseBulkStatement()
		void parseSingleRowUpdate(AST::BulkUpdateStatement& stmt);
		void parseSingleSetClause(AST::BulkUpdateStatement::UpdateSpec& updateSpec);
		std::unique_ptr<AST::BulkDeleteStatement> parseBulkDeleteStatement();
		void parseColumnDefinition(AST::CreateTableStatement& stmt);
		std::unique_ptr<AST::AlterTableStatement> parseAlterTableStatement();
		std::vector<std::unique_ptr<AST::Expression>> parseColumnList();
		std::vector<std::pair<std::unique_ptr<AST::Expression>,std::string>> parseColumnListAs();
		std::unique_ptr<AST::Expression> parseFromClause();
        // Case expressions in UPDATE
        std::unique_ptr<AST::Expression> parseCaseExpression();
        std::unique_ptr<AST::Expression> parseStatisticalFunction();
        std::unique_ptr<AST::Expression> parseWindowFunction();
        std::unique_ptr<AST::Expression> parseAIFunction();
		//std::unique_ptr<AST::Expression> parseExpression();
		std::unique_ptr<AST::Expression> parseBinaryExpression(int minPrecedence);
        std::unique_ptr<AST::Expression> parseLikePattern();
        std::unique_ptr<AST::Expression> parseCharacterClassPattern(const std::string& pattern);
		std::unique_ptr<AST::Expression> parsePrimaryExpression();
		std::unique_ptr<AST::Expression> parseIdentifier();
		std::unique_ptr<AST::Expression> parseLiteral();
		int getPrecedence(Token::Type type);
		bool isBinaryOperator(Token::Type type);
};
#endif
