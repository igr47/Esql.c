// database_schema.h
#pragma once
#ifndef DATABASE_SCHEMA_H
#define DATABASE_SCHEMA_H
#include <vector>
#include <string>
#include <stdexcept>
#include <unordered_map>

struct DatabaseSchema {
    struct Constraint {
	    enum Type { NOT_NULL,UNIQUE,PRIMARY_KEY,FOREIGN_KEY,CHECK,DEFAULT,AUTO_INCREAMENT };
	    Type type;
	    std::string name;
	    std::string value;
	    std::string reference_column;
	    std::string reference_table;

	    //Constraint(Type t,const std::string& n = "") : type(t), name(n) {}
    };
    struct Column {
    std::string name;
    enum Type { INTEGER, FLOAT, STRING, BOOLEAN, TEXT, VARCHAR, DATETIME } type;
    bool isNullable = true;
    bool hasDefault = false;
    bool isPrimaryKey = false;
    bool isUnique = false;
    bool autoIncreament = false;
    std::string defaultValue;
    size_t length = 0; // For VARCHAR types
    std::vector<Constraint> constraints;
    static Type parseType(const std::string& typeStr) {
        if (typeStr == "INT" || typeStr == "INTEGER") return INTEGER;
        if (typeStr == "FLOAT" || typeStr == "DOUBLE") return FLOAT;
        if (typeStr == "STRING") return STRING;
        if (typeStr == "BOOL" || typeStr == "BOOLEAN") return BOOLEAN;
        if (typeStr == "TEXT") return TEXT;
        if (typeStr.find("VARCHAR") == 0) return VARCHAR;
        if (typeStr == "DATETIME") return DATETIME;
        throw std::runtime_error("Unknown type: " + typeStr);
    }
};

    struct Table {
        std::string name;
        std::vector<Column> columns;
        std::string primaryKey;
    };

    std::unordered_map<std::string, Table> tables;
    
    void addTable(const Table& table);
    void createTable(const std::string& name, 
                   const std::vector<Column>& columns, 
                   const std::string& primarykey = "");
    void dropTable(const std::string& name);
    const Table* getTable(const std::string& name) const;
    bool tableExists(const std::string& name) const;
};
#endif
