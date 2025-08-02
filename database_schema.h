// database_schema.h
#pragma once
#ifndef DATABASE_SCHEMA_H
#define DATABASE_SCHEMA_H
#include <vector>
#include <string>
#include <unordered_map>

struct DatabaseSchema {
    struct Column {
        enum Type { INTEGER, FLOAT, STRING, BOOLEAN, TEXT };
        std::string name;
        Type type;
        bool isNullable = true;
        bool hasDefault = false;
        std::string defaultvalue;
        
        static Type parseType(const std::string& typeStr);
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
