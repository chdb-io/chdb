#include "../programs/local/chdb.hpp"
#include <iostream>
#include <string>

void connection_error()
{
    std::cout << "=== Testing Connection Errors ===\n";
    try {
        auto conn = chdb::Connection(std::vector<std::string>{"--invalid-flag"});
        std::cout << "Connection successful\n";
    } catch (const chdb::ChdbError& e) {
        std::cout << "Caught ChdbError: " << e.what() 
                  << " (Code: " << static_cast<int>(e.code()) << ")\n";
    } catch (const std::exception& e) {
        std::cout << "Caught standard exception: " << e.what() << std::endl;
    }
}

void query_error(const chdb::Connection & conn)
{
    std::cout << "\n=== Testing Query Errors ===\n";
    try
    {
        auto result = conn.query("SELECT * FROM non_existent_table");
        std::cout << "Query successful: " << result.str() << std::endl;
    }
    catch (const chdb::ChdbError & e)
    {
        std::cout << "Caught ChdbError: " << e.what() 
                  << " (Code: " << static_cast<int>(e.code()) << ")\n";
    }
}

void result_error_checking(const chdb::Connection & conn)
{
    std::cout << "\n=== Testing Result Error Checking ===\n";
    try {
        auto result = conn.query("CREATE TABLE if not exists test (id UInt32) ENGINE = Memory");
        result.throw_if_error();
        std::cout << "CREATE TABLE successful\n";
        
        auto insert_result = conn.query("INSERT INTO test VALUES (1), (2), (3)");
        insert_result.throw_if_error();
        std::cout << "INSERT successful\n";
        
        auto select_result = conn.query("SELECT * FROM test ORDER BY id");
        select_result.throw_if_error();
        std::cout << "SELECT result: " << select_result.str() << std::endl;

        auto select_result2 = conn.query("SELECT * FROM test ORDER BY id1");
        select_result2.throw_if_error();
        std::cout << "SELECT result: " << select_result2.str() << std::endl;
    } catch (const chdb::ChdbError& e) {
        std::cout << "Caught ChdbError during result checking: " << e.what() << std::endl;
    }
}

void different_error_types(const chdb::Connection & conn)
{
    std::cout << "\n=== Testing Different Error Types ===\n";
    
    std::vector<std::string> test_queries = {
        "INVALID SQL SYNTAX",
        "SELECT * FROM non_existent_table", 
        "CREATE TABLE ♥️ (id)"
    };

    for (const auto& query : test_queries) {
        try {
            std::cout << "Executing: " << query << std::endl;
            auto result = conn.query(query);
            result.throw_if_error();
            std::cout << "Success: " << result.str() << std::endl;
        } catch (const chdb::ChdbError& e) {
            std::cout << "ChdbError: " << e.what() 
                      << " (Code: " << static_cast<int>(e.code()) << ")\n";
        } catch (const std::exception& e) {
            std::cout << "Standard exception: " << e.what() << std::endl;
        }
    }
}

int main() {
    try {
        connection_error();
        auto conn = chdb::connect(":memory:");
        query_error(conn);
        result_error_checking(conn);
        different_error_types(conn);
        std::cout << "\n=== Error Handling Demo Complete ===\n";
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Unexpected error: " << e.what() << std::endl;
        return 1;
    }
}
