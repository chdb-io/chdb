#include "../programs/local/chdb.hpp"
#include <iostream>
#include <string>
#include <vector>

int main() {
    try {
        auto conn = chdb::connect(":memory:");
        
        conn.query(R"(
            CREATE TABLE users (
                id UInt32,
                name String,
                age UInt32,
                city String
            ) ENGINE = Memory
        )");
        
        conn.query(R"(
            INSERT INTO users VALUES 
            (1, 'Alice', 25, 'New York'),
            (2, 'Bob', 30, 'San Francisco'),
            (3, 'Charlie', 35, 'Chicago'),
            (4, 'Diana', 28, 'Boston')
        )");
        
        std::cout << "=== All Users (CSV Format) ===\n";
        auto result1 = conn.query("SELECT * FROM users ORDER BY id", "CSV");
        std::cout << result1.str() << std::endl;
        
        std::cout << "=== Users Over 30 (JSON Format) ===\n";
        auto result2 = conn.query("SELECT name, age, city FROM users WHERE age > 30", "JSON");
        std::cout << result2.str() << std::endl;
        
        std::cout << "=== Aggregation Query ===\n";
        auto result3 = conn.query(R"(
            SELECT 
                city,
                count() as user_count,
                avg(age) as avg_age
            FROM users 
            GROUP BY city 
            ORDER BY user_count DESC
        )");
        std::cout << result3.str() << std::endl;
        
        std::cout << "=== Query with Parameterized Values ===\n";
        std::string min_age = "25";
        std::string query = "SELECT name, age FROM users WHERE age >= " + min_age;
        auto result4 = conn.query(query);
        std::cout << result4.str() << std::endl;
        
        std::cout << "\n=== Query Statistics ===\n";
        std::cout << "Last query elapsed time: " << result4.elapsed() << " seconds\n";
        std::cout << "Rows read: " << result4.rows_read() << std::endl;
        std::cout << "Storage rows read: " << result4.storage_rows_read() << std::endl;
        
        return 0;
    } catch (const chdb::ChdbError& e) {
        std::cerr << "ChDB Error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}