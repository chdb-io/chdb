#include "../programs/local/chdb.hpp"
#include <iostream>
#include <string>

int main() {
    try {
        auto conn = chdb::connect(":memory:");
        
        conn.query(R"(
            CREATE TABLE large_dataset (
                id UInt64,
                value String,
                timestamp DateTime
            ) ENGINE = Memory
        )");
        
        std::cout << "Generating sample data...\n";
        conn.query(R"(
            INSERT INTO large_dataset 
            SELECT 
                number as id,
                concat('value_', toString(number)) as value,
                now() - interval number second as timestamp
            FROM numbers(1000)
        )");
        
        std::cout << "\n=== Regular Query (loads all data at once) ===\n";
        auto regular_result = conn.query("SELECT id, value FROM large_dataset WHERE id < 10 ORDER BY id");
        std::cout << "Regular query result size: " << regular_result.size() << " bytes\n";
        std::cout << regular_result.str() << std::endl;
        
        std::cout << "\n=== Streaming Query (processes data in chunks) ===\n";
        auto stream_result = conn.stream_query("SELECT id, value FROM large_dataset WHERE id >= 10 AND id < 20 ORDER BY id");
        
        std::cout << "Processing streaming results:\n";
        int batch_count = 0;
        size_t total_size = 0;
        
        while (true) {
            try {
                auto chunk = conn.stream_fetch(stream_result);
                
                if (chunk.size() == 0) {
                    std::cout << "Stream ended (no more data)\n";
                    break;
                }
                
                batch_count++;
                total_size += chunk.size();
                
                std::cout << "Batch " << batch_count << " (size: " << chunk.size() << " bytes):\n";
                std::cout << chunk.str() << std::endl;
                
                if (batch_count > 10) {
                    std::cout << "Stopping early (processed 10 batches)\n";
                    conn.stream_cancel(stream_result);
                    break;
                }
                
            } catch (const chdb::ChdbError& e) {
                std::cout << "Stream error: " << e.what() << std::endl;
                break;
            }
        }
        
        std::cout << "\nStreaming summary:\n";
        std::cout << "Total batches processed: " << batch_count << std::endl;
        std::cout << "Total data size: " << total_size << " bytes\n";
        
        std::cout << "\n=== Using Stream Iterator (C++ range-based for loop) ===\n";
        auto stream_result2 = conn.stream_query("SELECT id, concat('item_', toString(id)) as name FROM numbers(5)");
        
        chdb::Stream stream(conn, std::move(stream_result2));
        int item_count = 0;
        
        for (auto& chunk : stream) {
            item_count++;
            std::cout << "Chunk " << item_count << ":\n" << chunk.str() << std::endl;
            
            if (item_count >= 3) {
                std::cout << "Breaking early from iterator loop\n";
                break;
            }
        }
        
        return 0;
        
    } catch (const chdb::ChdbError& e) {
        std::cerr << "ChDB Error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}