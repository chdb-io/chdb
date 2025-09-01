#include "../programs/local/chdb.hpp"
#include <iostream>
#include <string>

int main() {
    try {
        auto conn = chdb::connect(":memory:");

        auto result = conn.query("SELECT 'Hello, chDB!' as message");

        std::cout << "Query result:\n";
        std::cout << result.str() << std::endl;

        std::cout << "\nQuery statistics:\n";
        std::cout << "Elapsed time: " << result.elapsed() << " seconds\n";
        std::cout << "Rows read: " << result.rows_read() << std::endl;
        std::cout << "Bytes read: " << result.bytes_read() << std::endl;

        return 0;
    } catch (const chdb::ChdbError& e) {
        std::cerr << "ChDB Error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}