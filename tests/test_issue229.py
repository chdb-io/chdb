import unittest
import threading
from chdb import session

thread_count = 5
insert_count = 15
return_results = [None] * thread_count

def perform_operations(index):
    sess = session.Session()
    print(f"Performing operations in session {index}, path = {sess._path}")

    # Create a local database
    sess.query("CREATE DATABASE local", "Debug")

    # Create a table within the local database
    sess.query(
        """
    USE local; 
    CREATE TABLE IF NOT EXISTS knowledge_base_portal_interface_event
    (
        timestamp DateTime64,
        company_id Int64,
        event_type String,
        locale String,
        article_id Int64 DEFAULT 0,
    )
    ENGINE = MergeTree
    ORDER BY (company_id, locale, timestamp)
    COMMENT 'This table represents a store of knowledge base portal interface events';
    """
    )

    # Insert multiple entries into the table
    for i in range(insert_count):
        # print(f"Inserting entry {i} into the table in session {index}")
        sess.query(
            f"""
        INSERT INTO knowledge_base_portal_interface_event
            FORMAT JSONEachRow [{{"company_id": {i+index}, "locale": "en", "timestamp": 1717780952772, "event_type": "article_update", "article_id": 7}},{{"company_id": {
                i + index + 100
                }, "locale": "en", "timestamp": 1717780952772, "event_type": "article_update", "article_id": 7}}]"""
        )

    print(f"Inserted {insert_count} entries into the table in session {index}")

    # Retrieve all entries from the table
    results = sess.query(
        "SELECT * FROM knowledge_base_portal_interface_event", "JSONObjectEachRow"
    )
    print("Session Query Result:", results)
    return_results[index] = str(results)

    # Cleanup session
    sess.cleanup()


class TestIssue229(unittest.TestCase):
    def test_issue229(self):
        # Create multiple threads to perform operations
        threads = []
        results = []
        for i in range(thread_count):
            threads.append(threading.Thread(target=perform_operations, args=(i,)))

        for thread in threads:
            thread.start()

        # Wait for all threads to complete, and collect results returned by each thread
        for thread in threads:
            thread.join()

        # Check if all threads have returned results
        for i in range(thread_count):
            lines = return_results[i].split("\n")
            self.assertGreater(len(lines), 2 * insert_count)


if __name__ == "__main__":
    unittest.main()
