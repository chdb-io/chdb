import threading
from chdb import session


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
    for i in range(15):
        sess.query(
            f"""
        INSERT INTO knowledge_base_portal_interface_event
            FORMAT JSONEachRow [{{"company_id": {i+index}, "locale": "en", "timestamp": 1717780952772, "event_type": "article_update", "article_id": 7}},{{"company_id": {
                i + index + 100
                }, "locale": "en", "timestamp": 1717780952772, "event_type": "article_update", "article_id": 7}}]"""
        )

    # Retrieve all entries from the table
    results = sess.query(
        "SELECT * FROM knowledge_base_portal_interface_event", "JSONObjectEachRow"
    )
    print("Session Query Result:", results)

    # Cleanup session
    sess.cleanup()


# Create multiple threads to perform operations
threads = []
for i in range(5):
    threads.append(threading.Thread(target=perform_operations, args=(i,)))

for thread in threads:
    thread.start()

for thread in threads:
    thread.join()

# for i in range(5):
#     perform_operations(i)
