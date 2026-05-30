-- Target ClickHouse instance — referenced as `localhost:9001` from tests and
-- as `ch2:9000` from inside the docker network.
--
-- Distinct credentials from ch1 on purpose: a misrouted connection (e.g. a
-- code path that accidentally sends source creds to the target) gets
-- rejected immediately rather than silently passing.

CREATE DATABASE IF NOT EXISTS chdb_test_2;

CREATE USER IF NOT EXISTS remote_user_2
    IDENTIFIED WITH plaintext_password BY 'test456'
    HOST ANY;

GRANT ALL ON chdb_test_2.* TO remote_user_2;
GRANT SELECT ON system.* TO remote_user_2;
