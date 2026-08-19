-- Source ClickHouse instance — referenced as `localhost:9000` from tests and
-- as `ch1:9000` from inside the docker network.
--
-- Files in /docker-entrypoint-initdb.d/ are executed by the official image
-- exactly once on first container startup, after the server is up. If you
-- need to re-run them, recreate the volume (`docker compose down -v`).

CREATE DATABASE IF NOT EXISTS chdb_test;

CREATE USER IF NOT EXISTS remote_user
    IDENTIFIED WITH plaintext_password BY 'test123'
    HOST ANY;

GRANT ALL ON chdb_test.* TO remote_user;

-- The library probes existence via `SELECT count() FROM remote(..., 'system',
-- 'tables', ...)`, so the test user must be able to read system metadata.
GRANT SELECT ON system.* TO remote_user;

-- chdb.deploy (tests/udf_deploy) registers executable UDFs by writing
-- artifacts into the mounted UDF dirs and then running SYSTEM RELOAD
-- FUNCTIONS over HTTP as this user.
GRANT SYSTEM RELOAD FUNCTION ON *.* TO remote_user;
