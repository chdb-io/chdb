import unittest
import shutil

import chdb

class TestAIQuery(unittest.TestCase):
    def setUp(self):
        base_url = "https://openrouter.ai/api"
        ai_model = "z-ai/glm-4.5-air:free"
        # API key should be set in environment variable `OPENAI_API_KEY`
        # Explicitly set provider=openai so the engine honors the custom base URL
        connection_str = f"file::memory:?ai_provider=openai&ai_base_url={base_url}&ai_model={ai_model}"
        self.conn = chdb.connect(connection_str)
        self.conn.query("CREATE TABLE users (id UInt32, name String) ENGINE = Memory")
        self.conn.query("INSERT INTO users VALUES (1, 'alice'), (2, 'bob'), (3, 'carol')")

    def tearDown(self):
        self.conn.close()

    def test_ai_query_lists_users_in_order(self):
        """
        Run an AI-generated query via generate_sql. If AI is unavailable/configured,
        the test is skipped to avoid false failures.
        """
        prompt = "List all rows from the users table ordered by id ascending."
        try:
            generated_sql = self.conn.generate_sql(prompt)
            df = self.conn.query(generated_sql, format="DataFrame")

            self.assertFalse(df.empty)
            self.assertListEqual(list(df["id"]), [1, 2, 3])
            self.assertListEqual(list(df["name"]), ["alice", "bob", "carol"])
        except Exception as exc:
            message = str(exc).lower()
            if (
                "ai sql generation is not available" in message
                or "ai sql generator is not configured" in message
                or "unknown ai provider" in message
                or "api key not provided" in message
                # "unsupported country" "unsupported_country_region_territory"
                or "unsupported" in message
                or "Syntax error" in message
                # musl linux generic error message
                or "Caught an unknown exception" in message
            ):
                self.skipTest("AI provider not configured/enabled or not suppported")
            raise



if __name__ == "__main__":
    unittest.main()
