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
        self.prompt = "List all rows from the users table ordered by id ascending."

    def tearDown(self):
        self.conn.close()

    def _run_ai_or_skip(self, fn):
        """
        Execute an AI-assisted call, skipping the test on common provider/configuration failures.
        """
        try:
            return fn()
        except Exception as exc:
            message = str(exc).lower()
            if (
                "ai sql generation is not available" in message
                or "ai sql generator is not configured" in message
                or "unknown ai provider" in message
                or "api key not provided" in message
                # "unsupported country" "unsupported_country_region_territory"
                or "unsupported" in message
                or "syntax error" in message
                # musl linux generic error message
                or "caught an unknown exception" in message
            ):
                self.skipTest("AI provider not configured/enabled or not suppported")
            raise

    def test_ai_query_lists_users_in_order(self):
        """
        Run an AI-generated query via generate_sql. If AI is unavailable/configured,
        the test is skipped to avoid false failures.
        """
        def _gen_sql():
            generated_sql = self.conn.generate_sql(self.prompt)
            return self.conn.query(generated_sql, format="DataFrame")

        df = self._run_ai_or_skip(_gen_sql)
        self.assertFalse(df.empty)
        self.assertListEqual(list(df["id"]), [1, 2, 3])
        self.assertListEqual(list(df["name"]), ["alice", "bob", "carol"])

    def test_ai_ask_runs_prompt_and_returns_dataframe(self):
        """
        Run a prompt end-to-end via ask() using default DataFrame output. Skip if
        AI is unavailable/configured.
        """
        df = self._run_ai_or_skip(lambda: self.conn.ask(self.prompt, format="DataFrame"))

        self.assertFalse(df.empty)
        self.assertListEqual(list(df["id"]), [1, 2, 3])
        self.assertListEqual(list(df["name"]), ["alice", "bob", "carol"])



if __name__ == "__main__":
    unittest.main()
