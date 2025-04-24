#!python3

import unittest
import os
import signal
import threading
import time
from chdb import session

class TestSignalHandler(unittest.TestCase):
    def setUp(self) -> None:
        self.sess = session.Session()
        self.signal_received = False
        return super().setUp()

    def tearDown(self) -> None:
        self.sess.close()
        return super().tearDown()

    def background_sender(self):
        time.sleep(5)
        print("send signal")
        os.kill(os.getpid(), signal.SIGINT)

    def test_signal_response(self):
        sender_thread = threading.Thread(target=self.background_sender, daemon=True)
        sender_thread.start()

        start_time = time.time()
        try:
            while time.time() - start_time < 10:
                time.sleep(0.1)
                self.sess.query("SELECT 1")
        except KeyboardInterrupt:
            print("receive signal")
            self.signal_received = True

        self.assertTrue(self.signal_received)

if __name__ == "__main__":
    unittest.main()
