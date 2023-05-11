#!python3

import os
import time
import hashlib
import unittest
import chdb
import zipfile
import urllib.request

from timeout_decorator import timeout

csv_url = "https://media.githubusercontent.com/media/datablist/sample-csv-files/main/files/organizations/organizations-2000000.zip"


# download csv file, and unzip it
def download_and_extract(url, save_path):
    print("Downloading file...")
    urllib.request.urlretrieve(url, save_path)

    print("Extracting file...")
    with zipfile.ZipFile(save_path, "r") as zip_ref:
        zip_ref.extractall(os.path.dirname(save_path))

    print("Done!")


# @timeout(60, use_signals=False)

import signal


def payload():
    now = time.time()
    res = chdb.query(
        'select Name, count(*) cnt from file("organizations-2000000.csv", CSVWithNames) group by Name order by cnt desc',
        "CSV",
    )
    # calculate md5 of the result
    hash_out = hashlib.md5(res.get_memview().tobytes()).hexdigest()
    print("output length: ", len(res.get_memview().tobytes()))
    if hash_out != "60833f6ba30f2892f1fda976b2088570":
        print(res.get_memview().tobytes().decode("utf-8"))
        raise Exception(f"md5 not match {hash_out}")
    used_time = time.time() - now
    print("used time: ", used_time)


class TimeoutTestRunner(unittest.TextTestRunner):
    def __init__(self, timeout=60, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timeout = timeout

    def run(self, test):
        class TimeoutException(Exception):
            pass

        def handler(signum, frame):
            print("Timeout after {} seconds".format(self.timeout))
            raise TimeoutException("Timeout after {} seconds".format(self.timeout))

        old_handler = signal.signal(signal.SIGALRM, handler)
        signal.alarm(self.timeout)

        result = super().run(test)

        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        return result


class TestAggOnCSVSpeed(unittest.TestCase):
    def setUp(self):
        download_and_extract(csv_url, "organizations-2000000.zip")

    def tearDown(self):
        os.remove("organizations-2000000.csv")
        os.remove("organizations-2000000.zip")

    def _test_agg(self, arg=None):
        payload()

    def test_agg(self):
        result = TimeoutTestRunner(timeout=20).run(self._test_agg)
        self.assertTrue(result.wasSuccessful(), "Test failed: took too long to execute")


if __name__ == "__main__":
    unittest.main()
