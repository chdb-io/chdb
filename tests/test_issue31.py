#!python3

import os
import time
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


@timeout(20, use_signals=False)
def payload():
    now = time.time()
    res = chdb.query(
        'select Name, count(*) cnt from file("organizations-2000000.csv", CSVWithNames) group by Name order by cnt desc',
        "CSV",
    )
    print(res.get_memview().tobytes().decode("utf-8"))
    used_time = time.time() - now
    print("used time: ", used_time)


class TestAggOnCSVSpeed(unittest.TestCase):
    def setUp(self):
        download_and_extract(csv_url, "organizations-2000000.zip")

    def tearDown(self):
        os.remove("organizations-2000000.csv")
        os.remove("organizations-2000000.zip")

    def test_agg(self):
        payload()


if __name__ == "__main__":
    unittest.main()
