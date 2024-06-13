#!python3

import chdb


class myReader(chdb.PyReader):
    def __init__(self, data):
        self.data = data
        self.cursor = 0
        super().__init__(data)

    def read(self, col_names, count):
        print("Python func read", col_names, count, self.cursor)
        if self.cursor >= len(self.data["a"]):
            return []
        block = [self.data[col] for col in col_names]
        self.cursor += len(block[0])
        return block


def bench():
    reader = myReader(
        {
            "a": [1, 2, 3, 4, 5, 6],
            "b": ["tom", "jerry", "auxten", "tom", "jerry", "auxten"],
        }
    )

    # chdb.query("SELECT b, sum(a) FROM Python('reader') GROUP BY b", "debug").show()
    print(chdb.query("SELECT b, sum(a) FROM Python('reader') GROUP BY b", "debug"))


bench()
