"""
join(on=left.col == right.col) of two SQL sources compares the two tables'
columns, not one column with itself (#755).

SELECT * of a SQL join keeps both tables' columns and names a right column
that clashes with a left one "<right table>.<column>", so the pandas mirror
renames the right frame's clashing columns the same way before merging.
"""

import os

import pandas as pd
import pytest

from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas


L = pd.DataFrame({"id": [1, 2, 3], "a": ["x", "y", "z"]})
R = pd.DataFrame({"id": [2, 3, 4], "b": ["p", "q", "s"]})
ORDERS = pd.DataFrame(
    {
        "id": [1, 2, 3, 4],
        "user_id": [2, 3, 3, 9],
        "k": [1, 1, 2, 2],
        "amount": [10, 20, 30, 40],
    }
)
USERS = pd.DataFrame({"id": [1, 2, 3], "k": [1, 1, 2], "name": ["A", "B", "C"]})
CITIES = pd.DataFrame({"k": [1, 2], "city": ["x", "y"]})


@pytest.fixture(params=["parquet", "csv"])
def files(tmp_path, request):
    paths = {}
    for name, df in [
        ("l", L),
        ("r", R),
        ("orders", ORDERS),
        ("users", USERS),
        ("cities", CITIES),
    ]:
        path = os.path.join(tmp_path, f"{name}.{request.param}")
        if request.param == "parquet":
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)
        paths[name] = path
    return paths


def test_join_on_same_column_name(files):
    dl, dr = DataStore.from_file(files["l"]), DataStore.from_file(files["r"])
    ds_result = dl.join(dr, on=dl.id == dr.id)

    assert 'ON "l"."id" = "r"."id"' in ds_result.to_sql()
    pd_result = L.merge(
        R.rename(columns={"id": "r.id"}), left_on="id", right_on="r.id"
    )
    assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


@pytest.mark.parametrize("reverse", [False, True])
def test_join_on_column_also_in_left_table(files, reverse):
    # Unqualified, "id" would be read from orders, which has an id too
    orders = DataStore.from_file(files["orders"])
    users = DataStore.from_file(files["users"])
    on = (users.id == orders.user_id) if reverse else (orders.user_id == users.id)
    ds_result = orders.join(users, on=on)

    pd_result = ORDERS.merge(
        USERS.rename(columns={"id": "users.id", "k": "users.k"}),
        left_on="user_id",
        right_on="users.id",
    )
    assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


def test_join_on_two_columns(files):
    orders = DataStore.from_file(files["orders"])
    users = DataStore.from_file(files["users"])
    ds_result = orders.join(
        users, on=(orders["user_id"] == users["id"]) & (orders["k"] == users["k"])
    )

    pd_result = ORDERS.merge(
        USERS.rename(columns={"id": "users.id", "k": "users.k"}),
        left_on=["user_id", "k"],
        right_on=["users.id", "users.k"],
    )
    assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


def test_filter_then_join_on_condition(files):
    orders = DataStore.from_file(files["orders"])
    users = DataStore.from_file(files["users"])
    ds_result = orders[orders.amount > 10].join(users, on=orders.user_id == users.id)

    pd_result = ORDERS[ORDERS["amount"] > 10].merge(
        USERS.rename(columns={"id": "users.id", "k": "users.k"}),
        left_on="user_id",
        right_on="users.id",
    )
    assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


def test_second_join_on_column_of_first_joined_table(files):
    orders = DataStore.from_file(files["orders"])
    users = DataStore.from_file(files["users"])
    cities = DataStore.from_file(files["cities"])
    ds_result = orders.join(users, on=orders.user_id == users.id).join(
        cities, on=users.k == cities.k
    )

    pd_result = (
        ORDERS.rename(columns={"id": "orders.id", "k": "orders.k"})
        .merge(
            USERS.rename(columns={"id": "users.id", "k": "users.k"}),
            left_on="user_id",
            right_on="users.id",
        )
        .merge(
            CITIES.rename(columns={"k": "cities.k"}),
            left_on="users.k",
            right_on="cities.k",
        )
    )
    assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)
