#pragma once

namespace DB
{

enum class TopNModel
{
    ROW_NUMBER       = 0,
    RANKER           = 1,
    DENSE_RANK       = 2
};

}
