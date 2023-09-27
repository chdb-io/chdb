#include <Processors/Transforms/MarkDistinctTransform.h>

#include <Columns/ColumnsNumber.h>
#include <DataTypes/DataTypesNumber.h>

namespace DB
{
namespace ErrorCodes
{
    extern const int INCORRECT_RESULT_OF_SCALAR_SUBQUERY;
}

Block MarkDistinctTransform::transformHeader(Block header, String marker_symbol)
{
    auto type_int = std::make_shared<DataTypeUInt8>();
    MutableColumnPtr column = type_int->createColumn();
    ColumnWithTypeAndName unique{std::move(column), type_int, marker_symbol};
    header.insert(unique);
    return header;
}

MarkDistinctTransform::MarkDistinctTransform(const Block & header_, String marker_symbol_, std::vector<String> distinct_symbols_)
    : ISimpleTransform(header_, transformHeader(header_, marker_symbol_), false)
    , distinct_symbols(std::move(distinct_symbols_))
    , distinct_set(SizeLimits{}, true, true)
{
    ColumnsWithTypeAndName distinct_columns;
    for (auto & distinct_symbol : distinct_symbols)
    {
        distinct_columns.emplace_back(getInputPort().getHeader().getByName(distinct_symbol));
    }
    Block mark_distinct_block{distinct_columns};
    distinct_set.setHeader(mark_distinct_block.getColumnsWithTypeAndName());
}

void MarkDistinctTransform::transform(Chunk & chunk)
{
    Block block = getInputPort().getHeader().cloneWithColumns(chunk.getColumns());

    ColumnsWithTypeAndName distinct_columns;
    for (auto & distinct_symbol : distinct_symbols)
    {
        distinct_columns.emplace_back(block.getByName(distinct_symbol));
    }

    Block mark_block{distinct_columns};

    ColumnUInt8::MutablePtr result = distinct_set.markDistinctBlock(mark_block);

    chunk.addColumn(std::move(result));
}

}
