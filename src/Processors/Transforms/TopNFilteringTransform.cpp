#include <Processors/Transforms/TopNFilteringTransform.h>

namespace DB
{

namespace ErrorCodes
{
    extern const int NOT_IMPLEMENTED;
}

TopNFilteringTransform::TopNFilteringTransform(
    const Block & header_, SortDescription description_, UInt64 size_, TopNModel model_)
    : ISimpleTransform(
            header_,
            header_,
            true),
    description(std::move(description_)),
    size(size_),
    model(model_)
{
    if (model != TopNModel::DENSE_RANK)
    {
        throw Exception("only support DENSE_RANK", ErrorCodes::NOT_IMPLEMENTED);
    }

    if (description.empty() || description.size() > 1)
    {
        throw Exception("only support one column now", ErrorCodes::NOT_IMPLEMENTED);
    }

    for (auto & sort_column_description: description)
    {
        if (sort_column_description.direction != 1)
        {
            throw Exception("descending not support now", ErrorCodes::NOT_IMPLEMENTED);
        }
    }
}

IProcessor::Status TopNFilteringTransform::prepare()
{
    auto status = ISimpleTransform::prepare();
    return status;
}

void TopNFilteringTransform::transform(Chunk & chunk)
{
    size_t num_rows_before_filtration = chunk.getNumRows();
    
    if (num_rows_before_filtration <= size)
        return;

    auto top_n_column_pos = getInputPort().getHeader().getPositionByName(description[0].column_name);
    auto columns = chunk.detachColumns();
    auto & top_n_column = columns[top_n_column_pos];

    size_t index = 0, count = 1;
    for (size_t i = 1; i < num_rows_before_filtration; i++)
    {
        if (top_n_column->compareAt(index, i, *top_n_column, -1) != 0)
        {
            index = i;
            count ++;
        }

        if (count > size)
        {
            for (size_t j = 0; j < columns.size(); ++j)
                columns[j] = columns[j]->cut(0, i);
            chunk.setColumns(std::move(columns), i);
            return;
        }
    }

    chunk.setColumns(std::move(columns), num_rows_before_filtration);
}

}
