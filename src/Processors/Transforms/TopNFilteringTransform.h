#pragma once

#include <Processors/ISimpleTransform.h>
#include <Core/SortDescription.h>
#include <Processors/QueryPlan/TopNModel.h>

namespace DB
{

class TopNFilteringTransform : public ISimpleTransform
{
public:
    TopNFilteringTransform(
        const Block & header_, SortDescription sort_description_, UInt64 size_, TopNModel model_);

    String getName() const override { return "TopNFilteringTransform"; }

    Status prepare() override;

protected:
    void transform(Chunk & chunk) override;

private:
    SortDescription description;
    UInt64 size;
    TopNModel model;
};

}
