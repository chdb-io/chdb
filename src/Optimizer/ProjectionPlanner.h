#pragma once
#include <QueryPlan/PlanNode.h>
#include <QueryPlan/ProjectionStep.h>
#include <Analyzers/TypeAnalyzer.h>

namespace DB
{

// an util class to simplify ProjectionStep building
class ProjectionPlanner
{
public:
    ProjectionPlanner(PlanNodePtr source_node_, ContextMutablePtr context_);

    std::pair<String, DataTypePtr> addColumn(ASTPtr column_expr);
    DataTypePtr getColumnType(const String & column_name)
    {
        return column_types.at(column_name);
    }
    PlanNodePtr build(const Names & output_columns = {});

private:
    ContextMutablePtr context;
    PlanNodePtr source_node;
    ASTMap<String> column_exprs;
    NameToType column_types;
    TypeAnalyzerPtr type_analyzer;
};

}
