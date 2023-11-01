#include <Optimizer/ProjectionPlanner.h>

namespace DB
{

ProjectionPlanner::ProjectionPlanner(PlanNodePtr source_node_, ContextMutablePtr context_):
    context(std::move(context_)),
    source_node(std::move(source_node_))
{
    for (const auto & name_and_type: source_node->getStep()->getOutputStream().getNamesAndTypes())
    {
        column_exprs.emplace(std::make_shared<ASTIdentifier>(name_and_type.name), name_and_type.name);
        column_types.emplace(name_and_type.name, name_and_type.type);
    }

    type_analyzer = std::make_shared<TypeAnalyzer>(TypeAnalyzer::create(context, column_types));
}

std::pair<String, DataTypePtr> ProjectionPlanner::addColumn(ASTPtr column_expr)
{
    if (auto it = column_exprs.find(column_expr); it != column_exprs.end())
    {
        auto & column_name = it->second;
        auto & column_type = column_types.at(column_name);
        return std::make_pair(column_name, column_type);
    }

    auto column_name = context->getSymbolAllocator()->newSymbol(column_expr);
    auto column_type = type_analyzer->getType(column_expr);
    column_exprs.emplace(column_expr, column_name);
    column_types.emplace(column_name, column_type);
    return std::make_pair(column_name, column_type);
}

PlanNodePtr ProjectionPlanner::build(const Names & output_columns)
{
    if (!output_columns.empty())
    {
        for (auto it = column_exprs.begin(); it != column_exprs.end();)
        {
            if (std::find(output_columns.begin(), output_columns.end(), it->second) == output_columns.end())
            {
                column_types.erase(it->second);
                it = column_exprs.erase(it);
            }
            else
            {
                it++;
            }
        }
    }

    Assignments assignments;

    for (const auto & [expr, name]: column_exprs)
        assignments.emplace_back(name, expr);

    auto step = std::make_shared<ProjectionStep>(source_node->getStep()->getOutputStream(), assignments, column_types);
    return source_node->addStep(context->nextNodeId(), std::move(step));
}

}
