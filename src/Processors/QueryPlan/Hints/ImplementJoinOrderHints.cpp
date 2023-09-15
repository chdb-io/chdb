#include <QueryPlan/Hints/ImplementJoinOrderHints.h>

#include <Optimizer/PredicateUtils.h>
#include <Optimizer/SymbolUtils.h>
#include <QueryPlan/FilterStep.h>
#include <QueryPlan/JoinStep.h>
#include <QueryPlan/PlanPattern.h>
#include <QueryPlan/ProjectionStep.h>
#include <Optimizer/Rewriter/SimpleReorderJoin.h>


#include <queue>
#include <utility>

namespace DB
{
void ImplementJoinOrderHints::rewrite(QueryPlan & plan, ContextMutablePtr context) const
{
    JoinOrderHintsVisitor visitor{context, plan.getCTEInfo()};
    Void v;
    auto result = VisitorUtil::accept(*plan.getPlanNode(), visitor, v);
    plan.update(result);
}

PlanNodePtr JoinOrderHintsVisitor::visitJoinNode(JoinNode & node, Void & v)
{
    if (node.getStep()->isOrdered())
        return node.shared_from_this();

    auto join_ptr = node.shared_from_this();
    auto hints_list = join_ptr->getStep()->getHints();

    PlanHintPtr order_hint;
    for (auto & hint : hints_list)
    {
        if (hint->getType() == HintCategory::JOIN_ORDER)
        {
            order_hint = hint;
            break;
        }
    }
    auto leading_hint = std::dynamic_pointer_cast<Leading>(order_hint);
    if (!leading_hint)
        return visitPlanNode(node, v);

    JoinGraph join_graph = JoinGraph::build(join_ptr, context, false, true);
    if (join_graph.size() < 2)
        return join_ptr;

    auto join_order = getJoinOrder(join_graph, leading_hint);

    if (join_order)
    {
        std::vector<String> output_symbols;
        for (auto & column : node.getStep()->getOutputStream().header)
        {
            output_symbols.emplace_back(column.name);
        }

        join_ptr = SimpleReorderJoinVisitor::buildJoinTree(output_symbols, join_graph, join_order, context);

        LOG_WARNING(&Poco::Logger::get("ImplementJoinOrderHints"), "Leading {} is implemented.", leading_hint->getJoinOrderString());
    }

    return join_ptr;
}

PlanNodePtr JoinOrderHintsVisitor::getJoinOrder(JoinGraph & graph, LeadingPtr & leading)
{
    Leading_RPN_List leading_table_id_list = buildLeadingList(graph, leading);
    if (leading_table_id_list.empty())
        return {};

    std::unordered_map<PlanNodeId, PlanNodePtr> id_to_node;
    for (auto & node : graph.getNodes())
    {
        id_to_node[node->getId()] = node;
    }

    std::unordered_map<PlanNodeId, std::unordered_set<PlanNodeId>> id_to_source_tables;
    for (auto & node : graph.getNodes())
    {
        id_to_source_tables[node->getId()].insert(node->getId());
    }

    std::unordered_map<PlanNodeId, PlanNodePtr> id_to_join_node;
    for (auto & node : graph.getNodes())
    {
        id_to_join_node[node->getId()] = node;
    }

    std::vector<PlanNodeId> stack;
    PlanNodePtr result;
    for (auto node_id : leading_table_id_list)
    {
        if (node_id != -1 && node_id != -2)
            stack.emplace_back(node_id);
        else
        {
            if (stack.size() < 2)
                return {};

            auto right_id = stack.back();
            stack.pop_back();
            auto left_id = stack.back();
            stack.pop_back();

            auto left_join_node = id_to_join_node[left_id];
            auto right_join_node = id_to_join_node[right_id];
            auto left_join_node_id = left_join_node->getId();
            auto right_join_node_id = right_join_node->getId();

            auto & left_tables = id_to_source_tables[left_join_node_id];
            auto & right_tables = id_to_source_tables[right_join_node_id];
            Names left_keys;
            Names right_keys;
            for (const auto & left_table_id : left_tables)
            {
                for (const auto & edge : graph.getEdges().at(left_table_id))
                {
                    if (right_tables.contains(edge.getTargetNode()->getId()))
                    {
                        left_keys.emplace_back(edge.getSourceSymbol());
                        right_keys.emplace_back(edge.getTargetSymbol());
                    }
                }
            }

            const DataStream & left_data_stream = left_join_node->getStep()->getOutputStream();
            const DataStream & right_data_stream = right_join_node->getStep()->getOutputStream();
            DataStreams streams = {left_data_stream, right_data_stream};

            auto left_header = left_data_stream.header;
            auto right_header = right_data_stream.header;
            NamesAndTypes output;
            for (const auto & item : left_header)
            {
                output.emplace_back(NameAndTypePair{item.name, item.type});
            }
            for (const auto & item : right_header)
            {
                output.emplace_back(NameAndTypePair{item.name, item.type});
            }

            auto new_join_step = std::make_shared<JoinStep>(
                streams,
                DataStream{.header = output},
                JoinKind::Inner,
                JoinStrictness::All,
                context->getSettingsRef().max_threads,
                context->getSettingsRef().optimize_read_in_order,
                left_keys,
                right_keys);

            if (node_id == -1)
                new_join_step->setOrdered(true);
            auto new_join_node
                = PlanNodeBase::createPlanNode(context->nextNodeId(), std::move(new_join_step), PlanNodes{left_join_node, right_join_node});

            for (auto left_table : left_tables)
            {
                id_to_join_node[left_table] = new_join_node;
                id_to_source_tables[new_join_node->getId()].insert(left_table);
            }
            for (auto right_table : right_tables)
            {
                id_to_join_node[right_table] = new_join_node;
                id_to_source_tables[new_join_node->getId()].insert(right_table);
            }
            id_to_join_node[new_join_node->getId()] = new_join_node;
            stack.emplace_back(new_join_node->getId());
            result = new_join_node;
        }
    }

    return result;
}

Leading_RPN_List JoinOrderHintsVisitor::buildLeadingList(JoinGraph & graph, LeadingPtr & leading)
{
    Leading_RPN_List leading_table_id_list;
    for (auto name : leading->getRPNList())
    {
        if (name == ",")
        {
            leading_table_id_list.emplace_back(-1);
            continue;
        }

        bool is_vaild_hint = false;
        for (auto & node : graph.getNodes())
        {
            TableNameVisitor name_visitor;
            String table_name;
            VisitorUtil::accept(*node, name_visitor, table_name);
            if (name == table_name)
            {
                leading_table_id_list.emplace_back(node->getId());
                is_vaild_hint = true;
                break;
            }
        }
        if (!is_vaild_hint)
            return {};
    }

    for (auto & node : graph.getNodes())
    {
        bool is_contain = false;
        for (auto node_id : leading_table_id_list)
        {
            if (node_id == node->getId())
            {
                is_contain = true;
                break;
            }
        }
        if (!is_contain)
        {
            leading_table_id_list.emplace_back(node->getId());
            leading_table_id_list.emplace_back(-2);
        }
    }
    return leading_table_id_list;
}

void TableNameVisitor::visitTableScanNode(TableScanNode & node, String & table_name)
{
    if (!node.getStep()->getTableAlias().empty())
        table_name = node.getStep()->getTableAlias();
    else
        table_name = node.getStep()->getTable();
}
void TableNameVisitor::visitProjectionNode(ProjectionNode & node, String & table_name)
{
    VisitorUtil::accept(node.getChildren()[0], *this, table_name);
}
void TableNameVisitor::visitFilterNode(FilterNode & node, String & table_name)
{
    VisitorUtil::accept(node.getChildren()[0], *this, table_name);
}
void TableNameVisitor::visitPlanNode(PlanNodeBase &, String &)
{
    return ;
}

}
