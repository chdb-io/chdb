#include <QueryPlan/Hints/Leading.h>
#include <QueryPlan/Hints/PlanHintFactory.h>
#include <Parsers/IAST.h>
#include <QueryPlan/PlanNode.h>


namespace DB
{

PlanHintPtr Leading::create(const SqlHint & sql_hint, const ContextMutablePtr & context)
{
    // check option is vaild
    Strings options = sql_hint.getOptions();
    if (options.empty())
        return {};
    if (options.size() > 2 && options.front() == "(" && options.back() == ")")
    {
        // Check if options is join pair form
        Strings stack;
        for (auto option : options)
        {
            if (option == "(" || option == "," )
            {
                stack.emplace_back(option);
            }
            else if (option == ")")
            {
                while (!stack.empty())
                {
                    auto top = stack.back();
                    stack.pop_back();
                    if (top == "(")
                        break;
                }
            }
        }
        if (!stack.empty())
            return {};
    }
    else
    {
        for (size_t i = 0; i < options.size(); ++i)
            if (options[i] == "," || options[i] == "(" || options[i] == ")")
                return {};
    }
    return std::make_shared<Leading>(sql_hint, context);
}

bool Leading::checkOptions(Strings & table_name_list) const
{
    if (table_name_list.empty() || table_list.empty())
        return false;

    if (table_name_list.size() < table_list.size())
        return false;

    bool contain_all_names = false;
    for (String table_name : table_list)
    {
        contain_all_names = false;
        for (size_t i = 0 ; i < table_name_list.size() ; ++i)
        {
            if (table_name == table_name_list[i])
                contain_all_names = true;
        }
        if (!contain_all_names)
            return false;
    }

    return contain_all_names;
}

bool Leading::canAttach(PlanNodeBase & node, HintOptions & hint_options) const
{
    auto & join_table_list = hint_options.table_name_list;
    if (node.getStep()->getType() != IQueryPlanStep::Type::Join)
        return false;

    if (!checkOptions(join_table_list))
        return false;

    return true;
}

// transform join pairs list to RPN format， comma means join. options< ( ( t1 , t2) , ( t3 t4 ) ) > => RPN_table_list<t1 t2 , t3 t4 , ,>
void Leading::transformJoinPairsToRPN(const Strings & options)
{
    Strings stack;
    for (auto option : options)
    {
        if (option != "(" &&  option != ")" && option != ",")
        {
            RPN_table_list.emplace_back(option);
            table_list.emplace_back(option);
        }
        else if (option == "(" || option == "," )
        {
            stack.emplace_back(option);
        }
        else if (option == ")")
        {
            while (!stack.empty())
            {
                auto top = stack.back();
                stack.pop_back();
                if (top == "(")
                    break;
                RPN_table_list.emplace_back(top);
            }
        }
    }
}

// transform table list to RPN format， comma means join. options<t1 t2 t3 t4> => RPN_table_list<t1 t2 , t3 , t4 ,>
void Leading::transformTableListToRPN(const Strings & options)
{
    for (size_t i = 0; i < options.size(); ++i)
    {
        RPN_table_list.emplace_back(options[i]);
        table_list.emplace_back(options[i]);
        if(i != 0)
            RPN_table_list.emplace_back(",");
    }
}

String Leading::getJoinOrderString() const
{
    String res = "";
    Strings stack;
    for (auto elemt : RPN_table_list)
    {
        if (elemt != ",")
        {
            stack.emplace_back(elemt);
        }
        else if (stack.size() >= 2)
        {
            String t1 = stack.back();
            stack.pop_back();
            String t2 = stack.back();
            stack.pop_back();
            res = "(" + t2 + "," + t1 + ")";
            stack.emplace_back(res);
        }
    }
    return res;
}

bool Leading::isVaildLeading(PlanNodeBase & node)
{
    Strings table_name_list;
    TableNamesVisitor visitor;
    VisitorUtil::accept(node, visitor, table_name_list);
    for (auto & table_name : table_list)
    {
        if (std::count(table_name_list.begin(), table_name_list.end(), table_name) != 1)
            return false;
    }
    return true;
}

void TableNamesVisitor::visitPlanNode(PlanNodeBase & node, Strings & table_names)
{
    for (const auto & item : node.getChildren())
    {
        Strings child_table_names;
        VisitorUtil::accept(*item, *this, child_table_names);
        table_names.insert(table_names.end(), child_table_names.begin(), child_table_names.end());
    }
}

void TableNamesVisitor::visitTableScanNode(TableScanNode & node, Strings & table_names)
{
    if (!node.getStep()->getTableAlias().empty())
        table_names.emplace_back(node.getStep()->getTableAlias());
    else
        table_names.emplace_back(node.getStep()->getTable());
}


void registerHintLeading(PlanHintFactory & factory)
{
    factory.registerPlanHint(Leading::name, &Leading::create, PlanHintFactory::CaseInsensitive);
}

}
