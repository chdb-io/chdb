#include <stack>

#include <Common/JSONBuilder.h>

#include <Interpreters/ActionsDAG.h>
#include <Interpreters/ArrayJoinAction.h>

#include <IO/Operators.h>
#include <IO/WriteBuffer.h>

#include <Processors/QueryPlan/BuildQueryPipelineSettings.h>
#include <Processors/QueryPlan/IQueryPlanStep.h>
#include <Processors/QueryPlan/Optimizations/Optimizations.h>
#include <Processors/QueryPlan/Optimizations/QueryPlanOptimizationSettings.h>
#include <Processors/QueryPlan/QueryPlan.h>
#include <Processors/QueryPlan/ReadFromMergeTree.h>
#include <Processors/QueryPlan/ITransformingStep.h>
#include <Processors/QueryPlan/QueryPlanVisitor.h>

#include <QueryPipeline/QueryPipelineBuilder.h>


namespace DB
{

namespace ErrorCodes
{
    extern const int LOGICAL_ERROR;
}

QueryPlan::QueryPlan() = default;
QueryPlan::~QueryPlan() = default;
QueryPlan::QueryPlan(QueryPlan &&) noexcept = default;
QueryPlan & QueryPlan::operator=(QueryPlan &&) noexcept = default;

QueryPlan::QueryPlan(PlanNodePtr root_, PlanNodeIdAllocatorPtr id_allocator_)
    : plan_node(std::move(root_)), id_allocator(std::move(id_allocator_))
{
}

QueryPlan::QueryPlan(PlanNodePtr root_, CTEInfo cte_info_, PlanNodeIdAllocatorPtr id_allocator_)
    : plan_node(std::move(root_)), cte_info(std::move(cte_info_)), id_allocator(std::move(id_allocator_))
{
}

void QueryPlan::checkInitialized() const
{
    if (!isInitialized())
        throw Exception(ErrorCodes::LOGICAL_ERROR, "QueryPlan was not initialized");
}

void QueryPlan::checkNotCompleted() const
{
    if (isCompleted())
        throw Exception(ErrorCodes::LOGICAL_ERROR, "QueryPlan was already completed");
}

bool QueryPlan::isCompleted() const
{
    return isInitialized() && !root->step->hasOutputStream();
}

const DataStream & QueryPlan::getCurrentDataStream() const
{
    checkInitialized();
    checkNotCompleted();
    return root->step->getOutputStream();
}

void QueryPlan::unitePlans(QueryPlanStepPtr step, std::vector<std::unique_ptr<QueryPlan>> plans)
{
    if (isInitialized())
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Cannot unite plans because current QueryPlan is already initialized");

    const auto & inputs = step->getInputStreams();
    size_t num_inputs = step->getInputStreams().size();
    if (num_inputs != plans.size())
        throw Exception(
            ErrorCodes::LOGICAL_ERROR,
            "Cannot unite QueryPlans using {} because step has different number of inputs. Has {} plans and {} inputs",
            step->getName(),
            plans.size(),
            num_inputs);

    for (size_t i = 0; i < num_inputs; ++i)
    {
        const auto & step_header = inputs[i].header;
        const auto & plan_header = plans[i]->getCurrentDataStream().header;
        if (!blocksHaveEqualStructure(step_header, plan_header))
            throw Exception(
                ErrorCodes::LOGICAL_ERROR,
                "Cannot unite QueryPlans using {} because it has incompatible header with plan {} plan header: {} step header: {}",
                step->getName(),
                root->step->getName(),
                plan_header.dumpStructure(),
                step_header.dumpStructure());
    }

    for (auto & plan : plans)
        nodes.splice(nodes.end(), std::move(plan->nodes));

    nodes.emplace_back(Node{.step = std::move(step)});
    root = &nodes.back();

    for (auto & plan : plans)
        root->children.emplace_back(plan->root);

    for (auto & plan : plans)
    {
        max_threads = std::max(max_threads, plan->max_threads);
        resources = std::move(plan->resources);
    }
}

void QueryPlan::addStep(QueryPlanStepPtr step)
{
    checkNotCompleted();

    size_t num_input_streams = step->getInputStreams().size();

    if (num_input_streams == 0)
    {
        if (isInitialized())
            throw Exception(
                ErrorCodes::LOGICAL_ERROR,
                "Cannot add step {} to QueryPlan because step has no inputs, but QueryPlan is already initialized",
                step->getName());

        nodes.emplace_back(Node{.step = std::move(step)});
        root = &nodes.back();
        return;
    }

    if (num_input_streams == 1)
    {
        if (!isInitialized())
            throw Exception(
                ErrorCodes::LOGICAL_ERROR,
                "Cannot add step {} to QueryPlan because step has input, but QueryPlan is not initialized",
                step->getName());

        const auto & root_header = root->step->getOutputStream().header;
        const auto & step_header = step->getInputStreams().front().header;
        if (!blocksHaveEqualStructure(root_header, step_header))
            throw Exception(
                ErrorCodes::LOGICAL_ERROR,
                "Cannot add step {} to QueryPlan because it has incompatible header with root step {} root header: {} step header: {}",
                step->getName(),
                root->step->getName(),
                root_header.dumpStructure(),
                step_header.dumpStructure());

        nodes.emplace_back(Node{.step = std::move(step), .children = {root}});
        root = &nodes.back();
        return;
    }

    throw Exception(
        ErrorCodes::LOGICAL_ERROR,
        "Cannot add step {} to QueryPlan because it has {} inputs but {} input expected",
        step->getName(),
        num_input_streams,
        isInitialized() ? 1 : 0);
}

void QueryPlan::addNode(Node && node_)
{
    nodes.emplace_back(std::move(node_));
}

void QueryPlan::addRoot(Node && node_)
{
    nodes.emplace_back(std::move(node_));
    root = &nodes.back();
}

/**
 * Remove nodes that have no step and children.
 * Refresh children of each node since this childen maybe removed.
 */
void QueryPlan::freshPlan()
{
    for (auto it = nodes.begin(); it != nodes.end();)
    {
        if (!it->step && it->children.empty())
            it = nodes.erase(it);
        else
            ++it;
    }

    std::unordered_set<Node *> exists_nodes;

    for (auto & node : nodes)
        exists_nodes.insert(&node);

    for (auto & node : nodes)
    {
        std::vector<Node *> freshed_children;
        for (auto & child : node.children)
        {
            if (exists_nodes.count(child))
                freshed_children.push_back(child);
        }
        node.children.swap(freshed_children);
    }
}

/**
 * Be careful, after we create a sub_plan, some nodes in the original plan have been deleted and deconstructed.
 * More precisely， nodes that moved to sub_plan are deleted.
 */
QueryPlan QueryPlan::getSubPlan(QueryPlan::Node * node_)
{
    QueryPlan sub_plan;

    std::stack<QueryPlan::Node *> plan_nodes;
    sub_plan.addRoot(Node{.step = node_->step, .children = node_->children, .id = node_->id});
    plan_nodes.push(sub_plan.getRoot());
    sub_plan.setResetStepId(reset_step_id);

    while (!plan_nodes.empty())
    {
        auto current = plan_nodes.top();
        plan_nodes.pop();

        std::vector<Node *> result_children;
        for (auto & child : current->children)
        {
            sub_plan.addNode(Node{.step = child->step, .children = child->children, .id = child->id});
            result_children.push_back(sub_plan.getLastNode());
            plan_nodes.push(sub_plan.getLastNode());
        }
        current->children.swap(result_children);
    }

    freshPlan();

    return sub_plan;
}
QueryPipelineBuilderPtr QueryPlan::buildQueryPipeline(
    const QueryPlanOptimizationSettings & optimization_settings,
    const BuildQueryPipelineSettings & build_pipeline_settings)
{
    checkInitialized();
    optimize(optimization_settings);

    struct Frame
    {
        Node * node = {};
        QueryPipelineBuilders pipelines = {};
    };

    QueryPipelineBuilderPtr last_pipeline;


    std::stack<Frame> stack;
    stack.push(Frame{.node = root});

    while (!stack.empty())
    {
        auto & frame = stack.top();

        if (last_pipeline)
        {
            frame.pipelines.emplace_back(std::move(last_pipeline));
            last_pipeline = nullptr;
        }

        size_t next_child = frame.pipelines.size();
        if (next_child == frame.node->children.size())
        {
            bool limit_max_threads = frame.pipelines.empty();
            last_pipeline = frame.node->step->updatePipeline(std::move(frame.pipelines), build_pipeline_settings);

            if (limit_max_threads && max_threads)
                last_pipeline->limitMaxThreads(max_threads);

            stack.pop();
        }
        else
            stack.push(Frame{.node = frame.node->children[next_child]});
    }

    last_pipeline->setProgressCallback(build_pipeline_settings.progress_callback);
    last_pipeline->setProcessListElement(build_pipeline_settings.process_list_element);
    last_pipeline->addResources(std::move(resources));

    return last_pipeline;
}

static void explainStep(const IQueryPlanStep & step, JSONBuilder::JSONMap & map, const QueryPlan::ExplainPlanOptions & options)
{
    map.add("Node Type", step.getName());

    if (options.description)
    {
        const auto & description = step.getStepDescription();
        if (!description.empty())
            map.add("Description", description);
    }

    if (options.header && step.hasOutputStream())
    {
        auto header_array = std::make_unique<JSONBuilder::JSONArray>();

        for (const auto & output_column : step.getOutputStream().header)
        {
            auto column_map = std::make_unique<JSONBuilder::JSONMap>();
            column_map->add("Name", output_column.name);
            if (output_column.type)
                column_map->add("Type", output_column.type->getName());

            header_array->add(std::move(column_map));
        }

        map.add("Header", std::move(header_array));
    }

    if (options.actions)
        step.describeActions(map);

    if (options.indexes)
        step.describeIndexes(map);
}

JSONBuilder::ItemPtr QueryPlan::explainPlan(const ExplainPlanOptions & options)
{
    checkInitialized();

    struct Frame
    {
        Node * node = {};
        size_t next_child = 0;
        std::unique_ptr<JSONBuilder::JSONMap> node_map = {};
        std::unique_ptr<JSONBuilder::JSONArray> children_array = {};
    };

    std::stack<Frame> stack;
    stack.push(Frame{.node = root});

    std::unique_ptr<JSONBuilder::JSONMap> tree;

    while (!stack.empty())
    {
        auto & frame = stack.top();

        if (frame.next_child == 0)
        {
            if (!frame.node->children.empty())
                frame.children_array = std::make_unique<JSONBuilder::JSONArray>();

            frame.node_map = std::make_unique<JSONBuilder::JSONMap>();
            explainStep(*frame.node->step, *frame.node_map, options);
        }

        if (frame.next_child < frame.node->children.size())
        {
            stack.push(Frame{frame.node->children[frame.next_child]});
            ++frame.next_child;
        }
        else
        {
            if (frame.children_array)
                frame.node_map->add("Plans", std::move(frame.children_array));

            tree.swap(frame.node_map);
            stack.pop();

            if (!stack.empty())
                stack.top().children_array->add(std::move(tree));
        }
    }

    return tree;
}

static void explainStep(
    const IQueryPlanStep & step,
    IQueryPlanStep::FormatSettings & settings,
    const QueryPlan::ExplainPlanOptions & options)
{
    std::string prefix(settings.offset, ' ');
    settings.out << prefix;
    settings.out << step.getName();

    const auto & description = step.getStepDescription();
    if (options.description && !description.empty())
        settings.out <<" (" << description << ')';

    settings.out.write('\n');

    if (options.header)
    {
        settings.out << prefix;

        if (!step.hasOutputStream())
            settings.out << "No header";
        else if (!step.getOutputStream().header)
            settings.out << "Empty header";
        else
        {
            settings.out << "Header: ";
            bool first = true;

            for (const auto & elem : step.getOutputStream().header)
            {
                if (!first)
                    settings.out << "\n" << prefix << "        ";

                first = false;
                elem.dumpNameAndType(settings.out);
            }
        }
        settings.out.write('\n');

    }

    if (options.sorting)
    {
        if (step.hasOutputStream())
        {
            settings.out << prefix << "Sorting (" << step.getOutputStream().sort_scope << ")";
            if (step.getOutputStream().sort_scope != DataStream::SortScope::None)
            {
                settings.out << ": ";
                dumpSortDescription(step.getOutputStream().sort_description, settings.out);
            }
            settings.out.write('\n');
        }
    }

    if (options.actions)
        step.describeActions(settings);

    if (options.indexes)
        step.describeIndexes(settings);
}

std::string debugExplainStep(const IQueryPlanStep & step)
{
    WriteBufferFromOwnString out;
    IQueryPlanStep::FormatSettings settings{.out = out};
    QueryPlan::ExplainPlanOptions options{.actions = true};
    explainStep(step, settings, options);
    return out.str();
}

void QueryPlan::explainPlan(WriteBuffer & buffer, const ExplainPlanOptions & options)
{
    checkInitialized();

    IQueryPlanStep::FormatSettings settings{.out = buffer, .write_header = options.header};

    struct Frame
    {
        Node * node = {};
        bool is_description_printed = false;
        size_t next_child = 0;
    };

    std::stack<Frame> stack;
    stack.push(Frame{.node = root});

    while (!stack.empty())
    {
        auto & frame = stack.top();

        if (!frame.is_description_printed)
        {
            settings.offset = (stack.size() - 1) * settings.indent;
            explainStep(*frame.node->step, settings, options);
            frame.is_description_printed = true;
        }

        if (frame.next_child < frame.node->children.size())
        {
            stack.push(Frame{frame.node->children[frame.next_child]});
            ++frame.next_child;
        }
        else
            stack.pop();
    }
}

static void explainPipelineStep(IQueryPlanStep & step, IQueryPlanStep::FormatSettings & settings)
{
    settings.out << String(settings.offset, settings.indent_char) << "(" << step.getName() << ")\n";

    size_t current_offset = settings.offset;
    step.describePipeline(settings);
    if (current_offset == settings.offset)
        settings.offset += settings.indent;
}

void QueryPlan::explainPipeline(WriteBuffer & buffer, const ExplainPipelineOptions & options)
{
    checkInitialized();

    IQueryPlanStep::FormatSettings settings{.out = buffer, .write_header = options.header};

    struct Frame
    {
        Node * node = {};
        size_t offset = 0;
        bool is_description_printed = false;
        size_t next_child = 0;
    };

    std::stack<Frame> stack;
    stack.push(Frame{.node = root});

    while (!stack.empty())
    {
        auto & frame = stack.top();

        if (!frame.is_description_printed)
        {
            settings.offset = frame.offset;
            explainPipelineStep(*frame.node->step, settings);
            frame.offset = settings.offset;
            frame.is_description_printed = true;
        }

        if (frame.next_child < frame.node->children.size())
        {
            stack.push(Frame{frame.node->children[frame.next_child], frame.offset});
            ++frame.next_child;
        }
        else
            stack.pop();
    }
}

static void updateDataStreams(QueryPlan::Node & root)
{
    class UpdateDataStreams : public QueryPlanVisitor<UpdateDataStreams, false>
    {
    public:
        explicit UpdateDataStreams(QueryPlan::Node * root_) : QueryPlanVisitor<UpdateDataStreams, false>(root_) { }

        static bool visitTopDownImpl(QueryPlan::Node * /*current_node*/, QueryPlan::Node * /*parent_node*/) { return true; }

        static void visitBottomUpImpl(QueryPlan::Node * current_node, QueryPlan::Node * parent_node)
        {
            if (!parent_node || parent_node->children.size() != 1)
                return;

            if (!current_node->step->hasOutputStream())
                return;

            if (auto * parent_transform_step = dynamic_cast<ITransformingStep *>(parent_node->step.get()); parent_transform_step)
                parent_transform_step->updateInputStream(current_node->step->getOutputStream());
        }
    };

    UpdateDataStreams(&root).visit();
}

void QueryPlan::optimize(const QueryPlanOptimizationSettings & optimization_settings)
{
    /// optimization need to be applied before "mergeExpressions" optimization
    /// it removes redundant sorting steps, but keep underlying expressions,
    /// so "mergeExpressions" optimization handles them afterwards
    if (optimization_settings.remove_redundant_sorting)
        QueryPlanOptimizations::tryRemoveRedundantSorting(root);

    QueryPlanOptimizations::optimizeTreeFirstPass(optimization_settings, *root, nodes);
    QueryPlanOptimizations::optimizeTreeSecondPass(optimization_settings, *root, nodes);

    updateDataStreams(*root);
}

void QueryPlan::explainEstimate(MutableColumns & columns)
{
    checkInitialized();

    struct EstimateCounters
    {
        std::string database_name;
        std::string table_name;
        UInt64 parts = 0;
        UInt64 rows = 0;
        UInt64 marks = 0;

        EstimateCounters(const std::string & database, const std::string & table) : database_name(database), table_name(table)
        {
        }
    };

    using CountersPtr = std::shared_ptr<EstimateCounters>;
    std::unordered_map<std::string, CountersPtr> counters;
    using processNodeFuncType = std::function<void(const Node * node)>;
    processNodeFuncType process_node = [&counters, &process_node] (const Node * node)
    {
        if (!node)
            return;
        if (const auto * step = dynamic_cast<ReadFromMergeTree*>(node->step.get()))
        {
            const auto & id = step->getStorageID();
            auto key = id.database_name + "." + id.table_name;
            auto it = counters.find(key);
            if (it == counters.end())
            {
                it = counters.insert({key, std::make_shared<EstimateCounters>(id.database_name, id.table_name)}).first;
            }
            it->second->parts += step->getSelectedParts();
            it->second->rows += step->getSelectedRows();
            it->second->marks += step->getSelectedMarks();
        }
        for (const auto * child : node->children)
            process_node(child);
    };
    process_node(root);

    for (const auto & counter : counters)
    {
        size_t index = 0;
        const auto & database_name = counter.second->database_name;
        const auto & table_name = counter.second->table_name;
        columns[index++]->insertData(database_name.c_str(), database_name.size());
        columns[index++]->insertData(table_name.c_str(), table_name.size());
        columns[index++]->insert(counter.second->parts);
        columns[index++]->insert(counter.second->rows);
        columns[index++]->insert(counter.second->marks);
    }
}

QueryPlan::Nodes QueryPlan::detachNodes(QueryPlan && plan)
{
    return std::move(plan.nodes);
}

PlanNodePtr QueryPlan::getPlanNodeById(PlanNodeId node_id) const
{
    if (plan_node)
        if (auto res = plan_node->getNodeById(node_id))
            return res;

    for (const auto & cte : cte_info.getCTEs())
    {
        if (auto res = cte.second->getNodeById(node_id))
            return res;
    }

    return nullptr;
}

}
