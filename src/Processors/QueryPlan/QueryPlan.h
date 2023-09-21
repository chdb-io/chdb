#pragma once

#include <Core/Names.h>
#include <Interpreters/Context_fwd.h>
#include <Columns/IColumn.h>
#include <QueryPipeline/QueryPlanResourceHolder.h>
#include <Processors/QueryPlan/CTEInfo.h>
#include <Processors/QueryPlan/PlanNodeIdAllocator.h>


#include <list>
#include <memory>
#include <set>
#include <vector>

namespace DB
{

class DataStream;

class IQueryPlanStep;
using QueryPlanStepPtr = std::shared_ptr<IQueryPlanStep>;

class QueryPipelineBuilder;
using QueryPipelineBuilderPtr = std::unique_ptr<QueryPipelineBuilder>;

class WriteBuffer;

class QueryPlan;
using QueryPlanPtr = std::unique_ptr<QueryPlan>;

class Pipe;

struct QueryPlanOptimizationSettings;
struct BuildQueryPipelineSettings;

namespace JSONBuilder
{
    class IItem;
    using ItemPtr = std::unique_ptr<IItem>;
}

/// A tree of query steps.
/// The goal of QueryPlan is to build QueryPipeline.
/// QueryPlan let delay pipeline creation which is helpful for pipeline-level optimizations.
class QueryPlan
{
public:
    QueryPlan();
    ~QueryPlan();
    QueryPlan(QueryPlan &&) noexcept;
    QueryPlan & operator=(QueryPlan &&) noexcept;

    void unitePlans(QueryPlanStepPtr step, std::vector<QueryPlanPtr> plans);
    void addStep(QueryPlanStepPtr step);

    bool isInitialized() const { return root != nullptr; } /// Tree is not empty
    bool isCompleted() const; /// Tree is not empty and root hasOutputStream()
    const DataStream & getCurrentDataStream() const; /// Checks that (isInitialized() && !isCompleted())

    void optimize(const QueryPlanOptimizationSettings & optimization_settings);

    QueryPipelineBuilderPtr buildQueryPipeline(
        const QueryPlanOptimizationSettings & optimization_settings,
        const BuildQueryPipelineSettings & build_pipeline_settings);

    struct ExplainPlanOptions
    {
        /// Add output header to step.
        bool header = false;
        /// Add description of step.
        bool description = true;
        /// Add detailed information about step actions.
        bool actions = false;
        /// Add information about indexes actions.
        bool indexes = false;
        /// Add information about sorting
        bool sorting = false;
    };

    struct ExplainPipelineOptions
    {
        /// Show header of output ports.
        bool header = false;
    };

    JSONBuilder::ItemPtr explainPlan(const ExplainPlanOptions & options);
    void explainPlan(WriteBuffer & buffer, const ExplainPlanOptions & options);
    void explainPipeline(WriteBuffer & buffer, const ExplainPipelineOptions & options);
    void explainEstimate(MutableColumns & columns);

    /// Do not allow to change the table while the pipeline alive.
    void addTableLock(TableLockHolder lock) { resources.table_locks.emplace_back(std::move(lock)); }
    void addInterpreterContext(std::shared_ptr<const Context> context) { resources.interpreter_context.emplace_back(std::move(context)); }
    void addStorageHolder(StoragePtr storage) { resources.storage_holders.emplace_back(std::move(storage)); }

    void addResources(QueryPlanResourceHolder resources_) { resources = std::move(resources_); }

    /// Set upper limit for the recommend number of threads. Will be applied to the newly-created pipelines.
    /// TODO: make it in a better way.
    void setMaxThreads(size_t max_threads_) { max_threads = max_threads_; }
    size_t getMaxThreads() const { return max_threads; }

    /// Tree node. Step and it's children.
    struct Node
    {
        QueryPlanStepPtr step;
        std::vector<Node *> children = {};
    };

    using Nodes = std::list<Node>;
    using CTEId = UInt32;
    using CTENodes = std::unordered_map<CTEId, Node *>;

    Nodes & getNodes() { return nodes; }

    Node * getRoot() { return root; }
    const Node * getRoot() const { return root; }
    PlanNodePtr getPlanNodeRoot() const { return plan_node; }
    void setRoot(Node * root_) { root = root_; }
    void setPlanNodeRoot(PlanNodePtr plan_node_) { plan_node = plan_node_; }
    CTENodes & getCTENodes() { return cte_nodes; }

    Node * getLastNode() { return &nodes.back(); }

    void addNode(QueryPlan::Node && node_);

    void addRoot(QueryPlan::Node && node_);
    UInt32 newPlanNodeId() { return (*max_node_id)++; }
    PlanNodePtr & getPlanNode() { return plan_node; }
    CTEInfo & getCTEInfo() { return cte_info; }
    PlanNodePtr getPlanNodeById(PlanNodeId node_id) const;
    const CTEInfo & getCTEInfo() const { return cte_info; }

    QueryPlan getSubPlan(QueryPlan::Node * node_);

    void freshPlan();

    size_t getSize() const { return nodes.size(); }

    void setResetStepId(bool reset_id) { reset_step_id = reset_id; }

    Node * getRootNode() const { return root; }
    static Nodes detachNodes(QueryPlan && plan);

private:
    QueryPlanResourceHolder resources;
    Nodes nodes;
    CTENodes cte_nodes;

    Node * root = nullptr;
    PlanNodePtr plan_node = nullptr;
    CTEInfo cte_info;
    PlanNodeIdAllocatorPtr id_allocator;

    void checkInitialized() const;
    void checkNotCompleted() const;

    /// Those fields are passed to QueryPipeline.
    size_t max_threads = 0;
    std::vector<std::shared_ptr<Context>> interpreter_context;
    std::shared_ptr<UInt32> max_node_id;
    //Whether reset step id in serialize()，use for explain analyze.
    bool reset_step_id = true;
};

std::string debugExplainStep(const IQueryPlanStep & step);

}
