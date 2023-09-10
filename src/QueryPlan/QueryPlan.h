/*
 * Copyright (2022) Bytedance Ltd. and/or its affiliates
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <Core/Names.h>
#include <Interpreters/Context_fwd.h>
#include <QueryPlan/CTEInfo.h>
#include <QueryPlan/PlanNodeIdAllocator.h>
#include <Poco/Logger.h>

#include <list>
#include <memory>
#include <set>
#include <vector>

namespace DB
{

class DataStream;

class IQueryPlanStep;

class QueryPipeline;
using QueryPipelinePtr = std::unique_ptr<QueryPipeline>;

class WriteBuffer;
class ReadBuffer;

class QueryPlan;
using QueryPlanPtr = std::unique_ptr<QueryPlan>;

class Pipe;

struct QueryPlanOptimizationSettings;
struct BuildQueryPipelineSettings;

class PlanNodeBase;
using PlanNodePtr = std::shared_ptr<PlanNodeBase>;
using PlanNodes = std::vector<PlanNodePtr>;

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
    QueryPlan(QueryPlan &&);
    QueryPlan(PlanNodePtr root, PlanNodeIdAllocatorPtr idAllocator);
    QueryPlan(PlanNodePtr root_, CTEInfo cte_info, PlanNodeIdAllocatorPtr id_allocator_);

    QueryPlan & operator=(QueryPlan &&);

    void allocateLocalTable(ContextPtr context);
    PlanNodeIdAllocatorPtr & getIdAllocator() { return id_allocator; }
    void update(PlanNodePtr plan) { plan_node = std::move(plan); }

    void unitePlans(QueryPlanStepPtr step, std::vector<QueryPlanPtr> plans);
    void addStep(QueryPlanStepPtr step, PlanNodes children = {});

    bool isInitialized() const { return root != nullptr; } /// Tree is not empty
    bool isCompleted() const; /// Tree is not empty and root hasOutputStream()
    const DataStream & getCurrentDataStream() const; /// Checks that (isInitialized() && !isCompleted())

    void optimize(const QueryPlanOptimizationSettings & optimization_settings);

    QueryPipelinePtr buildQueryPipeline(
        const QueryPlanOptimizationSettings & optimization_settings,
        const BuildQueryPipelineSettings & build_pipeline_settings);

    /// add step_id for processors
    static void updatePipelineStepInfo(QueryPipelinePtr & pipeline_ptr, QueryPlanStepPtr & step,size_t step_id);
    /// If initialized, build pipeline and convert to pipe. Otherwise, return empty pipe.
    Pipe convertToPipe(
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
    };

    struct ExplainPipelineOptions
    {
        /// Show header of output ports.
        bool header = false;
    };

    JSONBuilder::ItemPtr explainPlan(const ExplainPlanOptions & options);
    void explainPlan(WriteBuffer & buffer, const ExplainPlanOptions & options) const;
    void explainPipeline(WriteBuffer & buffer, const ExplainPipelineOptions & options) const;
    void explainPipelineWithOptimizer(WriteBuffer & buffer, const ExplainPipelineOptions & options) const;

    /// Set upper limit for the recommend number of threads. Will be applied to the newly-created pipelines.
    /// TODO: make it in a better way.
    void setMaxThreads(size_t max_threads_) { max_threads = max_threads_; }
    size_t getMaxThreads() const { return max_threads; }

    void addInterpreterContext(std::shared_ptr<Context> context);

    void serialize(WriteBuffer & buffer) const;
    void deserialize(ReadBuffer & buffer);

    /// Tree node. Step and it's children.
    struct Node
    {
        QueryPlanStepPtr step;
        std::vector<Node *> children = {};
        /**
         * Only used for serialize query plan for distributed query.
         */
        mutable size_t id;
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
private:
    Poco::Logger * log = &Poco::Logger::get("QueryPlan");
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
