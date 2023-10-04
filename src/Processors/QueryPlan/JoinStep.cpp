#include <Interpreters/IJoin.h>
#include <Interpreters/TableJoin.h>
#include <Processors/QueryPlan/JoinStep.h>
#include <Processors/Transforms/JoiningTransform.h>
#include <QueryPipeline/QueryPipelineBuilder.h>
#include <Common/typeid_cast.h>

namespace DB
{

namespace ErrorCodes
{
    extern const int LOGICAL_ERROR;
}

JoinPtr JoinStep::makeJoin(ContextPtr context)
{
    const auto & settings = context->getSettingsRef();
    auto table_join = std::make_shared<TableJoin>(settings, context->getTempDataOnDisk());

    // todo support storage join
    //    if (table_to_join.database_and_table_name)
    //    {
    //        auto joined_table_id = context->resolveStorageID(table_to_join.database_and_table_name);
    //        StoragePtr table = DatabaseCatalog::instance().tryGetTable(joined_table_id, context);
    //        if (table)
    //        {
    //            if (dynamic_cast<StorageJoin *>(table.get()) ||
    //                dynamic_cast<StorageDictionary *>(table.get()))
    //                table_join->joined_storage = table;
    //        }
    //    }
    table_join->deduplicateAndQualifyColumnNames(input_streams[0].header.getNameSet(), "");

    auto using_ast = std::make_shared<ASTExpressionList>();
    for (size_t index = 0; index < left_keys.size(); ++index)
    {
        ASTPtr left = std::make_shared<ASTIdentifier>(left_keys[index]);
        ASTPtr right = std::make_shared<ASTIdentifier>(right_keys[index]);
        if (has_using)
        {
            table_join->renames[left_keys[index]] = right_keys[index];
            // table_join->addUsingKey(left, settings.join_using_null_safe);
            table_join->addUsingKey(left);
            using_ast->children.emplace_back(left);
        }
        else
        {
            // table_join->addOnKeys(left, right, false);
            table_join->addOnKeys(left, right);
        }
    }

    if (has_using)
    {
        table_join->table_join.using_expression_list = using_ast;
    }

    for (const auto & item : output_stream->header)
    {
        if (!input_streams[0].header.has(item.name))
        {
            NameAndTypePair joined_column{item.name, item.type};
            table_join->addJoinedColumn(joined_column);
        }
    }

    table_join->setAsofInequality(asof_inequality);
    if (context->getSettings().enforce_all_join_to_any_join)
    {
        strictness = JoinStrictness::RightAny;
    }

    table_join->table_join.strictness = strictness;
    table_join->table_join.kind = isCrossJoin() ? JoinKind::Cross : kind;

    if (enforceNestLoopJoin())
    {
        if (!settings.enable_nested_loop_join)
            throw Exception("set enable_nested_loop_join=1 to enable outer join with filter", ErrorCodes::NOT_IMPLEMENTED);
        table_join->setJoinAlgorithm(JoinAlgorithm::NESTED_LOOP_JOIN);
        table_join->table_join.on_expression = filter->clone();
        table_join->table_join.kind = isCrossJoin() ? JoinKind::Inner : kind;
    }

    bool allow_merge_join = table_join->allowMergeJoin();

    /// HashJoin with Dictionary optimisation
    auto l_sample_block = input_streams[1].header;
    auto sample_block = input_streams[1].header;
    String dict_name;
    String key_name;
    if (table_join->forceNestedLoopJoin())
        return std::make_shared<NestedLoopJoin>(table_join, sample_block, context);
    else if (table_join->forceHashJoin() || (table_join->preferMergeJoin() && !allow_merge_join))
    {
        if (table_join->allowParallelHashJoin() && join_algorithm == JoinAlgorithm::PARALLEL_HASH)
        {
            LOG_TRACE(&Poco::Logger::get("JoinStep::makeJoin"), "will use ConcurrentHashJoin");
            return std::make_shared<ConcurrentHashJoin>(table_join, context->getSettings().max_threads, sample_block);
        }
        return std::make_shared<HashJoin>(table_join, sample_block);
    }
    else if (table_join->forceMergeJoin() || (table_join->preferMergeJoin() && allow_merge_join))
        return std::make_shared<MergeJoin>(table_join, sample_block);
    else if (table_join->forceGraceHashLoopJoin())
        return std::make_shared<GraceHashJoin>(context, table_join, l_sample_block, sample_block, context->getTempDataOnDisk());
    return std::make_shared<JoinSwitcher>(table_join, sample_block);
}

JoinStep::JoinStep(
    const DataStream & left_stream_,
    const DataStream & right_stream_,
    JoinPtr join_,
    size_t max_block_size_,
    size_t max_streams_,
    bool keep_left_read_in_order_,
    bool is_ordered_,
    PlanHints hints_)
    : join(std::move(join_))
    , max_block_size(max_block_size_)
    , max_streams(max_streams_)
    , keep_left_read_in_order(keep_left_read_in_order_)
    , is_ordered(is_ordered_)
{
    input_streams = {left_stream_, right_stream_};
    output_stream = DataStream
    {
        .header = JoiningTransform::transformHeader(left_stream_.header, join),
    };
    hints = std::move(hints_);
}

JoinStep::JoinStep(
    DataStreams input_streams_,
    DataStream output_stream_,
    JoinKind kind_,
    JoinStrictness strictness_,
    size_t max_streams_,
    bool keep_left_read_in_order_,
    Names left_keys_,
    Names right_keys_,
    ConstASTPtr filter_,
    bool has_using_,
    std::optional<std::vector<bool>> require_right_keys_,
    ASOF::Inequality asof_inequality_,
    DistributionType distribution_type_,
    JoinAlgorithm join_algorithm_,
    bool is_magic_,
    bool is_ordered_,
    PlanHints hints_)
    : kind(kind_)
    , strictness(strictness_)
    , max_streams(max_streams_)
    , keep_left_read_in_order(keep_left_read_in_order_)
    , left_keys(std::move(left_keys_))
    , right_keys(std::move(right_keys_))
    , filter(std::move(filter_))
    , has_using(has_using_)
    , require_right_keys(std::move(require_right_keys_))
    , asof_inequality(asof_inequality_)
    , distribution_type(distribution_type_)
    , join_algorithm(join_algorithm_)
    , is_magic(is_magic_)
    , is_ordered(is_ordered_)
{
    input_streams = std::move(input_streams_);
    output_stream = std::move(output_stream_);
    hints = std::move(hints_);
}

void JoinStep::setInputStreams(const DataStreams & input_streams_)
{
    input_streams = input_streams_;
}

QueryPipelineBuilderPtr JoinStep::updatePipeline(QueryPipelineBuilders pipelines, const BuildQueryPipelineSettings & settings)
{
    if (pipelines.size() != 2)
        throw Exception(ErrorCodes::LOGICAL_ERROR, "JoinStep expect two input steps");

    if (!join)
    {
        join = makeJoin(settings.context);
        max_block_size = settings.context->getSettingsRef().max_block_size;
    }

    QueryPipelineBuilderPtr pipeline;

    if (join->pipelineType() == JoinPipelineType::YShaped)
    {
        pipeline = QueryPipelineBuilder::joinPipelinesYShaped(
            std::move(pipelines[0]), std::move(pipelines[1]), join, output_stream->header, max_block_size, &processors);
        pipeline->resize(max_streams);
    }
    else
    {
        pipeline = QueryPipelineBuilder::joinPipelinesRightLeft(
            std::move(pipelines[0]),
            std::move(pipelines[1]),
            join,
            output_stream->header,
            max_block_size,
            max_streams,
            keep_left_read_in_order,
            &processors);
    }

    // if NestLoopJoin is choose, no need to add filter stream.
    if (filter && !PredicateUtils::isTruePredicate(filter) && join->getType() != JoinType::NestedLoop)
    {
        Names output;
        auto header = pipeline->getHeader();
        for (const auto & item : header)
            output.emplace_back(item.name);
        output.emplace_back(filter->getColumnName());

        auto actions_dag = createExpressionActions(settings.context, header.getNamesAndTypesList(), output, filter->clone());
        auto expression = std::make_shared<ExpressionActions>(actions_dag, settings.getActionsSettings());

        pipeline->addSimpleTransform(
            [&](const Block & input_header, QueryPipeline::StreamType stream_type)
            {
                bool on_totals = stream_type == QueryPipeline::StreamType::Totals;
                return std::make_shared<FilterTransform>(input_header, expression, filter->getColumnName(), true, on_totals);
            });
    }

    projection(*pipeline, output_stream->header, settings);
    return pipeline;
}

bool JoinStep::enforceNestLoopJoin() const
{
    if (filter && !PredicateUtils::isTruePredicate(filter))
    {
        bool strictness_join = strictness != JoinStrictness::Unspecified && strictness != JoinStrictness::All;
        bool outer_join = kind != JoinKind::Inner && kind != JoinKind::Cross;
        return strictness_join || outer_join;
    }
    return false;
}

bool JoinStep::enforceGraceHashJoin() const
{
    return false;
}

bool JoinStep::supportReorder(bool support_filter, bool support_cross) const
{
    if (!support_filter && !PredicateUtils::isTruePredicate(filter))
        return false;

    if (require_right_keys || has_using)
        return false;

    if (strictness != JoinStrictness::Unspecified && strictness != JoinStrictness::All)
        return false;

    bool cross_join = isCrossJoin();
    if (!support_cross && cross_join)
        return false;

    if (support_cross && cross_join)
        return !is_magic;

    return kind == JoinKind::Inner && !left_keys.empty() && !is_magic;
}

bool JoinStep::allowPushDownToRight() const
{
    return join->pipelineType() == JoinPipelineType::YShaped;
}

void JoinStep::describePipeline(FormatSettings & settings) const
{
    IQueryPlanStep::describePipeline(processors, settings);
}

void JoinStep::updateInputStream(const DataStream & new_input_stream_, size_t idx)
{
    if (idx == 0)
    {
        input_streams = {new_input_stream_, input_streams.at(1)};
        output_stream = DataStream
        {
            .header = JoiningTransform::transformHeader(new_input_stream_.header, join),
        };
    }
    else
    {
        input_streams = {input_streams.at(0), new_input_stream_};
    }
}

static ITransformingStep::Traits getStorageJoinTraits()
{
    return ITransformingStep::Traits{
        {
            .preserves_distinct_columns = false,
            .returns_single_stream = false,
            .preserves_number_of_streams = true,
            .preserves_sorting = false,
        },
        {
            .preserves_number_of_rows = false,
        }};
}

FilledJoinStep::FilledJoinStep(const DataStream & input_stream_, JoinPtr join_, size_t max_block_size_)
    : ITransformingStep(
        input_stream_,
        JoiningTransform::transformHeader(input_stream_.header, join_),
        getStorageJoinTraits())
    , join(std::move(join_))
    , max_block_size(max_block_size_)
{
    if (!join->isFilled())
        throw Exception(ErrorCodes::LOGICAL_ERROR, "FilledJoinStep expects Join to be filled");
}

void FilledJoinStep::setInputStreams(const DataStreams & input_streams_)
{
    input_streams = input_streams_;
    output_stream->header = JoiningTransform::transformHeader(input_streams_[0].header, join);
}

void FilledJoinStep::transformPipeline(QueryPipelineBuilder & pipeline, const BuildQueryPipelineSettings &)
{
    bool default_totals = false;
    if (!pipeline.hasTotals() && join->getTotals())
    {
        pipeline.addDefaultTotals();
        default_totals = true;
    }

    auto finish_counter = std::make_shared<JoiningTransform::FinishCounter>(pipeline.getNumStreams());

    pipeline.addSimpleTransform([&](const Block & header, QueryPipelineBuilder::StreamType stream_type)
    {
        bool on_totals = stream_type == QueryPipelineBuilder::StreamType::Totals;
        auto counter = on_totals ? nullptr : finish_counter;
        return std::make_shared<JoiningTransform>(
            header, output_stream->header, join, max_block_size, on_totals, default_totals, join_parallel_left_right, counter);
    });
}

void FilledJoinStep::updateOutputStream()
{
    output_stream = createOutputStream(
        input_streams.front(), JoiningTransform::transformHeader(input_streams.front().header, join), getDataStreamTraits());
}

std::shared_ptr<IQueryPlanStep> JoinStep::copy(ContextPtr) const
{
    return std::make_shared<JoinStep>(
        input_streams,
        output_stream.value(),
        kind,
        strictness,
        max_streams,
        keep_left_read_in_order,
        left_keys,
        right_keys,
        filter,
        has_using,
        require_right_keys,
        asof_inequality,
        distribution_type,
        join_algorithm,
        is_magic,
        is_ordered,
        hints);
}


bool JoinStep::mustReplicate() const
{
    if (left_keys.empty() && (kind == JoinKind::Inner || kind == JoinKind::Left || kind == JoinKind::Cross))
    {
        // There is nothing to partition on
        return true;
    }
    return false;
}

bool JoinStep::mustRepartition() const
{
    return kind == JoinKind::Right || kind == JoinKind::Full;
}


std::shared_ptr<IQueryPlanStep> FilledJoinStep::copy(ContextPtr) const
{
    return std::make_shared<FilledJoinStep>(input_streams[0], join, max_block_size);
}

}
