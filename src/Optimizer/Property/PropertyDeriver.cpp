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

#include <algorithm>
#include <memory>
#include <Optimizer/Property/PropertyDeriver.h>

#include <Core/Names.h>
#include <Interpreters/StorageID.h>
#include <Optimizer/ExpressionRewriter.h>
#include <Optimizer/Property/Property.h>
#include <Optimizer/SymbolsExtractor.h>
#include <Optimizer/DomainTranslator.h>
#include <Optimizer/Utils.h>
#include <Parsers/ASTLiteral.h>
#include <Parsers/IAST_fwd.h>
// #include <Processors/QueryPlan/ExchangeStep.h>
#include <Processors/QueryPlan/FilterStep.h>
#include <Processors/QueryPlan/ProjectionStep.h>
#include <Processors/QueryPlan/UnionStep.h>
#include <Storages/StorageDistributed.h>

namespace DB
{
namespace ErrorCodes
{
    extern const int OPTIMIZER_NONSUPPORT;
}

Property PropertyDeriver::deriveProperty(QueryPlanStepPtr step, ContextMutablePtr & context)
{
    PropertySet property_set;
    return deriveProperty(step, property_set, context);
}

Property PropertyDeriver::deriveProperty(QueryPlanStepPtr step, Property & input_property, ContextMutablePtr & context)
{
    PropertySet input_properties = std::vector<Property>();
    input_properties.emplace_back(input_property);
    return deriveProperty(step, input_properties, context);
}

Property PropertyDeriver::deriveProperty(QueryPlanStepPtr step, PropertySet & input_properties, ContextMutablePtr & context)
{
    DeriverContext deriver_context{input_properties, context};
    static DeriverVisitor visitor{};
    auto result = VisitorUtil::accept(step, visitor, deriver_context);

    CTEDescriptions cte_descriptions;
    for (auto & property : input_properties)
        for (auto & item : property.getCTEDescriptions())
            cte_descriptions.emplace(item);
    if (!cte_descriptions.empty())
        result.setCTEDescriptions(cte_descriptions);

    return result;
}

Property PropertyDeriver::deriveStorageProperty(const StoragePtr & storage, ContextMutablePtr & context)
{
    if (storage->getDatabaseName() == "system")
    {
        return Property{Partitioning(Partitioning::Handle::SINGLE), Partitioning(Partitioning::Handle::ARBITRARY)};
    }

    bool isBucketTable = storage->isBucketTable();

    Sorting sorting;
    for (const auto & col : storage->getInMemoryMetadataPtr()->getSortingKeyColumns())
    {
        sorting.emplace_back(SortColumn(col, SortOrder::ASC_NULLS_FIRST));
    }
    // TODO: fix bucket table property
    #if 0
    if (isBucketTable)
    {
        if (auto * merge_tree = dynamic_cast<MergeTreeMetaBase *>(storage.get()))
        {
            bool clustered;
            context->getCnchCatalog()->getTableClusterStatus(merge_tree->getStorageUUID(), clustered);
            if (clustered)
            {
                auto metadata = merge_tree->getInMemoryMetadataPtr();
                Names clusterBy = metadata->cluster_by_columns;
                UInt64 buckets = metadata->cluster_by_total_bucket_number;
                NameToNameMap translation;

                auto id_to_table = merge_tree->parseUnderlyingDictionaryTablse(merge_tree->settings.underlying_dictionary_tables);
                Names sec_cols;
                for (const auto & item : id_to_table)
                {
                    sec_cols.emplace_back(item.first);
                }

                return Property{
                    Partitioning{
                        Partitioning::Handle::BUCKET_TABLE, clusterBy, true, buckets, true, Partitioning::Component::ANY, sec_cols},
                    Partitioning{},
                    sorting};
            }
        }
    }
    #endif

    return Property{Partitioning(Partitioning::Handle::UNKNOWN), Partitioning(Partitioning::Handle::UNKNOWN), sorting};
}

Property DeriverVisitor::visitStep(const IQueryPlanStep &, DeriverContext & context)
{
    return context.getInput()[0].clearSorting();
}

Property DeriverVisitor::visitProjectionStep(const ProjectionStep & step, DeriverContext & context)
{
    auto assignments = step.getAssignments();
    std::unordered_map<String, String> identities = Utils::computeIdentityTranslations(assignments);
    std::unordered_map<String, String> revert_identifies;

    for (auto & item : identities)
    {
        revert_identifies[item.second] = item.first;
    }
    Property translated;
    if (!context.getInput().empty())
    {
        translated = context.getInput()[0].translate(revert_identifies);
    }

    // if partition columns are pruned, the output data has no property.
    if (translated.getNodePartitioning().getPartitioningColumns().size()
        != context.getInput()[0].getNodePartitioning().getPartitioningColumns().size())
    {
        return Property{};
    }

    if (translated.getStreamPartitioning().getPartitioningColumns().size()
        != context.getInput()[0].getStreamPartitioning().getPartitioningColumns().size())
    {
        // TODO stream partition
    }
    return translated;
}

Property DeriverVisitor::visitFilterStep(const FilterStep & step, DeriverContext & context)
{
    Predicate::DomainTranslator translator{context.getContext()};
    // TODO, remove clone. step.getFilter()->clone()
    Predicate::ExtractionResult result
        = translator.getExtractionResult(step.getFilter()->clone(), step.getOutputStream().header.getNamesAndTypes());
    std::optional<Predicate::FieldWithTypeMap> values = result.tuple_domain.extractFixedValues();
    if (values.has_value())
    {
        Property input = context.getInput()[0];
        std::map<String, FieldWithType> filter_values;
        const Constants & origin_constants = input.getConstants();
        for (const auto & value : origin_constants.getValues())
        {
            filter_values[value.first] = value.second;
        }
        for (auto & value : values.value())
        {
            filter_values[value.first] = value.second;
        }
        Constants filter_constants{filter_values};
        input.setConstants(filter_constants);
        return input;
    }

    return context.getInput()[0];
}

Property DeriverVisitor::visitJoinStep(const JoinStep & step, DeriverContext & context)
{
    std::unordered_map<String, String> identities;
    for (const auto & item : step.getOutputStream().header)
    {
        identities[item.name] = item.name;
    }
    
    Property translated;

    if (step.getKind() == JoinKind::Inner || step.getKind() == JoinKind::Cross) 
    {
        Property left_translated = context.getInput()[0].translate(identities);
        Property right_translated = context.getInput()[1].translate(identities);
        
        std::map<String, FieldWithType> filter_values;
        const Constants & left_constants = left_translated.getConstants();
        const Constants & right_constants = right_translated.getConstants();
        for (const auto & value : left_constants.getValues())
        {
            filter_values[value.first] = value.second;
        }
        for (const auto & value : right_constants.getValues())
        {
            filter_values[value.first] = value.second;
        }
        Constants filter_constants{filter_values};
        left_translated.setConstants(filter_constants);
        
        translated = left_translated;
        
        // if partition columns are pruned, the output data has no property.
        if (translated.getNodePartitioning().getPartitioningColumns().size()
            != context.getInput()[0].getNodePartitioning().getPartitioningColumns().size())
        {
            return Property{};
        }
        if (translated.getStreamPartitioning().getPartitioningColumns().size()
            != context.getInput()[0].getStreamPartitioning().getPartitioningColumns().size())
        {
            // TODO stream partition
        }
    }

    if (step.getKind() == JoinKind::Left) 
    {
        Property left_translated = context.getInput()[0].translate(identities);
        translated = left_translated;
        
        // if partition columns are pruned, the output data has no property.
        if (translated.getNodePartitioning().getPartitioningColumns().size()
            != context.getInput()[0].getNodePartitioning().getPartitioningColumns().size())
        {
            return Property{};
        }
        if (translated.getStreamPartitioning().getPartitioningColumns().size()
            != context.getInput()[0].getStreamPartitioning().getPartitioningColumns().size())
        {
            // TODO stream partition
        }
    } 
        
    if (step.getKind() == JoinKind::Right) 
    {
        Property right_translated = context.getInput()[1].translate(identities);
        translated = right_translated;
        
        // if partition columns are pruned, the output data has no property.
        if (translated.getNodePartitioning().getPartitioningColumns().size()
            != context.getInput()[1].getNodePartitioning().getPartitioningColumns().size())
        {
            return Property{};
        }
        if (translated.getStreamPartitioning().getPartitioningColumns().size()
            != context.getInput()[1].getStreamPartitioning().getPartitioningColumns().size())
        {
            // TODO stream partition
        }
    }  
    
    if (step.getKind() == JoinKind::Full) 
    {
        return Property{};
    }
    if (translated.getStreamPartitioning().getPartitioningColumns().size()
        != context.getInput()[0].getStreamPartitioning().getPartitioningColumns().size())
    {
        // TODO stream partition
    }

    if (step.enforceNestLoopJoin())
    {
        translated = translated.clearSorting();
    }
    return translated;
}

Property DeriverVisitor::visitArrayJoinStep(const ArrayJoinStep & , DeriverContext & context)
{
    return context.getInput()[0];
}

// TODO partition key inference, translate properties according to group by keys
Property DeriverVisitor::visitAggregatingStep(const AggregatingStep & step, DeriverContext & context)
{
    if (step.getKeys().empty())
    {
        return Property{Partitioning{Partitioning::Handle::SINGLE}, Partitioning{Partitioning::Handle::SINGLE}};
    }
    return context.getInput()[0].clearSorting();
}

Property DeriverVisitor::visitMarkDistinctStep(const MarkDistinctStep & step, DeriverContext & context)
{
    if (step.getDistinctSymbols().empty())
    {
        return Property{Partitioning{Partitioning::Handle::SINGLE}, Partitioning{Partitioning::Handle::SINGLE}};
    }
    return context.getInput()[0];
}

Property DeriverVisitor::visitMergingAggregatedStep(const MergingAggregatedStep &, DeriverContext & context)
{
    return context.getInput()[0].clearSorting();
}

Property DeriverVisitor::visitUnionStep(const UnionStep & step, DeriverContext & context)
{
    Property first_child_property = context.getInput()[0];
    if (first_child_property.getNodePartitioning().getPartitioningHandle() == Partitioning::Handle::SINGLE)
    {
        bool all_single = true;
        for (const auto & input : context.getInput())
        {
            all_single &= input.getNodePartitioning().getPartitioningHandle() == Partitioning::Handle::SINGLE;
        }

        if (all_single)
        {
            if (step.isLocal())
            {
                return Property{Partitioning{Partitioning::Handle::SINGLE}, Partitioning{Partitioning::Handle::SINGLE}};
            }
            else
            {
                return Property{Partitioning{Partitioning::Handle::SINGLE}};
            }
        }
    }

    std::vector<Property> transformed_children_prop;
    const auto & output_to_inputs = step.getOutToInputs();
    size_t index = 0;
    for (const auto & child_prop : context.getInput())
    {
        NameToNameMap mapping;
        for (const auto & output_to_input : output_to_inputs)
        {
            mapping[output_to_input.second[index]] = output_to_input.first;
        }
        index++;
        transformed_children_prop.emplace_back(child_prop.translate(mapping));
    }

    if (first_child_property.getNodePartitioning().getPartitioningHandle() == Partitioning::Handle::FIXED_HASH)
    {
        const Names & keys = first_child_property.getNodePartitioning().getPartitioningColumns();
        Names output_keys;
        bool match = true;
        for (const auto & transformed : transformed_children_prop)
        {
            if (transformed != transformed_children_prop[0])
            {
                match = false;
                break;
            }
        }

        if (match && keys.size() == transformed_children_prop[0].getNodePartitioning().getPartitioningColumns().size())
        {
            output_keys = transformed_children_prop[0].getNodePartitioning().getPartitioningColumns();
        }
        if (step.isLocal())
        {
            return Property{
                Partitioning{Partitioning::Handle::FIXED_HASH, output_keys, true, first_child_property.getNodePartitioning().getBuckets()},
                Partitioning{Partitioning::Handle::SINGLE}};
        }
        else
        {
            return Property{
                Partitioning{Partitioning::Handle::FIXED_HASH, output_keys, true, first_child_property.getNodePartitioning().getBuckets()}};
        }
    }
    return Property{};
}

Property DeriverVisitor::visitExceptStep(const ExceptStep &, DeriverContext & context)
{
    return context.getInput()[0].clearSorting();
}

Property DeriverVisitor::visitIntersectStep(const IntersectStep &, DeriverContext & context)
{
    return context.getInput()[0].clearSorting();
}

Property DeriverVisitor::visitExchangeStep(const ExchangeStep & step, DeriverContext & context)
{
    const ExchangeMode & mode = step.getExchangeMode();
    if (mode == ExchangeMode::GATHER)
    {
        Property output = context.getInput()[0];
        output.setNodePartitioning(Partitioning{Partitioning::Handle::SINGLE}); 
        return output.clearSorting();
    }

    if (mode == ExchangeMode::REPARTITION)
    {
        Property output = context.getInput()[0];
        output.setNodePartitioning(step.getSchema()); 
        return output.clearSorting();
    }

    if (mode == ExchangeMode::BROADCAST)
    {
        Property output = context.getInput()[0];
        output.setNodePartitioning(Partitioning{Partitioning::Handle::FIXED_BROADCAST}); 
        return output.clearSorting();
    }

    if (mode == ExchangeMode::LOCAL_NO_NEED_REPARTITION)
    {
        Property output = context.getInput()[0];
        output.setStreamPartitioning(Partitioning{});
        return output.clearSorting();
    }

    return context.getInput()[0].clearSorting();
}

Property DeriverVisitor::visitRemoteExchangeSourceStep(const RemoteExchangeSourceStep &, DeriverContext & context)
{
    return context.getInput()[0].clearSorting();
}

Property DeriverVisitor::visitTableScanStep(const TableScanStep & step, DeriverContext & context)
{
    NameToNameMap translation;
    for (const auto & item : step.getColumnAlias())
        translation.emplace(item.first, item.second);

    return PropertyDeriver::deriveStorageProperty(step.getStorage(), context.getContext()).translate(translation);
}

Property DeriverVisitor::visitReadNothingStep(const ReadNothingStep &, DeriverContext &)
{
    return Property{Partitioning(Partitioning::Handle::SINGLE), Partitioning(Partitioning::Handle::ARBITRARY)};
}

Property DeriverVisitor::visitValuesStep(const ValuesStep &, DeriverContext &)
{
    return Property{Partitioning(Partitioning::Handle::SINGLE), Partitioning(Partitioning::Handle::ARBITRARY)};
}

Property DeriverVisitor::visitLimitStep(const LimitStep &, DeriverContext & context)
{
    return Property{
        context.getInput()[0].getNodePartitioning(), Partitioning(Partitioning::Handle::SINGLE), context.getInput()[0].getSorting(), context.getInput()[0].getConstants()};
}

Property DeriverVisitor::visitLimitByStep(const LimitByStep &, DeriverContext & context)
{
    return context.getInput()[0];
}

Property DeriverVisitor::visitSortingStep(const SortingStep & step, DeriverContext & context)
{
    auto prop = context.getInput()[0];
    Sorting sorting;
    for (auto item : step.getSortDescription())
    {
        sorting.emplace_back(item);
    }
    prop.setSorting(sorting);
    return prop;
}

Property DeriverVisitor::visitMergeSortingStep(const MergeSortingStep &, DeriverContext & context)
{
    return context.getInput()[0];
}

Property DeriverVisitor::visitPartialSortingStep(const PartialSortingStep &, DeriverContext & context)
{
    return context.getInput()[0];
}

Property DeriverVisitor::visitMergingSortedStep(const MergingSortedStep &, DeriverContext & context)
{
    return Property{context.getInput()[0].getNodePartitioning(), Partitioning(Partitioning::Handle::SINGLE)};
}

Property DeriverVisitor::visitDistinctStep(const DistinctStep &, DeriverContext & context)
{
    return context.getInput()[0].clearSorting();
}

Property DeriverVisitor::visitExtremesStep(const ExtremesStep &, DeriverContext & context)
{
    return context.getInput()[0].clearSorting();
}

Property DeriverVisitor::visitWindowStep(const WindowStep &, DeriverContext & context)
{
    return context.getInput()[0].clearSorting();
}

Property DeriverVisitor::visitApplyStep(const ApplyStep &, DeriverContext & context)
{
    return context.getInput()[0].clearSorting();
}

Property DeriverVisitor::visitEnforceSingleRowStep(const EnforceSingleRowStep &, DeriverContext & context)
{
    return context.getInput()[0];
}

Property DeriverVisitor::visitAssignUniqueIdStep(const AssignUniqueIdStep &, DeriverContext & context)
{
    return context.getInput()[0].clearSorting();
}

Property DeriverVisitor::visitCTERefStep(const CTERefStep &, DeriverContext &)
{
    //    // CTERefStep output property can not be determined locally, it has been determined globally,
    //    //  Described in CTEDescription of property. If They don't match, we just ignore required property.
    //    // eg, input required property: <Repartition[B], CTE(0)=Repartition[A]> don't match,
    //    //    we ignore local required property Repartition[B] and prefer global property Repartition[A].
    //    CTEId cte_id = step.getId();
    //    Property cte_required_property = context.getRequiredProperty().getCTEDescriptions().at(cte_id).toProperty();
    //
    //    auto output_prop = cte_required_property.translate(step.getReverseOutputColumns());
    //    output_prop.getCTEDescriptions().emplace(cte_id, cte_required_property);
    //    return output_prop;
    throw Exception("CTERefStep is not supported", ErrorCodes::OPTIMIZER_NONSUPPORT);
}

Property DeriverVisitor::visitExplainAnalyzeStep(const ExplainAnalyzeStep &, DeriverContext & context)
{
    return context.getInput()[0];
}

Property DeriverVisitor::visitTopNFilteringStep(const TopNFilteringStep &, DeriverContext & context)
{
    return context.getInput()[0];
}

Property DeriverVisitor::visitFillingStep(const FillingStep &, DeriverContext & context)
{
    return context.getInput()[0];
}
}
