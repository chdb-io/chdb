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

#include <Parsers/IAST.h>
#include <QueryPlan/TableScanStep.h>
#include <Storages/IStorage_fwd.h>

namespace DB
{
/// this internal AST is only used by optimizer.
/// Represent the origin table and column of a identifier.
class ASTTableColumnReference : public IAST
{
public:
    const IStorage * storage;
    String column_name;
    // Used to identify different occurrence of a same table in self-join cases.
    const TableScanStep * step = nullptr;

    ASTTableColumnReference(const IStorage * storage_, String column_name_) : storage(storage_), column_name(std::move(column_name_))
    {
    }

    String getID(char delim) const override;

    void appendColumnName(WriteBuffer &) const override;

    ASTType getType() const override { return ASTType::ASTTableColumnReference; }

    ASTPtr clone() const override;

    const TableScanStep * getStep() const { return step; }
    void setStep(const TableScanStep * step_) { step = step_; }
};
}
