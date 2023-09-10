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

#include <Optimizer/tests/gtest_base_correctness_plan_test.h>
#include <QueryPlan/PlanPrinter.h>

#include <gtest/gtest.h>

using namespace DB;

class PlanCheckCorrectness : public ::testing::Test
{
public:
    static void SetUpTestSuite()
    {
        std::unordered_map<std::string, DB::Field> settings;
        settings.emplace("enable_execute_uncorrelated_subquery", 0);
        tester = std::make_shared<DB::CorrectnessPlanTest>(settings);
    }

    static std::string explain(const std::string & name)
    {
        return tester->explain(name);
    }

    static std::string expected(const std::string & name)
    {
        return tester->loadExplain(name);
    }

    static testing::AssertionResult equals(const std::string & actual, const std::string & expected)
    {
        if (actual == expected)
            return testing::AssertionSuccess();
        else
            return testing::AssertionFailure() << "\nExpected:\n" << expected << "\nActual:\n" << actual;
    }

    static std::shared_ptr<DB::CorrectnessPlanTest> tester;
};

std::shared_ptr<DB::CorrectnessPlanTest> PlanCheckCorrectness::tester;

DECLARE_GENERATE_TEST(PlanCheckCorrectness)

TEST_F(PlanCheckCorrectness, q1)
{
    EXPECT_TRUE(equals(explain("q1"), expected("q1")));
}

TEST_F(PlanCheckCorrectness, q2)
{
    EXPECT_TRUE(equals(explain("q2"), expected("q2")));
}

TEST_F(PlanCheckCorrectness, q3)
{
    EXPECT_TRUE(equals(explain("q3"), expected("q3")));
}

TEST_F(PlanCheckCorrectness, q4)
{
    EXPECT_TRUE(equals(explain("q4"), expected("q4")));
}

TEST_F(PlanCheckCorrectness, q5)
{
    EXPECT_TRUE(equals(explain("q5"), expected("q5")));
}

TEST_F(PlanCheckCorrectness, q6)
{
    EXPECT_TRUE(equals(explain("q6"), expected("q6")));
}

TEST_F(PlanCheckCorrectness, q7)
{
    EXPECT_TRUE(equals(explain("q7"), expected("q7")));
}

TEST_F(PlanCheckCorrectness, q8)
{
    EXPECT_TRUE(equals(explain("q8"), expected("q8")));
}

TEST_F(PlanCheckCorrectness, q9)
{
    EXPECT_TRUE(equals(explain("q9"), expected("q9")));
}

TEST_F(PlanCheckCorrectness, q10)
{
    EXPECT_TRUE(equals(explain("q10"), expected("q10")));
}

TEST_F(PlanCheckCorrectness, q11)
{
    EXPECT_TRUE(equals(explain("q11"), expected("q11")));
}

TEST_F(PlanCheckCorrectness, q12)
{
    EXPECT_TRUE(equals(explain("q12"), expected("q12")));
}

TEST_F(PlanCheckCorrectness, q13)
{
    EXPECT_TRUE(equals(explain("q13"), expected("q13")));
}

TEST_F(PlanCheckCorrectness, q14)
{
    EXPECT_TRUE(equals(explain("q14"), expected("q14")));
}

TEST_F(PlanCheckCorrectness, q15)
{
    EXPECT_TRUE(equals(explain("q15"), expected("q15")));
}
