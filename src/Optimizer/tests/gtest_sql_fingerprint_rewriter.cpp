#include <gtest/gtest.h>
#include <Parsers/SQLFingerprint.h>
#include <Poco/DigestStream.h>
#include <Poco/MD5Engine.h>

namespace DB
{
TEST(SQLFingerprint, generate)
{
    String query = "select * from tpcds.customer where c_customer_sk > 1 limit 1 settings enable_optimizer=1 format JSON";
    
    SQLFingerprint fingerprint;
    auto fingerprint_str = fingerprint.generate(query.data(), query.data() + query.size());

    String expect_query = "SELECT * FROM tpcds.customer WHERE c_customer_sk > '?' LIMIT '?'";
    Poco::MD5Engine md5;
    Poco::DigestOutputStream outstr(md5);
    outstr << expect_query;
    outstr.flush();
    String fingerprint_md5_str = Poco::DigestEngine::digestToHex(md5.digest());

    ASSERT_EQ(fingerprint_str, fingerprint_md5_str);

    String query1
        = "SELECT Websites.id, Websites.name, access_log.count, access_log.date FROM Websites  FULL OUTER JOIN access_log ON Websites.id=access_log.site_id where Websites.id > 100 limit 10 settings enable_optimizer=1 format JSON";

    auto fingerprint_str1 = fingerprint.generateIntermediateAST(query1.data(), query1.data() + query1.size());
    auto fingerprint_str1_md5 = fingerprint.generate(query1.data(), query1.data() + query1.size());

    String expect_query1 = "SELECT Websites.id, Websites.name, access_log.count, access_log.date FROM Websites FULL OUTER JOIN access_log ON Websites.id = access_log.site_id WHERE Websites.id > '?' LIMIT '?'";
    Poco::DigestOutputStream outstr1(md5);
    outstr << expect_query1;
    outstr.flush();
    String fingerprint_md5_str1 = Poco::DigestEngine::digestToHex(md5.digest());

    ASSERT_EQ(fingerprint_str1, expect_query1);
    ASSERT_EQ(fingerprint_str1_md5, fingerprint_md5_str1);

    String query2
        = "SELECT * FROM Sales.Store WHERE BusinessEntityID NOT IN (SELECT CustomerID FROM Sales.Customer WHERE TerritoryID = 5) settings enable_optimizer=1 format JSON";

    auto fingerprint_str2 = fingerprint.generateIntermediateAST(query2.data(), query2.data() + query2.size());
    auto fingerprint_str2_md5 = fingerprint.generate(query2.data(), query2.data() + query2.size());

    String expect_query2 = "SELECT * FROM Sales.Store WHERE BusinessEntityID NOT IN (SELECT CustomerID FROM Sales.Customer WHERE TerritoryID = '?')";
    Poco::DigestOutputStream outstr2(md5);
    outstr << expect_query2;
    outstr.flush();
    String fingerprint_md5_str2 = Poco::DigestEngine::digestToHex(md5.digest());

    ASSERT_EQ(fingerprint_str2, expect_query2);
    ASSERT_EQ(fingerprint_str2_md5, fingerprint_md5_str2);
}    
}
