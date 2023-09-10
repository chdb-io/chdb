#include <Statistics/SimplifyHistogram.h>

namespace DB::Statistics
{

Histogram simplifyHistogram(const Histogram & origin, double ndv_density_threshold, double range_density_threshold, bool is_integer)
{
    if (origin.getBucketSize() < 2)
        return origin;
    // merge adj bucket one by one,
    const auto & origin_buckets = origin.getBuckets();

    Buckets result_buckets;

    std::optional<Bucket> to_merge;
    for (const auto & right : origin_buckets)
    {
        bool should_merge;
        if (right.getCount() == 0)
            continue;

        // we will merge only when
        if (!to_merge)
        {
            to_merge = right;
            continue;
        }

        auto left = to_merge.value();

        double left_density = left.getCount() / left.getNumDistinct();
        double right_density = right.getCount() / right.getNumDistinct();

        if (auto [min, max] = std::minmax(left_density, right_density); max / min > 1 + ndv_density_threshold)
        {
            // ndv density is not good
            should_merge = false;
        }
        else
        {
            if (left.isSingleton() || right.isSingleton())
            {
                // for singleton bucket, we consider it is always range-valid
                // if ndv density is good
                should_merge = true;
            }
            else
            {
                auto li = is_integer ? left.isLowerClosed() + left.isUpperClosed() - 1 : 0;
                auto ri = is_integer ? right.isLowerClosed() + right.isUpperClosed() - 1 : 0;

                auto l = (left.getUpperBound() - left.getLowerBound() + li) / left.getCount();
                auto r = (right.getUpperBound() - right.getLowerBound() + ri) / right.getCount();

                auto [min_, max_] = std::minmax(l, r);
                should_merge = max_ / min_ <= 1 + range_density_threshold;
            }
        }
        if (should_merge)
        {
            auto ndv = left.getNumDistinct() + right.getNumDistinct();
            auto count = left.getCount() + right.getCount();
            Bucket new_bucket(left.getLowerBound(), right.getUpperBound(), ndv, count, left.isLowerClosed(), right.isUpperClosed());
            to_merge = new_bucket;
        }
        else
        {
            result_buckets.emplace_back(left);
            to_merge = right;
        }
    }

    if (to_merge)
    {
        result_buckets.emplace_back(to_merge.value());
    }

    return Histogram(result_buckets);
}

}
