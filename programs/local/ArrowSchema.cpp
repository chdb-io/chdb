#include "ArrowSchema.h"

#include <base/defines.h>

namespace CHDB
{

ArrowSchemaWrapper::~ArrowSchemaWrapper()
{
    if (arrow_schema.release != nullptr)
    {
        arrow_schema.release(&arrow_schema);
        chassert(!arrow_schema.release);
    }
}

} // namespace CHDB
