# try import pyarrow and pandas, if failed, raise ImportError with suggestion
try:
    import pyarrow as pa
    import pandas as pd
except ImportError as e:
    print(f'ImportError: {e}')
    print('Please install pyarrow and pandas via "pip install pyarrow pandas"')
    raise ImportError('Failed to import pyarrow or pandas') from None

# check if pandas version >= 2.0.0
if pd.__version__[0] < '2':
    print('Please upgrade pandas to version 2.0.0 or higher to have better performance')

from .query import *
