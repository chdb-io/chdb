Installation
============

chDB is an in-process OLAP SQL engine powered by ClickHouse, providing fast analytical queries directly within Python.

Requirements
------------

- **Python**: 3.8 or higher
- **Architecture**: 64-bit systems only
- **Platforms**: macOS and Linux (x86_64 and ARM64)

Quick Installation
------------------

Install chDB with a single command:

.. code-block:: bash

   pip install chdb

This installs the core chDB package with all essential functionality.

Optional Dependencies
---------------------

For enhanced functionality, install with optional dependencies:

**For DataFrame Support:**

.. code-block:: bash

   pip install chdb[pandas,pyarrow]

This enables:

- Direct pandas DataFrame querying
- PyArrow table support
- Efficient data interchange between formats

**For Development:**

.. code-block:: bash

   pip install chdb[dev]

This includes testing and development tools.

Supported Platforms
-------------------

chDB supports the following platforms:

**Linux**
  - x86_64 (Intel/AMD 64-bit)
  - ARM64 (AArch64)

**macOS**
  - x86_64 (Intel processors)
  - ARM64 (Apple Silicon: M1, M2, M3)

**Note**: Windows is not currently supported.

System Requirements
-------------------

**Minimum Requirements:**
  - **RAM**: 512MB available memory
  - **Disk**: 200MB free space for installation
  - **Python**: 3.8+ with pip

**Recommended:**
  - **RAM**: 2GB+ for processing large datasets
  - **Disk**: Additional space for data files and temporary storage
  - **Python**: 3.8+ for optimal performance

Verification
------------

Verify your installation works correctly:

.. code-block:: python

   import chdb
   
   # Test basic functionality
   result = chdb.query("SELECT 'chDB is working!' as message")
   print(result)
   
   # Check version
   print(f"chDB version: {chdb.__version__}")
   print(f"ClickHouse engine: {chdb.engine_version}")

You should see output similar to:

.. code-block:: text

   message
   chDB is working!
   
   chDB version: 3.6.0
   ClickHouse engine: 24.1.1.1

Troubleshooting
---------------

**Import Errors**

If you encounter import errors, ensure you have Python 3.8+ and a supported platform:

.. code-block:: python

   import sys
   import platform
   
   print(f"Python version: {sys.version}")
   print(f"Platform: {platform.platform()}")
   print(f"Architecture: {platform.architecture()}")

**Performance Issues**

For better performance with large datasets:

1. Ensure sufficient RAM is available
2. Use SSD storage for data files
3. Consider using chDB's connection-based API for repeated queries

**Getting Help**

If you encounter issues:

- Check the `GitHub Issues <https://github.com/chdb-io/chdb/issues>`_
- Visit our `Discord community <https://discord.gg/D2Daa2fM5K>`_
- Review the troubleshooting guide

Development Installation
------------------------

For contributing to chDB development:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/chdb-io/chdb.git
   cd chdb
   
   # Install in development mode
   pip install -e .[dev]
   
   # Run tests to verify installation
   python -m pytest tests/

This installs chDB in editable mode with development dependencies.
