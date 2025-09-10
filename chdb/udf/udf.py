import functools
import inspect
import os
import sys
import tempfile
import atexit
import shutil
import textwrap
from xml.etree import ElementTree as ET
import chdb


def generate_udf(func_name, args, return_type, udf_body):
    """Generate UDF configuration and executable script files.

    This function creates the necessary files for a User Defined Function (UDF) in chDB:
    1. A Python executable script that processes input data
    2. An XML configuration file that registers the UDF with ClickHouse

    Args:
        func_name (str): Name of the UDF function
        args (list): List of argument names for the function
        return_type (str): ClickHouse return type for the function
        udf_body (str): Python source code body of the UDF function

    Note:
        This function is typically called by the @chdb_udf decorator and should not
        be called directly by users.
    """
    # generate python script
    with open(f"{chdb.g_udf_path}/{func_name}.py", "w") as f:
        f.write(f"#!{sys.executable}\n")
        f.write("import sys\n")
        f.write("\n")
        for line in udf_body.split("\n"):
            f.write(f"{line}\n")
        f.write("\n")
        f.write("if __name__ == '__main__':\n")
        f.write("    for line in sys.stdin:\n")
        f.write("        args = line.strip().split('\t')\n")
        for i, arg in enumerate(args):
            f.write(f"        {arg} = args[{i}]\n")
        f.write(f"        print({func_name}({', '.join(args)}))\n")
        f.write("        sys.stdout.flush()\n")
    os.chmod(f"{chdb.g_udf_path}/{func_name}.py", 0o755)
    # generate xml file
    xml_file = f"{chdb.g_udf_path}/udf_config.xml"
    root = ET.Element("functions")
    if os.path.exists(xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
    function = ET.SubElement(root, "function")
    ET.SubElement(function, "type").text = "executable"
    ET.SubElement(function, "name").text = func_name
    ET.SubElement(function, "return_type").text = return_type
    ET.SubElement(function, "format").text = "TabSeparated"
    ET.SubElement(function, "command").text = f"{func_name}.py"
    for arg in args:
        argument = ET.SubElement(function, "argument")
        # We use TabSeparated format, so assume all arguments are strings
        ET.SubElement(argument, "type").text = "String"
        ET.SubElement(argument, "name").text = arg
    tree = ET.ElementTree(root)
    tree.write(xml_file)


def chdb_udf(return_type="String"):
    """Decorator for chDB Python UDF(User Defined Function).

    Args:
        return_type (str): Return type of the function. Default is "String".
            Should be one of the ClickHouse data types.

    Notes:
        1. The function should be stateless. Only UDFs are supported, not UDAFs.
        2. Default return type is String. The return type should be one of the ClickHouse data types.
        3. The function should take in arguments of type String. All arguments are strings.
        4. The function will be called for each line of input.
        5. The function should be pure python function. Import all modules used IN THE FUNCTION.
        6. Python interpreter used is the same as the one used to run the script.

    Example:
        .. code-block:: python

            @chdb_udf()
            def sum_udf(lhs, rhs):
                return int(lhs) + int(rhs)

            @chdb_udf()
            def func_use_json(arg):
                import json
                # ... use json module
    """

    def decorator(func):
        func_name = func.__name__
        sig = inspect.signature(func)
        args = list(sig.parameters.keys())
        src = inspect.getsource(func)
        src = textwrap.dedent(src)
        udf_body = src.split("\n", 1)[1]  # remove the first line "@chdb_udf()"
        # create tmp dir and make sure the dir is deleted when the process exits
        if chdb.g_udf_path == "":
            chdb.g_udf_path = tempfile.mkdtemp()

        # clean up the tmp dir on exit
        @atexit.register
        def _cleanup():
            try:
                shutil.rmtree(chdb.g_udf_path)
            except:  # noqa
                pass

        generate_udf(func_name, args, return_type, udf_body)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator
