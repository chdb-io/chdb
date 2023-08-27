import functools
import inspect
import os
import tempfile
from xml.etree import ElementTree as ET

tempdir = tempfile.TemporaryDirectory()
# os.chdir(tempdir.name)
os.chdir("user_scripts")
# print(f"Current working directory: {os.getcwd()}")

def generate_udf(func_name, args, return_type, udf_body):
    # generate python script
    with open(f"{func_name}.py", "w") as f:
        f.write("#!/usr/bin/python3\n")
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
    os.chmod(f"{func_name}.py", 0o755)
    # generate xml file
    xml_file = "udf_config.xml"
    root = ET.Element('functions')
    if os.path.exists(xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
    function = ET.SubElement(root, 'function')
    ET.SubElement(function, 'type').text = 'executable'
    ET.SubElement(function, 'name').text = func_name
    ET.SubElement(function, 'return_type').text = return_type
    ET.SubElement(function, 'format').text = 'TabSeparated'
    ET.SubElement(function, 'command').text = f"{func_name}.py"
    for arg in args:
        argument = ET.SubElement(function, 'argument')
        # We use TabSeparated format, so assume all arguments are strings
        ET.SubElement(argument, 'type').text = 'String'
        ET.SubElement(argument, 'name').text = arg
    tree = ET.ElementTree(root)
    tree.write(xml_file)

def to_clickhouse_udf(return_type="String"):
    def decorator(func):
        func_name = func.__name__
        sig = inspect.signature(func)
        args = list(sig.parameters.keys())
        src = inspect.getsource(func)
        udf_body = src.split("\n", 1)[1]
        generate_udf(func_name, args, return_type, udf_body)
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

