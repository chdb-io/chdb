import os
import sys
import re
import subprocess
import sysconfig
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import setuptools
from distutils import log

log.set_verbosity(log.DEBUG)


def get_python_ext_suffix():
    # Use Limited API suffix for cross-version compatibility
    return ".abi3.so"


# get the path of the current file
script_dir = os.path.dirname(os.path.abspath(__file__))
libdir = os.path.join(script_dir, "chdb")


def get_latest_git_tag(minor_ver_auto=False):
    try:
        # get latest tag commit
        completed_process = subprocess.run(
            ["git", "rev-list", "--tags", "--max-count=1"],
            capture_output=True,
            text=True,
        )
        if completed_process.returncode != 0:
            print(completed_process.stdout)
            print(completed_process.stderr)
            # get git version
            raise RuntimeError("Failed to get git latest tag commit ")
        output = completed_process.stdout.strip()
        # get latest tag name by commit
        completed_process = subprocess.run(
            ["git", "describe", "--tags", f"{output}"], capture_output=True, text=True
        )
        if completed_process.returncode != 0:
            print(completed_process.stdout)
            print(completed_process.stderr)
            # get git version
            raise RuntimeError("Failed to get git tag")
        output = completed_process.stdout.strip()
        # strip the v from the tag
        output = output[1:]
        parts = output.split(".")
        if len(parts) == 3:
            if minor_ver_auto:
                completed_process = subprocess.run(
                    ["git", "rev-list", "--count", f"v{output}..HEAD"],
                    capture_output=True,
                    text=True,
                )
                if completed_process.returncode != 0:
                    print(completed_process.stdout)
                    print(completed_process.stderr)
                    raise RuntimeError("Failed to get git rev-list")
                n = completed_process.stdout.strip()
                parts[2] = int(parts[2]) + int(n)
            return f"{parts[0]}.{parts[1]}.{parts[2]}"
    except Exception as e:
        print("Failed to get git tag. Error: ")
        print(e)
        raise

# Update version in pyproject.toml
def update_pyproject_version(version):
    pyproject_file = os.path.join(script_dir, "pyproject.toml")
    with open(pyproject_file, "r") as f:
        content = f.read()

    # Use regex to replace the version
    updated_content = re.sub(
        r'version\s*=\s*"[^"]*"', f'version = "{version}"', content
    )

    with open(pyproject_file, "w") as f:
        f.write(updated_content)


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".cpp") as f:
        f.write("int main (int argc, char **argv) { return 0; }")
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/2a] compiler flag.
    The c++2a is prefered over c++11 (when it is available).
    """
    if has_flag(compiler, "-std=c++2a"):
        return "-std=c++2a"
    elif has_flag(compiler, "-std=c++17"):
        return "-std=c++17"
    elif has_flag(compiler, "-std=c++14"):
        return "-std=c++14"
    elif has_flag(compiler, "-std=c++11"):
        return "-std=c++11"
    else:
        raise RuntimeError(
            "Unsupported compiler -- at least C++11 support " "is needed!"
        )


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""

    def build_extensions(self):
        # add the _chdb.cpython-37m-darwin.so or _chdb.cpython-39-x86_64-linux.so to the chdb package
        self.distribution.package_data["chdb"] = [
            "chdb/_chdb" + get_python_ext_suffix()
        ]
        # super().build_extensions()


# this will be executed by python setup.py bdist_wheel
if __name__ == "__main__":
    try:
        # get python extension file name
        chdb_so = libdir + "/_chdb" + get_python_ext_suffix()
        ext_modules = [
            Extension(
                "_chdb",
                sources=["programs/local/LocalChdb.cpp"],
                libraries=[],
                library_dirs=[libdir],
                extra_objects=[chdb_so],
                define_macros=[("Py_LIMITED_API", "0x03080000")],
                py_limited_api=True,
            ),
        ]
        # fix the version in chdb/__init__.py
        versionStr = get_latest_git_tag()
        # Call the function to update pyproject.toml
        # update_pyproject_version(versionStr)

        # scan the chdb directory and add all the .py and dynamic library files to the package
        pkg_files = []
        for root, dirs, files in os.walk(libdir):
            if "/build" in root or root.endswith("/build"):
                continue

            for file in files:
                if file.endswith(".py"):
                    pkg_files.append(os.path.join(root, file))
                # Include pybind11 nonlimitedapi libraries for all Python versions
                elif file.startswith("libpybind11nonlimitedapi_chdb_") and (file.endswith(".dylib") or file.endswith(".so")):
                    pkg_files.append(os.path.join(root, file))
                # Include pybind11 stub library
                elif file.startswith("libpybind11nonlimitedapi_stubs") and (file.endswith(".dylib") or file.endswith(".so")):
                    pkg_files.append(os.path.join(root, file))

        pkg_files.append(chdb_so)

        setup(
            packages=["chdb"],
            version=versionStr,
            include_package_data=False,
            package_data={"chdb": pkg_files},
            ext_modules=ext_modules,
            python_requires=">=3.8",
            install_requires=[
                "pyarrow>=13.0.0",
                "pandas>=2.1.0,<3.0.0",
            ],
            cmdclass={"build_ext": BuildExt},
            test_suite="tests",
            zip_safe=False,
            options={
                "bdist_wheel": {
                    "py_limited_api": "cp38",
                }
            },
        )
    except Exception as e:
        print("Build from setup.py failed. Error: ")
        print(e)
        raise
