import os
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools

if __name__ == "__main__":
    try:
        libdir = os.path.join(".", "pybind")
        library_file = os.path.join(libdir, "libchdb.so")
        source_files = ['chdb.cpp']
        extra_objects = []
        if os.path.exists(library_file):
            # if we have a prebuilt library file, use that.
            extra_objects.append(library_file)
        else:
            # otherwise, run build.sh to build the library file.
            # if this fails, the setup.py will fail.
            # os.system("bash pybind/build.sh")
            ret = os.system("bash pybind/build.sh")
            if ret != 0:
                raise RuntimeError("Build failed")
            extra_objects.append(library_file)
        
        setup(
            use_scm_version={"version_scheme": "no-guess-dev"},
            packages=['chdb'],
            package_data={'chdb': ['*.so']},
            exclude_package_data={'': ['*.pyc', 'src/**']},
        )
    except:  # noqa
        print(
            "\n\nAn error occurred while building the project, "
            "please ensure you have the most updated version of setuptools, "
            "setuptools_scm and wheel with:\n"
            "   pip install -U setuptools setuptools_scm wheel\n\n"
        )
        raise
