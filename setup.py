import os
import sys
import subprocess
import sysconfig
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import setuptools
from distutils import log

log.set_verbosity(log.DEBUG)

# get the path of the current file
script_dir = os.path.dirname(os.path.abspath(__file__))
libdir = os.path.join(script_dir, "chdb")

def get_latest_git_tag():
    try:
        completed_process = subprocess.run(['git', 'describe', '--tags', '--abbrev=0', '--match', 'v*'], capture_output=True, text=True)
        if completed_process.returncode != 0:
            print(completed_process.stdout)
            print(completed_process.stderr)
            # get git version
            raise RuntimeError("Failed to get git tag")
        output = completed_process.stdout.strip()
        #strip the v from the tag
        output = output[1:]
        parts = output.split('.')
        if len(parts) > 2:
            completed_process = subprocess.run(['git', 'rev-list', '--count', f"v{output}..HEAD"], capture_output=True, text=True)
            if completed_process.returncode != 0:
                print(completed_process.stdout)
                print(completed_process.stderr)
                raise RuntimeError("Failed to get git rev-list")
            n = completed_process.stdout.strip()
            return f"{parts[0]}.{parts[1]}.{n}"
    except Exception as e:
        print("Failed to get git tag. Error: ")
        print(e)
        raise


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/2a] compiler flag.
    The c++2a is prefered over c++11 (when it is available).
    """
    if has_flag(compiler, '-std=c++2a'):
        return '-std=c++2a'
    elif has_flag(compiler, '-std=c++17'):
        return '-std=c++17'
    elif has_flag(compiler, '-std=c++14'):
        return '-std=c++14'
    elif has_flag(compiler, '-std=c++11'):
        return '-std=c++11'
    else:
        raise RuntimeError('Unsupported compiler -- at least C++11 support '
                           'is needed!')


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""

    def build_extensions(self):
        # Determine which compiler to use, if CC and CXX env exist, use them
        if os.environ.get('CC') is not None and os.environ.get('CXX') is not None:
            print("Using CC and CXX from env")
            print("CC: " + os.environ.get('CC'))
            print("CXX: " + os.environ.get('CXX'))
        if sys.platform == 'darwin':
            if os.system('which /usr/local/opt/llvm/bin/clang++ > /dev/null') == 0:
                os.environ['CC'] = '/usr/local/opt/llvm/bin/clang'
                os.environ['CXX'] = '/usr/local/opt/llvm/bin/clang++'
            else:
                raise RuntimeError("Must use brew clang++")
        elif sys.platform == 'linux':
            pass
            #os.environ['CC'] = 'clang-15'
            #os.environ['CXX'] = 'clang++-15'
        else:
            raise RuntimeError("Unsupported platform")

        #exec chdb/build.sh and print the output if it fails
        # Run the build script and capture its output
        completed_process = subprocess.run(["bash", "chdb/build.sh"], capture_output=True, text=True)
        # If it failed, print the output
        print(completed_process.stdout)
        print(completed_process.stderr)

        # Check the return code to see if the script failed
        if completed_process.returncode != 0:
            raise RuntimeError("Build failed")

        # add the _chdb.cpython-37m-darwin.so or _chdb.cpython-39-x86_64-linux.so to the chdb package
        self.distribution.package_data['chdb'] = [ "chdb/_chdb" + sysconfig.get_config_var('EXT_SUFFIX')]
        # super().build_extensions()


# this will be executed by python setup.py bdist_wheel
if __name__ == "__main__":
    try:
        # get python extension file name
        chdb_so = libdir + "/_chdb" + sysconfig.get_config_var('EXT_SUFFIX')
        ext_modules = [
            Extension(
                '_chdb',
                sources=["programs/local/LocalChdb.cpp"],
                language='c++',
                libraries=[],
                library_dirs=[libdir],
                extra_objects=[chdb_so],
            ),
        ]

        setup(
            packages=['chdb'],
            platforms=['manylinux2014_x86_64', 'macosx_11_0_x86_64', 'macosx_12_0_x86_64', 'macosx_12_0_arm64'],
            version=get_latest_git_tag(),
            package_data={'chdb': [chdb_so]},
            exclude_package_data={'': ['*.pyc', 'src/**']},
            ext_modules=ext_modules,
            install_requires=['pybind11>=2.6'],
            python_requires='>=3.7',
            cmdclass={'build_ext': BuildExt},
            test_suite="tests",
            zip_safe=False,
        )
    except Exception as e:
        print("Build from setup.py failed. Error: ")
        print(e)
        raise
