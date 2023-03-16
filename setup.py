import os
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import setuptools
from distutils import log

log.set_verbosity(log.DEBUG)

# get the path of the current file
script_dir = os.path.dirname(os.path.abspath(__file__))
libdir = os.path.join(script_dir, "chdb")


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
        # Determine which compiler to use, prefer clang++ over g++
        if os.system('which clang++ > /dev/null') == 0:
            print("Using Clang")
            # if macOS, prefer brew clang++ compiler over the system one
            if sys.platform == 'darwin':
                if os.system('which /usr/local/opt/llvm/bin/clang++ > /dev/null') == 0:
                    # distutils likes to use CC as cpp, so set CC to clang++
                    os.environ['CC'] = '/usr/local/opt/llvm/bin/clang++'
                    os.environ['CXX'] = '/usr/local/opt/llvm/bin/clang++'
            else:
                os.environ['CC'] = 'clang++'
                os.environ['CXX'] = 'clang++'
        else:
            print("Using GCC")
            os.environ['CC'] = 'gcc'
            os.environ['CXX'] = 'g++'

        ret = os.system("bash chdb/build.sh")
        if ret != 0:
            raise RuntimeError("Build failed")
        # add the _chdb.cpython-37m-darwin.so or _chdb.cpython-39-x86_64-linux.so to the chdb package
        self.distribution.package_data['chdb'] = ['*.so']


# this will be executed by python setup.py bdist_wheel
if __name__ == "__main__":
    try:
        ext_modules = [
            Extension(
                '_chdb',
                sources=['programs/local/LocalChdb.cpp'],
                language='c++',
                libraries=['chdb'],
                library_dirs=[libdir],
                # extra_objects=extra_objects,
            ),
        ]
        
        setup(
            use_scm_version={"version_scheme": "no-guess-dev"},
            packages=['chdb'],
            package_data={'chdb': ['*.so']},
            exclude_package_data={'': ['*.pyc', 'src/**']},
            ext_modules=ext_modules,
            install_requires=['pybind11>=2.6'],
            cmdclass={'build_ext': BuildExt},
            test_suite="tests",
            zip_safe=False,
        )
    except:  # noqa
        print(
            "\n\nAn error occurred while building the project, "
            "please ensure you have the most updated version of setuptools, "
            "setuptools_scm and wheel with:\n"
            "   pip install -U setuptools setuptools_scm wheel\n\n"
        )
        raise
