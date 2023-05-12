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
    internal_ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
    p = subprocess.run(['python3', '-c', "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"], capture_output=True, text=True)
    if p.returncode != 0:
        print("Failed to get EXT_SUFFIX via python3")
        return internal_ext_suffix
    py_ext_suffix = p.stdout.strip()
    if py_ext_suffix != internal_ext_suffix:
        print("EXT_SUFFIX mismatch")
        print("Internal EXT_SUFFIX: " + internal_ext_suffix)
        print("Python3 EXT_SUFFIX: " + py_ext_suffix)
        print("Current Python Path: " + sys.executable)
        print("Current Python Version: " + sys.version)
        print("Outside Python Path: " + subprocess.check_output(['which', 'python3']).decode('utf-8').strip())
        print("Outside Python Version: " + subprocess.check_output(['python3', '--version']).decode('utf-8').strip())
    return py_ext_suffix

# get the path of the current file
script_dir = os.path.dirname(os.path.abspath(__file__))
libdir = os.path.join(script_dir, "chdb")

def get_latest_git_tag(minor_ver_auto=False):
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
        if len(parts) == 3:
            if minor_ver_auto:
                completed_process = subprocess.run(['git', 'rev-list', '--count', f"v{output}..HEAD"], capture_output=True, text=True)
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

# replace the version in chdb/__init__.py, which is `chdb_version = (0, 1, 0)` by default
# regex replace the version string `chdb_version = (0, 1, 0)` with version parts
def fix_version_init(version):
    # split version string into parts
    p1, p2, p3 = version.split('.')
    init_file = os.path.join(script_dir, "chdb", "__init__.py")
    with open(init_file, "r+") as f:
        init_content = f.read()
        # regex replace the version string `chdb_version = (0, 1, 0)`
        regPattern = r"chdb_version = \(\d+, \d+, \d+\)"
        init_content = re.sub(regPattern, f"chdb_version = ({p1}, {p2}, {p3})", init_content)
        f.seek(0)
        f.write(init_content)
        f.truncate()


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
            try:
                import subprocess
                brew_prefix = subprocess.check_output('brew --prefix', shell=True).decode("utf-8").strip("\n")
            except Exception:
                raise RuntimeError("Must install brew")
            if os.system('which '+brew_prefix+'/opt/llvm/bin/clang++ > /dev/null') == 0:
                os.environ['CC'] = brew_prefix + '/opt/llvm/bin/clang'
                os.environ['CXX'] = brew_prefix + '/opt/llvm/bin/clang++'
            elif os.system('which '+brew_prefix+'/opt/llvm@15/bin/clang++ > /dev/null') == 0:
                os.environ['CC'] = brew_prefix + '/opt/llvm@15/bin/clang'
                os.environ['CXX'] = brew_prefix + '/opt/llvm@15/bin/clang++'
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
        self.distribution.package_data['chdb'] = [ "chdb/_chdb" + get_python_ext_suffix()]
        # super().build_extensions()


# this will be executed by python setup.py bdist_wheel
if __name__ == "__main__":
    try:
        # get python extension file name
        chdb_so = libdir + "/_chdb" + get_python_ext_suffix()
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
        # fix the version in chdb/__init__.py
        versionStr = get_latest_git_tag()
        fix_version_init(versionStr)
        setup(
            packages=['chdb'],
            version=versionStr,
            package_data={'chdb': [chdb_so]},
            exclude_package_data={'': ['*.pyc', 'src/**']},
            ext_modules=ext_modules,
            python_requires='>=3.7',
            cmdclass={'build_ext': BuildExt},
            test_suite="tests",
            zip_safe=False,
        )
    except Exception as e:
        print("Build from setup.py failed. Error: ")
        print(e)
        raise
