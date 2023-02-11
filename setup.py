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
    """Return the -std=c++[11/17] compiler flag.
    The c++17 is prefered over c++11 (when it is available).
    """
    if has_flag(compiler, '-std=c++17'):
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
    c_opts = {
        'unix': ['-O3', '-Wall', '-fPIC'],
        'msvc': [],
    }
    link_opts = {
        # we must use relative path here, otherwise the '_chdb' package will not be able to find the './libchdb.so' file
        # see the 'chdb/__init__.py' file for more details
        'unix': [],
        'msvc': [],
    }

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=12.0']
        link_opts['unix'] += ['-static-libstdc++', '-stdlib=libc++', '-mmacosx-version-min=12.0']
        link_opts['unix'] += ['-lchdb', '-Wl,-rpath,' + libdir]
        # link_opts['unix'].append('-dynamiclib')
    else:
        link_opts['unix'] += ['-shared', '-Wl,--exclude-libs,ALL', '-static-libgcc', '-static-libstdc++', '-stdlib=libstdc++']
        link_opts['unix'] += ['./libchdb.so']

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())

        # extend include dirs here (don't assume numpy/pybind11 are installed when first run, since
        # pip could have installed them as part of executing this script
        import pybind11
        for ext in self.extensions:
            ext.extra_compile_args.extend(opts)
            ext.extra_link_args.extend(self.link_opts.get(ct, []))
            ext.include_dirs.extend([
                # Path to pybind11 headers
                pybind11.get_include(),
            ])

        if sys.platform == 'darwin':
            if os.environ["CC"] == "/usr/local/opt/llvm/bin/clang++":
                clean_compiler = []
                for arg in self.compiler.compiler:
                    if arg == "-I/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include":
                        clean_compiler.append("-I/usr/local/opt/llvm/include/c++/v1")
                    else:
                        clean_compiler.append(arg)
                self.compiler.compiler = clean_compiler
                self.compiler.compiler_so = clean_compiler
                
            clean_linker = []
            for arg in self.compiler.linker_so:
                if arg == "-bundle":
                    clean_linker.append("-dynamiclib")
                # elif arg == "-undefined":
                #     continue
                # elif arg == "dynamic_lookup":
                #     continue
                elif arg == "-L/usr/local/opt/readline/lib":
                    continue
                elif arg == "-L/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib":
                    if os.environ["CC"] == "/usr/local/opt/llvm/bin/clang++":
                        clean_linker.append("-L/usr/local/opt/llvm/lib")
                    else:
                        clean_linker.append(arg)
                else:
                    clean_linker.append(arg)

            self.compiler.linker_so = clean_linker
        # change the current working directory to the path of the current file
        # and compile and link the './libchdb.so' then change the working directory back
        cwd = os.getcwd()
        os.chdir(libdir)
        build_ext.build_extensions(self)
        os.chdir(cwd)

        

if __name__ == "__main__":
    try:
        library_file = os.path.join(libdir, "libchdb.so")
        source_files = ['chdb.cpp']
        extra_objects = []
        if not os.path.exists(library_file):
            # if we have a prebuilt library file, use that.
            # extra_objects.append(library_file)
        # else:
            # otherwise, run build.sh to build the library file.
            # if this fails, the setup.py will fail.
            # os.system("bash chdb/build.sh")
            ret = os.system("bash chdb/build.sh")
            if ret != 0:
                raise RuntimeError("Build failed")
            # extra_objects.append(library_file)
        
        ext_modules = [
            Extension(
                '_chdb',
                source_files,
                include_dirs=['../', '../base', '../src', '../programs/local'],
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
