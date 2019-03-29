from distutils.core import setup, Extension
import sys
py_major_ver = sys.version_info[0]


class setupmodule:
    def __init__(self, name='noName'):
        self.name = name
        self.include_dirs = ['../usr/include/']
        self.extra_compile_args = []
        self.libraries = []
        self.library_dirs = ['../usr/lib']
        self.extra_link_args = []
        self.sources = [name+'.cpp']
        self.depends = [name+'.h']


isMAC = False
isOMP = True
ompLib = 'gomp'
boost_lib = 'boost_python'

if '--with-mac-omp' in sys.argv:
    isMAC = True
    ompLib = 'omp'
    idx = sys.argv.index('--with-mac-omp')
    sys.argv.pop(idx)
elif '--with-mac' in sys.argv:
    isMAC = True
    isOMP = False
    idx = sys.argv.index('--with-mac')
    sys.argv.pop(idx)

if py_major_ver == 3:
    boost_lib = boost_lib + '3'

modules = []

# m = setupmodule('csLCPLemkeSolver')
# if isMAC:
#     m.include_dirs.append('/usr/local/include/bullet/')
# else:
#     m.include_dirs.append('/usr/include/bullet/')
# m.libraries = ['boost_python', 'LinearMath', ompLib]
# if isMAC and isOMP:
#     m.extra_compile_args = ['-fopenmp', '-D __APPLE_OMP__']
# elif isOMP:
#     m.extra_compile_args = ['-fopenmp']
# else:
#     m.libraries.pop()
# m.sources = ['csLCPLemkeSolver.cpp', 'btLemkeSolver.cpp', 'btLemkeAlgorithm.cpp']
# m.depends = ['stdafx.h', 'csLCPLemkeSolver.h', 'btLemkeSolver.h', 'btLemkeAlgorithm.h']
# modules.append(m)

# m = setupmodule('csLCPDantzigSolver')
# if isMAC:
#     m.include_dirs.append('/usr/local/include/bullet/')
# else:
#     m.include_dirs.append('/usr/include/bullet/')
# m.libraries = ['boost_python', 'LinearMath', ompLib]
# if isMAC and isOMP:
#     m.extra_compile_args = ['-fopenmp', '-D __APPLE_OMP__']
# elif isOMP:
#     m.extra_compile_args = ['-fopenmp']
# else:
#     m.libraries.pop()
# m.sources = ['csLCPDantzigSolver.cpp', 'btDantzigLCP.cpp']
# m.depends = ['stdafx.h', 'csLCPDantzigSolver.h', 'btDantzigLCP.h', 'btDantzigSolver.h']
# modules.append(m)

m = setupmodule('csQPOASES')
if isMAC:
    m.libraries = [boost_lib, 'qpOASES', 'clapack', 'cblas', ompLib]
else:
    m.libraries = [boost_lib, 'qpOASES', 'lapack', 'blas', ompLib]
m.include_dirs.append('../usr/include/qpOASES')
if isMAC and isOMP:
    m.extra_compile_args = ['-Xpreprocessor', '-fopenmp', '-D __APPLE_OMP__']
elif isOMP:
    m.extra_compile_args = ['-fopenmp']
else:
    m.libraries.pop()
m.depends = ['stdafx.h']
modules.append(m)

'''
m = setupmodule('csEQP')
m.libraries = ['boost_python', 'LinearMath', ompLib]
if isMAC and isOMP:
    m.extra_compile_args = ['-fopenmp', '-D __APPLE_OMP__']
elif isOMP:
    m.extra_compile_args = ['-fopenmp']
else:
    m.libraries.pop()
m.library_dirs = ['../usr/lib']
m.sources = ['csLCPDantzigSolver.cpp', 'btDantzigLCP.cpp']
m.depends = ['stdafx.h', 'csLCPDantzigSolver.h', 'btDantzigLCP.h', 'btDantzigSolver.h']
# modules.append(m)
#m = ModuleInfo()
#modules.append(m)
#m.pkg_name = 'Optimization'
#m.module_name = 'csEQP'
#m.include_dirs = ['../external_libraries/BaseLib']
#m.libraries = ['BaseLib', 'AMD', 'gdi32', 'User32']
#m.library_dirs = ['Release', '../external_libraries/dependencies/UMFPACK5.2/', 'C:\Program Files\Microsoft Visual Studio .NET 2003\Vc7\PlatformSDK\Lib']
#m.depends_in_pkg_dir = ['BaseLibUtil.h', 'stdafx.h']
'''

for m in modules:
    ext_module = Extension(m.name,
                           include_dirs=m.include_dirs,
                           extra_compile_args=m.extra_compile_args,
                           extra_link_args=m.extra_link_args,
                           libraries=m.libraries,
                           library_dirs=m.library_dirs,
                           sources=m.sources,
                           depends=m.depends)

    setup(name=m.name, ext_modules=[ext_module])


'''
module_csLCPLemkeSolver = Extension('csLCPLemkeSolver',
        include_dirs = ['../usr/include/', '/usr/include/bullet/'],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-lgomp'],
        libraries = ['boost_python', 'LinearMath'],
        library_dirs = ['../usr/lib'],
        sources = ['csLCPLemkeSolver.cpp', 'btLemkeSolver.cpp', 'btLemkeAlgorithm.cpp'])
    
setup (name = 'csLCPLemkeSolver',
    version = '0.1',
    description = 'csLCPLemkeSolver',
    ext_modules = [module_csLCPLemkeSolver])

module_csLCPDantzigSolver = Extension('csLCPDantzigSolver',
        include_dirs = ['../usr/include/', '/usr/include/bullet', '/usr/local/include/libiomp'],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-lgomp'],
        libraries = ['boost_python', 'LinearMath'],
        library_dirs = ['../usr/lib'],
        sources = ['csLCPDantzigSolver.cpp','btDantzigLCP.cpp'])
    
setup (name = 'csLCPDantzigSolver',
    version = '0.1',
    description = 'csLCPDantzigSolver',
    ext_modules = [module_csLCPDantzigSolver])
'''
