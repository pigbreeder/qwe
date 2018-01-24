# from distutils.core import setup, Extension
# import numpy
#
# # define the extension module
# cos_module_np = Extension('cos_module_np', sources=['src/cos_module_np.c'],
#                           include_dirs=[numpy.get_include()])
# convUtil = Extension('convUtil', sources=['src/convUtil.c'],
#                           include_dirs=[numpy.get_include()])
# spam = Extension('spam', sources=['src/spam.c'],
#                           include_dirs=[numpy.get_include()])
# # run the setup
# setup(ext_modules=[cos_module_np,convUtil])

from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy
setup(ext_modules = cythonize(Extension(
    'convUtil',
    sources=['src/convUtil.pyx'],
    language='c',
    include_dirs=[numpy.get_include()],
    library_dirs=[],
    libraries=[],
    extra_compile_args=[],
    extra_link_args=[]
)))