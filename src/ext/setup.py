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
from Cython.Build import cythonize, build_ext
import numpy
convUtil = Extension(
    'convUtil',
    sources=['src/convUtil.pyx'],
    language='c',
    include_dirs=[numpy.get_include()],
    library_dirs=[],
    libraries=[],
    extra_compile_args=[],
    extra_link_args=[]
    )
sample=Extension('sample',
              ['src/convUtil_c.c'],
              include_dirs=[numpy.get_include()],
              define_macros=[],
              undef_macros=[],
              library_dirs=[],
              libraries=[]
              )
sample_pyx=Extension('sample_pyx',
              ['src/sample.pyx'],include_dirs=[numpy.get_include()])
setup(ext_modules = [sample_pyx,convUtil] ,
      cmdclass = {'build_ext': build_ext})