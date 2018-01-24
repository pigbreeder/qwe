
# import conv_util as cos_module_np
import numpy as np
# import pylab

x = np.arange(0, 2 * np.pi, 0.1)
print(x)
import spam
print(spam.system("ls -l"))
import convUtil
convUtil.c_ext_forward()
import cos_module_np
y = cos_module_np.cos_func_(x)
print(x,y)
