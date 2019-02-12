import numpy as np
import math
import tensorflow as tf

tf.get_variable()
a = np.random.randn(3, 3)
b = np.random.randn(3, 1)
c = a*b

if a[0][0] >= 0:
    print(0)
else:
    print(1)

print(str(0.5**3))

print(c)


