a = [[1,2],[3,4],[5,6]]
b = [[2,3],[4,5],[6,7]]
import numpy as np 
import tensorflow as tf
a = np.array(a)
b = np.array(b)
input = tf.concat((a, b), axis=1)
print(input.shape)
input1_shape = list(-1 if s.value is None else s.value for s in input.shape)
print(input1_shape)