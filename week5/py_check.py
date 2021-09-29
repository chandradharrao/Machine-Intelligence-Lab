import numpy as np
from pprint import pprint

a = np.asarray([[2.0,2],[2.0,2]])
b = np.asarray([[2,2.0],[2,2.0]])
print((a==b).all())

a = np.array([[1.0, 2.0], [3.0, 4.0]])
print("Trans")
pprint(np.transpose(a))

a = np.array([[1.0,1],[1,1]])
print(2*a)

print(np.testing.assert_array_almost_equal(a,b))