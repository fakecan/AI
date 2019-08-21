import numpy as np

x = np.array([ [ [1], [2] ],
                
               [ [3], [4] ] ] )


axis0 = x.sum(axis=0)
axis1 = x.sum(axis=1)
axis2 = x.sum(axis=2)

print("axis0 : ", axis0)
print("axis1 : ", axis1)
print("axis2 : ", axis2)
