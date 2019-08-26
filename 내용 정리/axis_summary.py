import numpy as np

x = np.array(range(32))
x = x.reshape(-1, 2, 4)
print("x.shape:", x.shape)  # (4, 2, 4)
# x.shape                   # (row, column, depth)
# print(x)
# #([ [ [ 0  1  2  3  ], [  4  5  6  7 ] ],
# #   [ [ 8  9  10 11 ], [ 12 13 14 15 ] ],
# #   [ [ 16 17 18 19 ], [ 20 21 22 23 ] ],
# #   [ [ 24 25 26 27 ], [ 28 29 30 31 ] ] ])

axis0 = x.sum(axis=0)
axis1 = x.sum(axis=1)
axis2 = x.sum(axis=2)

print("axis0.shape:", axis0.shape)  # (2, 4)
print("axis0:\n", axis0)
print("axis1.shape:", axis1.shape)  # (4, 4)
print("axis1:\n", axis1)
print("axis2.shape:", axis2.shape)  # (4, 2)
print("axis2:\n", axis2)
