import pickle
import os
from math import sin, cos, radians
import numpy as np

im = np.array( [ 100, 100, 100 ] )
angle = radians( 180 )

mat = np.array( [
    [ cos( angle ), 0, sin( angle ) ],
    [ 0, 1, 0 ],
    [ -sin( angle ), 0, cos( angle ) ]
] )

im = np.matmul( mat, im )

print( im )
