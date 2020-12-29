import preprocess
import consts as c
import numpy as np

human = [ [ 0, 0, 0, 0 ], [ 6, 13, 0, 1 ], [ 4, 13, 0, 1 ], [ 3, 10, 0, 1 ], [ 1, 11, 0, 1 ],
          [ 0, 0, 0, 0 ], [ 0, 0, 0, 0 ], [ 0, 0, 0, 0 ], [ 6, 8, 0, 1 ], [ 0, 0, 0, 0 ], [ 0, 0, 0, 0 ], [ 0, 0, 0, 0 ],
          [ 7, 8, 0, 1 ], [ 8, 5, 0, 1 ], [ 9, 1, 0, 5 ] ]

print( preprocess.estimateNotDetectedKeypoints( human ) )


print( human[ 3 ] )
