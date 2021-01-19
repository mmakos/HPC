import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

with np.load( "../../data/datasets/6filledVal.npz", allow_pickle=True ) as data:
    img = data[ 'images' ]
    lab = data[ 'labels' ]
print( "Dataset loaded." )

img = img.astype( np.int32 )

stat = []
dyn = []

for i in tqdm( range( len( img ) ) ):
    moveSum = np.zeros( shape=3 )
    for k in range( 9, 15 ):
        for f in range( 1, 32 ):
            moveSum = np.add( moveSum, np.fabs( np.subtract( img[ i, k, f ], img[ i, k, f - 1 ] ) ) )
    moveSum = moveSum[ 0 ] + moveSum[ 1 ] + moveSum[ 2 ]

    if lab[ i ] < 5:
        stat.append( moveSum )
    else:
        dyn.append( moveSum )

stat = np.array( stat )
dyn = np.array( dyn )

statHist = np.histogram( stat, bins=1000, range=( 0, 4000 ) )
dynHist = np.histogram( dyn, bins=1000, range=( 0, 4000 ) )

rests = [ [ np.sum( statHist[ 0 ][ i: ] ), np.sum( dynHist[ 0 ][ :i ] ) ] for i in range( len( statHist[ 0 ] ) ) ]
restsSum = [ ( i[ 0 ] / len( stat ) + i[ 1 ] / len( dyn ) ) / 2 for i in rests ]
idx = np.argmin( restsSum )
print( f"value = { statHist[ 1 ][ idx ] }\nstatic wrong = { rests[ idx ][ 0 ] }\t{ ( rests[ idx ][ 0 ] / len( stat ) ) * 100 }%"
       f"\ndynamic wrong = { rests[ idx ][ 1 ] }\t{ ( rests[ idx ][ 1 ] / len( dyn ) ) * 100 }%\n"
       f"all wrong = { ( rests[ idx ][ 1 ] + rests[ idx ][ 0 ] ) * 100 / ( len( stat ) + len( dyn ) ) }%")

static = plt.figure( 0 )
plt.hist( stat, bins=100, range=( 0, 4000 ), density=True )
dynamic = plt.figure( 1 )
plt.hist( dyn, bins=100, range=( 0, 4000 ), color='C1', density=True )
plt.show()

print( "done" )
