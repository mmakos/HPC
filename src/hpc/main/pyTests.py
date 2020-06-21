from operator import truediv

list1 = [ [ 3, 3, 3 ], [ 1, 2, 3 ], [ 12, 16, 18 ] ]
list2 = [ 1, 2, 3 ]

for key in list1:
    print( list( map( truediv, key, list2 ) ) )

# initializing lists
test_list1 = [ 3, 5, 2, 6, 4 ]
test_list2 = [ 7, 3, 4, 1, 5 ]

# printing original lists
print( "The original list 1 is : " + str( test_list1 ) )
print( "The original list 2 is : " + str( test_list2 ) )

# division of lists
# using map()
res = list( map( truediv, test_list1, test_list2 ) )

# printing result
print( "The division list is : " + str( res ) )

import sys
print( sys.path )
sys.path.append( "M:\\Zrut C\\Michal\\Documents - Done\\Studia\\INŻ\\HPC\\openpose-rep\\build\\python\\openpose\\Release" )
sys.path.append( "M:\\Zrut C\\Michal\\Documents - Done\\Studia\\INŻ\\HPC\\openpose-rep\\build\\bin" )
sys.path.append( "M:\\Zrut C\\Michal\\Documents - Done\\Studia\\INŻ\\HPC\\openpose-rep\\build\\x64\\Release" )

