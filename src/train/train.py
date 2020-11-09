import sys

sys.path.insert( 1, '../func' )
import model

print( "\n" )
m = model.getModel()
print( "\n\n" )
print( m.summary() )

