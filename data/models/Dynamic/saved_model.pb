бщ
гё
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
Ы
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
В
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
Н
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
╛
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*	2.4.0-rc22v2.4.0-rc1-25-g0b06f2927b8╜ё	
Д
conv2d_54/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv2d_54/kernel
}
$conv2d_54/kernel/Read/ReadVariableOpReadVariableOpconv2d_54/kernel*&
_output_shapes
:@*
dtype0
t
conv2d_54/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_54/bias
m
"conv2d_54/bias/Read/ReadVariableOpReadVariableOpconv2d_54/bias*
_output_shapes
:@*
dtype0
Д
conv2d_55/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv2d_55/kernel
}
$conv2d_55/kernel/Read/ReadVariableOpReadVariableOpconv2d_55/kernel*&
_output_shapes
:@@*
dtype0
t
conv2d_55/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_55/bias
m
"conv2d_55/bias/Read/ReadVariableOpReadVariableOpconv2d_55/bias*
_output_shapes
:@*
dtype0
Е
conv2d_56/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*!
shared_nameconv2d_56/kernel
~
$conv2d_56/kernel/Read/ReadVariableOpReadVariableOpconv2d_56/kernel*'
_output_shapes
:@А*
dtype0
u
conv2d_56/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv2d_56/bias
n
"conv2d_56/bias/Read/ReadVariableOpReadVariableOpconv2d_56/bias*
_output_shapes	
:А*
dtype0
Ж
conv2d_57/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*!
shared_nameconv2d_57/kernel

$conv2d_57/kernel/Read/ReadVariableOpReadVariableOpconv2d_57/kernel*(
_output_shapes
:АА*
dtype0
u
conv2d_57/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv2d_57/bias
n
"conv2d_57/bias/Read/ReadVariableOpReadVariableOpconv2d_57/bias*
_output_shapes	
:А*
dtype0
Ж
conv2d_58/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*!
shared_nameconv2d_58/kernel

$conv2d_58/kernel/Read/ReadVariableOpReadVariableOpconv2d_58/kernel*(
_output_shapes
:АА*
dtype0
u
conv2d_58/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv2d_58/bias
n
"conv2d_58/bias/Read/ReadVariableOpReadVariableOpconv2d_58/bias*
_output_shapes	
:А*
dtype0
Ж
conv2d_59/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*!
shared_nameconv2d_59/kernel

$conv2d_59/kernel/Read/ReadVariableOpReadVariableOpconv2d_59/kernel*(
_output_shapes
:АА*
dtype0
u
conv2d_59/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv2d_59/bias
n
"conv2d_59/bias/Read/ReadVariableOpReadVariableOpconv2d_59/bias*
_output_shapes	
:А*
dtype0
y
dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*
shared_namedense_9/kernel
r
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel*
_output_shapes
:	А*
dtype0
p
dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_9/bias
i
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
Т
Adam/conv2d_54/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv2d_54/kernel/m
Л
+Adam/conv2d_54/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_54/kernel/m*&
_output_shapes
:@*
dtype0
В
Adam/conv2d_54/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_54/bias/m
{
)Adam/conv2d_54/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_54/bias/m*
_output_shapes
:@*
dtype0
Т
Adam/conv2d_55/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv2d_55/kernel/m
Л
+Adam/conv2d_55/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_55/kernel/m*&
_output_shapes
:@@*
dtype0
В
Adam/conv2d_55/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_55/bias/m
{
)Adam/conv2d_55/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_55/bias/m*
_output_shapes
:@*
dtype0
У
Adam/conv2d_56/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*(
shared_nameAdam/conv2d_56/kernel/m
М
+Adam/conv2d_56/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_56/kernel/m*'
_output_shapes
:@А*
dtype0
Г
Adam/conv2d_56/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/conv2d_56/bias/m
|
)Adam/conv2d_56/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_56/bias/m*
_output_shapes	
:А*
dtype0
Ф
Adam/conv2d_57/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*(
shared_nameAdam/conv2d_57/kernel/m
Н
+Adam/conv2d_57/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_57/kernel/m*(
_output_shapes
:АА*
dtype0
Г
Adam/conv2d_57/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/conv2d_57/bias/m
|
)Adam/conv2d_57/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_57/bias/m*
_output_shapes	
:А*
dtype0
Ф
Adam/conv2d_58/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*(
shared_nameAdam/conv2d_58/kernel/m
Н
+Adam/conv2d_58/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_58/kernel/m*(
_output_shapes
:АА*
dtype0
Г
Adam/conv2d_58/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/conv2d_58/bias/m
|
)Adam/conv2d_58/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_58/bias/m*
_output_shapes	
:А*
dtype0
Ф
Adam/conv2d_59/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*(
shared_nameAdam/conv2d_59/kernel/m
Н
+Adam/conv2d_59/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_59/kernel/m*(
_output_shapes
:АА*
dtype0
Г
Adam/conv2d_59/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/conv2d_59/bias/m
|
)Adam/conv2d_59/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_59/bias/m*
_output_shapes	
:А*
dtype0
З
Adam/dense_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*&
shared_nameAdam/dense_9/kernel/m
А
)Adam/dense_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/m*
_output_shapes
:	А*
dtype0
~
Adam/dense_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_9/bias/m
w
'Adam/dense_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/m*
_output_shapes
:*
dtype0
Т
Adam/conv2d_54/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv2d_54/kernel/v
Л
+Adam/conv2d_54/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_54/kernel/v*&
_output_shapes
:@*
dtype0
В
Adam/conv2d_54/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_54/bias/v
{
)Adam/conv2d_54/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_54/bias/v*
_output_shapes
:@*
dtype0
Т
Adam/conv2d_55/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv2d_55/kernel/v
Л
+Adam/conv2d_55/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_55/kernel/v*&
_output_shapes
:@@*
dtype0
В
Adam/conv2d_55/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_55/bias/v
{
)Adam/conv2d_55/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_55/bias/v*
_output_shapes
:@*
dtype0
У
Adam/conv2d_56/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*(
shared_nameAdam/conv2d_56/kernel/v
М
+Adam/conv2d_56/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_56/kernel/v*'
_output_shapes
:@А*
dtype0
Г
Adam/conv2d_56/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/conv2d_56/bias/v
|
)Adam/conv2d_56/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_56/bias/v*
_output_shapes	
:А*
dtype0
Ф
Adam/conv2d_57/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*(
shared_nameAdam/conv2d_57/kernel/v
Н
+Adam/conv2d_57/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_57/kernel/v*(
_output_shapes
:АА*
dtype0
Г
Adam/conv2d_57/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/conv2d_57/bias/v
|
)Adam/conv2d_57/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_57/bias/v*
_output_shapes	
:А*
dtype0
Ф
Adam/conv2d_58/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*(
shared_nameAdam/conv2d_58/kernel/v
Н
+Adam/conv2d_58/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_58/kernel/v*(
_output_shapes
:АА*
dtype0
Г
Adam/conv2d_58/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/conv2d_58/bias/v
|
)Adam/conv2d_58/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_58/bias/v*
_output_shapes	
:А*
dtype0
Ф
Adam/conv2d_59/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*(
shared_nameAdam/conv2d_59/kernel/v
Н
+Adam/conv2d_59/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_59/kernel/v*(
_output_shapes
:АА*
dtype0
Г
Adam/conv2d_59/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/conv2d_59/bias/v
|
)Adam/conv2d_59/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_59/bias/v*
_output_shapes	
:А*
dtype0
З
Adam/dense_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*&
shared_nameAdam/dense_9/kernel/v
А
)Adam/dense_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/v*
_output_shapes
:	А*
dtype0
~
Adam/dense_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_9/bias/v
w
'Adam/dense_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
ЄO
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*нO
valueгOBаO BЩO
й
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer-8

layer_with_weights-6

layer-9
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
 	keras_api
h

!kernel
"bias
#regularization_losses
$	variables
%trainable_variables
&	keras_api
h

'kernel
(bias
)regularization_losses
*	variables
+trainable_variables
,	keras_api
R
-regularization_losses
.	variables
/trainable_variables
0	keras_api
h

1kernel
2bias
3regularization_losses
4	variables
5trainable_variables
6	keras_api
h

7kernel
8bias
9regularization_losses
:	variables
;trainable_variables
<	keras_api
R
=regularization_losses
>	variables
?trainable_variables
@	keras_api
h

Akernel
Bbias
Cregularization_losses
D	variables
Etrainable_variables
F	keras_api
╪
Giter

Hbeta_1

Ibeta_2
	Jdecay
Klearning_ratemОmПmРmС!mТ"mУ'mФ(mХ1mЦ2mЧ7mШ8mЩAmЪBmЫvЬvЭvЮvЯ!vа"vб'vв(vг1vд2vе7vж8vзAvиBvй
 
f
0
1
2
3
!4
"5
'6
(7
18
29
710
811
A12
B13
f
0
1
2
3
!4
"5
'6
(7
18
29
710
811
A12
B13
н
Llayer_metrics
regularization_losses
Mlayer_regularization_losses
Nnon_trainable_variables
trainable_variables

Olayers
Pmetrics
	variables
 
\Z
VARIABLE_VALUEconv2d_54/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_54/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
н
Qlayer_metrics
regularization_losses
	variables
Rlayer_regularization_losses
Snon_trainable_variables
trainable_variables

Tlayers
Umetrics
\Z
VARIABLE_VALUEconv2d_55/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_55/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
н
Vlayer_metrics
regularization_losses
	variables
Wlayer_regularization_losses
Xnon_trainable_variables
trainable_variables

Ylayers
Zmetrics
 
 
 
н
[layer_metrics
regularization_losses
	variables
\layer_regularization_losses
]non_trainable_variables
trainable_variables

^layers
_metrics
\Z
VARIABLE_VALUEconv2d_56/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_56/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

!0
"1

!0
"1
н
`layer_metrics
#regularization_losses
$	variables
alayer_regularization_losses
bnon_trainable_variables
%trainable_variables

clayers
dmetrics
\Z
VARIABLE_VALUEconv2d_57/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_57/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

'0
(1

'0
(1
н
elayer_metrics
)regularization_losses
*	variables
flayer_regularization_losses
gnon_trainable_variables
+trainable_variables

hlayers
imetrics
 
 
 
н
jlayer_metrics
-regularization_losses
.	variables
klayer_regularization_losses
lnon_trainable_variables
/trainable_variables

mlayers
nmetrics
\Z
VARIABLE_VALUEconv2d_58/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_58/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

10
21

10
21
н
olayer_metrics
3regularization_losses
4	variables
player_regularization_losses
qnon_trainable_variables
5trainable_variables

rlayers
smetrics
\Z
VARIABLE_VALUEconv2d_59/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_59/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

70
81

70
81
н
tlayer_metrics
9regularization_losses
:	variables
ulayer_regularization_losses
vnon_trainable_variables
;trainable_variables

wlayers
xmetrics
 
 
 
н
ylayer_metrics
=regularization_losses
>	variables
zlayer_regularization_losses
{non_trainable_variables
?trainable_variables

|layers
}metrics
ZX
VARIABLE_VALUEdense_9/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_9/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

A0
B1

A0
B1
░
~layer_metrics
Cregularization_losses
D	variables
layer_regularization_losses
Аnon_trainable_variables
Etrainable_variables
Бlayers
Вmetrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
F
0
1
2
3
4
5
6
7
	8

9

Г0
Д1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

Еtotal

Жcount
З	variables
И	keras_api
I

Йtotal

Кcount
Л
_fn_kwargs
М	variables
Н	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

Е0
Ж1

З	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

Й0
К1

М	variables
}
VARIABLE_VALUEAdam/conv2d_54/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_54/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_55/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_55/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_56/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_56/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_57/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_57/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_58/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_58/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_59/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_59/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_9/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_9/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_54/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_54/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_55/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_55/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_56/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_56/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_57/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_57/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_58/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_58/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_59/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_59/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_9/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_9/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Т
serving_default_conv2d_54_inputPlaceholder*/
_output_shapes
:          *
dtype0*$
shape:          
├
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_54_inputconv2d_54/kernelconv2d_54/biasconv2d_55/kernelconv2d_55/biasconv2d_56/kernelconv2d_56/biasconv2d_57/kernelconv2d_57/biasconv2d_58/kernelconv2d_58/biasconv2d_59/kernelconv2d_59/biasdense_9/kerneldense_9/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *-
f(R&
$__inference_signature_wrapper_376115
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
┤
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_54/kernel/Read/ReadVariableOp"conv2d_54/bias/Read/ReadVariableOp$conv2d_55/kernel/Read/ReadVariableOp"conv2d_55/bias/Read/ReadVariableOp$conv2d_56/kernel/Read/ReadVariableOp"conv2d_56/bias/Read/ReadVariableOp$conv2d_57/kernel/Read/ReadVariableOp"conv2d_57/bias/Read/ReadVariableOp$conv2d_58/kernel/Read/ReadVariableOp"conv2d_58/bias/Read/ReadVariableOp$conv2d_59/kernel/Read/ReadVariableOp"conv2d_59/bias/Read/ReadVariableOp"dense_9/kernel/Read/ReadVariableOp dense_9/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/conv2d_54/kernel/m/Read/ReadVariableOp)Adam/conv2d_54/bias/m/Read/ReadVariableOp+Adam/conv2d_55/kernel/m/Read/ReadVariableOp)Adam/conv2d_55/bias/m/Read/ReadVariableOp+Adam/conv2d_56/kernel/m/Read/ReadVariableOp)Adam/conv2d_56/bias/m/Read/ReadVariableOp+Adam/conv2d_57/kernel/m/Read/ReadVariableOp)Adam/conv2d_57/bias/m/Read/ReadVariableOp+Adam/conv2d_58/kernel/m/Read/ReadVariableOp)Adam/conv2d_58/bias/m/Read/ReadVariableOp+Adam/conv2d_59/kernel/m/Read/ReadVariableOp)Adam/conv2d_59/bias/m/Read/ReadVariableOp)Adam/dense_9/kernel/m/Read/ReadVariableOp'Adam/dense_9/bias/m/Read/ReadVariableOp+Adam/conv2d_54/kernel/v/Read/ReadVariableOp)Adam/conv2d_54/bias/v/Read/ReadVariableOp+Adam/conv2d_55/kernel/v/Read/ReadVariableOp)Adam/conv2d_55/bias/v/Read/ReadVariableOp+Adam/conv2d_56/kernel/v/Read/ReadVariableOp)Adam/conv2d_56/bias/v/Read/ReadVariableOp+Adam/conv2d_57/kernel/v/Read/ReadVariableOp)Adam/conv2d_57/bias/v/Read/ReadVariableOp+Adam/conv2d_58/kernel/v/Read/ReadVariableOp)Adam/conv2d_58/bias/v/Read/ReadVariableOp+Adam/conv2d_59/kernel/v/Read/ReadVariableOp)Adam/conv2d_59/bias/v/Read/ReadVariableOp)Adam/dense_9/kernel/v/Read/ReadVariableOp'Adam/dense_9/bias/v/Read/ReadVariableOpConst*@
Tin9
725	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *(
f#R!
__inference__traced_save_376611
│

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_54/kernelconv2d_54/biasconv2d_55/kernelconv2d_55/biasconv2d_56/kernelconv2d_56/biasconv2d_57/kernelconv2d_57/biasconv2d_58/kernelconv2d_58/biasconv2d_59/kernelconv2d_59/biasdense_9/kerneldense_9/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv2d_54/kernel/mAdam/conv2d_54/bias/mAdam/conv2d_55/kernel/mAdam/conv2d_55/bias/mAdam/conv2d_56/kernel/mAdam/conv2d_56/bias/mAdam/conv2d_57/kernel/mAdam/conv2d_57/bias/mAdam/conv2d_58/kernel/mAdam/conv2d_58/bias/mAdam/conv2d_59/kernel/mAdam/conv2d_59/bias/mAdam/dense_9/kernel/mAdam/dense_9/bias/mAdam/conv2d_54/kernel/vAdam/conv2d_54/bias/vAdam/conv2d_55/kernel/vAdam/conv2d_55/bias/vAdam/conv2d_56/kernel/vAdam/conv2d_56/bias/vAdam/conv2d_57/kernel/vAdam/conv2d_57/bias/vAdam/conv2d_58/kernel/vAdam/conv2d_58/bias/vAdam/conv2d_59/kernel/vAdam/conv2d_59/bias/vAdam/dense_9/kernel/vAdam/dense_9/bias/v*?
Tin8
624*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *+
f&R$
"__inference__traced_restore_376774еЩ
╧

▐
E__inference_conv2d_55_layer_call_and_return_conditional_losses_375724

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpг
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          @*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          @2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:          @2
ReluЯ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:          @2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:          @::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          @
 
_user_specified_nameinputs
▓
M
1__inference_max_pooling2d_19_layer_call_fn_375669

inputs
identityЁ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_max_pooling2d_19_layer_call_and_return_conditional_losses_3756632
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
╪

▐
E__inference_conv2d_57_layer_call_and_return_conditional_losses_376366

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:          А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:          А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:          А2
Reluа
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:          А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:          А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:          А
 
_user_specified_nameinputs
╨0
Ъ
H__inference_sequential_9_layer_call_and_return_conditional_losses_375879
conv2d_54_input
conv2d_54_375708
conv2d_54_375710
conv2d_55_375735
conv2d_55_375737
conv2d_56_375763
conv2d_56_375765
conv2d_57_375790
conv2d_57_375792
conv2d_58_375818
conv2d_58_375820
conv2d_59_375845
conv2d_59_375847
dense_9_375873
dense_9_375875
identityИв!conv2d_54/StatefulPartitionedCallв!conv2d_55/StatefulPartitionedCallв!conv2d_56/StatefulPartitionedCallв!conv2d_57/StatefulPartitionedCallв!conv2d_58/StatefulPartitionedCallв!conv2d_59/StatefulPartitionedCallвdense_9/StatefulPartitionedCallн
!conv2d_54/StatefulPartitionedCallStatefulPartitionedCallconv2d_54_inputconv2d_54_375708conv2d_54_375710*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_54_layer_call_and_return_conditional_losses_3756972#
!conv2d_54/StatefulPartitionedCall╚
!conv2d_55/StatefulPartitionedCallStatefulPartitionedCall*conv2d_54/StatefulPartitionedCall:output:0conv2d_55_375735conv2d_55_375737*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_55_layer_call_and_return_conditional_losses_3757242#
!conv2d_55/StatefulPartitionedCallЫ
 max_pooling2d_18/PartitionedCallPartitionedCall*conv2d_55/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_max_pooling2d_18_layer_call_and_return_conditional_losses_3756512"
 max_pooling2d_18/PartitionedCall╚
!conv2d_56/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_18/PartitionedCall:output:0conv2d_56_375763conv2d_56_375765*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:          А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_56_layer_call_and_return_conditional_losses_3757522#
!conv2d_56/StatefulPartitionedCall╔
!conv2d_57/StatefulPartitionedCallStatefulPartitionedCall*conv2d_56/StatefulPartitionedCall:output:0conv2d_57_375790conv2d_57_375792*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:          А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_57_layer_call_and_return_conditional_losses_3757792#
!conv2d_57/StatefulPartitionedCallЬ
 max_pooling2d_19/PartitionedCallPartitionedCall*conv2d_57/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_max_pooling2d_19_layer_call_and_return_conditional_losses_3756632"
 max_pooling2d_19/PartitionedCall╚
!conv2d_58/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_19/PartitionedCall:output:0conv2d_58_375818conv2d_58_375820*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_58_layer_call_and_return_conditional_losses_3758072#
!conv2d_58/StatefulPartitionedCall╔
!conv2d_59/StatefulPartitionedCallStatefulPartitionedCall*conv2d_58/StatefulPartitionedCall:output:0conv2d_59_375845conv2d_59_375847*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_59_layer_call_and_return_conditional_losses_3758342#
!conv2d_59/StatefulPartitionedCall▓
*global_average_pooling2d_9/PartitionedCallPartitionedCall*conv2d_59/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *_
fZRX
V__inference_global_average_pooling2d_9_layer_call_and_return_conditional_losses_3756762,
*global_average_pooling2d_9/PartitionedCall┐
dense_9/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling2d_9/PartitionedCall:output:0dense_9_375873dense_9_375875*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_3758622!
dense_9/StatefulPartitionedCallЎ
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0"^conv2d_54/StatefulPartitionedCall"^conv2d_55/StatefulPartitionedCall"^conv2d_56/StatefulPartitionedCall"^conv2d_57/StatefulPartitionedCall"^conv2d_58/StatefulPartitionedCall"^conv2d_59/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:          ::::::::::::::2F
!conv2d_54/StatefulPartitionedCall!conv2d_54/StatefulPartitionedCall2F
!conv2d_55/StatefulPartitionedCall!conv2d_55/StatefulPartitionedCall2F
!conv2d_56/StatefulPartitionedCall!conv2d_56/StatefulPartitionedCall2F
!conv2d_57/StatefulPartitionedCall!conv2d_57/StatefulPartitionedCall2F
!conv2d_58/StatefulPartitionedCall!conv2d_58/StatefulPartitionedCall2F
!conv2d_59/StatefulPartitionedCall!conv2d_59/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:` \
/
_output_shapes
:          
)
_user_specified_nameconv2d_54_input
■	
╣
-__inference_sequential_9_layer_call_fn_376262

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identityИвStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_sequential_9_layer_call_and_return_conditional_losses_3759662
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:          ::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
Щ

┬
-__inference_sequential_9_layer_call_fn_376072
conv2d_54_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identityИвStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCallconv2d_54_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_sequential_9_layer_call_and_return_conditional_losses_3760412
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:          ::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:          
)
_user_specified_nameconv2d_54_input
╪

▐
E__inference_conv2d_59_layer_call_and_return_conditional_losses_376406

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         А2
Reluа
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
┤M
═	
H__inference_sequential_9_layer_call_and_return_conditional_losses_376229

inputs,
(conv2d_54_conv2d_readvariableop_resource-
)conv2d_54_biasadd_readvariableop_resource,
(conv2d_55_conv2d_readvariableop_resource-
)conv2d_55_biasadd_readvariableop_resource,
(conv2d_56_conv2d_readvariableop_resource-
)conv2d_56_biasadd_readvariableop_resource,
(conv2d_57_conv2d_readvariableop_resource-
)conv2d_57_biasadd_readvariableop_resource,
(conv2d_58_conv2d_readvariableop_resource-
)conv2d_58_biasadd_readvariableop_resource,
(conv2d_59_conv2d_readvariableop_resource-
)conv2d_59_biasadd_readvariableop_resource*
&dense_9_matmul_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource
identityИв conv2d_54/BiasAdd/ReadVariableOpвconv2d_54/Conv2D/ReadVariableOpв conv2d_55/BiasAdd/ReadVariableOpвconv2d_55/Conv2D/ReadVariableOpв conv2d_56/BiasAdd/ReadVariableOpвconv2d_56/Conv2D/ReadVariableOpв conv2d_57/BiasAdd/ReadVariableOpвconv2d_57/Conv2D/ReadVariableOpв conv2d_58/BiasAdd/ReadVariableOpвconv2d_58/Conv2D/ReadVariableOpв conv2d_59/BiasAdd/ReadVariableOpвconv2d_59/Conv2D/ReadVariableOpвdense_9/BiasAdd/ReadVariableOpвdense_9/MatMul/ReadVariableOp│
conv2d_54/Conv2D/ReadVariableOpReadVariableOp(conv2d_54_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02!
conv2d_54/Conv2D/ReadVariableOp┴
conv2d_54/Conv2DConv2Dinputs'conv2d_54/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          @*
paddingSAME*
strides
2
conv2d_54/Conv2Dк
 conv2d_54/BiasAdd/ReadVariableOpReadVariableOp)conv2d_54_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_54/BiasAdd/ReadVariableOp░
conv2d_54/BiasAddBiasAddconv2d_54/Conv2D:output:0(conv2d_54/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          @2
conv2d_54/BiasAdd~
conv2d_54/ReluReluconv2d_54/BiasAdd:output:0*
T0*/
_output_shapes
:          @2
conv2d_54/Relu│
conv2d_55/Conv2D/ReadVariableOpReadVariableOp(conv2d_55_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_55/Conv2D/ReadVariableOp╫
conv2d_55/Conv2DConv2Dconv2d_54/Relu:activations:0'conv2d_55/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          @*
paddingSAME*
strides
2
conv2d_55/Conv2Dк
 conv2d_55/BiasAdd/ReadVariableOpReadVariableOp)conv2d_55_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_55/BiasAdd/ReadVariableOp░
conv2d_55/BiasAddBiasAddconv2d_55/Conv2D:output:0(conv2d_55/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          @2
conv2d_55/BiasAdd~
conv2d_55/ReluReluconv2d_55/BiasAdd:output:0*
T0*/
_output_shapes
:          @2
conv2d_55/Relu╔
max_pooling2d_18/MaxPoolMaxPoolconv2d_55/Relu:activations:0*/
_output_shapes
:          @*
ksize
*
paddingSAME*
strides
2
max_pooling2d_18/MaxPool┤
conv2d_56/Conv2D/ReadVariableOpReadVariableOp(conv2d_56_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02!
conv2d_56/Conv2D/ReadVariableOp▌
conv2d_56/Conv2DConv2D!max_pooling2d_18/MaxPool:output:0'conv2d_56/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:          А*
paddingSAME*
strides
2
conv2d_56/Conv2Dл
 conv2d_56/BiasAdd/ReadVariableOpReadVariableOp)conv2d_56_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_56/BiasAdd/ReadVariableOp▒
conv2d_56/BiasAddBiasAddconv2d_56/Conv2D:output:0(conv2d_56/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:          А2
conv2d_56/BiasAdd
conv2d_56/ReluReluconv2d_56/BiasAdd:output:0*
T0*0
_output_shapes
:          А2
conv2d_56/Relu╡
conv2d_57/Conv2D/ReadVariableOpReadVariableOp(conv2d_57_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02!
conv2d_57/Conv2D/ReadVariableOp╪
conv2d_57/Conv2DConv2Dconv2d_56/Relu:activations:0'conv2d_57/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:          А*
paddingSAME*
strides
2
conv2d_57/Conv2Dл
 conv2d_57/BiasAdd/ReadVariableOpReadVariableOp)conv2d_57_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_57/BiasAdd/ReadVariableOp▒
conv2d_57/BiasAddBiasAddconv2d_57/Conv2D:output:0(conv2d_57/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:          А2
conv2d_57/BiasAdd
conv2d_57/ReluReluconv2d_57/BiasAdd:output:0*
T0*0
_output_shapes
:          А2
conv2d_57/Relu╩
max_pooling2d_19/MaxPoolMaxPoolconv2d_57/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingSAME*
strides
2
max_pooling2d_19/MaxPool╡
conv2d_58/Conv2D/ReadVariableOpReadVariableOp(conv2d_58_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02!
conv2d_58/Conv2D/ReadVariableOp▌
conv2d_58/Conv2DConv2D!max_pooling2d_19/MaxPool:output:0'conv2d_58/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
conv2d_58/Conv2Dл
 conv2d_58/BiasAdd/ReadVariableOpReadVariableOp)conv2d_58_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_58/BiasAdd/ReadVariableOp▒
conv2d_58/BiasAddBiasAddconv2d_58/Conv2D:output:0(conv2d_58/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
conv2d_58/BiasAdd
conv2d_58/ReluReluconv2d_58/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
conv2d_58/Relu╡
conv2d_59/Conv2D/ReadVariableOpReadVariableOp(conv2d_59_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02!
conv2d_59/Conv2D/ReadVariableOp╪
conv2d_59/Conv2DConv2Dconv2d_58/Relu:activations:0'conv2d_59/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
conv2d_59/Conv2Dл
 conv2d_59/BiasAdd/ReadVariableOpReadVariableOp)conv2d_59_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_59/BiasAdd/ReadVariableOp▒
conv2d_59/BiasAddBiasAddconv2d_59/Conv2D:output:0(conv2d_59/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
conv2d_59/BiasAdd
conv2d_59/ReluReluconv2d_59/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
conv2d_59/Relu╖
1global_average_pooling2d_9/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      23
1global_average_pooling2d_9/Mean/reduction_indices╫
global_average_pooling2d_9/MeanMeanconv2d_59/Relu:activations:0:global_average_pooling2d_9/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:         А2!
global_average_pooling2d_9/Meanж
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
dense_9/MatMul/ReadVariableOpн
dense_9/MatMulMatMul(global_average_pooling2d_9/Mean:output:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_9/MatMulд
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_9/BiasAdd/ReadVariableOpб
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_9/BiasAddy
dense_9/SoftmaxSoftmaxdense_9/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_9/Softmax╠
IdentityIdentitydense_9/Softmax:softmax:0!^conv2d_54/BiasAdd/ReadVariableOp ^conv2d_54/Conv2D/ReadVariableOp!^conv2d_55/BiasAdd/ReadVariableOp ^conv2d_55/Conv2D/ReadVariableOp!^conv2d_56/BiasAdd/ReadVariableOp ^conv2d_56/Conv2D/ReadVariableOp!^conv2d_57/BiasAdd/ReadVariableOp ^conv2d_57/Conv2D/ReadVariableOp!^conv2d_58/BiasAdd/ReadVariableOp ^conv2d_58/Conv2D/ReadVariableOp!^conv2d_59/BiasAdd/ReadVariableOp ^conv2d_59/Conv2D/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:          ::::::::::::::2D
 conv2d_54/BiasAdd/ReadVariableOp conv2d_54/BiasAdd/ReadVariableOp2B
conv2d_54/Conv2D/ReadVariableOpconv2d_54/Conv2D/ReadVariableOp2D
 conv2d_55/BiasAdd/ReadVariableOp conv2d_55/BiasAdd/ReadVariableOp2B
conv2d_55/Conv2D/ReadVariableOpconv2d_55/Conv2D/ReadVariableOp2D
 conv2d_56/BiasAdd/ReadVariableOp conv2d_56/BiasAdd/ReadVariableOp2B
conv2d_56/Conv2D/ReadVariableOpconv2d_56/Conv2D/ReadVariableOp2D
 conv2d_57/BiasAdd/ReadVariableOp conv2d_57/BiasAdd/ReadVariableOp2B
conv2d_57/Conv2D/ReadVariableOpconv2d_57/Conv2D/ReadVariableOp2D
 conv2d_58/BiasAdd/ReadVariableOp conv2d_58/BiasAdd/ReadVariableOp2B
conv2d_58/Conv2D/ReadVariableOpconv2d_58/Conv2D/ReadVariableOp2D
 conv2d_59/BiasAdd/ReadVariableOp conv2d_59/BiasAdd/ReadVariableOp2B
conv2d_59/Conv2D/ReadVariableOpconv2d_59/Conv2D/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
Б

*__inference_conv2d_55_layer_call_fn_376335

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_55_layer_call_and_return_conditional_losses_3757242
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:          @2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:          @::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          @
 
_user_specified_nameinputs
╪

▐
E__inference_conv2d_58_layer_call_and_return_conditional_losses_375807

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         А2
Reluа
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
╡0
С
H__inference_sequential_9_layer_call_and_return_conditional_losses_376041

inputs
conv2d_54_376002
conv2d_54_376004
conv2d_55_376007
conv2d_55_376009
conv2d_56_376013
conv2d_56_376015
conv2d_57_376018
conv2d_57_376020
conv2d_58_376024
conv2d_58_376026
conv2d_59_376029
conv2d_59_376031
dense_9_376035
dense_9_376037
identityИв!conv2d_54/StatefulPartitionedCallв!conv2d_55/StatefulPartitionedCallв!conv2d_56/StatefulPartitionedCallв!conv2d_57/StatefulPartitionedCallв!conv2d_58/StatefulPartitionedCallв!conv2d_59/StatefulPartitionedCallвdense_9/StatefulPartitionedCallд
!conv2d_54/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_54_376002conv2d_54_376004*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_54_layer_call_and_return_conditional_losses_3756972#
!conv2d_54/StatefulPartitionedCall╚
!conv2d_55/StatefulPartitionedCallStatefulPartitionedCall*conv2d_54/StatefulPartitionedCall:output:0conv2d_55_376007conv2d_55_376009*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_55_layer_call_and_return_conditional_losses_3757242#
!conv2d_55/StatefulPartitionedCallЫ
 max_pooling2d_18/PartitionedCallPartitionedCall*conv2d_55/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_max_pooling2d_18_layer_call_and_return_conditional_losses_3756512"
 max_pooling2d_18/PartitionedCall╚
!conv2d_56/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_18/PartitionedCall:output:0conv2d_56_376013conv2d_56_376015*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:          А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_56_layer_call_and_return_conditional_losses_3757522#
!conv2d_56/StatefulPartitionedCall╔
!conv2d_57/StatefulPartitionedCallStatefulPartitionedCall*conv2d_56/StatefulPartitionedCall:output:0conv2d_57_376018conv2d_57_376020*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:          А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_57_layer_call_and_return_conditional_losses_3757792#
!conv2d_57/StatefulPartitionedCallЬ
 max_pooling2d_19/PartitionedCallPartitionedCall*conv2d_57/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_max_pooling2d_19_layer_call_and_return_conditional_losses_3756632"
 max_pooling2d_19/PartitionedCall╚
!conv2d_58/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_19/PartitionedCall:output:0conv2d_58_376024conv2d_58_376026*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_58_layer_call_and_return_conditional_losses_3758072#
!conv2d_58/StatefulPartitionedCall╔
!conv2d_59/StatefulPartitionedCallStatefulPartitionedCall*conv2d_58/StatefulPartitionedCall:output:0conv2d_59_376029conv2d_59_376031*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_59_layer_call_and_return_conditional_losses_3758342#
!conv2d_59/StatefulPartitionedCall▓
*global_average_pooling2d_9/PartitionedCallPartitionedCall*conv2d_59/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *_
fZRX
V__inference_global_average_pooling2d_9_layer_call_and_return_conditional_losses_3756762,
*global_average_pooling2d_9/PartitionedCall┐
dense_9/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling2d_9/PartitionedCall:output:0dense_9_376035dense_9_376037*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_3758622!
dense_9/StatefulPartitionedCallЎ
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0"^conv2d_54/StatefulPartitionedCall"^conv2d_55/StatefulPartitionedCall"^conv2d_56/StatefulPartitionedCall"^conv2d_57/StatefulPartitionedCall"^conv2d_58/StatefulPartitionedCall"^conv2d_59/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:          ::::::::::::::2F
!conv2d_54/StatefulPartitionedCall!conv2d_54/StatefulPartitionedCall2F
!conv2d_55/StatefulPartitionedCall!conv2d_55/StatefulPartitionedCall2F
!conv2d_56/StatefulPartitionedCall!conv2d_56/StatefulPartitionedCall2F
!conv2d_57/StatefulPartitionedCall!conv2d_57/StatefulPartitionedCall2F
!conv2d_58/StatefulPartitionedCall!conv2d_58/StatefulPartitionedCall2F
!conv2d_59/StatefulPartitionedCall!conv2d_59/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
▓
M
1__inference_max_pooling2d_18_layer_call_fn_375657

inputs
identityЁ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_max_pooling2d_18_layer_call_and_return_conditional_losses_3756512
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
┤M
═	
H__inference_sequential_9_layer_call_and_return_conditional_losses_376172

inputs,
(conv2d_54_conv2d_readvariableop_resource-
)conv2d_54_biasadd_readvariableop_resource,
(conv2d_55_conv2d_readvariableop_resource-
)conv2d_55_biasadd_readvariableop_resource,
(conv2d_56_conv2d_readvariableop_resource-
)conv2d_56_biasadd_readvariableop_resource,
(conv2d_57_conv2d_readvariableop_resource-
)conv2d_57_biasadd_readvariableop_resource,
(conv2d_58_conv2d_readvariableop_resource-
)conv2d_58_biasadd_readvariableop_resource,
(conv2d_59_conv2d_readvariableop_resource-
)conv2d_59_biasadd_readvariableop_resource*
&dense_9_matmul_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource
identityИв conv2d_54/BiasAdd/ReadVariableOpвconv2d_54/Conv2D/ReadVariableOpв conv2d_55/BiasAdd/ReadVariableOpвconv2d_55/Conv2D/ReadVariableOpв conv2d_56/BiasAdd/ReadVariableOpвconv2d_56/Conv2D/ReadVariableOpв conv2d_57/BiasAdd/ReadVariableOpвconv2d_57/Conv2D/ReadVariableOpв conv2d_58/BiasAdd/ReadVariableOpвconv2d_58/Conv2D/ReadVariableOpв conv2d_59/BiasAdd/ReadVariableOpвconv2d_59/Conv2D/ReadVariableOpвdense_9/BiasAdd/ReadVariableOpвdense_9/MatMul/ReadVariableOp│
conv2d_54/Conv2D/ReadVariableOpReadVariableOp(conv2d_54_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02!
conv2d_54/Conv2D/ReadVariableOp┴
conv2d_54/Conv2DConv2Dinputs'conv2d_54/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          @*
paddingSAME*
strides
2
conv2d_54/Conv2Dк
 conv2d_54/BiasAdd/ReadVariableOpReadVariableOp)conv2d_54_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_54/BiasAdd/ReadVariableOp░
conv2d_54/BiasAddBiasAddconv2d_54/Conv2D:output:0(conv2d_54/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          @2
conv2d_54/BiasAdd~
conv2d_54/ReluReluconv2d_54/BiasAdd:output:0*
T0*/
_output_shapes
:          @2
conv2d_54/Relu│
conv2d_55/Conv2D/ReadVariableOpReadVariableOp(conv2d_55_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_55/Conv2D/ReadVariableOp╫
conv2d_55/Conv2DConv2Dconv2d_54/Relu:activations:0'conv2d_55/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          @*
paddingSAME*
strides
2
conv2d_55/Conv2Dк
 conv2d_55/BiasAdd/ReadVariableOpReadVariableOp)conv2d_55_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_55/BiasAdd/ReadVariableOp░
conv2d_55/BiasAddBiasAddconv2d_55/Conv2D:output:0(conv2d_55/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          @2
conv2d_55/BiasAdd~
conv2d_55/ReluReluconv2d_55/BiasAdd:output:0*
T0*/
_output_shapes
:          @2
conv2d_55/Relu╔
max_pooling2d_18/MaxPoolMaxPoolconv2d_55/Relu:activations:0*/
_output_shapes
:          @*
ksize
*
paddingSAME*
strides
2
max_pooling2d_18/MaxPool┤
conv2d_56/Conv2D/ReadVariableOpReadVariableOp(conv2d_56_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02!
conv2d_56/Conv2D/ReadVariableOp▌
conv2d_56/Conv2DConv2D!max_pooling2d_18/MaxPool:output:0'conv2d_56/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:          А*
paddingSAME*
strides
2
conv2d_56/Conv2Dл
 conv2d_56/BiasAdd/ReadVariableOpReadVariableOp)conv2d_56_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_56/BiasAdd/ReadVariableOp▒
conv2d_56/BiasAddBiasAddconv2d_56/Conv2D:output:0(conv2d_56/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:          А2
conv2d_56/BiasAdd
conv2d_56/ReluReluconv2d_56/BiasAdd:output:0*
T0*0
_output_shapes
:          А2
conv2d_56/Relu╡
conv2d_57/Conv2D/ReadVariableOpReadVariableOp(conv2d_57_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02!
conv2d_57/Conv2D/ReadVariableOp╪
conv2d_57/Conv2DConv2Dconv2d_56/Relu:activations:0'conv2d_57/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:          А*
paddingSAME*
strides
2
conv2d_57/Conv2Dл
 conv2d_57/BiasAdd/ReadVariableOpReadVariableOp)conv2d_57_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_57/BiasAdd/ReadVariableOp▒
conv2d_57/BiasAddBiasAddconv2d_57/Conv2D:output:0(conv2d_57/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:          А2
conv2d_57/BiasAdd
conv2d_57/ReluReluconv2d_57/BiasAdd:output:0*
T0*0
_output_shapes
:          А2
conv2d_57/Relu╩
max_pooling2d_19/MaxPoolMaxPoolconv2d_57/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingSAME*
strides
2
max_pooling2d_19/MaxPool╡
conv2d_58/Conv2D/ReadVariableOpReadVariableOp(conv2d_58_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02!
conv2d_58/Conv2D/ReadVariableOp▌
conv2d_58/Conv2DConv2D!max_pooling2d_19/MaxPool:output:0'conv2d_58/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
conv2d_58/Conv2Dл
 conv2d_58/BiasAdd/ReadVariableOpReadVariableOp)conv2d_58_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_58/BiasAdd/ReadVariableOp▒
conv2d_58/BiasAddBiasAddconv2d_58/Conv2D:output:0(conv2d_58/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
conv2d_58/BiasAdd
conv2d_58/ReluReluconv2d_58/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
conv2d_58/Relu╡
conv2d_59/Conv2D/ReadVariableOpReadVariableOp(conv2d_59_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02!
conv2d_59/Conv2D/ReadVariableOp╪
conv2d_59/Conv2DConv2Dconv2d_58/Relu:activations:0'conv2d_59/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
conv2d_59/Conv2Dл
 conv2d_59/BiasAdd/ReadVariableOpReadVariableOp)conv2d_59_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_59/BiasAdd/ReadVariableOp▒
conv2d_59/BiasAddBiasAddconv2d_59/Conv2D:output:0(conv2d_59/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
conv2d_59/BiasAdd
conv2d_59/ReluReluconv2d_59/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
conv2d_59/Relu╖
1global_average_pooling2d_9/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      23
1global_average_pooling2d_9/Mean/reduction_indices╫
global_average_pooling2d_9/MeanMeanconv2d_59/Relu:activations:0:global_average_pooling2d_9/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:         А2!
global_average_pooling2d_9/Meanж
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
dense_9/MatMul/ReadVariableOpн
dense_9/MatMulMatMul(global_average_pooling2d_9/Mean:output:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_9/MatMulд
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_9/BiasAdd/ReadVariableOpб
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_9/BiasAddy
dense_9/SoftmaxSoftmaxdense_9/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_9/Softmax╠
IdentityIdentitydense_9/Softmax:softmax:0!^conv2d_54/BiasAdd/ReadVariableOp ^conv2d_54/Conv2D/ReadVariableOp!^conv2d_55/BiasAdd/ReadVariableOp ^conv2d_55/Conv2D/ReadVariableOp!^conv2d_56/BiasAdd/ReadVariableOp ^conv2d_56/Conv2D/ReadVariableOp!^conv2d_57/BiasAdd/ReadVariableOp ^conv2d_57/Conv2D/ReadVariableOp!^conv2d_58/BiasAdd/ReadVariableOp ^conv2d_58/Conv2D/ReadVariableOp!^conv2d_59/BiasAdd/ReadVariableOp ^conv2d_59/Conv2D/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:          ::::::::::::::2D
 conv2d_54/BiasAdd/ReadVariableOp conv2d_54/BiasAdd/ReadVariableOp2B
conv2d_54/Conv2D/ReadVariableOpconv2d_54/Conv2D/ReadVariableOp2D
 conv2d_55/BiasAdd/ReadVariableOp conv2d_55/BiasAdd/ReadVariableOp2B
conv2d_55/Conv2D/ReadVariableOpconv2d_55/Conv2D/ReadVariableOp2D
 conv2d_56/BiasAdd/ReadVariableOp conv2d_56/BiasAdd/ReadVariableOp2B
conv2d_56/Conv2D/ReadVariableOpconv2d_56/Conv2D/ReadVariableOp2D
 conv2d_57/BiasAdd/ReadVariableOp conv2d_57/BiasAdd/ReadVariableOp2B
conv2d_57/Conv2D/ReadVariableOpconv2d_57/Conv2D/ReadVariableOp2D
 conv2d_58/BiasAdd/ReadVariableOp conv2d_58/BiasAdd/ReadVariableOp2B
conv2d_58/Conv2D/ReadVariableOpconv2d_58/Conv2D/ReadVariableOp2D
 conv2d_59/BiasAdd/ReadVariableOp conv2d_59/BiasAdd/ReadVariableOp2B
conv2d_59/Conv2D/ReadVariableOpconv2d_59/Conv2D/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
Б
h
L__inference_max_pooling2d_19_layer_call_and_return_conditional_losses_375663

inputs
identityм
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingSAME*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
▀
}
(__inference_dense_9_layer_call_fn_376435

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_3758622
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
щ	
╣
$__inference_signature_wrapper_376115
conv2d_54_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identityИвStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallconv2d_54_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В **
f%R#
!__inference__wrapped_model_3756452
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:          ::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:          
)
_user_specified_nameconv2d_54_input
°	
▄
C__inference_dense_9_layer_call_and_return_conditional_losses_376426

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         2	
SoftmaxЦ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╨0
Ъ
H__inference_sequential_9_layer_call_and_return_conditional_losses_375921
conv2d_54_input
conv2d_54_375882
conv2d_54_375884
conv2d_55_375887
conv2d_55_375889
conv2d_56_375893
conv2d_56_375895
conv2d_57_375898
conv2d_57_375900
conv2d_58_375904
conv2d_58_375906
conv2d_59_375909
conv2d_59_375911
dense_9_375915
dense_9_375917
identityИв!conv2d_54/StatefulPartitionedCallв!conv2d_55/StatefulPartitionedCallв!conv2d_56/StatefulPartitionedCallв!conv2d_57/StatefulPartitionedCallв!conv2d_58/StatefulPartitionedCallв!conv2d_59/StatefulPartitionedCallвdense_9/StatefulPartitionedCallн
!conv2d_54/StatefulPartitionedCallStatefulPartitionedCallconv2d_54_inputconv2d_54_375882conv2d_54_375884*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_54_layer_call_and_return_conditional_losses_3756972#
!conv2d_54/StatefulPartitionedCall╚
!conv2d_55/StatefulPartitionedCallStatefulPartitionedCall*conv2d_54/StatefulPartitionedCall:output:0conv2d_55_375887conv2d_55_375889*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_55_layer_call_and_return_conditional_losses_3757242#
!conv2d_55/StatefulPartitionedCallЫ
 max_pooling2d_18/PartitionedCallPartitionedCall*conv2d_55/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_max_pooling2d_18_layer_call_and_return_conditional_losses_3756512"
 max_pooling2d_18/PartitionedCall╚
!conv2d_56/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_18/PartitionedCall:output:0conv2d_56_375893conv2d_56_375895*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:          А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_56_layer_call_and_return_conditional_losses_3757522#
!conv2d_56/StatefulPartitionedCall╔
!conv2d_57/StatefulPartitionedCallStatefulPartitionedCall*conv2d_56/StatefulPartitionedCall:output:0conv2d_57_375898conv2d_57_375900*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:          А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_57_layer_call_and_return_conditional_losses_3757792#
!conv2d_57/StatefulPartitionedCallЬ
 max_pooling2d_19/PartitionedCallPartitionedCall*conv2d_57/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_max_pooling2d_19_layer_call_and_return_conditional_losses_3756632"
 max_pooling2d_19/PartitionedCall╚
!conv2d_58/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_19/PartitionedCall:output:0conv2d_58_375904conv2d_58_375906*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_58_layer_call_and_return_conditional_losses_3758072#
!conv2d_58/StatefulPartitionedCall╔
!conv2d_59/StatefulPartitionedCallStatefulPartitionedCall*conv2d_58/StatefulPartitionedCall:output:0conv2d_59_375909conv2d_59_375911*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_59_layer_call_and_return_conditional_losses_3758342#
!conv2d_59/StatefulPartitionedCall▓
*global_average_pooling2d_9/PartitionedCallPartitionedCall*conv2d_59/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *_
fZRX
V__inference_global_average_pooling2d_9_layer_call_and_return_conditional_losses_3756762,
*global_average_pooling2d_9/PartitionedCall┐
dense_9/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling2d_9/PartitionedCall:output:0dense_9_375915dense_9_375917*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_3758622!
dense_9/StatefulPartitionedCallЎ
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0"^conv2d_54/StatefulPartitionedCall"^conv2d_55/StatefulPartitionedCall"^conv2d_56/StatefulPartitionedCall"^conv2d_57/StatefulPartitionedCall"^conv2d_58/StatefulPartitionedCall"^conv2d_59/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:          ::::::::::::::2F
!conv2d_54/StatefulPartitionedCall!conv2d_54/StatefulPartitionedCall2F
!conv2d_55/StatefulPartitionedCall!conv2d_55/StatefulPartitionedCall2F
!conv2d_56/StatefulPartitionedCall!conv2d_56/StatefulPartitionedCall2F
!conv2d_57/StatefulPartitionedCall!conv2d_57/StatefulPartitionedCall2F
!conv2d_58/StatefulPartitionedCall!conv2d_58/StatefulPartitionedCall2F
!conv2d_59/StatefulPartitionedCall!conv2d_59/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:` \
/
_output_shapes
:          
)
_user_specified_nameconv2d_54_input
╧

▐
E__inference_conv2d_54_layer_call_and_return_conditional_losses_375697

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpг
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          @*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          @2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:          @2
ReluЯ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:          @2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:          ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
Б
h
L__inference_max_pooling2d_18_layer_call_and_return_conditional_losses_375651

inputs
identityм
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingSAME*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Щ

┬
-__inference_sequential_9_layer_call_fn_375997
conv2d_54_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identityИвStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCallconv2d_54_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_sequential_9_layer_call_and_return_conditional_losses_3759662
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:          ::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:          
)
_user_specified_nameconv2d_54_input
Г

*__inference_conv2d_56_layer_call_fn_376355

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:          А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_56_layer_call_and_return_conditional_losses_3757522
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:          А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:          @::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          @
 
_user_specified_nameinputs
Б

*__inference_conv2d_54_layer_call_fn_376315

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_54_layer_call_and_return_conditional_losses_3756972
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:          @2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:          ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
╧

▐
E__inference_conv2d_54_layer_call_and_return_conditional_losses_376306

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpг
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          @*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          @2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:          @2
ReluЯ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:          @2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:          ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
╪

▐
E__inference_conv2d_59_layer_call_and_return_conditional_losses_375834

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         А2
Reluа
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
╪

▐
E__inference_conv2d_57_layer_call_and_return_conditional_losses_375779

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:          А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:          А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:          А2
Reluа
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:          А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:          А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:          А
 
_user_specified_nameinputs
С
W
;__inference_global_average_pooling2d_9_layer_call_fn_375682

inputs
identityр
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:                  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *_
fZRX
V__inference_global_average_pooling2d_9_layer_call_and_return_conditional_losses_3756762
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:                  2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
°	
▄
C__inference_dense_9_layer_call_and_return_conditional_losses_375862

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         2	
SoftmaxЦ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
■	
╣
-__inference_sequential_9_layer_call_fn_376295

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identityИвStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_sequential_9_layer_call_and_return_conditional_losses_3760412
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:          ::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
Е

*__inference_conv2d_59_layer_call_fn_376415

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_59_layer_call_and_return_conditional_losses_3758342
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         А::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
╒

▐
E__inference_conv2d_56_layer_call_and_return_conditional_losses_376346

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpЦ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:          А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:          А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:          А2
Reluа
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:          А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:          @::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          @
 
_user_specified_nameinputs
Е

*__inference_conv2d_58_layer_call_fn_376395

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_58_layer_call_and_return_conditional_losses_3758072
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         А::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
фi
А
__inference__traced_save_376611
file_prefix/
+savev2_conv2d_54_kernel_read_readvariableop-
)savev2_conv2d_54_bias_read_readvariableop/
+savev2_conv2d_55_kernel_read_readvariableop-
)savev2_conv2d_55_bias_read_readvariableop/
+savev2_conv2d_56_kernel_read_readvariableop-
)savev2_conv2d_56_bias_read_readvariableop/
+savev2_conv2d_57_kernel_read_readvariableop-
)savev2_conv2d_57_bias_read_readvariableop/
+savev2_conv2d_58_kernel_read_readvariableop-
)savev2_conv2d_58_bias_read_readvariableop/
+savev2_conv2d_59_kernel_read_readvariableop-
)savev2_conv2d_59_bias_read_readvariableop-
)savev2_dense_9_kernel_read_readvariableop+
'savev2_dense_9_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_conv2d_54_kernel_m_read_readvariableop4
0savev2_adam_conv2d_54_bias_m_read_readvariableop6
2savev2_adam_conv2d_55_kernel_m_read_readvariableop4
0savev2_adam_conv2d_55_bias_m_read_readvariableop6
2savev2_adam_conv2d_56_kernel_m_read_readvariableop4
0savev2_adam_conv2d_56_bias_m_read_readvariableop6
2savev2_adam_conv2d_57_kernel_m_read_readvariableop4
0savev2_adam_conv2d_57_bias_m_read_readvariableop6
2savev2_adam_conv2d_58_kernel_m_read_readvariableop4
0savev2_adam_conv2d_58_bias_m_read_readvariableop6
2savev2_adam_conv2d_59_kernel_m_read_readvariableop4
0savev2_adam_conv2d_59_bias_m_read_readvariableop4
0savev2_adam_dense_9_kernel_m_read_readvariableop2
.savev2_adam_dense_9_bias_m_read_readvariableop6
2savev2_adam_conv2d_54_kernel_v_read_readvariableop4
0savev2_adam_conv2d_54_bias_v_read_readvariableop6
2savev2_adam_conv2d_55_kernel_v_read_readvariableop4
0savev2_adam_conv2d_55_bias_v_read_readvariableop6
2savev2_adam_conv2d_56_kernel_v_read_readvariableop4
0savev2_adam_conv2d_56_bias_v_read_readvariableop6
2savev2_adam_conv2d_57_kernel_v_read_readvariableop4
0savev2_adam_conv2d_57_bias_v_read_readvariableop6
2savev2_adam_conv2d_58_kernel_v_read_readvariableop4
0savev2_adam_conv2d_58_bias_v_read_readvariableop6
2savev2_adam_conv2d_59_kernel_v_read_readvariableop4
0savev2_adam_conv2d_59_bias_v_read_readvariableop4
0savev2_adam_dense_9_kernel_v_read_readvariableop2
.savev2_adam_dense_9_bias_v_read_readvariableop
savev2_const

identity_1ИвMergeV2CheckpointsП
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1Л
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameЇ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*Ж
value№B∙4B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesЁ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices╡
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_54_kernel_read_readvariableop)savev2_conv2d_54_bias_read_readvariableop+savev2_conv2d_55_kernel_read_readvariableop)savev2_conv2d_55_bias_read_readvariableop+savev2_conv2d_56_kernel_read_readvariableop)savev2_conv2d_56_bias_read_readvariableop+savev2_conv2d_57_kernel_read_readvariableop)savev2_conv2d_57_bias_read_readvariableop+savev2_conv2d_58_kernel_read_readvariableop)savev2_conv2d_58_bias_read_readvariableop+savev2_conv2d_59_kernel_read_readvariableop)savev2_conv2d_59_bias_read_readvariableop)savev2_dense_9_kernel_read_readvariableop'savev2_dense_9_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_conv2d_54_kernel_m_read_readvariableop0savev2_adam_conv2d_54_bias_m_read_readvariableop2savev2_adam_conv2d_55_kernel_m_read_readvariableop0savev2_adam_conv2d_55_bias_m_read_readvariableop2savev2_adam_conv2d_56_kernel_m_read_readvariableop0savev2_adam_conv2d_56_bias_m_read_readvariableop2savev2_adam_conv2d_57_kernel_m_read_readvariableop0savev2_adam_conv2d_57_bias_m_read_readvariableop2savev2_adam_conv2d_58_kernel_m_read_readvariableop0savev2_adam_conv2d_58_bias_m_read_readvariableop2savev2_adam_conv2d_59_kernel_m_read_readvariableop0savev2_adam_conv2d_59_bias_m_read_readvariableop0savev2_adam_dense_9_kernel_m_read_readvariableop.savev2_adam_dense_9_bias_m_read_readvariableop2savev2_adam_conv2d_54_kernel_v_read_readvariableop0savev2_adam_conv2d_54_bias_v_read_readvariableop2savev2_adam_conv2d_55_kernel_v_read_readvariableop0savev2_adam_conv2d_55_bias_v_read_readvariableop2savev2_adam_conv2d_56_kernel_v_read_readvariableop0savev2_adam_conv2d_56_bias_v_read_readvariableop2savev2_adam_conv2d_57_kernel_v_read_readvariableop0savev2_adam_conv2d_57_bias_v_read_readvariableop2savev2_adam_conv2d_58_kernel_v_read_readvariableop0savev2_adam_conv2d_58_bias_v_read_readvariableop2savev2_adam_conv2d_59_kernel_v_read_readvariableop0savev2_adam_conv2d_59_bias_v_read_readvariableop0savev2_adam_dense_9_kernel_v_read_readvariableop.savev2_adam_dense_9_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *B
dtypes8
624	2
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesб
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*п
_input_shapesЭ
Ъ: :@:@:@@:@:@А:А:АА:А:АА:А:АА:А:	А:: : : : : : : : : :@:@:@@:@:@А:А:АА:А:АА:А:АА:А:	А::@:@:@@:@:@А:А:АА:А:АА:А:АА:А:	А:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:-)
'
_output_shapes
:@А:!

_output_shapes	
:А:.*
(
_output_shapes
:АА:!

_output_shapes	
:А:.	*
(
_output_shapes
:АА:!


_output_shapes	
:А:.*
(
_output_shapes
:АА:!

_output_shapes	
:А:%!

_output_shapes
:	А: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:-)
'
_output_shapes
:@А:!

_output_shapes	
:А:.*
(
_output_shapes
:АА:!

_output_shapes	
:А:. *
(
_output_shapes
:АА:!!

_output_shapes	
:А:."*
(
_output_shapes
:АА:!#

_output_shapes	
:А:%$!

_output_shapes
:	А: %

_output_shapes
::,&(
&
_output_shapes
:@: '

_output_shapes
:@:,((
&
_output_shapes
:@@: )

_output_shapes
:@:-*)
'
_output_shapes
:@А:!+

_output_shapes	
:А:.,*
(
_output_shapes
:АА:!-

_output_shapes	
:А:..*
(
_output_shapes
:АА:!/

_output_shapes	
:А:.0*
(
_output_shapes
:АА:!1

_output_shapes	
:А:%2!

_output_shapes
:	А: 3

_output_shapes
::4

_output_shapes
: 
╠╓
┬
"__inference__traced_restore_376774
file_prefix%
!assignvariableop_conv2d_54_kernel%
!assignvariableop_1_conv2d_54_bias'
#assignvariableop_2_conv2d_55_kernel%
!assignvariableop_3_conv2d_55_bias'
#assignvariableop_4_conv2d_56_kernel%
!assignvariableop_5_conv2d_56_bias'
#assignvariableop_6_conv2d_57_kernel%
!assignvariableop_7_conv2d_57_bias'
#assignvariableop_8_conv2d_58_kernel%
!assignvariableop_9_conv2d_58_bias(
$assignvariableop_10_conv2d_59_kernel&
"assignvariableop_11_conv2d_59_bias&
"assignvariableop_12_dense_9_kernel$
 assignvariableop_13_dense_9_bias!
assignvariableop_14_adam_iter#
assignvariableop_15_adam_beta_1#
assignvariableop_16_adam_beta_2"
assignvariableop_17_adam_decay*
&assignvariableop_18_adam_learning_rate
assignvariableop_19_total
assignvariableop_20_count
assignvariableop_21_total_1
assignvariableop_22_count_1/
+assignvariableop_23_adam_conv2d_54_kernel_m-
)assignvariableop_24_adam_conv2d_54_bias_m/
+assignvariableop_25_adam_conv2d_55_kernel_m-
)assignvariableop_26_adam_conv2d_55_bias_m/
+assignvariableop_27_adam_conv2d_56_kernel_m-
)assignvariableop_28_adam_conv2d_56_bias_m/
+assignvariableop_29_adam_conv2d_57_kernel_m-
)assignvariableop_30_adam_conv2d_57_bias_m/
+assignvariableop_31_adam_conv2d_58_kernel_m-
)assignvariableop_32_adam_conv2d_58_bias_m/
+assignvariableop_33_adam_conv2d_59_kernel_m-
)assignvariableop_34_adam_conv2d_59_bias_m-
)assignvariableop_35_adam_dense_9_kernel_m+
'assignvariableop_36_adam_dense_9_bias_m/
+assignvariableop_37_adam_conv2d_54_kernel_v-
)assignvariableop_38_adam_conv2d_54_bias_v/
+assignvariableop_39_adam_conv2d_55_kernel_v-
)assignvariableop_40_adam_conv2d_55_bias_v/
+assignvariableop_41_adam_conv2d_56_kernel_v-
)assignvariableop_42_adam_conv2d_56_bias_v/
+assignvariableop_43_adam_conv2d_57_kernel_v-
)assignvariableop_44_adam_conv2d_57_bias_v/
+assignvariableop_45_adam_conv2d_58_kernel_v-
)assignvariableop_46_adam_conv2d_58_bias_v/
+assignvariableop_47_adam_conv2d_59_kernel_v-
)assignvariableop_48_adam_conv2d_59_bias_v-
)assignvariableop_49_adam_dense_9_kernel_v+
'assignvariableop_50_adam_dense_9_bias_v
identity_52ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_37вAssignVariableOp_38вAssignVariableOp_39вAssignVariableOp_4вAssignVariableOp_40вAssignVariableOp_41вAssignVariableOp_42вAssignVariableOp_43вAssignVariableOp_44вAssignVariableOp_45вAssignVariableOp_46вAssignVariableOp_47вAssignVariableOp_48вAssignVariableOp_49вAssignVariableOp_5вAssignVariableOp_50вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9·
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*Ж
value№B∙4B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesЎ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices▓
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ц
_output_shapes╙
╨::::::::::::::::::::::::::::::::::::::::::::::::::::*B
dtypes8
624	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identityа
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_54_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1ж
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_54_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2и
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_55_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3ж
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_55_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4и
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_56_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5ж
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_56_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6и
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_57_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7ж
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_57_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8и
AssignVariableOp_8AssignVariableOp#assignvariableop_8_conv2d_58_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9ж
AssignVariableOp_9AssignVariableOp!assignvariableop_9_conv2d_58_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10м
AssignVariableOp_10AssignVariableOp$assignvariableop_10_conv2d_59_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11к
AssignVariableOp_11AssignVariableOp"assignvariableop_11_conv2d_59_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12к
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_9_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13и
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_9_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_14е
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_iterIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15з
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_beta_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16з
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_beta_2Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17ж
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_decayIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18о
AssignVariableOp_18AssignVariableOp&assignvariableop_18_adam_learning_rateIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19б
AssignVariableOp_19AssignVariableOpassignvariableop_19_totalIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20б
AssignVariableOp_20AssignVariableOpassignvariableop_20_countIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21г
AssignVariableOp_21AssignVariableOpassignvariableop_21_total_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22г
AssignVariableOp_22AssignVariableOpassignvariableop_22_count_1Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23│
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_conv2d_54_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24▒
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_conv2d_54_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25│
AssignVariableOp_25AssignVariableOp+assignvariableop_25_adam_conv2d_55_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26▒
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_conv2d_55_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27│
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_conv2d_56_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28▒
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_conv2d_56_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29│
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_conv2d_57_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30▒
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_conv2d_57_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31│
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_conv2d_58_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32▒
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_conv2d_58_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33│
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_conv2d_59_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34▒
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_conv2d_59_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35▒
AssignVariableOp_35AssignVariableOp)assignvariableop_35_adam_dense_9_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36п
AssignVariableOp_36AssignVariableOp'assignvariableop_36_adam_dense_9_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37│
AssignVariableOp_37AssignVariableOp+assignvariableop_37_adam_conv2d_54_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38▒
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_conv2d_54_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39│
AssignVariableOp_39AssignVariableOp+assignvariableop_39_adam_conv2d_55_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40▒
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_conv2d_55_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41│
AssignVariableOp_41AssignVariableOp+assignvariableop_41_adam_conv2d_56_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42▒
AssignVariableOp_42AssignVariableOp)assignvariableop_42_adam_conv2d_56_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43│
AssignVariableOp_43AssignVariableOp+assignvariableop_43_adam_conv2d_57_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44▒
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_conv2d_57_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45│
AssignVariableOp_45AssignVariableOp+assignvariableop_45_adam_conv2d_58_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46▒
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adam_conv2d_58_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47│
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_conv2d_59_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48▒
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_conv2d_59_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49▒
AssignVariableOp_49AssignVariableOp)assignvariableop_49_adam_dense_9_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50п
AssignVariableOp_50AssignVariableOp'assignvariableop_50_adam_dense_9_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_509
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp└	
Identity_51Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_51│	
Identity_52IdentityIdentity_51:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_52"#
identity_52Identity_52:output:0*у
_input_shapes╤
╬: :::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
╡0
С
H__inference_sequential_9_layer_call_and_return_conditional_losses_375966

inputs
conv2d_54_375927
conv2d_54_375929
conv2d_55_375932
conv2d_55_375934
conv2d_56_375938
conv2d_56_375940
conv2d_57_375943
conv2d_57_375945
conv2d_58_375949
conv2d_58_375951
conv2d_59_375954
conv2d_59_375956
dense_9_375960
dense_9_375962
identityИв!conv2d_54/StatefulPartitionedCallв!conv2d_55/StatefulPartitionedCallв!conv2d_56/StatefulPartitionedCallв!conv2d_57/StatefulPartitionedCallв!conv2d_58/StatefulPartitionedCallв!conv2d_59/StatefulPartitionedCallвdense_9/StatefulPartitionedCallд
!conv2d_54/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_54_375927conv2d_54_375929*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_54_layer_call_and_return_conditional_losses_3756972#
!conv2d_54/StatefulPartitionedCall╚
!conv2d_55/StatefulPartitionedCallStatefulPartitionedCall*conv2d_54/StatefulPartitionedCall:output:0conv2d_55_375932conv2d_55_375934*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_55_layer_call_and_return_conditional_losses_3757242#
!conv2d_55/StatefulPartitionedCallЫ
 max_pooling2d_18/PartitionedCallPartitionedCall*conv2d_55/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_max_pooling2d_18_layer_call_and_return_conditional_losses_3756512"
 max_pooling2d_18/PartitionedCall╚
!conv2d_56/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_18/PartitionedCall:output:0conv2d_56_375938conv2d_56_375940*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:          А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_56_layer_call_and_return_conditional_losses_3757522#
!conv2d_56/StatefulPartitionedCall╔
!conv2d_57/StatefulPartitionedCallStatefulPartitionedCall*conv2d_56/StatefulPartitionedCall:output:0conv2d_57_375943conv2d_57_375945*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:          А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_57_layer_call_and_return_conditional_losses_3757792#
!conv2d_57/StatefulPartitionedCallЬ
 max_pooling2d_19/PartitionedCallPartitionedCall*conv2d_57/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_max_pooling2d_19_layer_call_and_return_conditional_losses_3756632"
 max_pooling2d_19/PartitionedCall╚
!conv2d_58/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_19/PartitionedCall:output:0conv2d_58_375949conv2d_58_375951*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_58_layer_call_and_return_conditional_losses_3758072#
!conv2d_58/StatefulPartitionedCall╔
!conv2d_59/StatefulPartitionedCallStatefulPartitionedCall*conv2d_58/StatefulPartitionedCall:output:0conv2d_59_375954conv2d_59_375956*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_59_layer_call_and_return_conditional_losses_3758342#
!conv2d_59/StatefulPartitionedCall▓
*global_average_pooling2d_9/PartitionedCallPartitionedCall*conv2d_59/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *_
fZRX
V__inference_global_average_pooling2d_9_layer_call_and_return_conditional_losses_3756762,
*global_average_pooling2d_9/PartitionedCall┐
dense_9/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling2d_9/PartitionedCall:output:0dense_9_375960dense_9_375962*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_3758622!
dense_9/StatefulPartitionedCallЎ
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0"^conv2d_54/StatefulPartitionedCall"^conv2d_55/StatefulPartitionedCall"^conv2d_56/StatefulPartitionedCall"^conv2d_57/StatefulPartitionedCall"^conv2d_58/StatefulPartitionedCall"^conv2d_59/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:          ::::::::::::::2F
!conv2d_54/StatefulPartitionedCall!conv2d_54/StatefulPartitionedCall2F
!conv2d_55/StatefulPartitionedCall!conv2d_55/StatefulPartitionedCall2F
!conv2d_56/StatefulPartitionedCall!conv2d_56/StatefulPartitionedCall2F
!conv2d_57/StatefulPartitionedCall!conv2d_57/StatefulPartitionedCall2F
!conv2d_58/StatefulPartitionedCall!conv2d_58/StatefulPartitionedCall2F
!conv2d_59/StatefulPartitionedCall!conv2d_59/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
фa
Ы
!__inference__wrapped_model_375645
conv2d_54_input9
5sequential_9_conv2d_54_conv2d_readvariableop_resource:
6sequential_9_conv2d_54_biasadd_readvariableop_resource9
5sequential_9_conv2d_55_conv2d_readvariableop_resource:
6sequential_9_conv2d_55_biasadd_readvariableop_resource9
5sequential_9_conv2d_56_conv2d_readvariableop_resource:
6sequential_9_conv2d_56_biasadd_readvariableop_resource9
5sequential_9_conv2d_57_conv2d_readvariableop_resource:
6sequential_9_conv2d_57_biasadd_readvariableop_resource9
5sequential_9_conv2d_58_conv2d_readvariableop_resource:
6sequential_9_conv2d_58_biasadd_readvariableop_resource9
5sequential_9_conv2d_59_conv2d_readvariableop_resource:
6sequential_9_conv2d_59_biasadd_readvariableop_resource7
3sequential_9_dense_9_matmul_readvariableop_resource8
4sequential_9_dense_9_biasadd_readvariableop_resource
identityИв-sequential_9/conv2d_54/BiasAdd/ReadVariableOpв,sequential_9/conv2d_54/Conv2D/ReadVariableOpв-sequential_9/conv2d_55/BiasAdd/ReadVariableOpв,sequential_9/conv2d_55/Conv2D/ReadVariableOpв-sequential_9/conv2d_56/BiasAdd/ReadVariableOpв,sequential_9/conv2d_56/Conv2D/ReadVariableOpв-sequential_9/conv2d_57/BiasAdd/ReadVariableOpв,sequential_9/conv2d_57/Conv2D/ReadVariableOpв-sequential_9/conv2d_58/BiasAdd/ReadVariableOpв,sequential_9/conv2d_58/Conv2D/ReadVariableOpв-sequential_9/conv2d_59/BiasAdd/ReadVariableOpв,sequential_9/conv2d_59/Conv2D/ReadVariableOpв+sequential_9/dense_9/BiasAdd/ReadVariableOpв*sequential_9/dense_9/MatMul/ReadVariableOp┌
,sequential_9/conv2d_54/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_54_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02.
,sequential_9/conv2d_54/Conv2D/ReadVariableOpё
sequential_9/conv2d_54/Conv2DConv2Dconv2d_54_input4sequential_9/conv2d_54/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          @*
paddingSAME*
strides
2
sequential_9/conv2d_54/Conv2D╤
-sequential_9/conv2d_54/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_54_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-sequential_9/conv2d_54/BiasAdd/ReadVariableOpф
sequential_9/conv2d_54/BiasAddBiasAdd&sequential_9/conv2d_54/Conv2D:output:05sequential_9/conv2d_54/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          @2 
sequential_9/conv2d_54/BiasAddе
sequential_9/conv2d_54/ReluRelu'sequential_9/conv2d_54/BiasAdd:output:0*
T0*/
_output_shapes
:          @2
sequential_9/conv2d_54/Relu┌
,sequential_9/conv2d_55/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_55_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,sequential_9/conv2d_55/Conv2D/ReadVariableOpЛ
sequential_9/conv2d_55/Conv2DConv2D)sequential_9/conv2d_54/Relu:activations:04sequential_9/conv2d_55/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          @*
paddingSAME*
strides
2
sequential_9/conv2d_55/Conv2D╤
-sequential_9/conv2d_55/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_55_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-sequential_9/conv2d_55/BiasAdd/ReadVariableOpф
sequential_9/conv2d_55/BiasAddBiasAdd&sequential_9/conv2d_55/Conv2D:output:05sequential_9/conv2d_55/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          @2 
sequential_9/conv2d_55/BiasAddе
sequential_9/conv2d_55/ReluRelu'sequential_9/conv2d_55/BiasAdd:output:0*
T0*/
_output_shapes
:          @2
sequential_9/conv2d_55/ReluЁ
%sequential_9/max_pooling2d_18/MaxPoolMaxPool)sequential_9/conv2d_55/Relu:activations:0*/
_output_shapes
:          @*
ksize
*
paddingSAME*
strides
2'
%sequential_9/max_pooling2d_18/MaxPool█
,sequential_9/conv2d_56/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_56_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02.
,sequential_9/conv2d_56/Conv2D/ReadVariableOpС
sequential_9/conv2d_56/Conv2DConv2D.sequential_9/max_pooling2d_18/MaxPool:output:04sequential_9/conv2d_56/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:          А*
paddingSAME*
strides
2
sequential_9/conv2d_56/Conv2D╥
-sequential_9/conv2d_56/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_56_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_9/conv2d_56/BiasAdd/ReadVariableOpх
sequential_9/conv2d_56/BiasAddBiasAdd&sequential_9/conv2d_56/Conv2D:output:05sequential_9/conv2d_56/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:          А2 
sequential_9/conv2d_56/BiasAddж
sequential_9/conv2d_56/ReluRelu'sequential_9/conv2d_56/BiasAdd:output:0*
T0*0
_output_shapes
:          А2
sequential_9/conv2d_56/Relu▄
,sequential_9/conv2d_57/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_57_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_9/conv2d_57/Conv2D/ReadVariableOpМ
sequential_9/conv2d_57/Conv2DConv2D)sequential_9/conv2d_56/Relu:activations:04sequential_9/conv2d_57/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:          А*
paddingSAME*
strides
2
sequential_9/conv2d_57/Conv2D╥
-sequential_9/conv2d_57/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_57_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_9/conv2d_57/BiasAdd/ReadVariableOpх
sequential_9/conv2d_57/BiasAddBiasAdd&sequential_9/conv2d_57/Conv2D:output:05sequential_9/conv2d_57/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:          А2 
sequential_9/conv2d_57/BiasAddж
sequential_9/conv2d_57/ReluRelu'sequential_9/conv2d_57/BiasAdd:output:0*
T0*0
_output_shapes
:          А2
sequential_9/conv2d_57/Reluё
%sequential_9/max_pooling2d_19/MaxPoolMaxPool)sequential_9/conv2d_57/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingSAME*
strides
2'
%sequential_9/max_pooling2d_19/MaxPool▄
,sequential_9/conv2d_58/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_58_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_9/conv2d_58/Conv2D/ReadVariableOpС
sequential_9/conv2d_58/Conv2DConv2D.sequential_9/max_pooling2d_19/MaxPool:output:04sequential_9/conv2d_58/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
sequential_9/conv2d_58/Conv2D╥
-sequential_9/conv2d_58/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_58_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_9/conv2d_58/BiasAdd/ReadVariableOpх
sequential_9/conv2d_58/BiasAddBiasAdd&sequential_9/conv2d_58/Conv2D:output:05sequential_9/conv2d_58/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2 
sequential_9/conv2d_58/BiasAddж
sequential_9/conv2d_58/ReluRelu'sequential_9/conv2d_58/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
sequential_9/conv2d_58/Relu▄
,sequential_9/conv2d_59/Conv2D/ReadVariableOpReadVariableOp5sequential_9_conv2d_59_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_9/conv2d_59/Conv2D/ReadVariableOpМ
sequential_9/conv2d_59/Conv2DConv2D)sequential_9/conv2d_58/Relu:activations:04sequential_9/conv2d_59/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
sequential_9/conv2d_59/Conv2D╥
-sequential_9/conv2d_59/BiasAdd/ReadVariableOpReadVariableOp6sequential_9_conv2d_59_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_9/conv2d_59/BiasAdd/ReadVariableOpх
sequential_9/conv2d_59/BiasAddBiasAdd&sequential_9/conv2d_59/Conv2D:output:05sequential_9/conv2d_59/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2 
sequential_9/conv2d_59/BiasAddж
sequential_9/conv2d_59/ReluRelu'sequential_9/conv2d_59/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
sequential_9/conv2d_59/Relu╤
>sequential_9/global_average_pooling2d_9/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2@
>sequential_9/global_average_pooling2d_9/Mean/reduction_indicesЛ
,sequential_9/global_average_pooling2d_9/MeanMean)sequential_9/conv2d_59/Relu:activations:0Gsequential_9/global_average_pooling2d_9/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:         А2.
,sequential_9/global_average_pooling2d_9/Mean═
*sequential_9/dense_9/MatMul/ReadVariableOpReadVariableOp3sequential_9_dense_9_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02,
*sequential_9/dense_9/MatMul/ReadVariableOpс
sequential_9/dense_9/MatMulMatMul5sequential_9/global_average_pooling2d_9/Mean:output:02sequential_9/dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
sequential_9/dense_9/MatMul╦
+sequential_9/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_9_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_9/dense_9/BiasAdd/ReadVariableOp╒
sequential_9/dense_9/BiasAddBiasAdd%sequential_9/dense_9/MatMul:product:03sequential_9/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
sequential_9/dense_9/BiasAddа
sequential_9/dense_9/SoftmaxSoftmax%sequential_9/dense_9/BiasAdd:output:0*
T0*'
_output_shapes
:         2
sequential_9/dense_9/SoftmaxП
IdentityIdentity&sequential_9/dense_9/Softmax:softmax:0.^sequential_9/conv2d_54/BiasAdd/ReadVariableOp-^sequential_9/conv2d_54/Conv2D/ReadVariableOp.^sequential_9/conv2d_55/BiasAdd/ReadVariableOp-^sequential_9/conv2d_55/Conv2D/ReadVariableOp.^sequential_9/conv2d_56/BiasAdd/ReadVariableOp-^sequential_9/conv2d_56/Conv2D/ReadVariableOp.^sequential_9/conv2d_57/BiasAdd/ReadVariableOp-^sequential_9/conv2d_57/Conv2D/ReadVariableOp.^sequential_9/conv2d_58/BiasAdd/ReadVariableOp-^sequential_9/conv2d_58/Conv2D/ReadVariableOp.^sequential_9/conv2d_59/BiasAdd/ReadVariableOp-^sequential_9/conv2d_59/Conv2D/ReadVariableOp,^sequential_9/dense_9/BiasAdd/ReadVariableOp+^sequential_9/dense_9/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:          ::::::::::::::2^
-sequential_9/conv2d_54/BiasAdd/ReadVariableOp-sequential_9/conv2d_54/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_54/Conv2D/ReadVariableOp,sequential_9/conv2d_54/Conv2D/ReadVariableOp2^
-sequential_9/conv2d_55/BiasAdd/ReadVariableOp-sequential_9/conv2d_55/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_55/Conv2D/ReadVariableOp,sequential_9/conv2d_55/Conv2D/ReadVariableOp2^
-sequential_9/conv2d_56/BiasAdd/ReadVariableOp-sequential_9/conv2d_56/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_56/Conv2D/ReadVariableOp,sequential_9/conv2d_56/Conv2D/ReadVariableOp2^
-sequential_9/conv2d_57/BiasAdd/ReadVariableOp-sequential_9/conv2d_57/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_57/Conv2D/ReadVariableOp,sequential_9/conv2d_57/Conv2D/ReadVariableOp2^
-sequential_9/conv2d_58/BiasAdd/ReadVariableOp-sequential_9/conv2d_58/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_58/Conv2D/ReadVariableOp,sequential_9/conv2d_58/Conv2D/ReadVariableOp2^
-sequential_9/conv2d_59/BiasAdd/ReadVariableOp-sequential_9/conv2d_59/BiasAdd/ReadVariableOp2\
,sequential_9/conv2d_59/Conv2D/ReadVariableOp,sequential_9/conv2d_59/Conv2D/ReadVariableOp2Z
+sequential_9/dense_9/BiasAdd/ReadVariableOp+sequential_9/dense_9/BiasAdd/ReadVariableOp2X
*sequential_9/dense_9/MatMul/ReadVariableOp*sequential_9/dense_9/MatMul/ReadVariableOp:` \
/
_output_shapes
:          
)
_user_specified_nameconv2d_54_input
╒

▐
E__inference_conv2d_56_layer_call_and_return_conditional_losses_375752

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpЦ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:          А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:          А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:          А2
Reluа
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:          А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:          @::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          @
 
_user_specified_nameinputs
╧

▐
E__inference_conv2d_55_layer_call_and_return_conditional_losses_376326

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpг
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          @*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          @2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:          @2
ReluЯ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:          @2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:          @::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          @
 
_user_specified_nameinputs
╪

▐
E__inference_conv2d_58_layer_call_and_return_conditional_losses_376386

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         А2
Reluа
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
Е

*__inference_conv2d_57_layer_call_fn_376375

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:          А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_57_layer_call_and_return_conditional_losses_3757792
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:          А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:          А::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:          А
 
_user_specified_nameinputs
╝
r
V__inference_global_average_pooling2d_9_layer_call_and_return_conditional_losses_375676

inputs
identityБ
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:                  2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:                  2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs"▒L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*┬
serving_defaultо
S
conv2d_54_input@
!serving_default_conv2d_54_input:0          ;
dense_90
StatefulPartitionedCall:0         tensorflow/serving/predict:┬с
╣b
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer-8

layer_with_weights-6

layer-9
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
+к&call_and_return_all_conditional_losses
л__call__
м_default_save_signature"│^
_tf_keras_sequentialФ^{"class_name": "Sequential", "name": "sequential_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 15, 32, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_54_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_54", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 15, 32, 3]}, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_55", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_18", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_56", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_57", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_19", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_58", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_59", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "GlobalAveragePooling2D", "config": {"name": "global_average_pooling2d_9", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 8, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 15, 32, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 15, 32, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_54_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_54", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 15, 32, 3]}, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_55", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_18", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_56", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_57", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_19", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_58", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_59", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "GlobalAveragePooling2D", "config": {"name": "global_average_pooling2d_9", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 8, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "sparse_categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ї


kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
+н&call_and_return_all_conditional_losses
о__call__"╬	
_tf_keras_layer┤	{"class_name": "Conv2D", "name": "conv2d_54", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 15, 32, 3]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_54", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 15, 32, 3]}, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 15, 32, 3]}}
Ў	

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
+п&call_and_return_all_conditional_losses
░__call__"╧
_tf_keras_layer╡{"class_name": "Conv2D", "name": "conv2d_55", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_55", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 15, 32, 64]}}
В
regularization_losses
	variables
trainable_variables
 	keras_api
+▒&call_and_return_all_conditional_losses
▓__call__"ё
_tf_keras_layer╫{"class_name": "MaxPooling2D", "name": "max_pooling2d_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_18", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ў	

!kernel
"bias
#regularization_losses
$	variables
%trainable_variables
&	keras_api
+│&call_and_return_all_conditional_losses
┤__call__"╨
_tf_keras_layer╢{"class_name": "Conv2D", "name": "conv2d_56", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_56", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 15, 32, 64]}}
∙	

'kernel
(bias
)regularization_losses
*	variables
+trainable_variables
,	keras_api
+╡&call_and_return_all_conditional_losses
╢__call__"╥
_tf_keras_layer╕{"class_name": "Conv2D", "name": "conv2d_57", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_57", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 15, 32, 128]}}
В
-regularization_losses
.	variables
/trainable_variables
0	keras_api
+╖&call_and_return_all_conditional_losses
╕__call__"ё
_tf_keras_layer╫{"class_name": "MaxPooling2D", "name": "max_pooling2d_19", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_19", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
°	

1kernel
2bias
3regularization_losses
4	variables
5trainable_variables
6	keras_api
+╣&call_and_return_all_conditional_losses
║__call__"╤
_tf_keras_layer╖{"class_name": "Conv2D", "name": "conv2d_58", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_58", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 16, 128]}}
°	

7kernel
8bias
9regularization_losses
:	variables
;trainable_variables
<	keras_api
+╗&call_and_return_all_conditional_losses
╝__call__"╤
_tf_keras_layer╖{"class_name": "Conv2D", "name": "conv2d_59", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_59", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 16, 256]}}
Щ
=regularization_losses
>	variables
?trainable_variables
@	keras_api
+╜&call_and_return_all_conditional_losses
╛__call__"И
_tf_keras_layerю{"class_name": "GlobalAveragePooling2D", "name": "global_average_pooling2d_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "global_average_pooling2d_9", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Ў

Akernel
Bbias
Cregularization_losses
D	variables
Etrainable_variables
F	keras_api
+┐&call_and_return_all_conditional_losses
└__call__"╧
_tf_keras_layer╡{"class_name": "Dense", "name": "dense_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 8, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
ы
Giter

Hbeta_1

Ibeta_2
	Jdecay
Klearning_ratemОmПmРmС!mТ"mУ'mФ(mХ1mЦ2mЧ7mШ8mЩAmЪBmЫvЬvЭvЮvЯ!vа"vб'vв(vг1vд2vе7vж8vзAvиBvй"
	optimizer
 "
trackable_list_wrapper
Ж
0
1
2
3
!4
"5
'6
(7
18
29
710
811
A12
B13"
trackable_list_wrapper
Ж
0
1
2
3
!4
"5
'6
(7
18
29
710
811
A12
B13"
trackable_list_wrapper
╬
Llayer_metrics
regularization_losses
Mlayer_regularization_losses
Nnon_trainable_variables
trainable_variables

Olayers
Pmetrics
	variables
л__call__
м_default_save_signature
+к&call_and_return_all_conditional_losses
'к"call_and_return_conditional_losses"
_generic_user_object
-
┴serving_default"
signature_map
*:(@2conv2d_54/kernel
:@2conv2d_54/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
░
Qlayer_metrics
regularization_losses
	variables
Rlayer_regularization_losses
Snon_trainable_variables
trainable_variables

Tlayers
Umetrics
о__call__
+н&call_and_return_all_conditional_losses
'н"call_and_return_conditional_losses"
_generic_user_object
*:(@@2conv2d_55/kernel
:@2conv2d_55/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
░
Vlayer_metrics
regularization_losses
	variables
Wlayer_regularization_losses
Xnon_trainable_variables
trainable_variables

Ylayers
Zmetrics
░__call__
+п&call_and_return_all_conditional_losses
'п"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
[layer_metrics
regularization_losses
	variables
\layer_regularization_losses
]non_trainable_variables
trainable_variables

^layers
_metrics
▓__call__
+▒&call_and_return_all_conditional_losses
'▒"call_and_return_conditional_losses"
_generic_user_object
+:)@А2conv2d_56/kernel
:А2conv2d_56/bias
 "
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
░
`layer_metrics
#regularization_losses
$	variables
alayer_regularization_losses
bnon_trainable_variables
%trainable_variables

clayers
dmetrics
┤__call__
+│&call_and_return_all_conditional_losses
'│"call_and_return_conditional_losses"
_generic_user_object
,:*АА2conv2d_57/kernel
:А2conv2d_57/bias
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
░
elayer_metrics
)regularization_losses
*	variables
flayer_regularization_losses
gnon_trainable_variables
+trainable_variables

hlayers
imetrics
╢__call__
+╡&call_and_return_all_conditional_losses
'╡"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
jlayer_metrics
-regularization_losses
.	variables
klayer_regularization_losses
lnon_trainable_variables
/trainable_variables

mlayers
nmetrics
╕__call__
+╖&call_and_return_all_conditional_losses
'╖"call_and_return_conditional_losses"
_generic_user_object
,:*АА2conv2d_58/kernel
:А2conv2d_58/bias
 "
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
░
olayer_metrics
3regularization_losses
4	variables
player_regularization_losses
qnon_trainable_variables
5trainable_variables

rlayers
smetrics
║__call__
+╣&call_and_return_all_conditional_losses
'╣"call_and_return_conditional_losses"
_generic_user_object
,:*АА2conv2d_59/kernel
:А2conv2d_59/bias
 "
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
░
tlayer_metrics
9regularization_losses
:	variables
ulayer_regularization_losses
vnon_trainable_variables
;trainable_variables

wlayers
xmetrics
╝__call__
+╗&call_and_return_all_conditional_losses
'╗"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
ylayer_metrics
=regularization_losses
>	variables
zlayer_regularization_losses
{non_trainable_variables
?trainable_variables

|layers
}metrics
╛__call__
+╜&call_and_return_all_conditional_losses
'╜"call_and_return_conditional_losses"
_generic_user_object
!:	А2dense_9/kernel
:2dense_9/bias
 "
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
│
~layer_metrics
Cregularization_losses
D	variables
layer_regularization_losses
Аnon_trainable_variables
Etrainable_variables
Бlayers
Вmetrics
└__call__
+┐&call_and_return_all_conditional_losses
'┐"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
	8

9"
trackable_list_wrapper
0
Г0
Д1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
┐

Еtotal

Жcount
З	variables
И	keras_api"Д
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
Л

Йtotal

Кcount
Л
_fn_kwargs
М	variables
Н	keras_api"┐
_tf_keras_metricд{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}
:  (2total
:  (2count
0
Е0
Ж1"
trackable_list_wrapper
.
З	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
Й0
К1"
trackable_list_wrapper
.
М	variables"
_generic_user_object
/:-@2Adam/conv2d_54/kernel/m
!:@2Adam/conv2d_54/bias/m
/:-@@2Adam/conv2d_55/kernel/m
!:@2Adam/conv2d_55/bias/m
0:.@А2Adam/conv2d_56/kernel/m
": А2Adam/conv2d_56/bias/m
1:/АА2Adam/conv2d_57/kernel/m
": А2Adam/conv2d_57/bias/m
1:/АА2Adam/conv2d_58/kernel/m
": А2Adam/conv2d_58/bias/m
1:/АА2Adam/conv2d_59/kernel/m
": А2Adam/conv2d_59/bias/m
&:$	А2Adam/dense_9/kernel/m
:2Adam/dense_9/bias/m
/:-@2Adam/conv2d_54/kernel/v
!:@2Adam/conv2d_54/bias/v
/:-@@2Adam/conv2d_55/kernel/v
!:@2Adam/conv2d_55/bias/v
0:.@А2Adam/conv2d_56/kernel/v
": А2Adam/conv2d_56/bias/v
1:/АА2Adam/conv2d_57/kernel/v
": А2Adam/conv2d_57/bias/v
1:/АА2Adam/conv2d_58/kernel/v
": А2Adam/conv2d_58/bias/v
1:/АА2Adam/conv2d_59/kernel/v
": А2Adam/conv2d_59/bias/v
&:$	А2Adam/dense_9/kernel/v
:2Adam/dense_9/bias/v
ю2ы
H__inference_sequential_9_layer_call_and_return_conditional_losses_376172
H__inference_sequential_9_layer_call_and_return_conditional_losses_375879
H__inference_sequential_9_layer_call_and_return_conditional_losses_376229
H__inference_sequential_9_layer_call_and_return_conditional_losses_375921└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
В2 
-__inference_sequential_9_layer_call_fn_375997
-__inference_sequential_9_layer_call_fn_376295
-__inference_sequential_9_layer_call_fn_376072
-__inference_sequential_9_layer_call_fn_376262└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
я2ь
!__inference__wrapped_model_375645╞
Л▓З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *6в3
1К.
conv2d_54_input          
я2ь
E__inference_conv2d_54_layer_call_and_return_conditional_losses_376306в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╘2╤
*__inference_conv2d_54_layer_call_fn_376315в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
я2ь
E__inference_conv2d_55_layer_call_and_return_conditional_losses_376326в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╘2╤
*__inference_conv2d_55_layer_call_fn_376335в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
┤2▒
L__inference_max_pooling2d_18_layer_call_and_return_conditional_losses_375651р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
Щ2Ц
1__inference_max_pooling2d_18_layer_call_fn_375657р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
я2ь
E__inference_conv2d_56_layer_call_and_return_conditional_losses_376346в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╘2╤
*__inference_conv2d_56_layer_call_fn_376355в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
я2ь
E__inference_conv2d_57_layer_call_and_return_conditional_losses_376366в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╘2╤
*__inference_conv2d_57_layer_call_fn_376375в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
┤2▒
L__inference_max_pooling2d_19_layer_call_and_return_conditional_losses_375663р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
Щ2Ц
1__inference_max_pooling2d_19_layer_call_fn_375669р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
я2ь
E__inference_conv2d_58_layer_call_and_return_conditional_losses_376386в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╘2╤
*__inference_conv2d_58_layer_call_fn_376395в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
я2ь
E__inference_conv2d_59_layer_call_and_return_conditional_losses_376406в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╘2╤
*__inference_conv2d_59_layer_call_fn_376415в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╛2╗
V__inference_global_average_pooling2d_9_layer_call_and_return_conditional_losses_375676р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
г2а
;__inference_global_average_pooling2d_9_layer_call_fn_375682р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
э2ъ
C__inference_dense_9_layer_call_and_return_conditional_losses_376426в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╥2╧
(__inference_dense_9_layer_call_fn_376435в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╙B╨
$__inference_signature_wrapper_376115conv2d_54_input"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 л
!__inference__wrapped_model_375645Е!"'(1278AB@в=
6в3
1К.
conv2d_54_input          
к "1к.
,
dense_9!К
dense_9         ╡
E__inference_conv2d_54_layer_call_and_return_conditional_losses_376306l7в4
-в*
(К%
inputs          
к "-в*
#К 
0          @
Ъ Н
*__inference_conv2d_54_layer_call_fn_376315_7в4
-в*
(К%
inputs          
к " К          @╡
E__inference_conv2d_55_layer_call_and_return_conditional_losses_376326l7в4
-в*
(К%
inputs          @
к "-в*
#К 
0          @
Ъ Н
*__inference_conv2d_55_layer_call_fn_376335_7в4
-в*
(К%
inputs          @
к " К          @╢
E__inference_conv2d_56_layer_call_and_return_conditional_losses_376346m!"7в4
-в*
(К%
inputs          @
к ".в+
$К!
0          А
Ъ О
*__inference_conv2d_56_layer_call_fn_376355`!"7в4
-в*
(К%
inputs          @
к "!К          А╖
E__inference_conv2d_57_layer_call_and_return_conditional_losses_376366n'(8в5
.в+
)К&
inputs          А
к ".в+
$К!
0          А
Ъ П
*__inference_conv2d_57_layer_call_fn_376375a'(8в5
.в+
)К&
inputs          А
к "!К          А╖
E__inference_conv2d_58_layer_call_and_return_conditional_losses_376386n128в5
.в+
)К&
inputs         А
к ".в+
$К!
0         А
Ъ П
*__inference_conv2d_58_layer_call_fn_376395a128в5
.в+
)К&
inputs         А
к "!К         А╖
E__inference_conv2d_59_layer_call_and_return_conditional_losses_376406n788в5
.в+
)К&
inputs         А
к ".в+
$К!
0         А
Ъ П
*__inference_conv2d_59_layer_call_fn_376415a788в5
.в+
)К&
inputs         А
к "!К         Ад
C__inference_dense_9_layer_call_and_return_conditional_losses_376426]AB0в-
&в#
!К
inputs         А
к "%в"
К
0         
Ъ |
(__inference_dense_9_layer_call_fn_376435PAB0в-
&в#
!К
inputs         А
к "К         ▀
V__inference_global_average_pooling2d_9_layer_call_and_return_conditional_losses_375676ДRвO
HвE
CК@
inputs4                                    
к ".в+
$К!
0                  
Ъ ╢
;__inference_global_average_pooling2d_9_layer_call_fn_375682wRвO
HвE
CК@
inputs4                                    
к "!К                  я
L__inference_max_pooling2d_18_layer_call_and_return_conditional_losses_375651ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╟
1__inference_max_pooling2d_18_layer_call_fn_375657СRвO
HвE
CК@
inputs4                                    
к ";К84                                    я
L__inference_max_pooling2d_19_layer_call_and_return_conditional_losses_375663ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╟
1__inference_max_pooling2d_19_layer_call_fn_375669СRвO
HвE
CК@
inputs4                                    
к ";К84                                    ╬
H__inference_sequential_9_layer_call_and_return_conditional_losses_375879Б!"'(1278ABHвE
>в;
1К.
conv2d_54_input          
p

 
к "%в"
К
0         
Ъ ╬
H__inference_sequential_9_layer_call_and_return_conditional_losses_375921Б!"'(1278ABHвE
>в;
1К.
conv2d_54_input          
p 

 
к "%в"
К
0         
Ъ ─
H__inference_sequential_9_layer_call_and_return_conditional_losses_376172x!"'(1278AB?в<
5в2
(К%
inputs          
p

 
к "%в"
К
0         
Ъ ─
H__inference_sequential_9_layer_call_and_return_conditional_losses_376229x!"'(1278AB?в<
5в2
(К%
inputs          
p 

 
к "%в"
К
0         
Ъ е
-__inference_sequential_9_layer_call_fn_375997t!"'(1278ABHвE
>в;
1К.
conv2d_54_input          
p

 
к "К         е
-__inference_sequential_9_layer_call_fn_376072t!"'(1278ABHвE
>в;
1К.
conv2d_54_input          
p 

 
к "К         Ь
-__inference_sequential_9_layer_call_fn_376262k!"'(1278AB?в<
5в2
(К%
inputs          
p

 
к "К         Ь
-__inference_sequential_9_layer_call_fn_376295k!"'(1278AB?в<
5в2
(К%
inputs          
p 

 
к "К         ┴
$__inference_signature_wrapper_376115Ш!"'(1278ABSвP
в 
IкF
D
conv2d_54_input1К.
conv2d_54_input          "1к.
,
dense_9!К
dense_9         