Õ´
è
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
R
Einsum
inputs"T*N
output"T"
equationstring"
Nint(0"	
Ttype
­
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
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

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
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	
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

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:
2		
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
¥
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
9
Softmax
logits"T
softmax"T"
Ttype:
2
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	
Á
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
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68¢¹
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
x
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_3/kernel
q
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes

:*
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:*
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
¿
1token_and_position_embedding/embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*B
shared_name31token_and_position_embedding/embedding/embeddings
¸
Etoken_and_position_embedding/embedding/embeddings/Read/ReadVariableOpReadVariableOp1token_and_position_embedding/embedding/embeddings*
_output_shapes
:	*
dtype0
Ã
3token_and_position_embedding/embedding_1/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*D
shared_name53token_and_position_embedding/embedding_1/embeddings
¼
Gtoken_and_position_embedding/embedding_1/embeddings/Read/ReadVariableOpReadVariableOp3token_and_position_embedding/embedding_1/embeddings*
_output_shapes
:	*
dtype0
Æ
3transformer_block/multi_head_attention/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53transformer_block/multi_head_attention/query/kernel
¿
Gtransformer_block/multi_head_attention/query/kernel/Read/ReadVariableOpReadVariableOp3transformer_block/multi_head_attention/query/kernel*"
_output_shapes
:*
dtype0
¾
1transformer_block/multi_head_attention/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*B
shared_name31transformer_block/multi_head_attention/query/bias
·
Etransformer_block/multi_head_attention/query/bias/Read/ReadVariableOpReadVariableOp1transformer_block/multi_head_attention/query/bias*
_output_shapes

:*
dtype0
Â
1transformer_block/multi_head_attention/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31transformer_block/multi_head_attention/key/kernel
»
Etransformer_block/multi_head_attention/key/kernel/Read/ReadVariableOpReadVariableOp1transformer_block/multi_head_attention/key/kernel*"
_output_shapes
:*
dtype0
º
/transformer_block/multi_head_attention/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*@
shared_name1/transformer_block/multi_head_attention/key/bias
³
Ctransformer_block/multi_head_attention/key/bias/Read/ReadVariableOpReadVariableOp/transformer_block/multi_head_attention/key/bias*
_output_shapes

:*
dtype0
Æ
3transformer_block/multi_head_attention/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53transformer_block/multi_head_attention/value/kernel
¿
Gtransformer_block/multi_head_attention/value/kernel/Read/ReadVariableOpReadVariableOp3transformer_block/multi_head_attention/value/kernel*"
_output_shapes
:*
dtype0
¾
1transformer_block/multi_head_attention/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*B
shared_name31transformer_block/multi_head_attention/value/bias
·
Etransformer_block/multi_head_attention/value/bias/Read/ReadVariableOpReadVariableOp1transformer_block/multi_head_attention/value/bias*
_output_shapes

:*
dtype0
Ü
>transformer_block/multi_head_attention/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*O
shared_name@>transformer_block/multi_head_attention/attention_output/kernel
Õ
Rtransformer_block/multi_head_attention/attention_output/kernel/Read/ReadVariableOpReadVariableOp>transformer_block/multi_head_attention/attention_output/kernel*"
_output_shapes
:*
dtype0
Ð
<transformer_block/multi_head_attention/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*M
shared_name><transformer_block/multi_head_attention/attention_output/bias
É
Ptransformer_block/multi_head_attention/attention_output/bias/Read/ReadVariableOpReadVariableOp<transformer_block/multi_head_attention/attention_output/bias*
_output_shapes
:*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
®
+transformer_block/layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+transformer_block/layer_normalization/gamma
§
?transformer_block/layer_normalization/gamma/Read/ReadVariableOpReadVariableOp+transformer_block/layer_normalization/gamma*
_output_shapes
:*
dtype0
¬
*transformer_block/layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*transformer_block/layer_normalization/beta
¥
>transformer_block/layer_normalization/beta/Read/ReadVariableOpReadVariableOp*transformer_block/layer_normalization/beta*
_output_shapes
:*
dtype0
²
-transformer_block/layer_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-transformer_block/layer_normalization_1/gamma
«
Atransformer_block/layer_normalization_1/gamma/Read/ReadVariableOpReadVariableOp-transformer_block/layer_normalization_1/gamma*
_output_shapes
:*
dtype0
°
,transformer_block/layer_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,transformer_block/layer_normalization_1/beta
©
@transformer_block/layer_normalization_1/beta/Read/ReadVariableOpReadVariableOp,transformer_block/layer_normalization_1/beta*
_output_shapes
:*
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

Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_2/kernel/m

)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/m
w
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes
:*
dtype0

Adam/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_3/kernel/m

)Adam/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/m
w
'Adam/dense_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/m*
_output_shapes
:*
dtype0
Í
8Adam/token_and_position_embedding/embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*I
shared_name:8Adam/token_and_position_embedding/embedding/embeddings/m
Æ
LAdam/token_and_position_embedding/embedding/embeddings/m/Read/ReadVariableOpReadVariableOp8Adam/token_and_position_embedding/embedding/embeddings/m*
_output_shapes
:	*
dtype0
Ñ
:Adam/token_and_position_embedding/embedding_1/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*K
shared_name<:Adam/token_and_position_embedding/embedding_1/embeddings/m
Ê
NAdam/token_and_position_embedding/embedding_1/embeddings/m/Read/ReadVariableOpReadVariableOp:Adam/token_and_position_embedding/embedding_1/embeddings/m*
_output_shapes
:	*
dtype0
Ô
:Adam/transformer_block/multi_head_attention/query/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*K
shared_name<:Adam/transformer_block/multi_head_attention/query/kernel/m
Í
NAdam/transformer_block/multi_head_attention/query/kernel/m/Read/ReadVariableOpReadVariableOp:Adam/transformer_block/multi_head_attention/query/kernel/m*"
_output_shapes
:*
dtype0
Ì
8Adam/transformer_block/multi_head_attention/query/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*I
shared_name:8Adam/transformer_block/multi_head_attention/query/bias/m
Å
LAdam/transformer_block/multi_head_attention/query/bias/m/Read/ReadVariableOpReadVariableOp8Adam/transformer_block/multi_head_attention/query/bias/m*
_output_shapes

:*
dtype0
Ð
8Adam/transformer_block/multi_head_attention/key/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8Adam/transformer_block/multi_head_attention/key/kernel/m
É
LAdam/transformer_block/multi_head_attention/key/kernel/m/Read/ReadVariableOpReadVariableOp8Adam/transformer_block/multi_head_attention/key/kernel/m*"
_output_shapes
:*
dtype0
È
6Adam/transformer_block/multi_head_attention/key/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*G
shared_name86Adam/transformer_block/multi_head_attention/key/bias/m
Á
JAdam/transformer_block/multi_head_attention/key/bias/m/Read/ReadVariableOpReadVariableOp6Adam/transformer_block/multi_head_attention/key/bias/m*
_output_shapes

:*
dtype0
Ô
:Adam/transformer_block/multi_head_attention/value/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*K
shared_name<:Adam/transformer_block/multi_head_attention/value/kernel/m
Í
NAdam/transformer_block/multi_head_attention/value/kernel/m/Read/ReadVariableOpReadVariableOp:Adam/transformer_block/multi_head_attention/value/kernel/m*"
_output_shapes
:*
dtype0
Ì
8Adam/transformer_block/multi_head_attention/value/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*I
shared_name:8Adam/transformer_block/multi_head_attention/value/bias/m
Å
LAdam/transformer_block/multi_head_attention/value/bias/m/Read/ReadVariableOpReadVariableOp8Adam/transformer_block/multi_head_attention/value/bias/m*
_output_shapes

:*
dtype0
ê
EAdam/transformer_block/multi_head_attention/attention_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*V
shared_nameGEAdam/transformer_block/multi_head_attention/attention_output/kernel/m
ã
YAdam/transformer_block/multi_head_attention/attention_output/kernel/m/Read/ReadVariableOpReadVariableOpEAdam/transformer_block/multi_head_attention/attention_output/kernel/m*"
_output_shapes
:*
dtype0
Þ
CAdam/transformer_block/multi_head_attention/attention_output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*T
shared_nameECAdam/transformer_block/multi_head_attention/attention_output/bias/m
×
WAdam/transformer_block/multi_head_attention/attention_output/bias/m/Read/ReadVariableOpReadVariableOpCAdam/transformer_block/multi_head_attention/attention_output/bias/m*
_output_shapes
:*
dtype0

Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_nameAdam/dense/kernel/m
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes

:*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:*
dtype0

Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:*
dtype0
¼
2Adam/transformer_block/layer_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*C
shared_name42Adam/transformer_block/layer_normalization/gamma/m
µ
FAdam/transformer_block/layer_normalization/gamma/m/Read/ReadVariableOpReadVariableOp2Adam/transformer_block/layer_normalization/gamma/m*
_output_shapes
:*
dtype0
º
1Adam/transformer_block/layer_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31Adam/transformer_block/layer_normalization/beta/m
³
EAdam/transformer_block/layer_normalization/beta/m/Read/ReadVariableOpReadVariableOp1Adam/transformer_block/layer_normalization/beta/m*
_output_shapes
:*
dtype0
À
4Adam/transformer_block/layer_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*E
shared_name64Adam/transformer_block/layer_normalization_1/gamma/m
¹
HAdam/transformer_block/layer_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp4Adam/transformer_block/layer_normalization_1/gamma/m*
_output_shapes
:*
dtype0
¾
3Adam/transformer_block/layer_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53Adam/transformer_block/layer_normalization_1/beta/m
·
GAdam/transformer_block/layer_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp3Adam/transformer_block/layer_normalization_1/beta/m*
_output_shapes
:*
dtype0

Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_2/kernel/v

)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/v
w
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes
:*
dtype0

Adam/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_3/kernel/v

)Adam/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/v
w
'Adam/dense_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/v*
_output_shapes
:*
dtype0
Í
8Adam/token_and_position_embedding/embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*I
shared_name:8Adam/token_and_position_embedding/embedding/embeddings/v
Æ
LAdam/token_and_position_embedding/embedding/embeddings/v/Read/ReadVariableOpReadVariableOp8Adam/token_and_position_embedding/embedding/embeddings/v*
_output_shapes
:	*
dtype0
Ñ
:Adam/token_and_position_embedding/embedding_1/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*K
shared_name<:Adam/token_and_position_embedding/embedding_1/embeddings/v
Ê
NAdam/token_and_position_embedding/embedding_1/embeddings/v/Read/ReadVariableOpReadVariableOp:Adam/token_and_position_embedding/embedding_1/embeddings/v*
_output_shapes
:	*
dtype0
Ô
:Adam/transformer_block/multi_head_attention/query/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*K
shared_name<:Adam/transformer_block/multi_head_attention/query/kernel/v
Í
NAdam/transformer_block/multi_head_attention/query/kernel/v/Read/ReadVariableOpReadVariableOp:Adam/transformer_block/multi_head_attention/query/kernel/v*"
_output_shapes
:*
dtype0
Ì
8Adam/transformer_block/multi_head_attention/query/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*I
shared_name:8Adam/transformer_block/multi_head_attention/query/bias/v
Å
LAdam/transformer_block/multi_head_attention/query/bias/v/Read/ReadVariableOpReadVariableOp8Adam/transformer_block/multi_head_attention/query/bias/v*
_output_shapes

:*
dtype0
Ð
8Adam/transformer_block/multi_head_attention/key/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8Adam/transformer_block/multi_head_attention/key/kernel/v
É
LAdam/transformer_block/multi_head_attention/key/kernel/v/Read/ReadVariableOpReadVariableOp8Adam/transformer_block/multi_head_attention/key/kernel/v*"
_output_shapes
:*
dtype0
È
6Adam/transformer_block/multi_head_attention/key/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*G
shared_name86Adam/transformer_block/multi_head_attention/key/bias/v
Á
JAdam/transformer_block/multi_head_attention/key/bias/v/Read/ReadVariableOpReadVariableOp6Adam/transformer_block/multi_head_attention/key/bias/v*
_output_shapes

:*
dtype0
Ô
:Adam/transformer_block/multi_head_attention/value/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*K
shared_name<:Adam/transformer_block/multi_head_attention/value/kernel/v
Í
NAdam/transformer_block/multi_head_attention/value/kernel/v/Read/ReadVariableOpReadVariableOp:Adam/transformer_block/multi_head_attention/value/kernel/v*"
_output_shapes
:*
dtype0
Ì
8Adam/transformer_block/multi_head_attention/value/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*I
shared_name:8Adam/transformer_block/multi_head_attention/value/bias/v
Å
LAdam/transformer_block/multi_head_attention/value/bias/v/Read/ReadVariableOpReadVariableOp8Adam/transformer_block/multi_head_attention/value/bias/v*
_output_shapes

:*
dtype0
ê
EAdam/transformer_block/multi_head_attention/attention_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*V
shared_nameGEAdam/transformer_block/multi_head_attention/attention_output/kernel/v
ã
YAdam/transformer_block/multi_head_attention/attention_output/kernel/v/Read/ReadVariableOpReadVariableOpEAdam/transformer_block/multi_head_attention/attention_output/kernel/v*"
_output_shapes
:*
dtype0
Þ
CAdam/transformer_block/multi_head_attention/attention_output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*T
shared_nameECAdam/transformer_block/multi_head_attention/attention_output/bias/v
×
WAdam/transformer_block/multi_head_attention/attention_output/bias/v/Read/ReadVariableOpReadVariableOpCAdam/transformer_block/multi_head_attention/attention_output/bias/v*
_output_shapes
:*
dtype0

Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_nameAdam/dense/kernel/v
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes

:*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:*
dtype0

Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:*
dtype0
¼
2Adam/transformer_block/layer_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*C
shared_name42Adam/transformer_block/layer_normalization/gamma/v
µ
FAdam/transformer_block/layer_normalization/gamma/v/Read/ReadVariableOpReadVariableOp2Adam/transformer_block/layer_normalization/gamma/v*
_output_shapes
:*
dtype0
º
1Adam/transformer_block/layer_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31Adam/transformer_block/layer_normalization/beta/v
³
EAdam/transformer_block/layer_normalization/beta/v/Read/ReadVariableOpReadVariableOp1Adam/transformer_block/layer_normalization/beta/v*
_output_shapes
:*
dtype0
À
4Adam/transformer_block/layer_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*E
shared_name64Adam/transformer_block/layer_normalization_1/gamma/v
¹
HAdam/transformer_block/layer_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp4Adam/transformer_block/layer_normalization_1/gamma/v*
_output_shapes
:*
dtype0
¾
3Adam/transformer_block/layer_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53Adam/transformer_block/layer_normalization_1/beta/v
·
GAdam/transformer_block/layer_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp3Adam/transformer_block/layer_normalization_1/beta/v*
_output_shapes
:*
dtype0

NoOpNoOp
â¼
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*¼
value¼B¼ B¼

layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
		optimizer

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
¬
	token_emb
pos_emb
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
Þ
att
ffn

layernorm1

layernorm2
dropout1
dropout2
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses*

&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses* 
¥
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0_random_generator
1__call__
*2&call_and_return_all_conditional_losses* 
¦

3kernel
4bias
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses*
¥
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?_random_generator
@__call__
*A&call_and_return_all_conditional_losses* 
¦

Bkernel
Cbias
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses*
ü
Jiter

Kbeta_1

Lbeta_2
	Mdecay
Nlearning_rate3mÚ4mÛBmÜCmÝOmÞPmßQmàRmáSmâTmãUmäVmåWmæXmçYmèZmé[mê\më]mì^mí_mî`mï3vð4vñBvòCvóOvôPvõQvöRv÷SvøTvùUvúVvûWvüXvýYvþZvÿ[v\v]v^v_v`v*
ª
O0
P1
Q2
R3
S4
T5
U6
V7
W8
X9
Y10
Z11
[12
\13
]14
^15
_16
`17
318
419
B20
C21*
ª
O0
P1
Q2
R3
S4
T5
U6
V7
W8
X9
Y10
Z11
[12
\13
]14
^15
_16
`17
318
419
B20
C21*
* 
°
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics

	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

fserving_default* 
 
O
embeddings
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses*
 
P
embeddings
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses*

O0
P1*

O0
P1*
* 

snon_trainable_variables

tlayers
umetrics
vlayer_regularization_losses
wlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
ý
x_query_dense
y
_key_dense
z_value_dense
{_softmax
|_dropout_layer
}_output_dense
~	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
è
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
¶
	axis
	]gamma
^beta
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
¶
	axis
	_gamma
`beta
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
¬
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
+ &call_and_return_all_conditional_losses* 
¬
¡	variables
¢trainable_variables
£regularization_losses
¤	keras_api
¥_random_generator
¦__call__
+§&call_and_return_all_conditional_losses* 
z
Q0
R1
S2
T3
U4
V5
W6
X7
Y8
Z9
[10
\11
]12
^13
_14
`15*
z
Q0
R1
S2
T3
U4
V5
W6
X7
Y8
Z9
[10
\11
]12
^13
_14
`15*
* 

¨non_trainable_variables
©layers
ªmetrics
 «layer_regularization_losses
¬layer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

­non_trainable_variables
®layers
¯metrics
 °layer_regularization_losses
±layer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

²non_trainable_variables
³layers
´metrics
 µlayer_regularization_losses
¶layer_metrics
,	variables
-trainable_variables
.regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses* 
* 
* 
* 
^X
VARIABLE_VALUEdense_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

30
41*

30
41*
* 

·non_trainable_variables
¸layers
¹metrics
 ºlayer_regularization_losses
»layer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

¼non_trainable_variables
½layers
¾metrics
 ¿layer_regularization_losses
Àlayer_metrics
;	variables
<trainable_variables
=regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses* 
* 
* 
* 
^X
VARIABLE_VALUEdense_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

B0
C1*

B0
C1*
* 

Ánon_trainable_variables
Âlayers
Ãmetrics
 Älayer_regularization_losses
Ålayer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE1token_and_position_embedding/embedding/embeddings&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE3token_and_position_embedding/embedding_1/embeddings&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE3transformer_block/multi_head_attention/query/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE1transformer_block/multi_head_attention/query/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE1transformer_block/multi_head_attention/key/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE/transformer_block/multi_head_attention/key/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE3transformer_block/multi_head_attention/value/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE1transformer_block/multi_head_attention/value/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE>transformer_block/multi_head_attention/attention_output/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE<transformer_block/multi_head_attention/attention_output/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
dense/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_1/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_1/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE+transformer_block/layer_normalization/gamma'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE*transformer_block/layer_normalization/beta'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE-transformer_block/layer_normalization_1/gamma'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE,transformer_block/layer_normalization_1/beta'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
* 
<
0
1
2
3
4
5
6
7*

Æ0
Ç1*
* 
* 
* 

O0*

O0*
* 

Ènon_trainable_variables
Élayers
Êmetrics
 Ëlayer_regularization_losses
Ìlayer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses*
* 
* 

P0*

P0*
* 

Ínon_trainable_variables
Îlayers
Ïmetrics
 Ðlayer_regularization_losses
Ñlayer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses*
* 
* 
* 

0
1*
* 
* 
* 
ß
Òpartial_output_shape
Ófull_output_shape

Qkernel
Rbias
Ô	variables
Õtrainable_variables
Öregularization_losses
×	keras_api
Ø__call__
+Ù&call_and_return_all_conditional_losses*
ß
Úpartial_output_shape
Ûfull_output_shape

Skernel
Tbias
Ü	variables
Ýtrainable_variables
Þregularization_losses
ß	keras_api
à__call__
+á&call_and_return_all_conditional_losses*
ß
âpartial_output_shape
ãfull_output_shape

Ukernel
Vbias
ä	variables
åtrainable_variables
æregularization_losses
ç	keras_api
è__call__
+é&call_and_return_all_conditional_losses*

ê	variables
ëtrainable_variables
ìregularization_losses
í	keras_api
î__call__
+ï&call_and_return_all_conditional_losses* 
¬
ð	variables
ñtrainable_variables
òregularization_losses
ó	keras_api
ô_random_generator
õ__call__
+ö&call_and_return_all_conditional_losses* 
ß
÷partial_output_shape
øfull_output_shape

Wkernel
Xbias
ù	variables
útrainable_variables
ûregularization_losses
ü	keras_api
ý__call__
+þ&call_and_return_all_conditional_losses*
<
Q0
R1
S2
T3
U4
V5
W6
X7*
<
Q0
R1
S2
T3
U4
V5
W6
X7*
* 

ÿnon_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
~	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
¬

Ykernel
Zbias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
¬

[kernel
\bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
 
Y0
Z1
[2
\3*
 
Y0
Z1
[2
\3*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 

]0
^1*

]0
^1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 

_0
`1*

_0
`1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
 layers
¡metrics
 ¢layer_regularization_losses
£layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 

¤non_trainable_variables
¥layers
¦metrics
 §layer_regularization_losses
¨layer_metrics
¡	variables
¢trainable_variables
£regularization_losses
¦__call__
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses* 
* 
* 
* 
* 
.
0
1
2
3
4
5*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<

©total

ªcount
«	variables
¬	keras_api*
M

­total

®count
¯
_fn_kwargs
°	variables
±	keras_api*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

Q0
R1*

Q0
R1*
* 

²non_trainable_variables
³layers
´metrics
 µlayer_regularization_losses
¶layer_metrics
Ô	variables
Õtrainable_variables
Öregularization_losses
Ø__call__
+Ù&call_and_return_all_conditional_losses
'Ù"call_and_return_conditional_losses*
* 
* 
* 
* 

S0
T1*

S0
T1*
* 

·non_trainable_variables
¸layers
¹metrics
 ºlayer_regularization_losses
»layer_metrics
Ü	variables
Ýtrainable_variables
Þregularization_losses
à__call__
+á&call_and_return_all_conditional_losses
'á"call_and_return_conditional_losses*
* 
* 
* 
* 

U0
V1*

U0
V1*
* 

¼non_trainable_variables
½layers
¾metrics
 ¿layer_regularization_losses
Àlayer_metrics
ä	variables
åtrainable_variables
æregularization_losses
è__call__
+é&call_and_return_all_conditional_losses
'é"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Ánon_trainable_variables
Âlayers
Ãmetrics
 Älayer_regularization_losses
Ålayer_metrics
ê	variables
ëtrainable_variables
ìregularization_losses
î__call__
+ï&call_and_return_all_conditional_losses
'ï"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

Ænon_trainable_variables
Çlayers
Èmetrics
 Élayer_regularization_losses
Êlayer_metrics
ð	variables
ñtrainable_variables
òregularization_losses
õ__call__
+ö&call_and_return_all_conditional_losses
'ö"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

W0
X1*

W0
X1*
* 

Ënon_trainable_variables
Ìlayers
Ímetrics
 Îlayer_regularization_losses
Ïlayer_metrics
ù	variables
útrainable_variables
ûregularization_losses
ý__call__
+þ&call_and_return_all_conditional_losses
'þ"call_and_return_conditional_losses*
* 
* 
* 
.
x0
y1
z2
{3
|4
}5*
* 
* 
* 

Y0
Z1*

Y0
Z1*
* 

Ðnon_trainable_variables
Ñlayers
Òmetrics
 Ólayer_regularization_losses
Ôlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 

[0
\1*

[0
\1*
* 

Õnon_trainable_variables
Ölayers
×metrics
 Ølayer_regularization_losses
Ùlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 

0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

©0
ª1*

«	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

­0
®1*

°	variables*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
{
VARIABLE_VALUEAdam/dense_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_3/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_3/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE8Adam/token_and_position_embedding/embedding/embeddings/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE:Adam/token_and_position_embedding/embedding_1/embeddings/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE:Adam/transformer_block/multi_head_attention/query/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE8Adam/transformer_block/multi_head_attention/query/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE8Adam/transformer_block/multi_head_attention/key/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE6Adam/transformer_block/multi_head_attention/key/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE:Adam/transformer_block/multi_head_attention/value/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE8Adam/transformer_block/multi_head_attention/value/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
¢
VARIABLE_VALUEEAdam/transformer_block/multi_head_attention/attention_output/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
 
VARIABLE_VALUECAdam/transformer_block/multi_head_attention/attention_output/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUEAdam/dense/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_1/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_1/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE2Adam/transformer_block/layer_normalization/gamma/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE1Adam/transformer_block/layer_normalization/beta/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE4Adam/transformer_block/layer_normalization_1/gamma/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE3Adam/transformer_block/layer_normalization_1/beta/mCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_3/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_3/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE8Adam/token_and_position_embedding/embedding/embeddings/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE:Adam/token_and_position_embedding/embedding_1/embeddings/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE:Adam/transformer_block/multi_head_attention/query/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE8Adam/transformer_block/multi_head_attention/query/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE8Adam/transformer_block/multi_head_attention/key/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE6Adam/transformer_block/multi_head_attention/key/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE:Adam/transformer_block/multi_head_attention/value/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE8Adam/transformer_block/multi_head_attention/value/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
¢
VARIABLE_VALUEEAdam/transformer_block/multi_head_attention/attention_output/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
 
VARIABLE_VALUECAdam/transformer_block/multi_head_attention/attention_output/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUEAdam/dense/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_1/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_1/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE2Adam/transformer_block/layer_normalization/gamma/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE1Adam/transformer_block/layer_normalization/beta/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE4Adam/transformer_block/layer_normalization_1/gamma/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE3Adam/transformer_block/layer_normalization_1/beta/vCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
serving_default_input_1Placeholder*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
 	
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_13token_and_position_embedding/embedding_1/embeddings1token_and_position_embedding/embedding/embeddings3transformer_block/multi_head_attention/query/kernel1transformer_block/multi_head_attention/query/bias1transformer_block/multi_head_attention/key/kernel/transformer_block/multi_head_attention/key/bias3transformer_block/multi_head_attention/value/kernel1transformer_block/multi_head_attention/value/bias>transformer_block/multi_head_attention/attention_output/kernel<transformer_block/multi_head_attention/attention_output/bias+transformer_block/layer_normalization/gamma*transformer_block/layer_normalization/betadense/kernel
dense/biasdense_1/kerneldense_1/bias-transformer_block/layer_normalization_1/gamma,transformer_block/layer_normalization_1/betadense_2/kerneldense_2/biasdense_3/kerneldense_3/bias*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference_signature_wrapper_9117
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
¦%
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOpEtoken_and_position_embedding/embedding/embeddings/Read/ReadVariableOpGtoken_and_position_embedding/embedding_1/embeddings/Read/ReadVariableOpGtransformer_block/multi_head_attention/query/kernel/Read/ReadVariableOpEtransformer_block/multi_head_attention/query/bias/Read/ReadVariableOpEtransformer_block/multi_head_attention/key/kernel/Read/ReadVariableOpCtransformer_block/multi_head_attention/key/bias/Read/ReadVariableOpGtransformer_block/multi_head_attention/value/kernel/Read/ReadVariableOpEtransformer_block/multi_head_attention/value/bias/Read/ReadVariableOpRtransformer_block/multi_head_attention/attention_output/kernel/Read/ReadVariableOpPtransformer_block/multi_head_attention/attention_output/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp?transformer_block/layer_normalization/gamma/Read/ReadVariableOp>transformer_block/layer_normalization/beta/Read/ReadVariableOpAtransformer_block/layer_normalization_1/gamma/Read/ReadVariableOp@transformer_block/layer_normalization_1/beta/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp)Adam/dense_3/kernel/m/Read/ReadVariableOp'Adam/dense_3/bias/m/Read/ReadVariableOpLAdam/token_and_position_embedding/embedding/embeddings/m/Read/ReadVariableOpNAdam/token_and_position_embedding/embedding_1/embeddings/m/Read/ReadVariableOpNAdam/transformer_block/multi_head_attention/query/kernel/m/Read/ReadVariableOpLAdam/transformer_block/multi_head_attention/query/bias/m/Read/ReadVariableOpLAdam/transformer_block/multi_head_attention/key/kernel/m/Read/ReadVariableOpJAdam/transformer_block/multi_head_attention/key/bias/m/Read/ReadVariableOpNAdam/transformer_block/multi_head_attention/value/kernel/m/Read/ReadVariableOpLAdam/transformer_block/multi_head_attention/value/bias/m/Read/ReadVariableOpYAdam/transformer_block/multi_head_attention/attention_output/kernel/m/Read/ReadVariableOpWAdam/transformer_block/multi_head_attention/attention_output/bias/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOpFAdam/transformer_block/layer_normalization/gamma/m/Read/ReadVariableOpEAdam/transformer_block/layer_normalization/beta/m/Read/ReadVariableOpHAdam/transformer_block/layer_normalization_1/gamma/m/Read/ReadVariableOpGAdam/transformer_block/layer_normalization_1/beta/m/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOp)Adam/dense_3/kernel/v/Read/ReadVariableOp'Adam/dense_3/bias/v/Read/ReadVariableOpLAdam/token_and_position_embedding/embedding/embeddings/v/Read/ReadVariableOpNAdam/token_and_position_embedding/embedding_1/embeddings/v/Read/ReadVariableOpNAdam/transformer_block/multi_head_attention/query/kernel/v/Read/ReadVariableOpLAdam/transformer_block/multi_head_attention/query/bias/v/Read/ReadVariableOpLAdam/transformer_block/multi_head_attention/key/kernel/v/Read/ReadVariableOpJAdam/transformer_block/multi_head_attention/key/bias/v/Read/ReadVariableOpNAdam/transformer_block/multi_head_attention/value/kernel/v/Read/ReadVariableOpLAdam/transformer_block/multi_head_attention/value/bias/v/Read/ReadVariableOpYAdam/transformer_block/multi_head_attention/attention_output/kernel/v/Read/ReadVariableOpWAdam/transformer_block/multi_head_attention/attention_output/bias/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOpFAdam/transformer_block/layer_normalization/gamma/v/Read/ReadVariableOpEAdam/transformer_block/layer_normalization/beta/v/Read/ReadVariableOpHAdam/transformer_block/layer_normalization_1/gamma/v/Read/ReadVariableOpGAdam/transformer_block/layer_normalization_1/beta/v/Read/ReadVariableOpConst*X
TinQ
O2M	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference__traced_save_10063
Å
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_2/kerneldense_2/biasdense_3/kerneldense_3/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate1token_and_position_embedding/embedding/embeddings3token_and_position_embedding/embedding_1/embeddings3transformer_block/multi_head_attention/query/kernel1transformer_block/multi_head_attention/query/bias1transformer_block/multi_head_attention/key/kernel/transformer_block/multi_head_attention/key/bias3transformer_block/multi_head_attention/value/kernel1transformer_block/multi_head_attention/value/bias>transformer_block/multi_head_attention/attention_output/kernel<transformer_block/multi_head_attention/attention_output/biasdense/kernel
dense/biasdense_1/kerneldense_1/bias+transformer_block/layer_normalization/gamma*transformer_block/layer_normalization/beta-transformer_block/layer_normalization_1/gamma,transformer_block/layer_normalization_1/betatotalcounttotal_1count_1Adam/dense_2/kernel/mAdam/dense_2/bias/mAdam/dense_3/kernel/mAdam/dense_3/bias/m8Adam/token_and_position_embedding/embedding/embeddings/m:Adam/token_and_position_embedding/embedding_1/embeddings/m:Adam/transformer_block/multi_head_attention/query/kernel/m8Adam/transformer_block/multi_head_attention/query/bias/m8Adam/transformer_block/multi_head_attention/key/kernel/m6Adam/transformer_block/multi_head_attention/key/bias/m:Adam/transformer_block/multi_head_attention/value/kernel/m8Adam/transformer_block/multi_head_attention/value/bias/mEAdam/transformer_block/multi_head_attention/attention_output/kernel/mCAdam/transformer_block/multi_head_attention/attention_output/bias/mAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/m2Adam/transformer_block/layer_normalization/gamma/m1Adam/transformer_block/layer_normalization/beta/m4Adam/transformer_block/layer_normalization_1/gamma/m3Adam/transformer_block/layer_normalization_1/beta/mAdam/dense_2/kernel/vAdam/dense_2/bias/vAdam/dense_3/kernel/vAdam/dense_3/bias/v8Adam/token_and_position_embedding/embedding/embeddings/v:Adam/token_and_position_embedding/embedding_1/embeddings/v:Adam/transformer_block/multi_head_attention/query/kernel/v8Adam/transformer_block/multi_head_attention/query/bias/v8Adam/transformer_block/multi_head_attention/key/kernel/v6Adam/transformer_block/multi_head_attention/key/bias/v:Adam/transformer_block/multi_head_attention/value/kernel/v8Adam/transformer_block/multi_head_attention/value/bias/vEAdam/transformer_block/multi_head_attention/attention_output/kernel/vCAdam/transformer_block/multi_head_attention/attention_output/bias/vAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/v2Adam/transformer_block/layer_normalization/gamma/v1Adam/transformer_block/layer_normalization/beta/v4Adam/transformer_block/layer_normalization_1/gamma/v3Adam/transformer_block/layer_normalization_1/beta/v*W
TinP
N2L*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_restore_10298ó
Ô¹
«9
!__inference__traced_restore_10298
file_prefix1
assignvariableop_dense_2_kernel:-
assignvariableop_1_dense_2_bias:3
!assignvariableop_2_dense_3_kernel:-
assignvariableop_3_dense_3_bias:&
assignvariableop_4_adam_iter:	 (
assignvariableop_5_adam_beta_1: (
assignvariableop_6_adam_beta_2: '
assignvariableop_7_adam_decay: /
%assignvariableop_8_adam_learning_rate: W
Dassignvariableop_9_token_and_position_embedding_embedding_embeddings:	Z
Gassignvariableop_10_token_and_position_embedding_embedding_1_embeddings:	]
Gassignvariableop_11_transformer_block_multi_head_attention_query_kernel:W
Eassignvariableop_12_transformer_block_multi_head_attention_query_bias:[
Eassignvariableop_13_transformer_block_multi_head_attention_key_kernel:U
Cassignvariableop_14_transformer_block_multi_head_attention_key_bias:]
Gassignvariableop_15_transformer_block_multi_head_attention_value_kernel:W
Eassignvariableop_16_transformer_block_multi_head_attention_value_bias:h
Rassignvariableop_17_transformer_block_multi_head_attention_attention_output_kernel:^
Passignvariableop_18_transformer_block_multi_head_attention_attention_output_bias:2
 assignvariableop_19_dense_kernel:,
assignvariableop_20_dense_bias:4
"assignvariableop_21_dense_1_kernel:.
 assignvariableop_22_dense_1_bias:M
?assignvariableop_23_transformer_block_layer_normalization_gamma:L
>assignvariableop_24_transformer_block_layer_normalization_beta:O
Aassignvariableop_25_transformer_block_layer_normalization_1_gamma:N
@assignvariableop_26_transformer_block_layer_normalization_1_beta:#
assignvariableop_27_total: #
assignvariableop_28_count: %
assignvariableop_29_total_1: %
assignvariableop_30_count_1: ;
)assignvariableop_31_adam_dense_2_kernel_m:5
'assignvariableop_32_adam_dense_2_bias_m:;
)assignvariableop_33_adam_dense_3_kernel_m:5
'assignvariableop_34_adam_dense_3_bias_m:_
Lassignvariableop_35_adam_token_and_position_embedding_embedding_embeddings_m:	a
Nassignvariableop_36_adam_token_and_position_embedding_embedding_1_embeddings_m:	d
Nassignvariableop_37_adam_transformer_block_multi_head_attention_query_kernel_m:^
Lassignvariableop_38_adam_transformer_block_multi_head_attention_query_bias_m:b
Lassignvariableop_39_adam_transformer_block_multi_head_attention_key_kernel_m:\
Jassignvariableop_40_adam_transformer_block_multi_head_attention_key_bias_m:d
Nassignvariableop_41_adam_transformer_block_multi_head_attention_value_kernel_m:^
Lassignvariableop_42_adam_transformer_block_multi_head_attention_value_bias_m:o
Yassignvariableop_43_adam_transformer_block_multi_head_attention_attention_output_kernel_m:e
Wassignvariableop_44_adam_transformer_block_multi_head_attention_attention_output_bias_m:9
'assignvariableop_45_adam_dense_kernel_m:3
%assignvariableop_46_adam_dense_bias_m:;
)assignvariableop_47_adam_dense_1_kernel_m:5
'assignvariableop_48_adam_dense_1_bias_m:T
Fassignvariableop_49_adam_transformer_block_layer_normalization_gamma_m:S
Eassignvariableop_50_adam_transformer_block_layer_normalization_beta_m:V
Hassignvariableop_51_adam_transformer_block_layer_normalization_1_gamma_m:U
Gassignvariableop_52_adam_transformer_block_layer_normalization_1_beta_m:;
)assignvariableop_53_adam_dense_2_kernel_v:5
'assignvariableop_54_adam_dense_2_bias_v:;
)assignvariableop_55_adam_dense_3_kernel_v:5
'assignvariableop_56_adam_dense_3_bias_v:_
Lassignvariableop_57_adam_token_and_position_embedding_embedding_embeddings_v:	a
Nassignvariableop_58_adam_token_and_position_embedding_embedding_1_embeddings_v:	d
Nassignvariableop_59_adam_transformer_block_multi_head_attention_query_kernel_v:^
Lassignvariableop_60_adam_transformer_block_multi_head_attention_query_bias_v:b
Lassignvariableop_61_adam_transformer_block_multi_head_attention_key_kernel_v:\
Jassignvariableop_62_adam_transformer_block_multi_head_attention_key_bias_v:d
Nassignvariableop_63_adam_transformer_block_multi_head_attention_value_kernel_v:^
Lassignvariableop_64_adam_transformer_block_multi_head_attention_value_bias_v:o
Yassignvariableop_65_adam_transformer_block_multi_head_attention_attention_output_kernel_v:e
Wassignvariableop_66_adam_transformer_block_multi_head_attention_attention_output_bias_v:9
'assignvariableop_67_adam_dense_kernel_v:3
%assignvariableop_68_adam_dense_bias_v:;
)assignvariableop_69_adam_dense_1_kernel_v:5
'assignvariableop_70_adam_dense_1_bias_v:T
Fassignvariableop_71_adam_transformer_block_layer_normalization_gamma_v:S
Eassignvariableop_72_adam_transformer_block_layer_normalization_beta_v:V
Hassignvariableop_73_adam_transformer_block_layer_normalization_1_gamma_v:U
Gassignvariableop_74_adam_transformer_block_layer_normalization_1_beta_v:
identity_76¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_70¢AssignVariableOp_71¢AssignVariableOp_72¢AssignVariableOp_73¢AssignVariableOp_74¢AssignVariableOp_8¢AssignVariableOp_9¶$
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*Ü#
valueÒ#BÏ#LB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*­
value£B LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Æ
_output_shapes³
°::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*Z
dtypesP
N2L	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_dense_2_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_2_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_3_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_3_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:³
AssignVariableOp_9AssignVariableOpDassignvariableop_9_token_and_position_embedding_embedding_embeddingsIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:¸
AssignVariableOp_10AssignVariableOpGassignvariableop_10_token_and_position_embedding_embedding_1_embeddingsIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:¸
AssignVariableOp_11AssignVariableOpGassignvariableop_11_transformer_block_multi_head_attention_query_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_12AssignVariableOpEassignvariableop_12_transformer_block_multi_head_attention_query_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_13AssignVariableOpEassignvariableop_13_transformer_block_multi_head_attention_key_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:´
AssignVariableOp_14AssignVariableOpCassignvariableop_14_transformer_block_multi_head_attention_key_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:¸
AssignVariableOp_15AssignVariableOpGassignvariableop_15_transformer_block_multi_head_attention_value_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_16AssignVariableOpEassignvariableop_16_transformer_block_multi_head_attention_value_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:Ã
AssignVariableOp_17AssignVariableOpRassignvariableop_17_transformer_block_multi_head_attention_attention_output_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Á
AssignVariableOp_18AssignVariableOpPassignvariableop_18_transformer_block_multi_head_attention_attention_output_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp assignvariableop_19_dense_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOpassignvariableop_20_dense_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp"assignvariableop_21_dense_1_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp assignvariableop_22_dense_1_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_23AssignVariableOp?assignvariableop_23_transformer_block_layer_normalization_gammaIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:¯
AssignVariableOp_24AssignVariableOp>assignvariableop_24_transformer_block_layer_normalization_betaIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:²
AssignVariableOp_25AssignVariableOpAassignvariableop_25_transformer_block_layer_normalization_1_gammaIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_26AssignVariableOp@assignvariableop_26_transformer_block_layer_normalization_1_betaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOpassignvariableop_27_totalIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOpassignvariableop_28_countIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOpassignvariableop_29_total_1Identity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOpassignvariableop_30_count_1Identity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_dense_2_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_dense_2_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp)assignvariableop_33_adam_dense_3_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp'assignvariableop_34_adam_dense_3_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:½
AssignVariableOp_35AssignVariableOpLassignvariableop_35_adam_token_and_position_embedding_embedding_embeddings_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:¿
AssignVariableOp_36AssignVariableOpNassignvariableop_36_adam_token_and_position_embedding_embedding_1_embeddings_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:¿
AssignVariableOp_37AssignVariableOpNassignvariableop_37_adam_transformer_block_multi_head_attention_query_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:½
AssignVariableOp_38AssignVariableOpLassignvariableop_38_adam_transformer_block_multi_head_attention_query_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:½
AssignVariableOp_39AssignVariableOpLassignvariableop_39_adam_transformer_block_multi_head_attention_key_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:»
AssignVariableOp_40AssignVariableOpJassignvariableop_40_adam_transformer_block_multi_head_attention_key_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:¿
AssignVariableOp_41AssignVariableOpNassignvariableop_41_adam_transformer_block_multi_head_attention_value_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:½
AssignVariableOp_42AssignVariableOpLassignvariableop_42_adam_transformer_block_multi_head_attention_value_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:Ê
AssignVariableOp_43AssignVariableOpYassignvariableop_43_adam_transformer_block_multi_head_attention_attention_output_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:È
AssignVariableOp_44AssignVariableOpWassignvariableop_44_adam_transformer_block_multi_head_attention_attention_output_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_45AssignVariableOp'assignvariableop_45_adam_dense_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_46AssignVariableOp%assignvariableop_46_adam_dense_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_47AssignVariableOp)assignvariableop_47_adam_dense_1_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_48AssignVariableOp'assignvariableop_48_adam_dense_1_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:·
AssignVariableOp_49AssignVariableOpFassignvariableop_49_adam_transformer_block_layer_normalization_gamma_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_50AssignVariableOpEassignvariableop_50_adam_transformer_block_layer_normalization_beta_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:¹
AssignVariableOp_51AssignVariableOpHassignvariableop_51_adam_transformer_block_layer_normalization_1_gamma_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:¸
AssignVariableOp_52AssignVariableOpGassignvariableop_52_adam_transformer_block_layer_normalization_1_beta_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_53AssignVariableOp)assignvariableop_53_adam_dense_2_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_54AssignVariableOp'assignvariableop_54_adam_dense_2_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_55AssignVariableOp)assignvariableop_55_adam_dense_3_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_56AssignVariableOp'assignvariableop_56_adam_dense_3_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:½
AssignVariableOp_57AssignVariableOpLassignvariableop_57_adam_token_and_position_embedding_embedding_embeddings_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:¿
AssignVariableOp_58AssignVariableOpNassignvariableop_58_adam_token_and_position_embedding_embedding_1_embeddings_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:¿
AssignVariableOp_59AssignVariableOpNassignvariableop_59_adam_transformer_block_multi_head_attention_query_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:½
AssignVariableOp_60AssignVariableOpLassignvariableop_60_adam_transformer_block_multi_head_attention_query_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:½
AssignVariableOp_61AssignVariableOpLassignvariableop_61_adam_transformer_block_multi_head_attention_key_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:»
AssignVariableOp_62AssignVariableOpJassignvariableop_62_adam_transformer_block_multi_head_attention_key_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:¿
AssignVariableOp_63AssignVariableOpNassignvariableop_63_adam_transformer_block_multi_head_attention_value_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:½
AssignVariableOp_64AssignVariableOpLassignvariableop_64_adam_transformer_block_multi_head_attention_value_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:Ê
AssignVariableOp_65AssignVariableOpYassignvariableop_65_adam_transformer_block_multi_head_attention_attention_output_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:È
AssignVariableOp_66AssignVariableOpWassignvariableop_66_adam_transformer_block_multi_head_attention_attention_output_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_67AssignVariableOp'assignvariableop_67_adam_dense_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_68AssignVariableOp%assignvariableop_68_adam_dense_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_69AssignVariableOp)assignvariableop_69_adam_dense_1_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_70AssignVariableOp'assignvariableop_70_adam_dense_1_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:·
AssignVariableOp_71AssignVariableOpFassignvariableop_71_adam_transformer_block_layer_normalization_gamma_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_72AssignVariableOpEassignvariableop_72_adam_transformer_block_layer_normalization_beta_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:¹
AssignVariableOp_73AssignVariableOpHassignvariableop_73_adam_transformer_block_layer_normalization_1_gamma_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:¸
AssignVariableOp_74AssignVariableOpGassignvariableop_74_adam_transformer_block_layer_normalization_1_beta_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Á
Identity_75Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_76IdentityIdentity_75:output:0^NoOp_1*
T0*
_output_shapes
: ®
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_76Identity_76:output:0*­
_input_shapes
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ë+
Æ	
?__inference_model_layer_call_and_return_conditional_losses_8605
input_14
!token_and_position_embedding_8553:	4
!token_and_position_embedding_8555:	,
transformer_block_8558:(
transformer_block_8560:,
transformer_block_8562:(
transformer_block_8564:,
transformer_block_8566:(
transformer_block_8568:,
transformer_block_8570:$
transformer_block_8572:$
transformer_block_8574:$
transformer_block_8576:(
transformer_block_8578:$
transformer_block_8580:(
transformer_block_8582:$
transformer_block_8584:$
transformer_block_8586:$
transformer_block_8588:
dense_2_8593:
dense_2_8595:
dense_3_8599:
dense_3_8601:
identity¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall¢!dropout_2/StatefulPartitionedCall¢!dropout_3/StatefulPartitionedCall¢4token_and_position_embedding/StatefulPartitionedCall¢)transformer_block/StatefulPartitionedCallÀ
4token_and_position_embedding/StatefulPartitionedCallStatefulPartitionedCallinput_1!token_and_position_embedding_8553!token_and_position_embedding_8555*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *_
fZRX
V__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_7738¶
)transformer_block/StatefulPartitionedCallStatefulPartitionedCall=token_and_position_embedding/StatefulPartitionedCall:output:0transformer_block_8558transformer_block_8560transformer_block_8562transformer_block_8564transformer_block_8566transformer_block_8568transformer_block_8570transformer_block_8572transformer_block_8574transformer_block_8576transformer_block_8578transformer_block_8580transformer_block_8582transformer_block_8584transformer_block_8586transformer_block_8588*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_transformer_block_layer_call_and_return_conditional_losses_8248
(global_average_pooling1d/PartitionedCallPartitionedCall2transformer_block/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_7704ó
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_8065
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_2_8593dense_2_8595*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_7924
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_3_layer_call_and_return_conditional_losses_8032
dense_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0dense_3_8599dense_3_8601*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_7948w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
NoOpNoOp ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall5^token_and_position_embedding/StatefulPartitionedCall*^transformer_block/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2l
4token_and_position_embedding/StatefulPartitionedCall4token_and_position_embedding/StatefulPartitionedCall2V
)transformer_block/StatefulPartitionedCall)transformer_block/StatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
é
©
;__inference_token_and_position_embedding_layer_call_fn_9126
x
unknown:	
	unknown_0:	
identity¢StatefulPartitionedCallë
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *_
fZRX
V__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_7738t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
ñ	
b
C__inference_dropout_2_layer_call_and_return_conditional_losses_9529

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦Ã
â
K__inference_transformer_block_layer_call_and_return_conditional_losses_8248

inputsV
@multi_head_attention_query_einsum_einsum_readvariableop_resource:H
6multi_head_attention_query_add_readvariableop_resource:T
>multi_head_attention_key_einsum_einsum_readvariableop_resource:F
4multi_head_attention_key_add_readvariableop_resource:V
@multi_head_attention_value_einsum_einsum_readvariableop_resource:H
6multi_head_attention_value_add_readvariableop_resource:a
Kmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource:O
Amulti_head_attention_attention_output_add_readvariableop_resource:G
9layer_normalization_batchnorm_mul_readvariableop_resource:C
5layer_normalization_batchnorm_readvariableop_resource:D
2sequential_dense_tensordot_readvariableop_resource:>
0sequential_dense_biasadd_readvariableop_resource:F
4sequential_dense_1_tensordot_readvariableop_resource:@
2sequential_dense_1_biasadd_readvariableop_resource:I
;layer_normalization_1_batchnorm_mul_readvariableop_resource:E
7layer_normalization_1_batchnorm_readvariableop_resource:
identity¢,layer_normalization/batchnorm/ReadVariableOp¢0layer_normalization/batchnorm/mul/ReadVariableOp¢.layer_normalization_1/batchnorm/ReadVariableOp¢2layer_normalization_1/batchnorm/mul/ReadVariableOp¢8multi_head_attention/attention_output/add/ReadVariableOp¢Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp¢+multi_head_attention/key/add/ReadVariableOp¢5multi_head_attention/key/einsum/Einsum/ReadVariableOp¢-multi_head_attention/query/add/ReadVariableOp¢7multi_head_attention/query/einsum/Einsum/ReadVariableOp¢-multi_head_attention/value/add/ReadVariableOp¢7multi_head_attention/value/einsum/Einsum/ReadVariableOp¢'sequential/dense/BiasAdd/ReadVariableOp¢)sequential/dense/Tensordot/ReadVariableOp¢)sequential/dense_1/BiasAdd/ReadVariableOp¢+sequential/dense_1/Tensordot/ReadVariableOp¼
7multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0Ü
(multi_head_attention/query/einsum/EinsumEinsuminputs?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equationabc,cde->abde¤
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0Ì
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0Ø
&multi_head_attention/key/einsum/EinsumEinsuminputs=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equationabc,cde->abde 
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0Æ
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0Ü
(multi_head_attention/value/einsum/EinsumEinsuminputs?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equationabc,cde->abde¤
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0Ì
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ìQ>£
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
"multi_head_attention/einsum/EinsumEinsum multi_head_attention/key/add:z:0multi_head_attention/Mul:z:0*
N*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equationaecd,abcd->acbe
$multi_head_attention/softmax/SoftmaxSoftmax+multi_head_attention/einsum/Einsum:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿå
$multi_head_attention/einsum_1/EinsumEinsum.multi_head_attention/softmax/Softmax:softmax:0"multi_head_attention/value/add:z:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equationacbe,aecd->abcdÒ
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equationabcd,cde->abe¶
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0é
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8? 
dropout/dropout/MulMul-multi_head_attention/attention_output/add:z:0dropout/dropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
dropout/dropout/ShapeShape-multi_head_attention/attention_output/add:z:0*
T0*
_output_shapes
:¡
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ã
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
addAddV2inputsdropout/dropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:¶
 layer_normalization/moments/meanMeanadd:z:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
-layer_normalization/moments/SquaredDifferenceSquaredDifferenceadd:z:01layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:è
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75¾
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Â
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#layer_normalization/batchnorm/mul_1Muladd:z:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0¾
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)sequential/dense/Tensordot/ReadVariableOpReadVariableOp2sequential_dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0i
sequential/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:p
sequential/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       w
 sequential/dense/Tensordot/ShapeShape'layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:j
(sequential/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ÿ
#sequential/dense/Tensordot/GatherV2GatherV2)sequential/dense/Tensordot/Shape:output:0(sequential/dense/Tensordot/free:output:01sequential/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
*sequential/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
%sequential/dense/Tensordot/GatherV2_1GatherV2)sequential/dense/Tensordot/Shape:output:0(sequential/dense/Tensordot/axes:output:03sequential/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:j
 sequential/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
sequential/dense/Tensordot/ProdProd,sequential/dense/Tensordot/GatherV2:output:0)sequential/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: l
"sequential/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: §
!sequential/dense/Tensordot/Prod_1Prod.sequential/dense/Tensordot/GatherV2_1:output:0+sequential/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: h
&sequential/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : à
!sequential/dense/Tensordot/concatConcatV2(sequential/dense/Tensordot/free:output:0(sequential/dense/Tensordot/axes:output:0/sequential/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:¬
 sequential/dense/Tensordot/stackPack(sequential/dense/Tensordot/Prod:output:0*sequential/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:½
$sequential/dense/Tensordot/transpose	Transpose'layer_normalization/batchnorm/add_1:z:0*sequential/dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
"sequential/dense/Tensordot/ReshapeReshape(sequential/dense/Tensordot/transpose:y:0)sequential/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ½
!sequential/dense/Tensordot/MatMulMatMul+sequential/dense/Tensordot/Reshape:output:01sequential/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"sequential/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:j
(sequential/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ë
#sequential/dense/Tensordot/concat_1ConcatV2,sequential/dense/Tensordot/GatherV2:output:0+sequential/dense/Tensordot/Const_2:output:01sequential/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:·
sequential/dense/TensordotReshape+sequential/dense/Tensordot/MatMul:product:0,sequential/dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0°
sequential/dense/BiasAddBiasAdd#sequential/dense/Tensordot:output:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
+sequential/dense_1/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_1_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0k
!sequential/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:r
!sequential/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       u
"sequential/dense_1/Tensordot/ShapeShape#sequential/dense/Relu:activations:0*
T0*
_output_shapes
:l
*sequential/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
%sequential/dense_1/Tensordot/GatherV2GatherV2+sequential/dense_1/Tensordot/Shape:output:0*sequential/dense_1/Tensordot/free:output:03sequential/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
,sequential/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
'sequential/dense_1/Tensordot/GatherV2_1GatherV2+sequential/dense_1/Tensordot/Shape:output:0*sequential/dense_1/Tensordot/axes:output:05sequential/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
"sequential/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: §
!sequential/dense_1/Tensordot/ProdProd.sequential/dense_1/Tensordot/GatherV2:output:0+sequential/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$sequential/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ­
#sequential/dense_1/Tensordot/Prod_1Prod0sequential/dense_1/Tensordot/GatherV2_1:output:0-sequential/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(sequential/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : è
#sequential/dense_1/Tensordot/concatConcatV2*sequential/dense_1/Tensordot/free:output:0*sequential/dense_1/Tensordot/axes:output:01sequential/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:²
"sequential/dense_1/Tensordot/stackPack*sequential/dense_1/Tensordot/Prod:output:0,sequential/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:½
&sequential/dense_1/Tensordot/transpose	Transpose#sequential/dense/Relu:activations:0,sequential/dense_1/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
$sequential/dense_1/Tensordot/ReshapeReshape*sequential/dense_1/Tensordot/transpose:y:0+sequential/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÃ
#sequential/dense_1/Tensordot/MatMulMatMul-sequential/dense_1/Tensordot/Reshape:output:03sequential/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
$sequential/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:l
*sequential/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ó
%sequential/dense_1/Tensordot/concat_1ConcatV2.sequential/dense_1/Tensordot/GatherV2:output:0-sequential/dense_1/Tensordot/Const_2:output:03sequential/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:½
sequential/dense_1/TensordotReshape-sequential/dense_1/Tensordot/MatMul:product:0.sequential/dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¶
sequential/dense_1/BiasAddBiasAdd%sequential/dense_1/Tensordot:output:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
dropout_1/dropout/MulMul#sequential/dense_1/BiasAdd:output:0 dropout_1/dropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
dropout_1/dropout/ShapeShape#sequential/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:¥
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=É
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
add_1AddV2'layer_normalization/batchnorm/add_1:z:0dropout_1/dropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:¼
"layer_normalization_1/moments/meanMean	add_1:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ»
/layer_normalization_1/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:î
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Ä
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0È
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%layer_normalization_1/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Ä
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
IdentityIdentity)layer_normalization_1/batchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿø
NoOpNoOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp*^sequential/dense/Tensordot/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp,^sequential/dense_1/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2t
8multi_head_attention/attention_output/add/ReadVariableOp8multi_head_attention/attention_output/add/ReadVariableOp2
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpBmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp2Z
+multi_head_attention/key/add/ReadVariableOp+multi_head_attention/key/add/ReadVariableOp2n
5multi_head_attention/key/einsum/Einsum/ReadVariableOp5multi_head_attention/key/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/query/add/ReadVariableOp-multi_head_attention/query/add/ReadVariableOp2r
7multi_head_attention/query/einsum/Einsum/ReadVariableOp7multi_head_attention/query/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/value/add/ReadVariableOp-multi_head_attention/value/add/ReadVariableOp2r
7multi_head_attention/value/einsum/Einsum/ReadVariableOp7multi_head_attention/value/einsum/Einsum/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2V
)sequential/dense/Tensordot/ReadVariableOp)sequential/dense/Tensordot/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2Z
+sequential/dense_1/Tensordot/ReadVariableOp+sequential/dense_1/Tensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¥
ö
?__inference_dense_layer_call_and_return_conditional_losses_7539

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ö
a
C__inference_dropout_3_layer_call_and_return_conditional_losses_7935

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í
a
(__inference_dropout_3_layer_call_fn_9559

inputs
identity¢StatefulPartitionedCall¾
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_3_layer_call_and_return_conditional_losses_8032o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ
Ó
$__inference_model_layer_call_fn_8002
input_1
unknown:	
	unknown_0:	
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_7955o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
ô(
ý
?__inference_model_layer_call_and_return_conditional_losses_7955

inputs4
!token_and_position_embedding_7739:	4
!token_and_position_embedding_7741:	,
transformer_block_7872:(
transformer_block_7874:,
transformer_block_7876:(
transformer_block_7878:,
transformer_block_7880:(
transformer_block_7882:,
transformer_block_7884:$
transformer_block_7886:$
transformer_block_7888:$
transformer_block_7890:(
transformer_block_7892:$
transformer_block_7894:(
transformer_block_7896:$
transformer_block_7898:$
transformer_block_7900:$
transformer_block_7902:
dense_2_7925:
dense_2_7927:
dense_3_7949:
dense_3_7951:
identity¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall¢4token_and_position_embedding/StatefulPartitionedCall¢)transformer_block/StatefulPartitionedCall¿
4token_and_position_embedding/StatefulPartitionedCallStatefulPartitionedCallinputs!token_and_position_embedding_7739!token_and_position_embedding_7741*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *_
fZRX
V__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_7738¶
)transformer_block/StatefulPartitionedCallStatefulPartitionedCall=token_and_position_embedding/StatefulPartitionedCall:output:0transformer_block_7872transformer_block_7874transformer_block_7876transformer_block_7878transformer_block_7880transformer_block_7882transformer_block_7884transformer_block_7886transformer_block_7888transformer_block_7890transformer_block_7892transformer_block_7894transformer_block_7896transformer_block_7898transformer_block_7900transformer_block_7902*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_transformer_block_layer_call_and_return_conditional_losses_7871
(global_average_pooling1d/PartitionedCallPartitionedCall2transformer_block/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_7704ã
dropout_2/PartitionedCallPartitionedCall1global_average_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_7911
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_2_7925dense_2_7927*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_7924Ú
dropout_3/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_3_layer_call_and_return_conditional_losses_7935
dense_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_3_7949dense_3_7951*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_7948w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿí
NoOpNoOp ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall5^token_and_position_embedding/StatefulPartitionedCall*^transformer_block/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2l
4token_and_position_embedding/StatefulPartitionedCall4token_and_position_embedding/StatefulPartitionedCall2V
)transformer_block/StatefulPartitionedCall)transformer_block/StatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¼

&__inference_dense_2_layer_call_fn_9538

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÖ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_7924o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
þ
å)
__inference__traced_save_10063
file_prefix-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopP
Lsavev2_token_and_position_embedding_embedding_embeddings_read_readvariableopR
Nsavev2_token_and_position_embedding_embedding_1_embeddings_read_readvariableopR
Nsavev2_transformer_block_multi_head_attention_query_kernel_read_readvariableopP
Lsavev2_transformer_block_multi_head_attention_query_bias_read_readvariableopP
Lsavev2_transformer_block_multi_head_attention_key_kernel_read_readvariableopN
Jsavev2_transformer_block_multi_head_attention_key_bias_read_readvariableopR
Nsavev2_transformer_block_multi_head_attention_value_kernel_read_readvariableopP
Lsavev2_transformer_block_multi_head_attention_value_bias_read_readvariableop]
Ysavev2_transformer_block_multi_head_attention_attention_output_kernel_read_readvariableop[
Wsavev2_transformer_block_multi_head_attention_attention_output_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableopJ
Fsavev2_transformer_block_layer_normalization_gamma_read_readvariableopI
Esavev2_transformer_block_layer_normalization_beta_read_readvariableopL
Hsavev2_transformer_block_layer_normalization_1_gamma_read_readvariableopK
Gsavev2_transformer_block_layer_normalization_1_beta_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop4
0savev2_adam_dense_3_kernel_m_read_readvariableop2
.savev2_adam_dense_3_bias_m_read_readvariableopW
Ssavev2_adam_token_and_position_embedding_embedding_embeddings_m_read_readvariableopY
Usavev2_adam_token_and_position_embedding_embedding_1_embeddings_m_read_readvariableopY
Usavev2_adam_transformer_block_multi_head_attention_query_kernel_m_read_readvariableopW
Ssavev2_adam_transformer_block_multi_head_attention_query_bias_m_read_readvariableopW
Ssavev2_adam_transformer_block_multi_head_attention_key_kernel_m_read_readvariableopU
Qsavev2_adam_transformer_block_multi_head_attention_key_bias_m_read_readvariableopY
Usavev2_adam_transformer_block_multi_head_attention_value_kernel_m_read_readvariableopW
Ssavev2_adam_transformer_block_multi_head_attention_value_bias_m_read_readvariableopd
`savev2_adam_transformer_block_multi_head_attention_attention_output_kernel_m_read_readvariableopb
^savev2_adam_transformer_block_multi_head_attention_attention_output_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableopQ
Msavev2_adam_transformer_block_layer_normalization_gamma_m_read_readvariableopP
Lsavev2_adam_transformer_block_layer_normalization_beta_m_read_readvariableopS
Osavev2_adam_transformer_block_layer_normalization_1_gamma_m_read_readvariableopR
Nsavev2_adam_transformer_block_layer_normalization_1_beta_m_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop4
0savev2_adam_dense_3_kernel_v_read_readvariableop2
.savev2_adam_dense_3_bias_v_read_readvariableopW
Ssavev2_adam_token_and_position_embedding_embedding_embeddings_v_read_readvariableopY
Usavev2_adam_token_and_position_embedding_embedding_1_embeddings_v_read_readvariableopY
Usavev2_adam_transformer_block_multi_head_attention_query_kernel_v_read_readvariableopW
Ssavev2_adam_transformer_block_multi_head_attention_query_bias_v_read_readvariableopW
Ssavev2_adam_transformer_block_multi_head_attention_key_kernel_v_read_readvariableopU
Qsavev2_adam_transformer_block_multi_head_attention_key_bias_v_read_readvariableopY
Usavev2_adam_transformer_block_multi_head_attention_value_kernel_v_read_readvariableopW
Ssavev2_adam_transformer_block_multi_head_attention_value_bias_v_read_readvariableopd
`savev2_adam_transformer_block_multi_head_attention_attention_output_kernel_v_read_readvariableopb
^savev2_adam_transformer_block_multi_head_attention_attention_output_bias_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableopQ
Msavev2_adam_transformer_block_layer_normalization_gamma_v_read_readvariableopP
Lsavev2_adam_transformer_block_layer_normalization_beta_v_read_readvariableopS
Osavev2_adam_transformer_block_layer_normalization_1_gamma_v_read_readvariableopR
Nsavev2_adam_transformer_block_layer_normalization_1_beta_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ³$
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*Ü#
valueÒ#BÏ#LB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*­
value£B LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B É(
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopLsavev2_token_and_position_embedding_embedding_embeddings_read_readvariableopNsavev2_token_and_position_embedding_embedding_1_embeddings_read_readvariableopNsavev2_transformer_block_multi_head_attention_query_kernel_read_readvariableopLsavev2_transformer_block_multi_head_attention_query_bias_read_readvariableopLsavev2_transformer_block_multi_head_attention_key_kernel_read_readvariableopJsavev2_transformer_block_multi_head_attention_key_bias_read_readvariableopNsavev2_transformer_block_multi_head_attention_value_kernel_read_readvariableopLsavev2_transformer_block_multi_head_attention_value_bias_read_readvariableopYsavev2_transformer_block_multi_head_attention_attention_output_kernel_read_readvariableopWsavev2_transformer_block_multi_head_attention_attention_output_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableopFsavev2_transformer_block_layer_normalization_gamma_read_readvariableopEsavev2_transformer_block_layer_normalization_beta_read_readvariableopHsavev2_transformer_block_layer_normalization_1_gamma_read_readvariableopGsavev2_transformer_block_layer_normalization_1_beta_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop0savev2_adam_dense_3_kernel_m_read_readvariableop.savev2_adam_dense_3_bias_m_read_readvariableopSsavev2_adam_token_and_position_embedding_embedding_embeddings_m_read_readvariableopUsavev2_adam_token_and_position_embedding_embedding_1_embeddings_m_read_readvariableopUsavev2_adam_transformer_block_multi_head_attention_query_kernel_m_read_readvariableopSsavev2_adam_transformer_block_multi_head_attention_query_bias_m_read_readvariableopSsavev2_adam_transformer_block_multi_head_attention_key_kernel_m_read_readvariableopQsavev2_adam_transformer_block_multi_head_attention_key_bias_m_read_readvariableopUsavev2_adam_transformer_block_multi_head_attention_value_kernel_m_read_readvariableopSsavev2_adam_transformer_block_multi_head_attention_value_bias_m_read_readvariableop`savev2_adam_transformer_block_multi_head_attention_attention_output_kernel_m_read_readvariableop^savev2_adam_transformer_block_multi_head_attention_attention_output_bias_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableopMsavev2_adam_transformer_block_layer_normalization_gamma_m_read_readvariableopLsavev2_adam_transformer_block_layer_normalization_beta_m_read_readvariableopOsavev2_adam_transformer_block_layer_normalization_1_gamma_m_read_readvariableopNsavev2_adam_transformer_block_layer_normalization_1_beta_m_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableop0savev2_adam_dense_3_kernel_v_read_readvariableop.savev2_adam_dense_3_bias_v_read_readvariableopSsavev2_adam_token_and_position_embedding_embedding_embeddings_v_read_readvariableopUsavev2_adam_token_and_position_embedding_embedding_1_embeddings_v_read_readvariableopUsavev2_adam_transformer_block_multi_head_attention_query_kernel_v_read_readvariableopSsavev2_adam_transformer_block_multi_head_attention_query_bias_v_read_readvariableopSsavev2_adam_transformer_block_multi_head_attention_key_kernel_v_read_readvariableopQsavev2_adam_transformer_block_multi_head_attention_key_bias_v_read_readvariableopUsavev2_adam_transformer_block_multi_head_attention_value_kernel_v_read_readvariableopSsavev2_adam_transformer_block_multi_head_attention_value_bias_v_read_readvariableop`savev2_adam_transformer_block_multi_head_attention_attention_output_kernel_v_read_readvariableop^savev2_adam_transformer_block_multi_head_attention_attention_output_bias_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableopMsavev2_adam_transformer_block_layer_normalization_gamma_v_read_readvariableopLsavev2_adam_transformer_block_layer_normalization_beta_v_read_readvariableopOsavev2_adam_transformer_block_layer_normalization_1_gamma_v_read_readvariableopNsavev2_adam_transformer_block_layer_normalization_1_beta_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *Z
dtypesP
N2L	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*
_input_shapes÷
ô: ::::: : : : : :	:	::::::::::::::::: : : : :::::	:	:::::::::::::::::::::	:	::::::::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :%
!

_output_shapes
:	:%!

_output_shapes
:	:($
"
_output_shapes
::$ 

_output_shapes

::($
"
_output_shapes
::$ 

_output_shapes

::($
"
_output_shapes
::$ 

_output_shapes

::($
"
_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$  

_output_shapes

:: !

_output_shapes
::$" 

_output_shapes

:: #

_output_shapes
::%$!

_output_shapes
:	:%%!

_output_shapes
:	:(&$
"
_output_shapes
::$' 

_output_shapes

::(($
"
_output_shapes
::$) 

_output_shapes

::(*$
"
_output_shapes
::$+ 

_output_shapes

::(,$
"
_output_shapes
:: -

_output_shapes
::$. 

_output_shapes

:: /

_output_shapes
::$0 

_output_shapes

:: 1

_output_shapes
:: 2

_output_shapes
:: 3

_output_shapes
:: 4

_output_shapes
:: 5

_output_shapes
::$6 

_output_shapes

:: 7

_output_shapes
::$8 

_output_shapes

:: 9

_output_shapes
::%:!

_output_shapes
:	:%;!

_output_shapes
:	:(<$
"
_output_shapes
::$= 

_output_shapes

::(>$
"
_output_shapes
::$? 

_output_shapes

::(@$
"
_output_shapes
::$A 

_output_shapes

::(B$
"
_output_shapes
:: C

_output_shapes
::$D 

_output_shapes

:: E

_output_shapes
::$F 

_output_shapes

:: G

_output_shapes
:: H

_output_shapes
:: I

_output_shapes
:: J

_output_shapes
:: K

_output_shapes
::L

_output_shapes
: 

n
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_9502

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
è+
Å	
?__inference_model_layer_call_and_return_conditional_losses_8399

inputs4
!token_and_position_embedding_8347:	4
!token_and_position_embedding_8349:	,
transformer_block_8352:(
transformer_block_8354:,
transformer_block_8356:(
transformer_block_8358:,
transformer_block_8360:(
transformer_block_8362:,
transformer_block_8364:$
transformer_block_8366:$
transformer_block_8368:$
transformer_block_8370:(
transformer_block_8372:$
transformer_block_8374:(
transformer_block_8376:$
transformer_block_8378:$
transformer_block_8380:$
transformer_block_8382:
dense_2_8387:
dense_2_8389:
dense_3_8393:
dense_3_8395:
identity¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall¢!dropout_2/StatefulPartitionedCall¢!dropout_3/StatefulPartitionedCall¢4token_and_position_embedding/StatefulPartitionedCall¢)transformer_block/StatefulPartitionedCall¿
4token_and_position_embedding/StatefulPartitionedCallStatefulPartitionedCallinputs!token_and_position_embedding_8347!token_and_position_embedding_8349*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *_
fZRX
V__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_7738¶
)transformer_block/StatefulPartitionedCallStatefulPartitionedCall=token_and_position_embedding/StatefulPartitionedCall:output:0transformer_block_8352transformer_block_8354transformer_block_8356transformer_block_8358transformer_block_8360transformer_block_8362transformer_block_8364transformer_block_8366transformer_block_8368transformer_block_8370transformer_block_8372transformer_block_8374transformer_block_8376transformer_block_8378transformer_block_8380transformer_block_8382*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_transformer_block_layer_call_and_return_conditional_losses_8248
(global_average_pooling1d/PartitionedCallPartitionedCall2transformer_block/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_7704ó
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_8065
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_2_8387dense_2_8389*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_7924
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_3_layer_call_and_return_conditional_losses_8032
dense_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0dense_3_8393dense_3_8395*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_7948w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
NoOpNoOp ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall5^token_and_position_embedding/StatefulPartitionedCall*^transformer_block/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2l
4token_and_position_embedding/StatefulPartitionedCall4token_and_position_embedding/StatefulPartitionedCall2V
)transformer_block/StatefulPartitionedCall)transformer_block/StatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦Ã
â
K__inference_transformer_block_layer_call_and_return_conditional_losses_9491

inputsV
@multi_head_attention_query_einsum_einsum_readvariableop_resource:H
6multi_head_attention_query_add_readvariableop_resource:T
>multi_head_attention_key_einsum_einsum_readvariableop_resource:F
4multi_head_attention_key_add_readvariableop_resource:V
@multi_head_attention_value_einsum_einsum_readvariableop_resource:H
6multi_head_attention_value_add_readvariableop_resource:a
Kmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource:O
Amulti_head_attention_attention_output_add_readvariableop_resource:G
9layer_normalization_batchnorm_mul_readvariableop_resource:C
5layer_normalization_batchnorm_readvariableop_resource:D
2sequential_dense_tensordot_readvariableop_resource:>
0sequential_dense_biasadd_readvariableop_resource:F
4sequential_dense_1_tensordot_readvariableop_resource:@
2sequential_dense_1_biasadd_readvariableop_resource:I
;layer_normalization_1_batchnorm_mul_readvariableop_resource:E
7layer_normalization_1_batchnorm_readvariableop_resource:
identity¢,layer_normalization/batchnorm/ReadVariableOp¢0layer_normalization/batchnorm/mul/ReadVariableOp¢.layer_normalization_1/batchnorm/ReadVariableOp¢2layer_normalization_1/batchnorm/mul/ReadVariableOp¢8multi_head_attention/attention_output/add/ReadVariableOp¢Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp¢+multi_head_attention/key/add/ReadVariableOp¢5multi_head_attention/key/einsum/Einsum/ReadVariableOp¢-multi_head_attention/query/add/ReadVariableOp¢7multi_head_attention/query/einsum/Einsum/ReadVariableOp¢-multi_head_attention/value/add/ReadVariableOp¢7multi_head_attention/value/einsum/Einsum/ReadVariableOp¢'sequential/dense/BiasAdd/ReadVariableOp¢)sequential/dense/Tensordot/ReadVariableOp¢)sequential/dense_1/BiasAdd/ReadVariableOp¢+sequential/dense_1/Tensordot/ReadVariableOp¼
7multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0Ü
(multi_head_attention/query/einsum/EinsumEinsuminputs?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equationabc,cde->abde¤
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0Ì
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0Ø
&multi_head_attention/key/einsum/EinsumEinsuminputs=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equationabc,cde->abde 
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0Æ
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0Ü
(multi_head_attention/value/einsum/EinsumEinsuminputs?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equationabc,cde->abde¤
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0Ì
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ìQ>£
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
"multi_head_attention/einsum/EinsumEinsum multi_head_attention/key/add:z:0multi_head_attention/Mul:z:0*
N*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equationaecd,abcd->acbe
$multi_head_attention/softmax/SoftmaxSoftmax+multi_head_attention/einsum/Einsum:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿå
$multi_head_attention/einsum_1/EinsumEinsum.multi_head_attention/softmax/Softmax:softmax:0"multi_head_attention/value/add:z:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equationacbe,aecd->abcdÒ
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equationabcd,cde->abe¶
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0é
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8? 
dropout/dropout/MulMul-multi_head_attention/attention_output/add:z:0dropout/dropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
dropout/dropout/ShapeShape-multi_head_attention/attention_output/add:z:0*
T0*
_output_shapes
:¡
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ã
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
addAddV2inputsdropout/dropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:¶
 layer_normalization/moments/meanMeanadd:z:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
-layer_normalization/moments/SquaredDifferenceSquaredDifferenceadd:z:01layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:è
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75¾
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Â
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#layer_normalization/batchnorm/mul_1Muladd:z:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0¾
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)sequential/dense/Tensordot/ReadVariableOpReadVariableOp2sequential_dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0i
sequential/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:p
sequential/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       w
 sequential/dense/Tensordot/ShapeShape'layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:j
(sequential/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ÿ
#sequential/dense/Tensordot/GatherV2GatherV2)sequential/dense/Tensordot/Shape:output:0(sequential/dense/Tensordot/free:output:01sequential/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
*sequential/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
%sequential/dense/Tensordot/GatherV2_1GatherV2)sequential/dense/Tensordot/Shape:output:0(sequential/dense/Tensordot/axes:output:03sequential/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:j
 sequential/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
sequential/dense/Tensordot/ProdProd,sequential/dense/Tensordot/GatherV2:output:0)sequential/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: l
"sequential/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: §
!sequential/dense/Tensordot/Prod_1Prod.sequential/dense/Tensordot/GatherV2_1:output:0+sequential/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: h
&sequential/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : à
!sequential/dense/Tensordot/concatConcatV2(sequential/dense/Tensordot/free:output:0(sequential/dense/Tensordot/axes:output:0/sequential/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:¬
 sequential/dense/Tensordot/stackPack(sequential/dense/Tensordot/Prod:output:0*sequential/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:½
$sequential/dense/Tensordot/transpose	Transpose'layer_normalization/batchnorm/add_1:z:0*sequential/dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
"sequential/dense/Tensordot/ReshapeReshape(sequential/dense/Tensordot/transpose:y:0)sequential/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ½
!sequential/dense/Tensordot/MatMulMatMul+sequential/dense/Tensordot/Reshape:output:01sequential/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"sequential/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:j
(sequential/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ë
#sequential/dense/Tensordot/concat_1ConcatV2,sequential/dense/Tensordot/GatherV2:output:0+sequential/dense/Tensordot/Const_2:output:01sequential/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:·
sequential/dense/TensordotReshape+sequential/dense/Tensordot/MatMul:product:0,sequential/dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0°
sequential/dense/BiasAddBiasAdd#sequential/dense/Tensordot:output:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
+sequential/dense_1/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_1_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0k
!sequential/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:r
!sequential/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       u
"sequential/dense_1/Tensordot/ShapeShape#sequential/dense/Relu:activations:0*
T0*
_output_shapes
:l
*sequential/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
%sequential/dense_1/Tensordot/GatherV2GatherV2+sequential/dense_1/Tensordot/Shape:output:0*sequential/dense_1/Tensordot/free:output:03sequential/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
,sequential/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
'sequential/dense_1/Tensordot/GatherV2_1GatherV2+sequential/dense_1/Tensordot/Shape:output:0*sequential/dense_1/Tensordot/axes:output:05sequential/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
"sequential/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: §
!sequential/dense_1/Tensordot/ProdProd.sequential/dense_1/Tensordot/GatherV2:output:0+sequential/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$sequential/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ­
#sequential/dense_1/Tensordot/Prod_1Prod0sequential/dense_1/Tensordot/GatherV2_1:output:0-sequential/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(sequential/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : è
#sequential/dense_1/Tensordot/concatConcatV2*sequential/dense_1/Tensordot/free:output:0*sequential/dense_1/Tensordot/axes:output:01sequential/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:²
"sequential/dense_1/Tensordot/stackPack*sequential/dense_1/Tensordot/Prod:output:0,sequential/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:½
&sequential/dense_1/Tensordot/transpose	Transpose#sequential/dense/Relu:activations:0,sequential/dense_1/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
$sequential/dense_1/Tensordot/ReshapeReshape*sequential/dense_1/Tensordot/transpose:y:0+sequential/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÃ
#sequential/dense_1/Tensordot/MatMulMatMul-sequential/dense_1/Tensordot/Reshape:output:03sequential/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
$sequential/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:l
*sequential/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ó
%sequential/dense_1/Tensordot/concat_1ConcatV2.sequential/dense_1/Tensordot/GatherV2:output:0-sequential/dense_1/Tensordot/Const_2:output:03sequential/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:½
sequential/dense_1/TensordotReshape-sequential/dense_1/Tensordot/MatMul:product:0.sequential/dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¶
sequential/dense_1/BiasAddBiasAdd%sequential/dense_1/Tensordot:output:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
dropout_1/dropout/MulMul#sequential/dense_1/BiasAdd:output:0 dropout_1/dropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
dropout_1/dropout/ShapeShape#sequential/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:¥
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=É
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
add_1AddV2'layer_normalization/batchnorm/add_1:z:0dropout_1/dropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:¼
"layer_normalization_1/moments/meanMean	add_1:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ»
/layer_normalization_1/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:î
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Ä
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0È
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%layer_normalization_1/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Ä
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
IdentityIdentity)layer_normalization_1/batchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿø
NoOpNoOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp*^sequential/dense/Tensordot/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp,^sequential/dense_1/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2t
8multi_head_attention/attention_output/add/ReadVariableOp8multi_head_attention/attention_output/add/ReadVariableOp2
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpBmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp2Z
+multi_head_attention/key/add/ReadVariableOp+multi_head_attention/key/add/ReadVariableOp2n
5multi_head_attention/key/einsum/Einsum/ReadVariableOp5multi_head_attention/key/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/query/add/ReadVariableOp-multi_head_attention/query/add/ReadVariableOp2r
7multi_head_attention/query/einsum/Einsum/ReadVariableOp7multi_head_attention/query/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/value/add/ReadVariableOp-multi_head_attention/value/add/ReadVariableOp2r
7multi_head_attention/value/einsum/Einsum/ReadVariableOp7multi_head_attention/value/einsum/Einsum/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2V
)sequential/dense/Tensordot/ReadVariableOp)sequential/dense/Tensordot/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2Z
+sequential/dense_1/Tensordot/ReadVariableOp+sequential/dense_1/Tensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¼

&__inference_dense_3_layer_call_fn_9585

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÖ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_7948o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


ò
A__inference_dense_2_layer_call_and_return_conditional_losses_9549

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

n
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_7704

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

D
(__inference_dropout_3_layer_call_fn_9554

inputs
identity®
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_3_layer_call_and_return_conditional_losses_7935`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ«
ý
__inference__wrapped_model_7501
input_1W
Dmodel_token_and_position_embedding_embedding_1_embedding_lookup_7347:	U
Bmodel_token_and_position_embedding_embedding_embedding_lookup_7353:	n
Xmodel_transformer_block_multi_head_attention_query_einsum_einsum_readvariableop_resource:`
Nmodel_transformer_block_multi_head_attention_query_add_readvariableop_resource:l
Vmodel_transformer_block_multi_head_attention_key_einsum_einsum_readvariableop_resource:^
Lmodel_transformer_block_multi_head_attention_key_add_readvariableop_resource:n
Xmodel_transformer_block_multi_head_attention_value_einsum_einsum_readvariableop_resource:`
Nmodel_transformer_block_multi_head_attention_value_add_readvariableop_resource:y
cmodel_transformer_block_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource:g
Ymodel_transformer_block_multi_head_attention_attention_output_add_readvariableop_resource:_
Qmodel_transformer_block_layer_normalization_batchnorm_mul_readvariableop_resource:[
Mmodel_transformer_block_layer_normalization_batchnorm_readvariableop_resource:\
Jmodel_transformer_block_sequential_dense_tensordot_readvariableop_resource:V
Hmodel_transformer_block_sequential_dense_biasadd_readvariableop_resource:^
Lmodel_transformer_block_sequential_dense_1_tensordot_readvariableop_resource:X
Jmodel_transformer_block_sequential_dense_1_biasadd_readvariableop_resource:a
Smodel_transformer_block_layer_normalization_1_batchnorm_mul_readvariableop_resource:]
Omodel_transformer_block_layer_normalization_1_batchnorm_readvariableop_resource:>
,model_dense_2_matmul_readvariableop_resource:;
-model_dense_2_biasadd_readvariableop_resource:>
,model_dense_3_matmul_readvariableop_resource:;
-model_dense_3_biasadd_readvariableop_resource:
identity¢$model/dense_2/BiasAdd/ReadVariableOp¢#model/dense_2/MatMul/ReadVariableOp¢$model/dense_3/BiasAdd/ReadVariableOp¢#model/dense_3/MatMul/ReadVariableOp¢=model/token_and_position_embedding/embedding/embedding_lookup¢?model/token_and_position_embedding/embedding_1/embedding_lookup¢Dmodel/transformer_block/layer_normalization/batchnorm/ReadVariableOp¢Hmodel/transformer_block/layer_normalization/batchnorm/mul/ReadVariableOp¢Fmodel/transformer_block/layer_normalization_1/batchnorm/ReadVariableOp¢Jmodel/transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp¢Pmodel/transformer_block/multi_head_attention/attention_output/add/ReadVariableOp¢Zmodel/transformer_block/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp¢Cmodel/transformer_block/multi_head_attention/key/add/ReadVariableOp¢Mmodel/transformer_block/multi_head_attention/key/einsum/Einsum/ReadVariableOp¢Emodel/transformer_block/multi_head_attention/query/add/ReadVariableOp¢Omodel/transformer_block/multi_head_attention/query/einsum/Einsum/ReadVariableOp¢Emodel/transformer_block/multi_head_attention/value/add/ReadVariableOp¢Omodel/transformer_block/multi_head_attention/value/einsum/Einsum/ReadVariableOp¢?model/transformer_block/sequential/dense/BiasAdd/ReadVariableOp¢Amodel/transformer_block/sequential/dense/Tensordot/ReadVariableOp¢Amodel/transformer_block/sequential/dense_1/BiasAdd/ReadVariableOp¢Cmodel/transformer_block/sequential/dense_1/Tensordot/ReadVariableOp_
(model/token_and_position_embedding/ShapeShapeinput_1*
T0*
_output_shapes
:
6model/token_and_position_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
8model/token_and_position_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
8model/token_and_position_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
0model/token_and_position_embedding/strided_sliceStridedSlice1model/token_and_position_embedding/Shape:output:0?model/token_and_position_embedding/strided_slice/stack:output:0Amodel/token_and_position_embedding/strided_slice/stack_1:output:0Amodel/token_and_position_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskp
.model/token_and_position_embedding/range/startConst*
_output_shapes
: *
dtype0*
value	B : p
.model/token_and_position_embedding/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :û
(model/token_and_position_embedding/rangeRange7model/token_and_position_embedding/range/start:output:09model/token_and_position_embedding/strided_slice:output:07model/token_and_position_embedding/range/delta:output:0*
_output_shapes	
:á
?model/token_and_position_embedding/embedding_1/embedding_lookupResourceGatherDmodel_token_and_position_embedding_embedding_1_embedding_lookup_73471model/token_and_position_embedding/range:output:0*
Tindices0*W
_classM
KIloc:@model/token_and_position_embedding/embedding_1/embedding_lookup/7347*
_output_shapes
:	*
dtype0¡
Hmodel/token_and_position_embedding/embedding_1/embedding_lookup/IdentityIdentityHmodel/token_and_position_embedding/embedding_1/embedding_lookup:output:0*
T0*W
_classM
KIloc:@model/token_and_position_embedding/embedding_1/embedding_lookup/7347*
_output_shapes
:	Ó
Jmodel/token_and_position_embedding/embedding_1/embedding_lookup/Identity_1IdentityQmodel/token_and_position_embedding/embedding_1/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	
1model/token_and_position_embedding/embedding/CastCastinput_1*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿì
=model/token_and_position_embedding/embedding/embedding_lookupResourceGatherBmodel_token_and_position_embedding_embedding_embedding_lookup_73535model/token_and_position_embedding/embedding/Cast:y:0*
Tindices0*U
_classK
IGloc:@model/token_and_position_embedding/embedding/embedding_lookup/7353*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0¨
Fmodel/token_and_position_embedding/embedding/embedding_lookup/IdentityIdentityFmodel/token_and_position_embedding/embedding/embedding_lookup:output:0*
T0*U
_classK
IGloc:@model/token_and_position_embedding/embedding/embedding_lookup/7353*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
Hmodel/token_and_position_embedding/embedding/embedding_lookup/Identity_1IdentityOmodel/token_and_position_embedding/embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&model/token_and_position_embedding/addAddV2Qmodel/token_and_position_embedding/embedding/embedding_lookup/Identity_1:output:0Smodel/token_and_position_embedding/embedding_1/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿì
Omodel/transformer_block/multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOpXmodel_transformer_block_multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0°
@model/transformer_block/multi_head_attention/query/einsum/EinsumEinsum*model/token_and_position_embedding/add:z:0Wmodel/transformer_block/multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equationabc,cde->abdeÔ
Emodel/transformer_block/multi_head_attention/query/add/ReadVariableOpReadVariableOpNmodel_transformer_block_multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0
6model/transformer_block/multi_head_attention/query/addAddV2Imodel/transformer_block/multi_head_attention/query/einsum/Einsum:output:0Mmodel/transformer_block/multi_head_attention/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
Mmodel/transformer_block/multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOpVmodel_transformer_block_multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0¬
>model/transformer_block/multi_head_attention/key/einsum/EinsumEinsum*model/token_and_position_embedding/add:z:0Umodel/transformer_block/multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equationabc,cde->abdeÐ
Cmodel/transformer_block/multi_head_attention/key/add/ReadVariableOpReadVariableOpLmodel_transformer_block_multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0
4model/transformer_block/multi_head_attention/key/addAddV2Gmodel/transformer_block/multi_head_attention/key/einsum/Einsum:output:0Kmodel/transformer_block/multi_head_attention/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿì
Omodel/transformer_block/multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOpXmodel_transformer_block_multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0°
@model/transformer_block/multi_head_attention/value/einsum/EinsumEinsum*model/token_and_position_embedding/add:z:0Wmodel/transformer_block/multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equationabc,cde->abdeÔ
Emodel/transformer_block/multi_head_attention/value/add/ReadVariableOpReadVariableOpNmodel_transformer_block_multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0
6model/transformer_block/multi_head_attention/value/addAddV2Imodel/transformer_block/multi_head_attention/value/einsum/Einsum:output:0Mmodel/transformer_block/multi_head_attention/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
2model/transformer_block/multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ìQ>ë
0model/transformer_block/multi_head_attention/MulMul:model/transformer_block/multi_head_attention/query/add:z:0;model/transformer_block/multi_head_attention/Mul/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:model/transformer_block/multi_head_attention/einsum/EinsumEinsum8model/transformer_block/multi_head_attention/key/add:z:04model/transformer_block/multi_head_attention/Mul:z:0*
N*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equationaecd,abcd->acbeÈ
<model/transformer_block/multi_head_attention/softmax/SoftmaxSoftmaxCmodel/transformer_block/multi_head_attention/einsum/Einsum:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ
=model/transformer_block/multi_head_attention/dropout/IdentityIdentityFmodel/transformer_block/multi_head_attention/softmax/Softmax:softmax:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
<model/transformer_block/multi_head_attention/einsum_1/EinsumEinsumFmodel/transformer_block/multi_head_attention/dropout/Identity:output:0:model/transformer_block/multi_head_attention/value/add:z:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equationacbe,aecd->abcd
Zmodel/transformer_block/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpcmodel_transformer_block_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0Ý
Kmodel/transformer_block/multi_head_attention/attention_output/einsum/EinsumEinsumEmodel/transformer_block/multi_head_attention/einsum_1/Einsum:output:0bmodel/transformer_block/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equationabcd,cde->abeæ
Pmodel/transformer_block/multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpYmodel_transformer_block_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0±
Amodel/transformer_block/multi_head_attention/attention_output/addAddV2Tmodel/transformer_block/multi_head_attention/attention_output/einsum/Einsum:output:0Xmodel/transformer_block/multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
(model/transformer_block/dropout/IdentityIdentityEmodel/transformer_block/multi_head_attention/attention_output/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
model/transformer_block/addAddV2*model/token_and_position_embedding/add:z:01model/transformer_block/dropout/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Jmodel/transformer_block/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:þ
8model/transformer_block/layer_normalization/moments/meanMeanmodel/transformer_block/add:z:0Smodel/transformer_block/layer_normalization/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(Ê
@model/transformer_block/layer_normalization/moments/StopGradientStopGradientAmodel/transformer_block/layer_normalization/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿý
Emodel/transformer_block/layer_normalization/moments/SquaredDifferenceSquaredDifferencemodel/transformer_block/add:z:0Imodel/transformer_block/layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Nmodel/transformer_block/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:°
<model/transformer_block/layer_normalization/moments/varianceMeanImodel/transformer_block/layer_normalization/moments/SquaredDifference:z:0Wmodel/transformer_block/layer_normalization/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(
;model/transformer_block/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75
9model/transformer_block/layer_normalization/batchnorm/addAddV2Emodel/transformer_block/layer_normalization/moments/variance:output:0Dmodel/transformer_block/layer_normalization/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
;model/transformer_block/layer_normalization/batchnorm/RsqrtRsqrt=model/transformer_block/layer_normalization/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÖ
Hmodel/transformer_block/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOpQmodel_transformer_block_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0
9model/transformer_block/layer_normalization/batchnorm/mulMul?model/transformer_block/layer_normalization/batchnorm/Rsqrt:y:0Pmodel/transformer_block/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ
;model/transformer_block/layer_normalization/batchnorm/mul_1Mulmodel/transformer_block/add:z:0=model/transformer_block/layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿû
;model/transformer_block/layer_normalization/batchnorm/mul_2MulAmodel/transformer_block/layer_normalization/moments/mean:output:0=model/transformer_block/layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÎ
Dmodel/transformer_block/layer_normalization/batchnorm/ReadVariableOpReadVariableOpMmodel_transformer_block_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0
9model/transformer_block/layer_normalization/batchnorm/subSubLmodel/transformer_block/layer_normalization/batchnorm/ReadVariableOp:value:0?model/transformer_block/layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿû
;model/transformer_block/layer_normalization/batchnorm/add_1AddV2?model/transformer_block/layer_normalization/batchnorm/mul_1:z:0=model/transformer_block/layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ
Amodel/transformer_block/sequential/dense/Tensordot/ReadVariableOpReadVariableOpJmodel_transformer_block_sequential_dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0
7model/transformer_block/sequential/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
7model/transformer_block/sequential/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       §
8model/transformer_block/sequential/dense/Tensordot/ShapeShape?model/transformer_block/layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:
@model/transformer_block/sequential/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
;model/transformer_block/sequential/dense/Tensordot/GatherV2GatherV2Amodel/transformer_block/sequential/dense/Tensordot/Shape:output:0@model/transformer_block/sequential/dense/Tensordot/free:output:0Imodel/transformer_block/sequential/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
Bmodel/transformer_block/sequential/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ã
=model/transformer_block/sequential/dense/Tensordot/GatherV2_1GatherV2Amodel/transformer_block/sequential/dense/Tensordot/Shape:output:0@model/transformer_block/sequential/dense/Tensordot/axes:output:0Kmodel/transformer_block/sequential/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
8model/transformer_block/sequential/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: é
7model/transformer_block/sequential/dense/Tensordot/ProdProdDmodel/transformer_block/sequential/dense/Tensordot/GatherV2:output:0Amodel/transformer_block/sequential/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 
:model/transformer_block/sequential/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ï
9model/transformer_block/sequential/dense/Tensordot/Prod_1ProdFmodel/transformer_block/sequential/dense/Tensordot/GatherV2_1:output:0Cmodel/transformer_block/sequential/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 
>model/transformer_block/sequential/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : À
9model/transformer_block/sequential/dense/Tensordot/concatConcatV2@model/transformer_block/sequential/dense/Tensordot/free:output:0@model/transformer_block/sequential/dense/Tensordot/axes:output:0Gmodel/transformer_block/sequential/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:ô
8model/transformer_block/sequential/dense/Tensordot/stackPack@model/transformer_block/sequential/dense/Tensordot/Prod:output:0Bmodel/transformer_block/sequential/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
<model/transformer_block/sequential/dense/Tensordot/transpose	Transpose?model/transformer_block/layer_normalization/batchnorm/add_1:z:0Bmodel/transformer_block/sequential/dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:model/transformer_block/sequential/dense/Tensordot/ReshapeReshape@model/transformer_block/sequential/dense/Tensordot/transpose:y:0Amodel/transformer_block/sequential/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
9model/transformer_block/sequential/dense/Tensordot/MatMulMatMulCmodel/transformer_block/sequential/dense/Tensordot/Reshape:output:0Imodel/transformer_block/sequential/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:model/transformer_block/sequential/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:
@model/transformer_block/sequential/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ë
;model/transformer_block/sequential/dense/Tensordot/concat_1ConcatV2Dmodel/transformer_block/sequential/dense/Tensordot/GatherV2:output:0Cmodel/transformer_block/sequential/dense/Tensordot/Const_2:output:0Imodel/transformer_block/sequential/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:ÿ
2model/transformer_block/sequential/dense/TensordotReshapeCmodel/transformer_block/sequential/dense/Tensordot/MatMul:product:0Dmodel/transformer_block/sequential/dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
?model/transformer_block/sequential/dense/BiasAdd/ReadVariableOpReadVariableOpHmodel_transformer_block_sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ø
0model/transformer_block/sequential/dense/BiasAddBiasAdd;model/transformer_block/sequential/dense/Tensordot:output:0Gmodel/transformer_block/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
-model/transformer_block/sequential/dense/ReluRelu9model/transformer_block/sequential/dense/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
Cmodel/transformer_block/sequential/dense_1/Tensordot/ReadVariableOpReadVariableOpLmodel_transformer_block_sequential_dense_1_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0
9model/transformer_block/sequential/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
9model/transformer_block/sequential/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ¥
:model/transformer_block/sequential/dense_1/Tensordot/ShapeShape;model/transformer_block/sequential/dense/Relu:activations:0*
T0*
_output_shapes
:
Bmodel/transformer_block/sequential/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ç
=model/transformer_block/sequential/dense_1/Tensordot/GatherV2GatherV2Cmodel/transformer_block/sequential/dense_1/Tensordot/Shape:output:0Bmodel/transformer_block/sequential/dense_1/Tensordot/free:output:0Kmodel/transformer_block/sequential/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
Dmodel/transformer_block/sequential/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ë
?model/transformer_block/sequential/dense_1/Tensordot/GatherV2_1GatherV2Cmodel/transformer_block/sequential/dense_1/Tensordot/Shape:output:0Bmodel/transformer_block/sequential/dense_1/Tensordot/axes:output:0Mmodel/transformer_block/sequential/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
:model/transformer_block/sequential/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ï
9model/transformer_block/sequential/dense_1/Tensordot/ProdProdFmodel/transformer_block/sequential/dense_1/Tensordot/GatherV2:output:0Cmodel/transformer_block/sequential/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 
<model/transformer_block/sequential/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: õ
;model/transformer_block/sequential/dense_1/Tensordot/Prod_1ProdHmodel/transformer_block/sequential/dense_1/Tensordot/GatherV2_1:output:0Emodel/transformer_block/sequential/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 
@model/transformer_block/sequential/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : È
;model/transformer_block/sequential/dense_1/Tensordot/concatConcatV2Bmodel/transformer_block/sequential/dense_1/Tensordot/free:output:0Bmodel/transformer_block/sequential/dense_1/Tensordot/axes:output:0Imodel/transformer_block/sequential/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:ú
:model/transformer_block/sequential/dense_1/Tensordot/stackPackBmodel/transformer_block/sequential/dense_1/Tensordot/Prod:output:0Dmodel/transformer_block/sequential/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
>model/transformer_block/sequential/dense_1/Tensordot/transpose	Transpose;model/transformer_block/sequential/dense/Relu:activations:0Dmodel/transformer_block/sequential/dense_1/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
<model/transformer_block/sequential/dense_1/Tensordot/ReshapeReshapeBmodel/transformer_block/sequential/dense_1/Tensordot/transpose:y:0Cmodel/transformer_block/sequential/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
;model/transformer_block/sequential/dense_1/Tensordot/MatMulMatMulEmodel/transformer_block/sequential/dense_1/Tensordot/Reshape:output:0Kmodel/transformer_block/sequential/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
<model/transformer_block/sequential/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:
Bmodel/transformer_block/sequential/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ó
=model/transformer_block/sequential/dense_1/Tensordot/concat_1ConcatV2Fmodel/transformer_block/sequential/dense_1/Tensordot/GatherV2:output:0Emodel/transformer_block/sequential/dense_1/Tensordot/Const_2:output:0Kmodel/transformer_block/sequential/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
4model/transformer_block/sequential/dense_1/TensordotReshapeEmodel/transformer_block/sequential/dense_1/Tensordot/MatMul:product:0Fmodel/transformer_block/sequential/dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
Amodel/transformer_block/sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOpJmodel_transformer_block_sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0þ
2model/transformer_block/sequential/dense_1/BiasAddBiasAdd=model/transformer_block/sequential/dense_1/Tensordot:output:0Imodel/transformer_block/sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
*model/transformer_block/dropout_1/IdentityIdentity;model/transformer_block/sequential/dense_1/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÓ
model/transformer_block/add_1AddV2?model/transformer_block/layer_normalization/batchnorm/add_1:z:03model/transformer_block/dropout_1/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Lmodel/transformer_block/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
:model/transformer_block/layer_normalization_1/moments/meanMean!model/transformer_block/add_1:z:0Umodel/transformer_block/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(Î
Bmodel/transformer_block/layer_normalization_1/moments/StopGradientStopGradientCmodel/transformer_block/layer_normalization_1/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Gmodel/transformer_block/layer_normalization_1/moments/SquaredDifferenceSquaredDifference!model/transformer_block/add_1:z:0Kmodel/transformer_block/layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Pmodel/transformer_block/layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:¶
>model/transformer_block/layer_normalization_1/moments/varianceMeanKmodel/transformer_block/layer_normalization_1/moments/SquaredDifference:z:0Ymodel/transformer_block/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(
=model/transformer_block/layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75
;model/transformer_block/layer_normalization_1/batchnorm/addAddV2Gmodel/transformer_block/layer_normalization_1/moments/variance:output:0Fmodel/transformer_block/layer_normalization_1/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¾
=model/transformer_block/layer_normalization_1/batchnorm/RsqrtRsqrt?model/transformer_block/layer_normalization_1/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
Jmodel/transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpSmodel_transformer_block_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0
;model/transformer_block/layer_normalization_1/batchnorm/mulMulAmodel/transformer_block/layer_normalization_1/batchnorm/Rsqrt:y:0Rmodel/transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿß
=model/transformer_block/layer_normalization_1/batchnorm/mul_1Mul!model/transformer_block/add_1:z:0?model/transformer_block/layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
=model/transformer_block/layer_normalization_1/batchnorm/mul_2MulCmodel/transformer_block/layer_normalization_1/moments/mean:output:0?model/transformer_block/layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÒ
Fmodel/transformer_block/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOpOmodel_transformer_block_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0
;model/transformer_block/layer_normalization_1/batchnorm/subSubNmodel/transformer_block/layer_normalization_1/batchnorm/ReadVariableOp:value:0Amodel/transformer_block/layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
=model/transformer_block/layer_normalization_1/batchnorm/add_1AddV2Amodel/transformer_block/layer_normalization_1/batchnorm/mul_1:z:0?model/transformer_block/layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
5model/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :à
#model/global_average_pooling1d/MeanMeanAmodel/transformer_block/layer_normalization_1/batchnorm/add_1:z:0>model/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model/dropout_2/IdentityIdentity,model/global_average_pooling1d/Mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#model/dense_2/MatMul/ReadVariableOpReadVariableOp,model_dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0 
model/dense_2/MatMulMatMul!model/dropout_2/Identity:output:0+model/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0 
model/dense_2/BiasAddBiasAddmodel/dense_2/MatMul:product:0,model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
model/dense_2/ReluRelumodel/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
model/dropout_3/IdentityIdentity model/dense_2/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#model/dense_3/MatMul/ReadVariableOpReadVariableOp,model_dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype0 
model/dense_3/MatMulMatMul!model/dropout_3/Identity:output:0+model/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$model/dense_3/BiasAdd/ReadVariableOpReadVariableOp-model_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0 
model/dense_3/BiasAddBiasAddmodel/dense_3/MatMul:product:0,model/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
model/dense_3/SigmoidSigmoidmodel/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentitymodel/dense_3/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp%^model/dense_2/BiasAdd/ReadVariableOp$^model/dense_2/MatMul/ReadVariableOp%^model/dense_3/BiasAdd/ReadVariableOp$^model/dense_3/MatMul/ReadVariableOp>^model/token_and_position_embedding/embedding/embedding_lookup@^model/token_and_position_embedding/embedding_1/embedding_lookupE^model/transformer_block/layer_normalization/batchnorm/ReadVariableOpI^model/transformer_block/layer_normalization/batchnorm/mul/ReadVariableOpG^model/transformer_block/layer_normalization_1/batchnorm/ReadVariableOpK^model/transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpQ^model/transformer_block/multi_head_attention/attention_output/add/ReadVariableOp[^model/transformer_block/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpD^model/transformer_block/multi_head_attention/key/add/ReadVariableOpN^model/transformer_block/multi_head_attention/key/einsum/Einsum/ReadVariableOpF^model/transformer_block/multi_head_attention/query/add/ReadVariableOpP^model/transformer_block/multi_head_attention/query/einsum/Einsum/ReadVariableOpF^model/transformer_block/multi_head_attention/value/add/ReadVariableOpP^model/transformer_block/multi_head_attention/value/einsum/Einsum/ReadVariableOp@^model/transformer_block/sequential/dense/BiasAdd/ReadVariableOpB^model/transformer_block/sequential/dense/Tensordot/ReadVariableOpB^model/transformer_block/sequential/dense_1/BiasAdd/ReadVariableOpD^model/transformer_block/sequential/dense_1/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 2L
$model/dense_2/BiasAdd/ReadVariableOp$model/dense_2/BiasAdd/ReadVariableOp2J
#model/dense_2/MatMul/ReadVariableOp#model/dense_2/MatMul/ReadVariableOp2L
$model/dense_3/BiasAdd/ReadVariableOp$model/dense_3/BiasAdd/ReadVariableOp2J
#model/dense_3/MatMul/ReadVariableOp#model/dense_3/MatMul/ReadVariableOp2~
=model/token_and_position_embedding/embedding/embedding_lookup=model/token_and_position_embedding/embedding/embedding_lookup2
?model/token_and_position_embedding/embedding_1/embedding_lookup?model/token_and_position_embedding/embedding_1/embedding_lookup2
Dmodel/transformer_block/layer_normalization/batchnorm/ReadVariableOpDmodel/transformer_block/layer_normalization/batchnorm/ReadVariableOp2
Hmodel/transformer_block/layer_normalization/batchnorm/mul/ReadVariableOpHmodel/transformer_block/layer_normalization/batchnorm/mul/ReadVariableOp2
Fmodel/transformer_block/layer_normalization_1/batchnorm/ReadVariableOpFmodel/transformer_block/layer_normalization_1/batchnorm/ReadVariableOp2
Jmodel/transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpJmodel/transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp2¤
Pmodel/transformer_block/multi_head_attention/attention_output/add/ReadVariableOpPmodel/transformer_block/multi_head_attention/attention_output/add/ReadVariableOp2¸
Zmodel/transformer_block/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpZmodel/transformer_block/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2
Cmodel/transformer_block/multi_head_attention/key/add/ReadVariableOpCmodel/transformer_block/multi_head_attention/key/add/ReadVariableOp2
Mmodel/transformer_block/multi_head_attention/key/einsum/Einsum/ReadVariableOpMmodel/transformer_block/multi_head_attention/key/einsum/Einsum/ReadVariableOp2
Emodel/transformer_block/multi_head_attention/query/add/ReadVariableOpEmodel/transformer_block/multi_head_attention/query/add/ReadVariableOp2¢
Omodel/transformer_block/multi_head_attention/query/einsum/Einsum/ReadVariableOpOmodel/transformer_block/multi_head_attention/query/einsum/Einsum/ReadVariableOp2
Emodel/transformer_block/multi_head_attention/value/add/ReadVariableOpEmodel/transformer_block/multi_head_attention/value/add/ReadVariableOp2¢
Omodel/transformer_block/multi_head_attention/value/einsum/Einsum/ReadVariableOpOmodel/transformer_block/multi_head_attention/value/einsum/Einsum/ReadVariableOp2
?model/transformer_block/sequential/dense/BiasAdd/ReadVariableOp?model/transformer_block/sequential/dense/BiasAdd/ReadVariableOp2
Amodel/transformer_block/sequential/dense/Tensordot/ReadVariableOpAmodel/transformer_block/sequential/dense/Tensordot/ReadVariableOp2
Amodel/transformer_block/sequential/dense_1/BiasAdd/ReadVariableOpAmodel/transformer_block/sequential/dense_1/BiasAdd/ReadVariableOp2
Cmodel/transformer_block/sequential/dense_1/Tensordot/ReadVariableOpCmodel/transformer_block/sequential/dense_1/Tensordot/ReadVariableOp:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
ª
Ì
)__inference_sequential_layer_call_fn_9609

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_7582t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


D__inference_sequential_layer_call_and_return_conditional_losses_7680
dense_input

dense_7669:

dense_7671:
dense_1_7674:
dense_1_7676:
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCallè
dense/StatefulPartitionedCallStatefulPartitionedCalldense_input
dense_7669
dense_7671*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_7539
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_7674dense_1_7676*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_7575|
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:Y U
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namedense_input
¹
Ñ
)__inference_sequential_layer_call_fn_7593
dense_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_7582t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namedense_input
¥
ö
?__inference_dense_layer_call_and_return_conditional_losses_9776

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

D
(__inference_dropout_2_layer_call_fn_9507

inputs
identity®
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_7911`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª
Ì
)__inference_sequential_layer_call_fn_9622

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_7642t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ

D__inference_sequential_layer_call_and_return_conditional_losses_7642

inputs

dense_7631:

dense_7633:
dense_1_7636:
dense_1_7638:
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCallã
dense/StatefulPartitionedCallStatefulPartitionedCallinputs
dense_7631
dense_7633*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_7539
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_7636dense_1_7638*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_7575|
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ=
Ë
D__inference_sequential_layer_call_and_return_conditional_losses_9736

inputs9
'dense_tensordot_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:;
)dense_1_tensordot_readvariableop_resource:5
'dense_1_biasadd_readvariableop_resource:
identity¢dense/BiasAdd/ReadVariableOp¢dense/Tensordot/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢ dense_1/Tensordot/ReadVariableOp
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0^
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:e
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       K
dense/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:_
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ó
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ×
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:_
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ´
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense/Tensordot/transpose	Transposeinputsdense/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:_
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿa

dense/ReluReludense/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0`
dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       _
dense_1/Tensordot/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:a
dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Û
dense_1/Tensordot/GatherV2GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/free:output:0(dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_1/Tensordot/GatherV2_1GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/axes:output:0*dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_1/Tensordot/transpose	Transposedense/Relu:activations:0!dense_1/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÊ
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Î
°
0__inference_transformer_block_layer_call_fn_9187

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:
identity¢StatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_transformer_block_layer_call_and_return_conditional_losses_7871t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Ñ
"__inference_signature_wrapper_9117
input_1
unknown:	
	unknown_0:	
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:
identity¢StatefulPartitionedCallÄ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__wrapped_model_7501o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
ñ	
b
C__inference_dropout_2_layer_call_and_return_conditional_losses_8065

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷
S
7__inference_global_average_pooling1d_layer_call_fn_9496

inputs
identityÆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_7704i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷(
þ
?__inference_model_layer_call_and_return_conditional_losses_8550
input_14
!token_and_position_embedding_8498:	4
!token_and_position_embedding_8500:	,
transformer_block_8503:(
transformer_block_8505:,
transformer_block_8507:(
transformer_block_8509:,
transformer_block_8511:(
transformer_block_8513:,
transformer_block_8515:$
transformer_block_8517:$
transformer_block_8519:$
transformer_block_8521:(
transformer_block_8523:$
transformer_block_8525:(
transformer_block_8527:$
transformer_block_8529:$
transformer_block_8531:$
transformer_block_8533:
dense_2_8538:
dense_2_8540:
dense_3_8544:
dense_3_8546:
identity¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall¢4token_and_position_embedding/StatefulPartitionedCall¢)transformer_block/StatefulPartitionedCallÀ
4token_and_position_embedding/StatefulPartitionedCallStatefulPartitionedCallinput_1!token_and_position_embedding_8498!token_and_position_embedding_8500*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *_
fZRX
V__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_7738¶
)transformer_block/StatefulPartitionedCallStatefulPartitionedCall=token_and_position_embedding/StatefulPartitionedCall:output:0transformer_block_8503transformer_block_8505transformer_block_8507transformer_block_8509transformer_block_8511transformer_block_8513transformer_block_8515transformer_block_8517transformer_block_8519transformer_block_8521transformer_block_8523transformer_block_8525transformer_block_8527transformer_block_8529transformer_block_8531transformer_block_8533*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_transformer_block_layer_call_and_return_conditional_losses_7871
(global_average_pooling1d/PartitionedCallPartitionedCall2transformer_block/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_7704ã
dropout_2/PartitionedCallPartitionedCall1global_average_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_7911
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_2_8538dense_2_8540*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_7924Ú
dropout_3/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_3_layer_call_and_return_conditional_losses_7935
dense_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_3_8544dense_3_8546*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_7948w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿí
NoOpNoOp ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall5^token_and_position_embedding/StatefulPartitionedCall*^transformer_block/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2l
4token_and_position_embedding/StatefulPartitionedCall4token_and_position_embedding/StatefulPartitionedCall2V
)transformer_block/StatefulPartitionedCall)transformer_block/StatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
²
Ò
$__inference_model_layer_call_fn_8660

inputs
unknown:	
	unknown_0:	
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:
identity¢StatefulPartitionedCallã
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_7955o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ
â
K__inference_transformer_block_layer_call_and_return_conditional_losses_7871

inputsV
@multi_head_attention_query_einsum_einsum_readvariableop_resource:H
6multi_head_attention_query_add_readvariableop_resource:T
>multi_head_attention_key_einsum_einsum_readvariableop_resource:F
4multi_head_attention_key_add_readvariableop_resource:V
@multi_head_attention_value_einsum_einsum_readvariableop_resource:H
6multi_head_attention_value_add_readvariableop_resource:a
Kmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource:O
Amulti_head_attention_attention_output_add_readvariableop_resource:G
9layer_normalization_batchnorm_mul_readvariableop_resource:C
5layer_normalization_batchnorm_readvariableop_resource:D
2sequential_dense_tensordot_readvariableop_resource:>
0sequential_dense_biasadd_readvariableop_resource:F
4sequential_dense_1_tensordot_readvariableop_resource:@
2sequential_dense_1_biasadd_readvariableop_resource:I
;layer_normalization_1_batchnorm_mul_readvariableop_resource:E
7layer_normalization_1_batchnorm_readvariableop_resource:
identity¢,layer_normalization/batchnorm/ReadVariableOp¢0layer_normalization/batchnorm/mul/ReadVariableOp¢.layer_normalization_1/batchnorm/ReadVariableOp¢2layer_normalization_1/batchnorm/mul/ReadVariableOp¢8multi_head_attention/attention_output/add/ReadVariableOp¢Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp¢+multi_head_attention/key/add/ReadVariableOp¢5multi_head_attention/key/einsum/Einsum/ReadVariableOp¢-multi_head_attention/query/add/ReadVariableOp¢7multi_head_attention/query/einsum/Einsum/ReadVariableOp¢-multi_head_attention/value/add/ReadVariableOp¢7multi_head_attention/value/einsum/Einsum/ReadVariableOp¢'sequential/dense/BiasAdd/ReadVariableOp¢)sequential/dense/Tensordot/ReadVariableOp¢)sequential/dense_1/BiasAdd/ReadVariableOp¢+sequential/dense_1/Tensordot/ReadVariableOp¼
7multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0Ü
(multi_head_attention/query/einsum/EinsumEinsuminputs?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equationabc,cde->abde¤
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0Ì
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0Ø
&multi_head_attention/key/einsum/EinsumEinsuminputs=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equationabc,cde->abde 
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0Æ
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0Ü
(multi_head_attention/value/einsum/EinsumEinsuminputs?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equationabc,cde->abde¤
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0Ì
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ìQ>£
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
"multi_head_attention/einsum/EinsumEinsum multi_head_attention/key/add:z:0multi_head_attention/Mul:z:0*
N*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equationaecd,abcd->acbe
$multi_head_attention/softmax/SoftmaxSoftmax+multi_head_attention/einsum/Einsum:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%multi_head_attention/dropout/IdentityIdentity.multi_head_attention/softmax/Softmax:softmax:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿå
$multi_head_attention/einsum_1/EinsumEinsum.multi_head_attention/dropout/Identity:output:0"multi_head_attention/value/add:z:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equationacbe,aecd->abcdÒ
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equationabcd,cde->abe¶
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0é
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout/IdentityIdentity-multi_head_attention/attention_output/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
addAddV2inputsdropout/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:¶
 layer_normalization/moments/meanMeanadd:z:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
-layer_normalization/moments/SquaredDifferenceSquaredDifferenceadd:z:01layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:è
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75¾
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Â
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#layer_normalization/batchnorm/mul_1Muladd:z:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0¾
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)sequential/dense/Tensordot/ReadVariableOpReadVariableOp2sequential_dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0i
sequential/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:p
sequential/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       w
 sequential/dense/Tensordot/ShapeShape'layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:j
(sequential/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ÿ
#sequential/dense/Tensordot/GatherV2GatherV2)sequential/dense/Tensordot/Shape:output:0(sequential/dense/Tensordot/free:output:01sequential/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
*sequential/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
%sequential/dense/Tensordot/GatherV2_1GatherV2)sequential/dense/Tensordot/Shape:output:0(sequential/dense/Tensordot/axes:output:03sequential/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:j
 sequential/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
sequential/dense/Tensordot/ProdProd,sequential/dense/Tensordot/GatherV2:output:0)sequential/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: l
"sequential/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: §
!sequential/dense/Tensordot/Prod_1Prod.sequential/dense/Tensordot/GatherV2_1:output:0+sequential/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: h
&sequential/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : à
!sequential/dense/Tensordot/concatConcatV2(sequential/dense/Tensordot/free:output:0(sequential/dense/Tensordot/axes:output:0/sequential/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:¬
 sequential/dense/Tensordot/stackPack(sequential/dense/Tensordot/Prod:output:0*sequential/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:½
$sequential/dense/Tensordot/transpose	Transpose'layer_normalization/batchnorm/add_1:z:0*sequential/dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
"sequential/dense/Tensordot/ReshapeReshape(sequential/dense/Tensordot/transpose:y:0)sequential/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ½
!sequential/dense/Tensordot/MatMulMatMul+sequential/dense/Tensordot/Reshape:output:01sequential/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"sequential/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:j
(sequential/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ë
#sequential/dense/Tensordot/concat_1ConcatV2,sequential/dense/Tensordot/GatherV2:output:0+sequential/dense/Tensordot/Const_2:output:01sequential/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:·
sequential/dense/TensordotReshape+sequential/dense/Tensordot/MatMul:product:0,sequential/dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0°
sequential/dense/BiasAddBiasAdd#sequential/dense/Tensordot:output:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
+sequential/dense_1/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_1_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0k
!sequential/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:r
!sequential/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       u
"sequential/dense_1/Tensordot/ShapeShape#sequential/dense/Relu:activations:0*
T0*
_output_shapes
:l
*sequential/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
%sequential/dense_1/Tensordot/GatherV2GatherV2+sequential/dense_1/Tensordot/Shape:output:0*sequential/dense_1/Tensordot/free:output:03sequential/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
,sequential/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
'sequential/dense_1/Tensordot/GatherV2_1GatherV2+sequential/dense_1/Tensordot/Shape:output:0*sequential/dense_1/Tensordot/axes:output:05sequential/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
"sequential/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: §
!sequential/dense_1/Tensordot/ProdProd.sequential/dense_1/Tensordot/GatherV2:output:0+sequential/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$sequential/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ­
#sequential/dense_1/Tensordot/Prod_1Prod0sequential/dense_1/Tensordot/GatherV2_1:output:0-sequential/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(sequential/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : è
#sequential/dense_1/Tensordot/concatConcatV2*sequential/dense_1/Tensordot/free:output:0*sequential/dense_1/Tensordot/axes:output:01sequential/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:²
"sequential/dense_1/Tensordot/stackPack*sequential/dense_1/Tensordot/Prod:output:0,sequential/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:½
&sequential/dense_1/Tensordot/transpose	Transpose#sequential/dense/Relu:activations:0,sequential/dense_1/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
$sequential/dense_1/Tensordot/ReshapeReshape*sequential/dense_1/Tensordot/transpose:y:0+sequential/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÃ
#sequential/dense_1/Tensordot/MatMulMatMul-sequential/dense_1/Tensordot/Reshape:output:03sequential/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
$sequential/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:l
*sequential/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ó
%sequential/dense_1/Tensordot/concat_1ConcatV2.sequential/dense_1/Tensordot/GatherV2:output:0-sequential/dense_1/Tensordot/Const_2:output:03sequential/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:½
sequential/dense_1/TensordotReshape-sequential/dense_1/Tensordot/MatMul:product:0.sequential/dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¶
sequential/dense_1/BiasAddBiasAdd%sequential/dense_1/Tensordot:output:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
dropout_1/IdentityIdentity#sequential/dense_1/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
add_1AddV2'layer_normalization/batchnorm/add_1:z:0dropout_1/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:¼
"layer_normalization_1/moments/meanMean	add_1:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ»
/layer_normalization_1/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:î
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Ä
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0È
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%layer_normalization_1/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Ä
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
IdentityIdentity)layer_normalization_1/batchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿø
NoOpNoOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp*^sequential/dense/Tensordot/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp,^sequential/dense_1/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2t
8multi_head_attention/attention_output/add/ReadVariableOp8multi_head_attention/attention_output/add/ReadVariableOp2
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpBmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp2Z
+multi_head_attention/key/add/ReadVariableOp+multi_head_attention/key/add/ReadVariableOp2n
5multi_head_attention/key/einsum/Einsum/ReadVariableOp5multi_head_attention/key/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/query/add/ReadVariableOp-multi_head_attention/query/add/ReadVariableOp2r
7multi_head_attention/query/einsum/Einsum/ReadVariableOp7multi_head_attention/query/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/value/add/ReadVariableOp-multi_head_attention/value/add/ReadVariableOp2r
7multi_head_attention/value/einsum/Einsum/ReadVariableOp7multi_head_attention/value/einsum/Einsum/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2V
)sequential/dense/Tensordot/ReadVariableOp)sequential/dense/Tensordot/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2Z
+sequential/dense_1/Tensordot/ReadVariableOp+sequential/dense_1/Tensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
á

V__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_7738
x4
!embedding_1_embedding_lookup_7725:	2
embedding_embedding_lookup_7731:	
identity¢embedding/embedding_lookup¢embedding_1/embedding_lookup6
ShapeShapex*
T0*
_output_shapes
:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :o
rangeRangerange/start:output:0strided_slice:output:0range/delta:output:0*
_output_shapes	
:Õ
embedding_1/embedding_lookupResourceGather!embedding_1_embedding_lookup_7725range:output:0*
Tindices0*4
_class*
(&loc:@embedding_1/embedding_lookup/7725*
_output_shapes
:	*
dtype0¸
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*4
_class*
(&loc:@embedding_1/embedding_lookup/7725*
_output_shapes
:	
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	[
embedding/CastCastx*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
embedding/embedding_lookupResourceGatherembedding_embedding_lookup_7731embedding/Cast:y:0*
Tindices0*2
_class(
&$loc:@embedding/embedding_lookup/7731*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0¿
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*2
_class(
&$loc:@embedding/embedding_lookup/7731*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
addAddV2.embedding/embedding_lookup/Identity_1:output:00embedding_1/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
IdentityIdentityadd:z:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^embedding/embedding_lookup^embedding_1/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 28
embedding/embedding_lookupembedding/embedding_lookup2<
embedding_1/embedding_lookupembedding_1/embedding_lookup:K G
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex


ò
A__inference_dense_3_layer_call_and_return_conditional_losses_7948

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ
Ó
$__inference_model_layer_call_fn_8495
input_1
unknown:	
	unknown_0:	
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_8399o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Ñ=
Ë
D__inference_sequential_layer_call_and_return_conditional_losses_9679

inputs9
'dense_tensordot_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:;
)dense_1_tensordot_readvariableop_resource:5
'dense_1_biasadd_readvariableop_resource:
identity¢dense/BiasAdd/ReadVariableOp¢dense/Tensordot/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢ dense_1/Tensordot/ReadVariableOp
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0^
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:e
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       K
dense/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:_
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ó
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ×
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:_
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ´
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense/Tensordot/transpose	Transposeinputsdense/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:_
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿa

dense/ReluReludense/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0`
dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       _
dense_1/Tensordot/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:a
dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Û
dense_1/Tensordot/GatherV2GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/free:output:0(dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_1/Tensordot/GatherV2_1GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/axes:output:0*dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_1/Tensordot/transpose	Transposedense/Relu:activations:0!dense_1/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÊ
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
á

V__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_9150
x4
!embedding_1_embedding_lookup_9137:	2
embedding_embedding_lookup_9143:	
identity¢embedding/embedding_lookup¢embedding_1/embedding_lookup6
ShapeShapex*
T0*
_output_shapes
:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :o
rangeRangerange/start:output:0strided_slice:output:0range/delta:output:0*
_output_shapes	
:Õ
embedding_1/embedding_lookupResourceGather!embedding_1_embedding_lookup_9137range:output:0*
Tindices0*4
_class*
(&loc:@embedding_1/embedding_lookup/9137*
_output_shapes
:	*
dtype0¸
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*4
_class*
(&loc:@embedding_1/embedding_lookup/9137*
_output_shapes
:	
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	[
embedding/CastCastx*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
embedding/embedding_lookupResourceGatherembedding_embedding_lookup_9143embedding/Cast:y:0*
Tindices0*2
_class(
&$loc:@embedding/embedding_lookup/9143*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0¿
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*2
_class(
&$loc:@embedding/embedding_lookup/9143*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
addAddV2.embedding/embedding_lookup/Identity_1:output:00embedding_1/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
IdentityIdentityadd:z:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^embedding/embedding_lookup^embedding_1/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 28
embedding/embedding_lookupembedding/embedding_lookup2<
embedding_1/embedding_lookupembedding_1/embedding_lookup:K G
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
Ð

&__inference_dense_1_layer_call_fn_9785

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_7575t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ì¸

?__inference_model_layer_call_and_return_conditional_losses_9066

inputsQ
>token_and_position_embedding_embedding_1_embedding_lookup_8885:	O
<token_and_position_embedding_embedding_embedding_lookup_8891:	h
Rtransformer_block_multi_head_attention_query_einsum_einsum_readvariableop_resource:Z
Htransformer_block_multi_head_attention_query_add_readvariableop_resource:f
Ptransformer_block_multi_head_attention_key_einsum_einsum_readvariableop_resource:X
Ftransformer_block_multi_head_attention_key_add_readvariableop_resource:h
Rtransformer_block_multi_head_attention_value_einsum_einsum_readvariableop_resource:Z
Htransformer_block_multi_head_attention_value_add_readvariableop_resource:s
]transformer_block_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource:a
Stransformer_block_multi_head_attention_attention_output_add_readvariableop_resource:Y
Ktransformer_block_layer_normalization_batchnorm_mul_readvariableop_resource:U
Gtransformer_block_layer_normalization_batchnorm_readvariableop_resource:V
Dtransformer_block_sequential_dense_tensordot_readvariableop_resource:P
Btransformer_block_sequential_dense_biasadd_readvariableop_resource:X
Ftransformer_block_sequential_dense_1_tensordot_readvariableop_resource:R
Dtransformer_block_sequential_dense_1_biasadd_readvariableop_resource:[
Mtransformer_block_layer_normalization_1_batchnorm_mul_readvariableop_resource:W
Itransformer_block_layer_normalization_1_batchnorm_readvariableop_resource:8
&dense_2_matmul_readvariableop_resource:5
'dense_2_biasadd_readvariableop_resource:8
&dense_3_matmul_readvariableop_resource:5
'dense_3_biasadd_readvariableop_resource:
identity¢dense_2/BiasAdd/ReadVariableOp¢dense_2/MatMul/ReadVariableOp¢dense_3/BiasAdd/ReadVariableOp¢dense_3/MatMul/ReadVariableOp¢7token_and_position_embedding/embedding/embedding_lookup¢9token_and_position_embedding/embedding_1/embedding_lookup¢>transformer_block/layer_normalization/batchnorm/ReadVariableOp¢Btransformer_block/layer_normalization/batchnorm/mul/ReadVariableOp¢@transformer_block/layer_normalization_1/batchnorm/ReadVariableOp¢Dtransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp¢Jtransformer_block/multi_head_attention/attention_output/add/ReadVariableOp¢Ttransformer_block/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp¢=transformer_block/multi_head_attention/key/add/ReadVariableOp¢Gtransformer_block/multi_head_attention/key/einsum/Einsum/ReadVariableOp¢?transformer_block/multi_head_attention/query/add/ReadVariableOp¢Itransformer_block/multi_head_attention/query/einsum/Einsum/ReadVariableOp¢?transformer_block/multi_head_attention/value/add/ReadVariableOp¢Itransformer_block/multi_head_attention/value/einsum/Einsum/ReadVariableOp¢9transformer_block/sequential/dense/BiasAdd/ReadVariableOp¢;transformer_block/sequential/dense/Tensordot/ReadVariableOp¢;transformer_block/sequential/dense_1/BiasAdd/ReadVariableOp¢=transformer_block/sequential/dense_1/Tensordot/ReadVariableOpX
"token_and_position_embedding/ShapeShapeinputs*
T0*
_output_shapes
:
0token_and_position_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ|
2token_and_position_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: |
2token_and_position_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:â
*token_and_position_embedding/strided_sliceStridedSlice+token_and_position_embedding/Shape:output:09token_and_position_embedding/strided_slice/stack:output:0;token_and_position_embedding/strided_slice/stack_1:output:0;token_and_position_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(token_and_position_embedding/range/startConst*
_output_shapes
: *
dtype0*
value	B : j
(token_and_position_embedding/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :ã
"token_and_position_embedding/rangeRange1token_and_position_embedding/range/start:output:03token_and_position_embedding/strided_slice:output:01token_and_position_embedding/range/delta:output:0*
_output_shapes	
:É
9token_and_position_embedding/embedding_1/embedding_lookupResourceGather>token_and_position_embedding_embedding_1_embedding_lookup_8885+token_and_position_embedding/range:output:0*
Tindices0*Q
_classG
ECloc:@token_and_position_embedding/embedding_1/embedding_lookup/8885*
_output_shapes
:	*
dtype0
Btoken_and_position_embedding/embedding_1/embedding_lookup/IdentityIdentityBtoken_and_position_embedding/embedding_1/embedding_lookup:output:0*
T0*Q
_classG
ECloc:@token_and_position_embedding/embedding_1/embedding_lookup/8885*
_output_shapes
:	Ç
Dtoken_and_position_embedding/embedding_1/embedding_lookup/Identity_1IdentityKtoken_and_position_embedding/embedding_1/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	}
+token_and_position_embedding/embedding/CastCastinputs*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ
7token_and_position_embedding/embedding/embedding_lookupResourceGather<token_and_position_embedding_embedding_embedding_lookup_8891/token_and_position_embedding/embedding/Cast:y:0*
Tindices0*O
_classE
CAloc:@token_and_position_embedding/embedding/embedding_lookup/8891*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0
@token_and_position_embedding/embedding/embedding_lookup/IdentityIdentity@token_and_position_embedding/embedding/embedding_lookup:output:0*
T0*O
_classE
CAloc:@token_and_position_embedding/embedding/embedding_lookup/8891*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
Btoken_and_position_embedding/embedding/embedding_lookup/Identity_1IdentityItoken_and_position_embedding/embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿü
 token_and_position_embedding/addAddV2Ktoken_and_position_embedding/embedding/embedding_lookup/Identity_1:output:0Mtoken_and_position_embedding/embedding_1/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
Itransformer_block/multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOpRtransformer_block_multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0
:transformer_block/multi_head_attention/query/einsum/EinsumEinsum$token_and_position_embedding/add:z:0Qtransformer_block/multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equationabc,cde->abdeÈ
?transformer_block/multi_head_attention/query/add/ReadVariableOpReadVariableOpHtransformer_block_multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0
0transformer_block/multi_head_attention/query/addAddV2Ctransformer_block/multi_head_attention/query/einsum/Einsum:output:0Gtransformer_block/multi_head_attention/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
Gtransformer_block/multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOpPtransformer_block_multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0
8transformer_block/multi_head_attention/key/einsum/EinsumEinsum$token_and_position_embedding/add:z:0Otransformer_block/multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equationabc,cde->abdeÄ
=transformer_block/multi_head_attention/key/add/ReadVariableOpReadVariableOpFtransformer_block_multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0ü
.transformer_block/multi_head_attention/key/addAddV2Atransformer_block/multi_head_attention/key/einsum/Einsum:output:0Etransformer_block/multi_head_attention/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
Itransformer_block/multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOpRtransformer_block_multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0
:transformer_block/multi_head_attention/value/einsum/EinsumEinsum$token_and_position_embedding/add:z:0Qtransformer_block/multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equationabc,cde->abdeÈ
?transformer_block/multi_head_attention/value/add/ReadVariableOpReadVariableOpHtransformer_block_multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0
0transformer_block/multi_head_attention/value/addAddV2Ctransformer_block/multi_head_attention/value/einsum/Einsum:output:0Gtransformer_block/multi_head_attention/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
,transformer_block/multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ìQ>Ù
*transformer_block/multi_head_attention/MulMul4transformer_block/multi_head_attention/query/add:z:05transformer_block/multi_head_attention/Mul/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
4transformer_block/multi_head_attention/einsum/EinsumEinsum2transformer_block/multi_head_attention/key/add:z:0.transformer_block/multi_head_attention/Mul:z:0*
N*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equationaecd,abcd->acbe¼
6transformer_block/multi_head_attention/softmax/SoftmaxSoftmax=transformer_block/multi_head_attention/einsum/Einsum:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
6transformer_block/multi_head_attention/einsum_1/EinsumEinsum@transformer_block/multi_head_attention/softmax/Softmax:softmax:04transformer_block/multi_head_attention/value/add:z:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equationacbe,aecd->abcdö
Ttransformer_block/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOp]transformer_block_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0Ë
Etransformer_block/multi_head_attention/attention_output/einsum/EinsumEinsum?transformer_block/multi_head_attention/einsum_1/Einsum:output:0\transformer_block/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equationabcd,cde->abeÚ
Jtransformer_block/multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpStransformer_block_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0
;transformer_block/multi_head_attention/attention_output/addAddV2Ntransformer_block/multi_head_attention/attention_output/einsum/Einsum:output:0Rtransformer_block/multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
'transformer_block/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?Ö
%transformer_block/dropout/dropout/MulMul?transformer_block/multi_head_attention/attention_output/add:z:00transformer_block/dropout/dropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'transformer_block/dropout/dropout/ShapeShape?transformer_block/multi_head_attention/attention_output/add:z:0*
T0*
_output_shapes
:Å
>transformer_block/dropout/dropout/random_uniform/RandomUniformRandomUniform0transformer_block/dropout/dropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0u
0transformer_block/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=ù
.transformer_block/dropout/dropout/GreaterEqualGreaterEqualGtransformer_block/dropout/dropout/random_uniform/RandomUniform:output:09transformer_block/dropout/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
&transformer_block/dropout/dropout/CastCast2transformer_block/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
'transformer_block/dropout/dropout/Mul_1Mul)transformer_block/dropout/dropout/Mul:z:0*transformer_block/dropout/dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
transformer_block/addAddV2$token_and_position_embedding/add:z:0+transformer_block/dropout/dropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Dtransformer_block/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ì
2transformer_block/layer_normalization/moments/meanMeantransformer_block/add:z:0Mtransformer_block/layer_normalization/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(¾
:transformer_block/layer_normalization/moments/StopGradientStopGradient;transformer_block/layer_normalization/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿë
?transformer_block/layer_normalization/moments/SquaredDifferenceSquaredDifferencetransformer_block/add:z:0Ctransformer_block/layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Htransformer_block/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
6transformer_block/layer_normalization/moments/varianceMeanCtransformer_block/layer_normalization/moments/SquaredDifference:z:0Qtransformer_block/layer_normalization/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(z
5transformer_block/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75ô
3transformer_block/layer_normalization/batchnorm/addAddV2?transformer_block/layer_normalization/moments/variance:output:0>transformer_block/layer_normalization/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
5transformer_block/layer_normalization/batchnorm/RsqrtRsqrt7transformer_block/layer_normalization/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÊ
Btransformer_block/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOpKtransformer_block_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0ø
3transformer_block/layer_normalization/batchnorm/mulMul9transformer_block/layer_normalization/batchnorm/Rsqrt:y:0Jtransformer_block/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÇ
5transformer_block/layer_normalization/batchnorm/mul_1Multransformer_block/add:z:07transformer_block/layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿé
5transformer_block/layer_normalization/batchnorm/mul_2Mul;transformer_block/layer_normalization/moments/mean:output:07transformer_block/layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
>transformer_block/layer_normalization/batchnorm/ReadVariableOpReadVariableOpGtransformer_block_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0ô
3transformer_block/layer_normalization/batchnorm/subSubFtransformer_block/layer_normalization/batchnorm/ReadVariableOp:value:09transformer_block/layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿé
5transformer_block/layer_normalization/batchnorm/add_1AddV29transformer_block/layer_normalization/batchnorm/mul_1:z:07transformer_block/layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
;transformer_block/sequential/dense/Tensordot/ReadVariableOpReadVariableOpDtransformer_block_sequential_dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0{
1transformer_block/sequential/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
1transformer_block/sequential/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       
2transformer_block/sequential/dense/Tensordot/ShapeShape9transformer_block/layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:|
:transformer_block/sequential/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
5transformer_block/sequential/dense/Tensordot/GatherV2GatherV2;transformer_block/sequential/dense/Tensordot/Shape:output:0:transformer_block/sequential/dense/Tensordot/free:output:0Ctransformer_block/sequential/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:~
<transformer_block/sequential/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ë
7transformer_block/sequential/dense/Tensordot/GatherV2_1GatherV2;transformer_block/sequential/dense/Tensordot/Shape:output:0:transformer_block/sequential/dense/Tensordot/axes:output:0Etransformer_block/sequential/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:|
2transformer_block/sequential/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ×
1transformer_block/sequential/dense/Tensordot/ProdProd>transformer_block/sequential/dense/Tensordot/GatherV2:output:0;transformer_block/sequential/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: ~
4transformer_block/sequential/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Ý
3transformer_block/sequential/dense/Tensordot/Prod_1Prod@transformer_block/sequential/dense/Tensordot/GatherV2_1:output:0=transformer_block/sequential/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: z
8transformer_block/sequential/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¨
3transformer_block/sequential/dense/Tensordot/concatConcatV2:transformer_block/sequential/dense/Tensordot/free:output:0:transformer_block/sequential/dense/Tensordot/axes:output:0Atransformer_block/sequential/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:â
2transformer_block/sequential/dense/Tensordot/stackPack:transformer_block/sequential/dense/Tensordot/Prod:output:0<transformer_block/sequential/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:ó
6transformer_block/sequential/dense/Tensordot/transpose	Transpose9transformer_block/layer_normalization/batchnorm/add_1:z:0<transformer_block/sequential/dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿó
4transformer_block/sequential/dense/Tensordot/ReshapeReshape:transformer_block/sequential/dense/Tensordot/transpose:y:0;transformer_block/sequential/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿó
3transformer_block/sequential/dense/Tensordot/MatMulMatMul=transformer_block/sequential/dense/Tensordot/Reshape:output:0Ctransformer_block/sequential/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
4transformer_block/sequential/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:|
:transformer_block/sequential/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ³
5transformer_block/sequential/dense/Tensordot/concat_1ConcatV2>transformer_block/sequential/dense/Tensordot/GatherV2:output:0=transformer_block/sequential/dense/Tensordot/Const_2:output:0Ctransformer_block/sequential/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:í
,transformer_block/sequential/dense/TensordotReshape=transformer_block/sequential/dense/Tensordot/MatMul:product:0>transformer_block/sequential/dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸
9transformer_block/sequential/dense/BiasAdd/ReadVariableOpReadVariableOpBtransformer_block_sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0æ
*transformer_block/sequential/dense/BiasAddBiasAdd5transformer_block/sequential/dense/Tensordot:output:0Atransformer_block/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'transformer_block/sequential/dense/ReluRelu3transformer_block/sequential/dense/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
=transformer_block/sequential/dense_1/Tensordot/ReadVariableOpReadVariableOpFtransformer_block_sequential_dense_1_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0}
3transformer_block/sequential/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
3transformer_block/sequential/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       
4transformer_block/sequential/dense_1/Tensordot/ShapeShape5transformer_block/sequential/dense/Relu:activations:0*
T0*
_output_shapes
:~
<transformer_block/sequential/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ï
7transformer_block/sequential/dense_1/Tensordot/GatherV2GatherV2=transformer_block/sequential/dense_1/Tensordot/Shape:output:0<transformer_block/sequential/dense_1/Tensordot/free:output:0Etransformer_block/sequential/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
>transformer_block/sequential/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ó
9transformer_block/sequential/dense_1/Tensordot/GatherV2_1GatherV2=transformer_block/sequential/dense_1/Tensordot/Shape:output:0<transformer_block/sequential/dense_1/Tensordot/axes:output:0Gtransformer_block/sequential/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:~
4transformer_block/sequential/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Ý
3transformer_block/sequential/dense_1/Tensordot/ProdProd@transformer_block/sequential/dense_1/Tensordot/GatherV2:output:0=transformer_block/sequential/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 
6transformer_block/sequential/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ã
5transformer_block/sequential/dense_1/Tensordot/Prod_1ProdBtransformer_block/sequential/dense_1/Tensordot/GatherV2_1:output:0?transformer_block/sequential/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: |
:transformer_block/sequential/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : °
5transformer_block/sequential/dense_1/Tensordot/concatConcatV2<transformer_block/sequential/dense_1/Tensordot/free:output:0<transformer_block/sequential/dense_1/Tensordot/axes:output:0Ctransformer_block/sequential/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:è
4transformer_block/sequential/dense_1/Tensordot/stackPack<transformer_block/sequential/dense_1/Tensordot/Prod:output:0>transformer_block/sequential/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:ó
8transformer_block/sequential/dense_1/Tensordot/transpose	Transpose5transformer_block/sequential/dense/Relu:activations:0>transformer_block/sequential/dense_1/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿù
6transformer_block/sequential/dense_1/Tensordot/ReshapeReshape<transformer_block/sequential/dense_1/Tensordot/transpose:y:0=transformer_block/sequential/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿù
5transformer_block/sequential/dense_1/Tensordot/MatMulMatMul?transformer_block/sequential/dense_1/Tensordot/Reshape:output:0Etransformer_block/sequential/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
6transformer_block/sequential/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:~
<transformer_block/sequential/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : »
7transformer_block/sequential/dense_1/Tensordot/concat_1ConcatV2@transformer_block/sequential/dense_1/Tensordot/GatherV2:output:0?transformer_block/sequential/dense_1/Tensordot/Const_2:output:0Etransformer_block/sequential/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:ó
.transformer_block/sequential/dense_1/TensordotReshape?transformer_block/sequential/dense_1/Tensordot/MatMul:product:0@transformer_block/sequential/dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
;transformer_block/sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOpDtransformer_block_sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ì
,transformer_block/sequential/dense_1/BiasAddBiasAdd7transformer_block/sequential/dense_1/Tensordot:output:0Ctransformer_block/sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
)transformer_block/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?Ð
'transformer_block/dropout_1/dropout/MulMul5transformer_block/sequential/dense_1/BiasAdd:output:02transformer_block/dropout_1/dropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)transformer_block/dropout_1/dropout/ShapeShape5transformer_block/sequential/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:É
@transformer_block/dropout_1/dropout/random_uniform/RandomUniformRandomUniform2transformer_block/dropout_1/dropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0w
2transformer_block/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=ÿ
0transformer_block/dropout_1/dropout/GreaterEqualGreaterEqualItransformer_block/dropout_1/dropout/random_uniform/RandomUniform:output:0;transformer_block/dropout_1/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
(transformer_block/dropout_1/dropout/CastCast4transformer_block/dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
)transformer_block/dropout_1/dropout/Mul_1Mul+transformer_block/dropout_1/dropout/Mul:z:0,transformer_block/dropout_1/dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÁ
transformer_block/add_1AddV29transformer_block/layer_normalization/batchnorm/add_1:z:0-transformer_block/dropout_1/dropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ftransformer_block/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ò
4transformer_block/layer_normalization_1/moments/meanMeantransformer_block/add_1:z:0Otransformer_block/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(Â
<transformer_block/layer_normalization_1/moments/StopGradientStopGradient=transformer_block/layer_normalization_1/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿñ
Atransformer_block/layer_normalization_1/moments/SquaredDifferenceSquaredDifferencetransformer_block/add_1:z:0Etransformer_block/layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Jtransformer_block/layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:¤
8transformer_block/layer_normalization_1/moments/varianceMeanEtransformer_block/layer_normalization_1/moments/SquaredDifference:z:0Stransformer_block/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(|
7transformer_block/layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75ú
5transformer_block/layer_normalization_1/batchnorm/addAddV2Atransformer_block/layer_normalization_1/moments/variance:output:0@transformer_block/layer_normalization_1/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
7transformer_block/layer_normalization_1/batchnorm/RsqrtRsqrt9transformer_block/layer_normalization_1/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÎ
Dtransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpMtransformer_block_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0þ
5transformer_block/layer_normalization_1/batchnorm/mulMul;transformer_block/layer_normalization_1/batchnorm/Rsqrt:y:0Ltransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ
7transformer_block/layer_normalization_1/batchnorm/mul_1Multransformer_block/add_1:z:09transformer_block/layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿï
7transformer_block/layer_normalization_1/batchnorm/mul_2Mul=transformer_block/layer_normalization_1/moments/mean:output:09transformer_block/layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
@transformer_block/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOpItransformer_block_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0ú
5transformer_block/layer_normalization_1/batchnorm/subSubHtransformer_block/layer_normalization_1/batchnorm/ReadVariableOp:value:0;transformer_block/layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿï
7transformer_block/layer_normalization_1/batchnorm/add_1AddV2;transformer_block/layer_normalization_1/batchnorm/mul_1:z:09transformer_block/layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :Î
global_average_pooling1d/MeanMean;transformer_block/layer_normalization_1/batchnorm/add_1:z:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
dropout_2/dropout/MulMul&global_average_pooling1d/Mean:output:0 dropout_2/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
dropout_2/dropout/ShapeShape&global_average_pooling1d/Mean:output:0*
T0*
_output_shapes
: 
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0e
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ä
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_2/MatMulMatMuldropout_2/dropout/Mul_1:z:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
dropout_3/dropout/MulMuldense_2/Relu:activations:0 dropout_3/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
dropout_3/dropout/ShapeShapedense_2/Relu:activations:0*
T0*
_output_shapes
: 
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0e
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ä
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_3/MatMulMatMuldropout_3/dropout/Mul_1:z:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_3/SigmoidSigmoiddense_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitydense_3/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp8^token_and_position_embedding/embedding/embedding_lookup:^token_and_position_embedding/embedding_1/embedding_lookup?^transformer_block/layer_normalization/batchnorm/ReadVariableOpC^transformer_block/layer_normalization/batchnorm/mul/ReadVariableOpA^transformer_block/layer_normalization_1/batchnorm/ReadVariableOpE^transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpK^transformer_block/multi_head_attention/attention_output/add/ReadVariableOpU^transformer_block/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp>^transformer_block/multi_head_attention/key/add/ReadVariableOpH^transformer_block/multi_head_attention/key/einsum/Einsum/ReadVariableOp@^transformer_block/multi_head_attention/query/add/ReadVariableOpJ^transformer_block/multi_head_attention/query/einsum/Einsum/ReadVariableOp@^transformer_block/multi_head_attention/value/add/ReadVariableOpJ^transformer_block/multi_head_attention/value/einsum/Einsum/ReadVariableOp:^transformer_block/sequential/dense/BiasAdd/ReadVariableOp<^transformer_block/sequential/dense/Tensordot/ReadVariableOp<^transformer_block/sequential/dense_1/BiasAdd/ReadVariableOp>^transformer_block/sequential/dense_1/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2r
7token_and_position_embedding/embedding/embedding_lookup7token_and_position_embedding/embedding/embedding_lookup2v
9token_and_position_embedding/embedding_1/embedding_lookup9token_and_position_embedding/embedding_1/embedding_lookup2
>transformer_block/layer_normalization/batchnorm/ReadVariableOp>transformer_block/layer_normalization/batchnorm/ReadVariableOp2
Btransformer_block/layer_normalization/batchnorm/mul/ReadVariableOpBtransformer_block/layer_normalization/batchnorm/mul/ReadVariableOp2
@transformer_block/layer_normalization_1/batchnorm/ReadVariableOp@transformer_block/layer_normalization_1/batchnorm/ReadVariableOp2
Dtransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpDtransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp2
Jtransformer_block/multi_head_attention/attention_output/add/ReadVariableOpJtransformer_block/multi_head_attention/attention_output/add/ReadVariableOp2¬
Ttransformer_block/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpTtransformer_block/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2~
=transformer_block/multi_head_attention/key/add/ReadVariableOp=transformer_block/multi_head_attention/key/add/ReadVariableOp2
Gtransformer_block/multi_head_attention/key/einsum/Einsum/ReadVariableOpGtransformer_block/multi_head_attention/key/einsum/Einsum/ReadVariableOp2
?transformer_block/multi_head_attention/query/add/ReadVariableOp?transformer_block/multi_head_attention/query/add/ReadVariableOp2
Itransformer_block/multi_head_attention/query/einsum/Einsum/ReadVariableOpItransformer_block/multi_head_attention/query/einsum/Einsum/ReadVariableOp2
?transformer_block/multi_head_attention/value/add/ReadVariableOp?transformer_block/multi_head_attention/value/add/ReadVariableOp2
Itransformer_block/multi_head_attention/value/einsum/Einsum/ReadVariableOpItransformer_block/multi_head_attention/value/einsum/Einsum/ReadVariableOp2v
9transformer_block/sequential/dense/BiasAdd/ReadVariableOp9transformer_block/sequential/dense/BiasAdd/ReadVariableOp2z
;transformer_block/sequential/dense/Tensordot/ReadVariableOp;transformer_block/sequential/dense/Tensordot/ReadVariableOp2z
;transformer_block/sequential/dense_1/BiasAdd/ReadVariableOp;transformer_block/sequential/dense_1/BiasAdd/ReadVariableOp2~
=transformer_block/sequential/dense_1/Tensordot/ReadVariableOp=transformer_block/sequential/dense_1/Tensordot/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
´

?__inference_model_layer_call_and_return_conditional_losses_8874

inputsQ
>token_and_position_embedding_embedding_1_embedding_lookup_8720:	O
<token_and_position_embedding_embedding_embedding_lookup_8726:	h
Rtransformer_block_multi_head_attention_query_einsum_einsum_readvariableop_resource:Z
Htransformer_block_multi_head_attention_query_add_readvariableop_resource:f
Ptransformer_block_multi_head_attention_key_einsum_einsum_readvariableop_resource:X
Ftransformer_block_multi_head_attention_key_add_readvariableop_resource:h
Rtransformer_block_multi_head_attention_value_einsum_einsum_readvariableop_resource:Z
Htransformer_block_multi_head_attention_value_add_readvariableop_resource:s
]transformer_block_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource:a
Stransformer_block_multi_head_attention_attention_output_add_readvariableop_resource:Y
Ktransformer_block_layer_normalization_batchnorm_mul_readvariableop_resource:U
Gtransformer_block_layer_normalization_batchnorm_readvariableop_resource:V
Dtransformer_block_sequential_dense_tensordot_readvariableop_resource:P
Btransformer_block_sequential_dense_biasadd_readvariableop_resource:X
Ftransformer_block_sequential_dense_1_tensordot_readvariableop_resource:R
Dtransformer_block_sequential_dense_1_biasadd_readvariableop_resource:[
Mtransformer_block_layer_normalization_1_batchnorm_mul_readvariableop_resource:W
Itransformer_block_layer_normalization_1_batchnorm_readvariableop_resource:8
&dense_2_matmul_readvariableop_resource:5
'dense_2_biasadd_readvariableop_resource:8
&dense_3_matmul_readvariableop_resource:5
'dense_3_biasadd_readvariableop_resource:
identity¢dense_2/BiasAdd/ReadVariableOp¢dense_2/MatMul/ReadVariableOp¢dense_3/BiasAdd/ReadVariableOp¢dense_3/MatMul/ReadVariableOp¢7token_and_position_embedding/embedding/embedding_lookup¢9token_and_position_embedding/embedding_1/embedding_lookup¢>transformer_block/layer_normalization/batchnorm/ReadVariableOp¢Btransformer_block/layer_normalization/batchnorm/mul/ReadVariableOp¢@transformer_block/layer_normalization_1/batchnorm/ReadVariableOp¢Dtransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp¢Jtransformer_block/multi_head_attention/attention_output/add/ReadVariableOp¢Ttransformer_block/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp¢=transformer_block/multi_head_attention/key/add/ReadVariableOp¢Gtransformer_block/multi_head_attention/key/einsum/Einsum/ReadVariableOp¢?transformer_block/multi_head_attention/query/add/ReadVariableOp¢Itransformer_block/multi_head_attention/query/einsum/Einsum/ReadVariableOp¢?transformer_block/multi_head_attention/value/add/ReadVariableOp¢Itransformer_block/multi_head_attention/value/einsum/Einsum/ReadVariableOp¢9transformer_block/sequential/dense/BiasAdd/ReadVariableOp¢;transformer_block/sequential/dense/Tensordot/ReadVariableOp¢;transformer_block/sequential/dense_1/BiasAdd/ReadVariableOp¢=transformer_block/sequential/dense_1/Tensordot/ReadVariableOpX
"token_and_position_embedding/ShapeShapeinputs*
T0*
_output_shapes
:
0token_and_position_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ|
2token_and_position_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: |
2token_and_position_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:â
*token_and_position_embedding/strided_sliceStridedSlice+token_and_position_embedding/Shape:output:09token_and_position_embedding/strided_slice/stack:output:0;token_and_position_embedding/strided_slice/stack_1:output:0;token_and_position_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(token_and_position_embedding/range/startConst*
_output_shapes
: *
dtype0*
value	B : j
(token_and_position_embedding/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :ã
"token_and_position_embedding/rangeRange1token_and_position_embedding/range/start:output:03token_and_position_embedding/strided_slice:output:01token_and_position_embedding/range/delta:output:0*
_output_shapes	
:É
9token_and_position_embedding/embedding_1/embedding_lookupResourceGather>token_and_position_embedding_embedding_1_embedding_lookup_8720+token_and_position_embedding/range:output:0*
Tindices0*Q
_classG
ECloc:@token_and_position_embedding/embedding_1/embedding_lookup/8720*
_output_shapes
:	*
dtype0
Btoken_and_position_embedding/embedding_1/embedding_lookup/IdentityIdentityBtoken_and_position_embedding/embedding_1/embedding_lookup:output:0*
T0*Q
_classG
ECloc:@token_and_position_embedding/embedding_1/embedding_lookup/8720*
_output_shapes
:	Ç
Dtoken_and_position_embedding/embedding_1/embedding_lookup/Identity_1IdentityKtoken_and_position_embedding/embedding_1/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	}
+token_and_position_embedding/embedding/CastCastinputs*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ
7token_and_position_embedding/embedding/embedding_lookupResourceGather<token_and_position_embedding_embedding_embedding_lookup_8726/token_and_position_embedding/embedding/Cast:y:0*
Tindices0*O
_classE
CAloc:@token_and_position_embedding/embedding/embedding_lookup/8726*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0
@token_and_position_embedding/embedding/embedding_lookup/IdentityIdentity@token_and_position_embedding/embedding/embedding_lookup:output:0*
T0*O
_classE
CAloc:@token_and_position_embedding/embedding/embedding_lookup/8726*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
Btoken_and_position_embedding/embedding/embedding_lookup/Identity_1IdentityItoken_and_position_embedding/embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿü
 token_and_position_embedding/addAddV2Ktoken_and_position_embedding/embedding/embedding_lookup/Identity_1:output:0Mtoken_and_position_embedding/embedding_1/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
Itransformer_block/multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOpRtransformer_block_multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0
:transformer_block/multi_head_attention/query/einsum/EinsumEinsum$token_and_position_embedding/add:z:0Qtransformer_block/multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equationabc,cde->abdeÈ
?transformer_block/multi_head_attention/query/add/ReadVariableOpReadVariableOpHtransformer_block_multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0
0transformer_block/multi_head_attention/query/addAddV2Ctransformer_block/multi_head_attention/query/einsum/Einsum:output:0Gtransformer_block/multi_head_attention/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
Gtransformer_block/multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOpPtransformer_block_multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0
8transformer_block/multi_head_attention/key/einsum/EinsumEinsum$token_and_position_embedding/add:z:0Otransformer_block/multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equationabc,cde->abdeÄ
=transformer_block/multi_head_attention/key/add/ReadVariableOpReadVariableOpFtransformer_block_multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0ü
.transformer_block/multi_head_attention/key/addAddV2Atransformer_block/multi_head_attention/key/einsum/Einsum:output:0Etransformer_block/multi_head_attention/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
Itransformer_block/multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOpRtransformer_block_multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0
:transformer_block/multi_head_attention/value/einsum/EinsumEinsum$token_and_position_embedding/add:z:0Qtransformer_block/multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equationabc,cde->abdeÈ
?transformer_block/multi_head_attention/value/add/ReadVariableOpReadVariableOpHtransformer_block_multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0
0transformer_block/multi_head_attention/value/addAddV2Ctransformer_block/multi_head_attention/value/einsum/Einsum:output:0Gtransformer_block/multi_head_attention/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
,transformer_block/multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ìQ>Ù
*transformer_block/multi_head_attention/MulMul4transformer_block/multi_head_attention/query/add:z:05transformer_block/multi_head_attention/Mul/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
4transformer_block/multi_head_attention/einsum/EinsumEinsum2transformer_block/multi_head_attention/key/add:z:0.transformer_block/multi_head_attention/Mul:z:0*
N*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equationaecd,abcd->acbe¼
6transformer_block/multi_head_attention/softmax/SoftmaxSoftmax=transformer_block/multi_head_attention/einsum/Einsum:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÁ
7transformer_block/multi_head_attention/dropout/IdentityIdentity@transformer_block/multi_head_attention/softmax/Softmax:softmax:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
6transformer_block/multi_head_attention/einsum_1/EinsumEinsum@transformer_block/multi_head_attention/dropout/Identity:output:04transformer_block/multi_head_attention/value/add:z:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equationacbe,aecd->abcdö
Ttransformer_block/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOp]transformer_block_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0Ë
Etransformer_block/multi_head_attention/attention_output/einsum/EinsumEinsum?transformer_block/multi_head_attention/einsum_1/Einsum:output:0\transformer_block/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equationabcd,cde->abeÚ
Jtransformer_block/multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpStransformer_block_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0
;transformer_block/multi_head_attention/attention_output/addAddV2Ntransformer_block/multi_head_attention/attention_output/einsum/Einsum:output:0Rtransformer_block/multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
"transformer_block/dropout/IdentityIdentity?transformer_block/multi_head_attention/attention_output/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
transformer_block/addAddV2$token_and_position_embedding/add:z:0+transformer_block/dropout/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Dtransformer_block/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ì
2transformer_block/layer_normalization/moments/meanMeantransformer_block/add:z:0Mtransformer_block/layer_normalization/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(¾
:transformer_block/layer_normalization/moments/StopGradientStopGradient;transformer_block/layer_normalization/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿë
?transformer_block/layer_normalization/moments/SquaredDifferenceSquaredDifferencetransformer_block/add:z:0Ctransformer_block/layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Htransformer_block/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
6transformer_block/layer_normalization/moments/varianceMeanCtransformer_block/layer_normalization/moments/SquaredDifference:z:0Qtransformer_block/layer_normalization/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(z
5transformer_block/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75ô
3transformer_block/layer_normalization/batchnorm/addAddV2?transformer_block/layer_normalization/moments/variance:output:0>transformer_block/layer_normalization/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
5transformer_block/layer_normalization/batchnorm/RsqrtRsqrt7transformer_block/layer_normalization/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÊ
Btransformer_block/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOpKtransformer_block_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0ø
3transformer_block/layer_normalization/batchnorm/mulMul9transformer_block/layer_normalization/batchnorm/Rsqrt:y:0Jtransformer_block/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÇ
5transformer_block/layer_normalization/batchnorm/mul_1Multransformer_block/add:z:07transformer_block/layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿé
5transformer_block/layer_normalization/batchnorm/mul_2Mul;transformer_block/layer_normalization/moments/mean:output:07transformer_block/layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
>transformer_block/layer_normalization/batchnorm/ReadVariableOpReadVariableOpGtransformer_block_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0ô
3transformer_block/layer_normalization/batchnorm/subSubFtransformer_block/layer_normalization/batchnorm/ReadVariableOp:value:09transformer_block/layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿé
5transformer_block/layer_normalization/batchnorm/add_1AddV29transformer_block/layer_normalization/batchnorm/mul_1:z:07transformer_block/layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
;transformer_block/sequential/dense/Tensordot/ReadVariableOpReadVariableOpDtransformer_block_sequential_dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0{
1transformer_block/sequential/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
1transformer_block/sequential/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       
2transformer_block/sequential/dense/Tensordot/ShapeShape9transformer_block/layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:|
:transformer_block/sequential/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
5transformer_block/sequential/dense/Tensordot/GatherV2GatherV2;transformer_block/sequential/dense/Tensordot/Shape:output:0:transformer_block/sequential/dense/Tensordot/free:output:0Ctransformer_block/sequential/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:~
<transformer_block/sequential/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ë
7transformer_block/sequential/dense/Tensordot/GatherV2_1GatherV2;transformer_block/sequential/dense/Tensordot/Shape:output:0:transformer_block/sequential/dense/Tensordot/axes:output:0Etransformer_block/sequential/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:|
2transformer_block/sequential/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ×
1transformer_block/sequential/dense/Tensordot/ProdProd>transformer_block/sequential/dense/Tensordot/GatherV2:output:0;transformer_block/sequential/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: ~
4transformer_block/sequential/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Ý
3transformer_block/sequential/dense/Tensordot/Prod_1Prod@transformer_block/sequential/dense/Tensordot/GatherV2_1:output:0=transformer_block/sequential/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: z
8transformer_block/sequential/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¨
3transformer_block/sequential/dense/Tensordot/concatConcatV2:transformer_block/sequential/dense/Tensordot/free:output:0:transformer_block/sequential/dense/Tensordot/axes:output:0Atransformer_block/sequential/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:â
2transformer_block/sequential/dense/Tensordot/stackPack:transformer_block/sequential/dense/Tensordot/Prod:output:0<transformer_block/sequential/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:ó
6transformer_block/sequential/dense/Tensordot/transpose	Transpose9transformer_block/layer_normalization/batchnorm/add_1:z:0<transformer_block/sequential/dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿó
4transformer_block/sequential/dense/Tensordot/ReshapeReshape:transformer_block/sequential/dense/Tensordot/transpose:y:0;transformer_block/sequential/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿó
3transformer_block/sequential/dense/Tensordot/MatMulMatMul=transformer_block/sequential/dense/Tensordot/Reshape:output:0Ctransformer_block/sequential/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
4transformer_block/sequential/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:|
:transformer_block/sequential/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ³
5transformer_block/sequential/dense/Tensordot/concat_1ConcatV2>transformer_block/sequential/dense/Tensordot/GatherV2:output:0=transformer_block/sequential/dense/Tensordot/Const_2:output:0Ctransformer_block/sequential/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:í
,transformer_block/sequential/dense/TensordotReshape=transformer_block/sequential/dense/Tensordot/MatMul:product:0>transformer_block/sequential/dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸
9transformer_block/sequential/dense/BiasAdd/ReadVariableOpReadVariableOpBtransformer_block_sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0æ
*transformer_block/sequential/dense/BiasAddBiasAdd5transformer_block/sequential/dense/Tensordot:output:0Atransformer_block/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'transformer_block/sequential/dense/ReluRelu3transformer_block/sequential/dense/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
=transformer_block/sequential/dense_1/Tensordot/ReadVariableOpReadVariableOpFtransformer_block_sequential_dense_1_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0}
3transformer_block/sequential/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
3transformer_block/sequential/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       
4transformer_block/sequential/dense_1/Tensordot/ShapeShape5transformer_block/sequential/dense/Relu:activations:0*
T0*
_output_shapes
:~
<transformer_block/sequential/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ï
7transformer_block/sequential/dense_1/Tensordot/GatherV2GatherV2=transformer_block/sequential/dense_1/Tensordot/Shape:output:0<transformer_block/sequential/dense_1/Tensordot/free:output:0Etransformer_block/sequential/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
>transformer_block/sequential/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ó
9transformer_block/sequential/dense_1/Tensordot/GatherV2_1GatherV2=transformer_block/sequential/dense_1/Tensordot/Shape:output:0<transformer_block/sequential/dense_1/Tensordot/axes:output:0Gtransformer_block/sequential/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:~
4transformer_block/sequential/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Ý
3transformer_block/sequential/dense_1/Tensordot/ProdProd@transformer_block/sequential/dense_1/Tensordot/GatherV2:output:0=transformer_block/sequential/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 
6transformer_block/sequential/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ã
5transformer_block/sequential/dense_1/Tensordot/Prod_1ProdBtransformer_block/sequential/dense_1/Tensordot/GatherV2_1:output:0?transformer_block/sequential/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: |
:transformer_block/sequential/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : °
5transformer_block/sequential/dense_1/Tensordot/concatConcatV2<transformer_block/sequential/dense_1/Tensordot/free:output:0<transformer_block/sequential/dense_1/Tensordot/axes:output:0Ctransformer_block/sequential/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:è
4transformer_block/sequential/dense_1/Tensordot/stackPack<transformer_block/sequential/dense_1/Tensordot/Prod:output:0>transformer_block/sequential/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:ó
8transformer_block/sequential/dense_1/Tensordot/transpose	Transpose5transformer_block/sequential/dense/Relu:activations:0>transformer_block/sequential/dense_1/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿù
6transformer_block/sequential/dense_1/Tensordot/ReshapeReshape<transformer_block/sequential/dense_1/Tensordot/transpose:y:0=transformer_block/sequential/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿù
5transformer_block/sequential/dense_1/Tensordot/MatMulMatMul?transformer_block/sequential/dense_1/Tensordot/Reshape:output:0Etransformer_block/sequential/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
6transformer_block/sequential/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:~
<transformer_block/sequential/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : »
7transformer_block/sequential/dense_1/Tensordot/concat_1ConcatV2@transformer_block/sequential/dense_1/Tensordot/GatherV2:output:0?transformer_block/sequential/dense_1/Tensordot/Const_2:output:0Etransformer_block/sequential/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:ó
.transformer_block/sequential/dense_1/TensordotReshape?transformer_block/sequential/dense_1/Tensordot/MatMul:product:0@transformer_block/sequential/dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
;transformer_block/sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOpDtransformer_block_sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ì
,transformer_block/sequential/dense_1/BiasAddBiasAdd7transformer_block/sequential/dense_1/Tensordot:output:0Ctransformer_block/sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$transformer_block/dropout_1/IdentityIdentity5transformer_block/sequential/dense_1/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÁ
transformer_block/add_1AddV29transformer_block/layer_normalization/batchnorm/add_1:z:0-transformer_block/dropout_1/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ftransformer_block/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ò
4transformer_block/layer_normalization_1/moments/meanMeantransformer_block/add_1:z:0Otransformer_block/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(Â
<transformer_block/layer_normalization_1/moments/StopGradientStopGradient=transformer_block/layer_normalization_1/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿñ
Atransformer_block/layer_normalization_1/moments/SquaredDifferenceSquaredDifferencetransformer_block/add_1:z:0Etransformer_block/layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Jtransformer_block/layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:¤
8transformer_block/layer_normalization_1/moments/varianceMeanEtransformer_block/layer_normalization_1/moments/SquaredDifference:z:0Stransformer_block/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(|
7transformer_block/layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75ú
5transformer_block/layer_normalization_1/batchnorm/addAddV2Atransformer_block/layer_normalization_1/moments/variance:output:0@transformer_block/layer_normalization_1/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
7transformer_block/layer_normalization_1/batchnorm/RsqrtRsqrt9transformer_block/layer_normalization_1/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÎ
Dtransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpMtransformer_block_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0þ
5transformer_block/layer_normalization_1/batchnorm/mulMul;transformer_block/layer_normalization_1/batchnorm/Rsqrt:y:0Ltransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ
7transformer_block/layer_normalization_1/batchnorm/mul_1Multransformer_block/add_1:z:09transformer_block/layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿï
7transformer_block/layer_normalization_1/batchnorm/mul_2Mul=transformer_block/layer_normalization_1/moments/mean:output:09transformer_block/layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
@transformer_block/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOpItransformer_block_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0ú
5transformer_block/layer_normalization_1/batchnorm/subSubHtransformer_block/layer_normalization_1/batchnorm/ReadVariableOp:value:0;transformer_block/layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿï
7transformer_block/layer_normalization_1/batchnorm/add_1AddV2;transformer_block/layer_normalization_1/batchnorm/mul_1:z:09transformer_block/layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :Î
global_average_pooling1d/MeanMean;transformer_block/layer_normalization_1/batchnorm/add_1:z:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
dropout_2/IdentityIdentity&global_average_pooling1d/Mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_2/MatMulMatMuldropout_2/Identity:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
dropout_3/IdentityIdentitydense_2/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_3/MatMulMatMuldropout_3/Identity:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_3/SigmoidSigmoiddense_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitydense_3/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp8^token_and_position_embedding/embedding/embedding_lookup:^token_and_position_embedding/embedding_1/embedding_lookup?^transformer_block/layer_normalization/batchnorm/ReadVariableOpC^transformer_block/layer_normalization/batchnorm/mul/ReadVariableOpA^transformer_block/layer_normalization_1/batchnorm/ReadVariableOpE^transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpK^transformer_block/multi_head_attention/attention_output/add/ReadVariableOpU^transformer_block/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp>^transformer_block/multi_head_attention/key/add/ReadVariableOpH^transformer_block/multi_head_attention/key/einsum/Einsum/ReadVariableOp@^transformer_block/multi_head_attention/query/add/ReadVariableOpJ^transformer_block/multi_head_attention/query/einsum/Einsum/ReadVariableOp@^transformer_block/multi_head_attention/value/add/ReadVariableOpJ^transformer_block/multi_head_attention/value/einsum/Einsum/ReadVariableOp:^transformer_block/sequential/dense/BiasAdd/ReadVariableOp<^transformer_block/sequential/dense/Tensordot/ReadVariableOp<^transformer_block/sequential/dense_1/BiasAdd/ReadVariableOp>^transformer_block/sequential/dense_1/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2r
7token_and_position_embedding/embedding/embedding_lookup7token_and_position_embedding/embedding/embedding_lookup2v
9token_and_position_embedding/embedding_1/embedding_lookup9token_and_position_embedding/embedding_1/embedding_lookup2
>transformer_block/layer_normalization/batchnorm/ReadVariableOp>transformer_block/layer_normalization/batchnorm/ReadVariableOp2
Btransformer_block/layer_normalization/batchnorm/mul/ReadVariableOpBtransformer_block/layer_normalization/batchnorm/mul/ReadVariableOp2
@transformer_block/layer_normalization_1/batchnorm/ReadVariableOp@transformer_block/layer_normalization_1/batchnorm/ReadVariableOp2
Dtransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpDtransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp2
Jtransformer_block/multi_head_attention/attention_output/add/ReadVariableOpJtransformer_block/multi_head_attention/attention_output/add/ReadVariableOp2¬
Ttransformer_block/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpTtransformer_block/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2~
=transformer_block/multi_head_attention/key/add/ReadVariableOp=transformer_block/multi_head_attention/key/add/ReadVariableOp2
Gtransformer_block/multi_head_attention/key/einsum/Einsum/ReadVariableOpGtransformer_block/multi_head_attention/key/einsum/Einsum/ReadVariableOp2
?transformer_block/multi_head_attention/query/add/ReadVariableOp?transformer_block/multi_head_attention/query/add/ReadVariableOp2
Itransformer_block/multi_head_attention/query/einsum/Einsum/ReadVariableOpItransformer_block/multi_head_attention/query/einsum/Einsum/ReadVariableOp2
?transformer_block/multi_head_attention/value/add/ReadVariableOp?transformer_block/multi_head_attention/value/add/ReadVariableOp2
Itransformer_block/multi_head_attention/value/einsum/Einsum/ReadVariableOpItransformer_block/multi_head_attention/value/einsum/Einsum/ReadVariableOp2v
9transformer_block/sequential/dense/BiasAdd/ReadVariableOp9transformer_block/sequential/dense/BiasAdd/ReadVariableOp2z
;transformer_block/sequential/dense/Tensordot/ReadVariableOp;transformer_block/sequential/dense/Tensordot/ReadVariableOp2z
;transformer_block/sequential/dense_1/BiasAdd/ReadVariableOp;transformer_block/sequential/dense_1/BiasAdd/ReadVariableOp2~
=transformer_block/sequential/dense_1/Tensordot/ReadVariableOp=transformer_block/sequential/dense_1/Tensordot/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


D__inference_sequential_layer_call_and_return_conditional_losses_7694
dense_input

dense_7683:

dense_7685:
dense_1_7688:
dense_1_7690:
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCallè
dense/StatefulPartitionedCallStatefulPartitionedCalldense_input
dense_7683
dense_7685*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_7539
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_7688dense_1_7690*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_7575|
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:Y U
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namedense_input
Î
ø
A__inference_dense_1_layer_call_and_return_conditional_losses_9815

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í
a
(__inference_dropout_2_layer_call_fn_9512

inputs
identity¢StatefulPartitionedCall¾
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_8065o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Î
ø
A__inference_dense_1_layer_call_and_return_conditional_losses_7575

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ	
b
C__inference_dropout_3_layer_call_and_return_conditional_losses_8032

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
²
Ò
$__inference_model_layer_call_fn_8709

inputs
unknown:	
	unknown_0:	
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:
identity¢StatefulPartitionedCallã
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_8399o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


ò
A__inference_dense_2_layer_call_and_return_conditional_losses_7924

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ö
a
C__inference_dropout_2_layer_call_and_return_conditional_losses_9517

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ

D__inference_sequential_layer_call_and_return_conditional_losses_7582

inputs

dense_7540:

dense_7542:
dense_1_7576:
dense_1_7578:
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCallã
dense/StatefulPartitionedCallStatefulPartitionedCallinputs
dense_7540
dense_7542*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_7539
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_7576dense_1_7578*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_7575|
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Î
°
0__inference_transformer_block_layer_call_fn_9224

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:
identity¢StatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_transformer_block_layer_call_and_return_conditional_losses_8248t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¹
Ñ
)__inference_sequential_layer_call_fn_7666
dense_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_7642t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namedense_input
Ö
a
C__inference_dropout_3_layer_call_and_return_conditional_losses_9564

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ì

$__inference_dense_layer_call_fn_9745

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÙ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_7539t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ
â
K__inference_transformer_block_layer_call_and_return_conditional_losses_9351

inputsV
@multi_head_attention_query_einsum_einsum_readvariableop_resource:H
6multi_head_attention_query_add_readvariableop_resource:T
>multi_head_attention_key_einsum_einsum_readvariableop_resource:F
4multi_head_attention_key_add_readvariableop_resource:V
@multi_head_attention_value_einsum_einsum_readvariableop_resource:H
6multi_head_attention_value_add_readvariableop_resource:a
Kmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource:O
Amulti_head_attention_attention_output_add_readvariableop_resource:G
9layer_normalization_batchnorm_mul_readvariableop_resource:C
5layer_normalization_batchnorm_readvariableop_resource:D
2sequential_dense_tensordot_readvariableop_resource:>
0sequential_dense_biasadd_readvariableop_resource:F
4sequential_dense_1_tensordot_readvariableop_resource:@
2sequential_dense_1_biasadd_readvariableop_resource:I
;layer_normalization_1_batchnorm_mul_readvariableop_resource:E
7layer_normalization_1_batchnorm_readvariableop_resource:
identity¢,layer_normalization/batchnorm/ReadVariableOp¢0layer_normalization/batchnorm/mul/ReadVariableOp¢.layer_normalization_1/batchnorm/ReadVariableOp¢2layer_normalization_1/batchnorm/mul/ReadVariableOp¢8multi_head_attention/attention_output/add/ReadVariableOp¢Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp¢+multi_head_attention/key/add/ReadVariableOp¢5multi_head_attention/key/einsum/Einsum/ReadVariableOp¢-multi_head_attention/query/add/ReadVariableOp¢7multi_head_attention/query/einsum/Einsum/ReadVariableOp¢-multi_head_attention/value/add/ReadVariableOp¢7multi_head_attention/value/einsum/Einsum/ReadVariableOp¢'sequential/dense/BiasAdd/ReadVariableOp¢)sequential/dense/Tensordot/ReadVariableOp¢)sequential/dense_1/BiasAdd/ReadVariableOp¢+sequential/dense_1/Tensordot/ReadVariableOp¼
7multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0Ü
(multi_head_attention/query/einsum/EinsumEinsuminputs?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equationabc,cde->abde¤
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0Ì
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0Ø
&multi_head_attention/key/einsum/EinsumEinsuminputs=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equationabc,cde->abde 
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0Æ
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0Ü
(multi_head_attention/value/einsum/EinsumEinsuminputs?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equationabc,cde->abde¤
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0Ì
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ìQ>£
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
"multi_head_attention/einsum/EinsumEinsum multi_head_attention/key/add:z:0multi_head_attention/Mul:z:0*
N*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equationaecd,abcd->acbe
$multi_head_attention/softmax/SoftmaxSoftmax+multi_head_attention/einsum/Einsum:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%multi_head_attention/dropout/IdentityIdentity.multi_head_attention/softmax/Softmax:softmax:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿå
$multi_head_attention/einsum_1/EinsumEinsum.multi_head_attention/dropout/Identity:output:0"multi_head_attention/value/add:z:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equationacbe,aecd->abcdÒ
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equationabcd,cde->abe¶
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0é
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout/IdentityIdentity-multi_head_attention/attention_output/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
addAddV2inputsdropout/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:¶
 layer_normalization/moments/meanMeanadd:z:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
-layer_normalization/moments/SquaredDifferenceSquaredDifferenceadd:z:01layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:è
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75¾
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Â
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#layer_normalization/batchnorm/mul_1Muladd:z:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0¾
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)sequential/dense/Tensordot/ReadVariableOpReadVariableOp2sequential_dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0i
sequential/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:p
sequential/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       w
 sequential/dense/Tensordot/ShapeShape'layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:j
(sequential/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ÿ
#sequential/dense/Tensordot/GatherV2GatherV2)sequential/dense/Tensordot/Shape:output:0(sequential/dense/Tensordot/free:output:01sequential/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
*sequential/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
%sequential/dense/Tensordot/GatherV2_1GatherV2)sequential/dense/Tensordot/Shape:output:0(sequential/dense/Tensordot/axes:output:03sequential/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:j
 sequential/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
sequential/dense/Tensordot/ProdProd,sequential/dense/Tensordot/GatherV2:output:0)sequential/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: l
"sequential/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: §
!sequential/dense/Tensordot/Prod_1Prod.sequential/dense/Tensordot/GatherV2_1:output:0+sequential/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: h
&sequential/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : à
!sequential/dense/Tensordot/concatConcatV2(sequential/dense/Tensordot/free:output:0(sequential/dense/Tensordot/axes:output:0/sequential/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:¬
 sequential/dense/Tensordot/stackPack(sequential/dense/Tensordot/Prod:output:0*sequential/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:½
$sequential/dense/Tensordot/transpose	Transpose'layer_normalization/batchnorm/add_1:z:0*sequential/dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
"sequential/dense/Tensordot/ReshapeReshape(sequential/dense/Tensordot/transpose:y:0)sequential/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ½
!sequential/dense/Tensordot/MatMulMatMul+sequential/dense/Tensordot/Reshape:output:01sequential/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"sequential/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:j
(sequential/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ë
#sequential/dense/Tensordot/concat_1ConcatV2,sequential/dense/Tensordot/GatherV2:output:0+sequential/dense/Tensordot/Const_2:output:01sequential/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:·
sequential/dense/TensordotReshape+sequential/dense/Tensordot/MatMul:product:0,sequential/dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0°
sequential/dense/BiasAddBiasAdd#sequential/dense/Tensordot:output:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
+sequential/dense_1/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_1_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0k
!sequential/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:r
!sequential/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       u
"sequential/dense_1/Tensordot/ShapeShape#sequential/dense/Relu:activations:0*
T0*
_output_shapes
:l
*sequential/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
%sequential/dense_1/Tensordot/GatherV2GatherV2+sequential/dense_1/Tensordot/Shape:output:0*sequential/dense_1/Tensordot/free:output:03sequential/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
,sequential/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
'sequential/dense_1/Tensordot/GatherV2_1GatherV2+sequential/dense_1/Tensordot/Shape:output:0*sequential/dense_1/Tensordot/axes:output:05sequential/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
"sequential/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: §
!sequential/dense_1/Tensordot/ProdProd.sequential/dense_1/Tensordot/GatherV2:output:0+sequential/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$sequential/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ­
#sequential/dense_1/Tensordot/Prod_1Prod0sequential/dense_1/Tensordot/GatherV2_1:output:0-sequential/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(sequential/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : è
#sequential/dense_1/Tensordot/concatConcatV2*sequential/dense_1/Tensordot/free:output:0*sequential/dense_1/Tensordot/axes:output:01sequential/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:²
"sequential/dense_1/Tensordot/stackPack*sequential/dense_1/Tensordot/Prod:output:0,sequential/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:½
&sequential/dense_1/Tensordot/transpose	Transpose#sequential/dense/Relu:activations:0,sequential/dense_1/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
$sequential/dense_1/Tensordot/ReshapeReshape*sequential/dense_1/Tensordot/transpose:y:0+sequential/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÃ
#sequential/dense_1/Tensordot/MatMulMatMul-sequential/dense_1/Tensordot/Reshape:output:03sequential/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
$sequential/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:l
*sequential/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ó
%sequential/dense_1/Tensordot/concat_1ConcatV2.sequential/dense_1/Tensordot/GatherV2:output:0-sequential/dense_1/Tensordot/Const_2:output:03sequential/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:½
sequential/dense_1/TensordotReshape-sequential/dense_1/Tensordot/MatMul:product:0.sequential/dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¶
sequential/dense_1/BiasAddBiasAdd%sequential/dense_1/Tensordot:output:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
dropout_1/IdentityIdentity#sequential/dense_1/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
add_1AddV2'layer_normalization/batchnorm/add_1:z:0dropout_1/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:¼
"layer_normalization_1/moments/meanMean	add_1:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ»
/layer_normalization_1/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:î
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Ä
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0È
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%layer_normalization_1/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Ä
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
IdentityIdentity)layer_normalization_1/batchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿø
NoOpNoOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp*^sequential/dense/Tensordot/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp,^sequential/dense_1/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2t
8multi_head_attention/attention_output/add/ReadVariableOp8multi_head_attention/attention_output/add/ReadVariableOp2
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpBmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp2Z
+multi_head_attention/key/add/ReadVariableOp+multi_head_attention/key/add/ReadVariableOp2n
5multi_head_attention/key/einsum/Einsum/ReadVariableOp5multi_head_attention/key/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/query/add/ReadVariableOp-multi_head_attention/query/add/ReadVariableOp2r
7multi_head_attention/query/einsum/Einsum/ReadVariableOp7multi_head_attention/query/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/value/add/ReadVariableOp-multi_head_attention/value/add/ReadVariableOp2r
7multi_head_attention/value/einsum/Einsum/ReadVariableOp7multi_head_attention/value/einsum/Einsum/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2V
)sequential/dense/Tensordot/ReadVariableOp)sequential/dense/Tensordot/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2Z
+sequential/dense_1/Tensordot/ReadVariableOp+sequential/dense_1/Tensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ö
a
C__inference_dropout_2_layer_call_and_return_conditional_losses_7911

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ	
b
C__inference_dropout_3_layer_call_and_return_conditional_losses_9576

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


ò
A__inference_dense_3_layer_call_and_return_conditional_losses_9596

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*«
serving_default
<
input_11
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ;
dense_30
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:´Û
³
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
		optimizer

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
Á
	token_emb
pos_emb
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
ó
att
ffn

layernorm1

layernorm2
dropout1
dropout2
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses"
_tf_keras_layer
¼
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0_random_generator
1__call__
*2&call_and_return_all_conditional_losses"
_tf_keras_layer
»

3kernel
4bias
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses"
_tf_keras_layer
¼
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?_random_generator
@__call__
*A&call_and_return_all_conditional_losses"
_tf_keras_layer
»

Bkernel
Cbias
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses"
_tf_keras_layer

Jiter

Kbeta_1

Lbeta_2
	Mdecay
Nlearning_rate3mÚ4mÛBmÜCmÝOmÞPmßQmàRmáSmâTmãUmäVmåWmæXmçYmèZmé[mê\më]mì^mí_mî`mï3vð4vñBvòCvóOvôPvõQvöRv÷SvøTvùUvúVvûWvüXvýYvþZvÿ[v\v]v^v_v`v"
	optimizer
Æ
O0
P1
Q2
R3
S4
T5
U6
V7
W8
X9
Y10
Z11
[12
\13
]14
^15
_16
`17
318
419
B20
C21"
trackable_list_wrapper
Æ
O0
P1
Q2
R3
S4
T5
U6
V7
W8
X9
Y10
Z11
[12
\13
]14
^15
_16
`17
318
419
B20
C21"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics

	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Þ2Û
$__inference_model_layer_call_fn_8002
$__inference_model_layer_call_fn_8660
$__inference_model_layer_call_fn_8709
$__inference_model_layer_call_fn_8495À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ê2Ç
?__inference_model_layer_call_and_return_conditional_losses_8874
?__inference_model_layer_call_and_return_conditional_losses_9066
?__inference_model_layer_call_and_return_conditional_losses_8550
?__inference_model_layer_call_and_return_conditional_losses_8605À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÊBÇ
__inference__wrapped_model_7501input_1"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,
fserving_default"
signature_map
µ
O
embeddings
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses"
_tf_keras_layer
µ
P
embeddings
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses"
_tf_keras_layer
.
O0
P1"
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
snon_trainable_variables

tlayers
umetrics
vlayer_regularization_losses
wlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
à2Ý
;__inference_token_and_position_embedding_layer_call_fn_9126
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
û2ø
V__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_9150
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 

x_query_dense
y
_key_dense
z_value_dense
{_softmax
|_dropout_layer
}_output_dense
~	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer

layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_sequential
Ë
	axis
	]gamma
^beta
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ë
	axis
	_gamma
`beta
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
+ &call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
¡	variables
¢trainable_variables
£regularization_losses
¤	keras_api
¥_random_generator
¦__call__
+§&call_and_return_all_conditional_losses"
_tf_keras_layer

Q0
R1
S2
T3
U4
V5
W6
X7
Y8
Z9
[10
\11
]12
^13
_14
`15"
trackable_list_wrapper

Q0
R1
S2
T3
U4
V5
W6
X7
Y8
Z9
[10
\11
]12
^13
_14
`15"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¨non_trainable_variables
©layers
ªmetrics
 «layer_regularization_losses
¬layer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
2
0__inference_transformer_block_layer_call_fn_9187
0__inference_transformer_block_layer_call_fn_9224°
§²£
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ð2Í
K__inference_transformer_block_layer_call_and_return_conditional_losses_9351
K__inference_transformer_block_layer_call_and_return_conditional_losses_9491°
§²£
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
­non_trainable_variables
®layers
¯metrics
 °layer_regularization_losses
±layer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
î2ë
7__inference_global_average_pooling1d_layer_call_fn_9496¯
¦²¢
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults¢

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_9502¯
¦²¢
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults¢

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
²non_trainable_variables
³layers
´metrics
 µlayer_regularization_losses
¶layer_metrics
,	variables
-trainable_variables
.regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
(__inference_dropout_2_layer_call_fn_9507
(__inference_dropout_2_layer_call_fn_9512´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ä2Á
C__inference_dropout_2_layer_call_and_return_conditional_losses_9517
C__inference_dropout_2_layer_call_and_return_conditional_losses_9529´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 :2dense_2/kernel
:2dense_2/bias
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
²
·non_trainable_variables
¸layers
¹metrics
 ºlayer_regularization_losses
»layer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
Ð2Í
&__inference_dense_2_layer_call_fn_9538¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ë2è
A__inference_dense_2_layer_call_and_return_conditional_losses_9549¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
¼non_trainable_variables
½layers
¾metrics
 ¿layer_regularization_losses
Àlayer_metrics
;	variables
<trainable_variables
=regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
(__inference_dropout_3_layer_call_fn_9554
(__inference_dropout_3_layer_call_fn_9559´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ä2Á
C__inference_dropout_3_layer_call_and_return_conditional_losses_9564
C__inference_dropout_3_layer_call_and_return_conditional_losses_9576´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 :2dense_3/kernel
:2dense_3/bias
.
B0
C1"
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ánon_trainable_variables
Âlayers
Ãmetrics
 Älayer_regularization_losses
Ålayer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
Ð2Í
&__inference_dense_3_layer_call_fn_9585¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ë2è
A__inference_dense_3_layer_call_and_return_conditional_losses_9596¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
D:B	21token_and_position_embedding/embedding/embeddings
F:D	23token_and_position_embedding/embedding_1/embeddings
I:G23transformer_block/multi_head_attention/query/kernel
C:A21transformer_block/multi_head_attention/query/bias
G:E21transformer_block/multi_head_attention/key/kernel
A:?2/transformer_block/multi_head_attention/key/bias
I:G23transformer_block/multi_head_attention/value/kernel
C:A21transformer_block/multi_head_attention/value/bias
T:R2>transformer_block/multi_head_attention/attention_output/kernel
J:H2<transformer_block/multi_head_attention/attention_output/bias
:2dense/kernel
:2
dense/bias
 :2dense_1/kernel
:2dense_1/bias
9:72+transformer_block/layer_normalization/gamma
8:62*transformer_block/layer_normalization/beta
;:92-transformer_block/layer_normalization_1/gamma
::82,transformer_block/layer_normalization_1/beta
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
0
Æ0
Ç1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÉBÆ
"__inference_signature_wrapper_9117input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
'
O0"
trackable_list_wrapper
'
O0"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ènon_trainable_variables
Élayers
Êmetrics
 Ëlayer_regularization_losses
Ìlayer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
'
P0"
trackable_list_wrapper
'
P0"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ínon_trainable_variables
Îlayers
Ïmetrics
 Ðlayer_regularization_losses
Ñlayer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ô
Òpartial_output_shape
Ófull_output_shape

Qkernel
Rbias
Ô	variables
Õtrainable_variables
Öregularization_losses
×	keras_api
Ø__call__
+Ù&call_and_return_all_conditional_losses"
_tf_keras_layer
ô
Úpartial_output_shape
Ûfull_output_shape

Skernel
Tbias
Ü	variables
Ýtrainable_variables
Þregularization_losses
ß	keras_api
à__call__
+á&call_and_return_all_conditional_losses"
_tf_keras_layer
ô
âpartial_output_shape
ãfull_output_shape

Ukernel
Vbias
ä	variables
åtrainable_variables
æregularization_losses
ç	keras_api
è__call__
+é&call_and_return_all_conditional_losses"
_tf_keras_layer
«
ê	variables
ëtrainable_variables
ìregularization_losses
í	keras_api
î__call__
+ï&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
ð	variables
ñtrainable_variables
òregularization_losses
ó	keras_api
ô_random_generator
õ__call__
+ö&call_and_return_all_conditional_losses"
_tf_keras_layer
ô
÷partial_output_shape
øfull_output_shape

Wkernel
Xbias
ù	variables
útrainable_variables
ûregularization_losses
ü	keras_api
ý__call__
+þ&call_and_return_all_conditional_losses"
_tf_keras_layer
X
Q0
R1
S2
T3
U4
V5
W6
X7"
trackable_list_wrapper
X
Q0
R1
S2
T3
U4
V5
W6
X7"
trackable_list_wrapper
 "
trackable_list_wrapper
¶
ÿnon_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
~	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
2ÿü
ó²ï
FullArgSpece
args]Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults

 

 
p 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2ÿü
ó²ï
FullArgSpece
args]Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults

 

 
p 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Á

Ykernel
Zbias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

[kernel
\bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
<
Y0
Z1
[2
\3"
trackable_list_wrapper
<
Y0
Z1
[2
\3"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ò2ï
)__inference_sequential_layer_call_fn_7593
)__inference_sequential_layer_call_fn_9609
)__inference_sequential_layer_call_fn_9622
)__inference_sequential_layer_call_fn_7666À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Þ2Û
D__inference_sequential_layer_call_and_return_conditional_losses_9679
D__inference_sequential_layer_call_and_return_conditional_losses_9736
D__inference_sequential_layer_call_and_return_conditional_losses_7680
D__inference_sequential_layer_call_and_return_conditional_losses_7694À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
 layers
¡metrics
 ¢layer_regularization_losses
£layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
º2·´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
º2·´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¤non_trainable_variables
¥layers
¦metrics
 §layer_regularization_losses
¨layer_metrics
¡	variables
¢trainable_variables
£regularization_losses
¦__call__
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
º2·´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
º2·´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
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
R

©total

ªcount
«	variables
¬	keras_api"
_tf_keras_metric
c

­total

®count
¯
_fn_kwargs
°	variables
±	keras_api"
_tf_keras_metric
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
.
Q0
R1"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
²non_trainable_variables
³layers
´metrics
 µlayer_regularization_losses
¶layer_metrics
Ô	variables
Õtrainable_variables
Öregularization_losses
Ø__call__
+Ù&call_and_return_all_conditional_losses
'Ù"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
·non_trainable_variables
¸layers
¹metrics
 ºlayer_regularization_losses
»layer_metrics
Ü	variables
Ýtrainable_variables
Þregularization_losses
à__call__
+á&call_and_return_all_conditional_losses
'á"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¼non_trainable_variables
½layers
¾metrics
 ¿layer_regularization_losses
Àlayer_metrics
ä	variables
åtrainable_variables
æregularization_losses
è__call__
+é&call_and_return_all_conditional_losses
'é"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ánon_trainable_variables
Âlayers
Ãmetrics
 Älayer_regularization_losses
Ålayer_metrics
ê	variables
ëtrainable_variables
ìregularization_losses
î__call__
+ï&call_and_return_all_conditional_losses
'ï"call_and_return_conditional_losses"
_generic_user_object
µ2²¯
¦²¢
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults¢

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
µ2²¯
¦²¢
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults¢

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ænon_trainable_variables
Çlayers
Èmetrics
 Élayer_regularization_losses
Êlayer_metrics
ð	variables
ñtrainable_variables
òregularization_losses
õ__call__
+ö&call_and_return_all_conditional_losses
'ö"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
º2·´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
º2·´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ënon_trainable_variables
Ìlayers
Ímetrics
 Îlayer_regularization_losses
Ïlayer_metrics
ù	variables
útrainable_variables
ûregularization_losses
ý__call__
+þ&call_and_return_all_conditional_losses
'þ"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
J
x0
y1
z2
{3
|4
}5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
Y0
Z1"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ðnon_trainable_variables
Ñlayers
Òmetrics
 Ólayer_regularization_losses
Ôlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Î2Ë
$__inference_dense_layer_call_fn_9745¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
é2æ
?__inference_dense_layer_call_and_return_conditional_losses_9776¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
[0
\1"
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Õnon_trainable_variables
Ölayers
×metrics
 Ølayer_regularization_losses
Ùlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ð2Í
&__inference_dense_1_layer_call_fn_9785¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ë2è
A__inference_dense_1_layer_call_and_return_conditional_losses_9815¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
0
0
1"
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
:  (2total
:  (2count
0
©0
ª1"
trackable_list_wrapper
.
«	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
­0
®1"
trackable_list_wrapper
.
°	variables"
_generic_user_object
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
%:#2Adam/dense_2/kernel/m
:2Adam/dense_2/bias/m
%:#2Adam/dense_3/kernel/m
:2Adam/dense_3/bias/m
I:G	28Adam/token_and_position_embedding/embedding/embeddings/m
K:I	2:Adam/token_and_position_embedding/embedding_1/embeddings/m
N:L2:Adam/transformer_block/multi_head_attention/query/kernel/m
H:F28Adam/transformer_block/multi_head_attention/query/bias/m
L:J28Adam/transformer_block/multi_head_attention/key/kernel/m
F:D26Adam/transformer_block/multi_head_attention/key/bias/m
N:L2:Adam/transformer_block/multi_head_attention/value/kernel/m
H:F28Adam/transformer_block/multi_head_attention/value/bias/m
Y:W2EAdam/transformer_block/multi_head_attention/attention_output/kernel/m
O:M2CAdam/transformer_block/multi_head_attention/attention_output/bias/m
#:!2Adam/dense/kernel/m
:2Adam/dense/bias/m
%:#2Adam/dense_1/kernel/m
:2Adam/dense_1/bias/m
>:<22Adam/transformer_block/layer_normalization/gamma/m
=:;21Adam/transformer_block/layer_normalization/beta/m
@:>24Adam/transformer_block/layer_normalization_1/gamma/m
?:=23Adam/transformer_block/layer_normalization_1/beta/m
%:#2Adam/dense_2/kernel/v
:2Adam/dense_2/bias/v
%:#2Adam/dense_3/kernel/v
:2Adam/dense_3/bias/v
I:G	28Adam/token_and_position_embedding/embedding/embeddings/v
K:I	2:Adam/token_and_position_embedding/embedding_1/embeddings/v
N:L2:Adam/transformer_block/multi_head_attention/query/kernel/v
H:F28Adam/transformer_block/multi_head_attention/query/bias/v
L:J28Adam/transformer_block/multi_head_attention/key/kernel/v
F:D26Adam/transformer_block/multi_head_attention/key/bias/v
N:L2:Adam/transformer_block/multi_head_attention/value/kernel/v
H:F28Adam/transformer_block/multi_head_attention/value/bias/v
Y:W2EAdam/transformer_block/multi_head_attention/attention_output/kernel/v
O:M2CAdam/transformer_block/multi_head_attention/attention_output/bias/v
#:!2Adam/dense/kernel/v
:2Adam/dense/bias/v
%:#2Adam/dense_1/kernel/v
:2Adam/dense_1/bias/v
>:<22Adam/transformer_block/layer_normalization/gamma/v
=:;21Adam/transformer_block/layer_normalization/beta/v
@:>24Adam/transformer_block/layer_normalization_1/gamma/v
?:=23Adam/transformer_block/layer_normalization_1/beta/v¡
__inference__wrapped_model_7501~POQRSTUVWX]^YZ[\_`34BC1¢.
'¢$
"
input_1ÿÿÿÿÿÿÿÿÿ
ª "1ª.
,
dense_3!
dense_3ÿÿÿÿÿÿÿÿÿ«
A__inference_dense_1_layer_call_and_return_conditional_losses_9815f[\4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 
&__inference_dense_1_layer_call_fn_9785Y[\4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¡
A__inference_dense_2_layer_call_and_return_conditional_losses_9549\34/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 y
&__inference_dense_2_layer_call_fn_9538O34/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¡
A__inference_dense_3_layer_call_and_return_conditional_losses_9596\BC/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 y
&__inference_dense_3_layer_call_fn_9585OBC/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ©
?__inference_dense_layer_call_and_return_conditional_losses_9776fYZ4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 
$__inference_dense_layer_call_fn_9745YYZ4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ£
C__inference_dropout_2_layer_call_and_return_conditional_losses_9517\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 £
C__inference_dropout_2_layer_call_and_return_conditional_losses_9529\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 {
(__inference_dropout_2_layer_call_fn_9507O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ{
(__inference_dropout_2_layer_call_fn_9512O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ£
C__inference_dropout_3_layer_call_and_return_conditional_losses_9564\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 £
C__inference_dropout_3_layer_call_and_return_conditional_losses_9576\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 {
(__inference_dropout_3_layer_call_fn_9554O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ{
(__inference_dropout_3_layer_call_fn_9559O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿÑ
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_9502{I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ©
7__inference_global_average_pooling1d_layer_call_fn_9496nI¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
ª "!ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ½
?__inference_model_layer_call_and_return_conditional_losses_8550zPOQRSTUVWX]^YZ[\_`34BC9¢6
/¢,
"
input_1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ½
?__inference_model_layer_call_and_return_conditional_losses_8605zPOQRSTUVWX]^YZ[\_`34BC9¢6
/¢,
"
input_1ÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¼
?__inference_model_layer_call_and_return_conditional_losses_8874yPOQRSTUVWX]^YZ[\_`34BC8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¼
?__inference_model_layer_call_and_return_conditional_losses_9066yPOQRSTUVWX]^YZ[\_`34BC8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
$__inference_model_layer_call_fn_8002mPOQRSTUVWX]^YZ[\_`34BC9¢6
/¢,
"
input_1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
$__inference_model_layer_call_fn_8495mPOQRSTUVWX]^YZ[\_`34BC9¢6
/¢,
"
input_1ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
$__inference_model_layer_call_fn_8660lPOQRSTUVWX]^YZ[\_`34BC8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
$__inference_model_layer_call_fn_8709lPOQRSTUVWX]^YZ[\_`34BC8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ½
D__inference_sequential_layer_call_and_return_conditional_losses_7680uYZ[\A¢>
7¢4
*'
dense_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 ½
D__inference_sequential_layer_call_and_return_conditional_losses_7694uYZ[\A¢>
7¢4
*'
dense_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 ¸
D__inference_sequential_layer_call_and_return_conditional_losses_9679pYZ[\<¢9
2¢/
%"
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 ¸
D__inference_sequential_layer_call_and_return_conditional_losses_9736pYZ[\<¢9
2¢/
%"
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 
)__inference_sequential_layer_call_fn_7593hYZ[\A¢>
7¢4
*'
dense_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
)__inference_sequential_layer_call_fn_7666hYZ[\A¢>
7¢4
*'
dense_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
)__inference_sequential_layer_call_fn_9609cYZ[\<¢9
2¢/
%"
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
)__inference_sequential_layer_call_fn_9622cYZ[\<¢9
2¢/
%"
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ°
"__inference_signature_wrapper_9117POQRSTUVWX]^YZ[\_`34BC<¢9
¢ 
2ª/
-
input_1"
input_1ÿÿÿÿÿÿÿÿÿ"1ª.
,
dense_3!
dense_3ÿÿÿÿÿÿÿÿÿ·
V__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_9150]PO+¢(
!¢

xÿÿÿÿÿÿÿÿÿ
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 
;__inference_token_and_position_embedding_layer_call_fn_9126PPO+¢(
!¢

xÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÇ
K__inference_transformer_block_layer_call_and_return_conditional_losses_9351xQRSTUVWX]^YZ[\_`8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 Ç
K__inference_transformer_block_layer_call_and_return_conditional_losses_9491xQRSTUVWX]^YZ[\_`8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ
p
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 
0__inference_transformer_block_layer_call_fn_9187kQRSTUVWX]^YZ[\_`8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
0__inference_transformer_block_layer_call_fn_9224kQRSTUVWX]^YZ[\_`8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ