require 'hdf5'
nn = require "nn"
d = 10

f = hdf5.open('chorales.hdf5', 'r')
X  = f:read('X'):all()
y  = f:read('y'):all()
nX = X:size(1) -- number of training examples
nd = X:size(2) -- feature space
ny = y:size(1) -- number of answers
indices = f:read('indices'):all()
min_index = indices[1][1] -- should be 1
max_index = indices[1][2] -- largest unique index for feature representation
output_size = indices[1][3] -- should be the largest index for chords
hidden_size = 25
--print(indices[1][2], nX)
first_example = X[1]
matrix_V = nn.LookupTable(max_index, d)
out = matrix_V:forward(first_example)
sum = nn.Sum(1)
--print(sum:forward(out)) ~ this returns a length-4 Tensor that represents the embedding


-- Putting together the Softmax (multinomial logistic regression for neural networks) --


-- Extract the embedding
function_f = nn.Sequential()
function_f:add(matrix_V)
function_f:add(sum)

-- Model extracts the embedding and then feeds forwards
model = nn.Sequential()
model:add(function_f)
model:add(nn.Linear(d, output_size)) -- max_index (~117) input units, hidden_size ouput units
model:add(nn.LogSoftMax())

pred = model:forward(first_example)


criterion = nn.ClassNLLCriterion() -- Two classes
loss = criterion:forward(pred, y[1])
print(loss)


---- Training ---
model:reset()
learning_rate = 0.01
for epoch = 1, 20 do 
	nll = 0
    model:zeroGradParameters()
    for j = 1, X:size(1) do
    end
end



