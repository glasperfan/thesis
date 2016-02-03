-- eval_tree.lua

--[[
	Experiment 2B: Hierarchal Softmax
--]]

-- Other libraries
require 'nn'
require 'nnx'
require 'hdf5'

--- Neural network ---
function make_model(max_index)
	-- Embedding sequence
	local embedding = nn.Sequential()
	embedding:add(nn.LookupTable(max_index, embedding_size))
	embedding:add(nn.Sum(1))

	-- Feed forward sequence
	local model = nn.Sequential()
	model:add(embedding)
	
	model:add(nn.Linear(embedding_size, hidden_size))
	model:add(nn.Tanh())
	smt = nn.SoftMaxTree(hidden_size, tree, root_id, _, _, true)

	-- Criterion: negative log likelihood
	local criterion = nn.TreeNLLCriterion()

	return model, criterion, smt
end


--- Train the Model ---
function train(Xtrain, ytrain, Xtst, ytst, model, criterion, smt)
	for epoch = 1, epochs do
		model:zeroGradParameters()
		smt:zeroGradParameters()
		local res = model:forward(Xtrain)
		local out = smt:forward{res, ytrain}
		local nll = criterion:forward(out, ytrain)
		local deriv = criterion:backward(out, ytrain)
		local smt_deriv = smt:backward({res, ytrain}, deriv)
		local gradInput = smt_deriv[1]
		model:backward(Xtrain, gradInput)
		smt:updateParameters(learning_rate)
		model:updateParameters(learning_rate)
		print("Epoch:", epoch, nll)
		eval(Xtrain, ytrain, model, criterion, smt)
		eval(Xtst, ytst, model, criterion, smt)
	end
end



-- Evaluation numeral subtask --
function eval(Xt, yt, model, criterion, smt)
	-- Collect numeral predictions
	local res = model:forward(Xt)
	local out = smt:forward{res, yt}
	local nll = criterion:forward(out, yt)
	print("Test NLL: ", nll)
end


function range(min, max)
	local arr = {}
	for i = min, max - 1 do table.insert(arr, i) end
	return torch.IntTensor(arr)
end

-- Computes the leaf index in the tree
function lookup(y1, y2, y3, y4)
	level1 = tree[root_id]
	node1 = level1[y1]
	level2 = tree[node1]
	node2 = level2[y2]
	level3 = tree[node2]
	node3 = level3[y3]
	level4 = tree[node3]
	leaf = level4[y4]
	return leaf
end


function main() 
	-- Contants
	embedding_size = 250
	hidden_size = 250
	epochs = 20
	learning_rate = 0.01

	-- Load data
	local f = hdf5.open("data/chorales.hdf5")
	local Xtrain = f:read('Xtrain'):all()
	local Xdev = f:read('Xdev'):all()
	local Xtest = f:read('Xtest'):all()
	local ytrain = f:read('ytrain'):all()
	local ydev = f:read('ydev'):all()
	local ytest = f:read('ytest'):all()
	local yall = torch.cat(ytrain, torch.cat(ydev, ytest, 1), 1)
	f:close()
	
	local f = hdf5.open("data/choralestree.hdf5")
	local ytrainTree = f:read('ytrainTree'):all()
	local ydevTree = f:read('ydevTree'):all()
	local ytestTree = f:read('ytestTree'):all()
	local yallTree = torch.cat(ytrainTree, torch.cat(ydevTree, ytestTree, 1), 1)
	f:close()
	
	-- Select the training data, excluding the final two columns (features related to previous input) --
	Xtrain = Xtrain[{ {}, {1,10} }]
	Xdev = Xdev[{ {}, {1,10} }]
	Xtest = Xtest[{ {}, {1,10} }]
	Xall = torch.cat(Xtrain, torch.cat(Xdev, Xtest, 1), 1)
	local max_index = torch.max(Xall)

	-- Create the tree --
	counter = 1
	tree = {}
	root_id = 410725
	numeralSize = yall[{ {}, 1 }]:max()
	inversionSize = yall[{ {}, 2 }]:max()
	altoSize = yall[{ {}, 3 }]:max()
	tenorSize = yall[{ {}, 4 }]:max()

	tree[root_id] = range(counter, counter + numeralSize)
	counter = counter + numeralSize
	for el = 1, numeralSize do
	    tree[el] = range(counter, counter + inversionSize)
	    counter = counter + inversionSize
	end
	inv_min, inv_max = tree[tree[root_id][1]][1], counter - 1
	for el = inv_min, inv_max do
	    tree[el] = range(counter, counter + altoSize)
	    counter = counter + altoSize
	end
	alt_min, alt_max = tree[inv_min][1], counter - 1
	for el = alt_min, alt_max do
	    tree[el] = range(counter, counter + tenorSize)
	    counter = counter + tenorSize
	end


	-- Create global models and criterion
	model, criterion, smt = make_model(max_index)

	-- Train
	train(Xtrain, ytrainTree, Xtest, ytestTree, model, criterion, smt)
end

main()