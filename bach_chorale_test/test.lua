require 'hdf5'
nn = require "nn"


--- Make the Model ---
function make_model(max_index, embedding_size, hidden_size, output_size)
	lookup = nn.LookupTable(max_index, embedding_size)
	sum = nn.Sum(1)
	hidden_size = (embedding_size + output_size) / 2

	-- Embedding sequence
	function_f = nn.Sequential()
	function_f:add(lookup)
	function_f:add(sum)

	-- Feed forward sequence
	model = nn.Sequential()
	model:add(function_f)
	model:add(nn.Linear(embedding_size, hidden_size))
	model:add(nn.Tanh())
	model:add(nn.Linear(hidden_size, output_size))
	model:add(nn.LogSoftMax())

	return model
end

---- Training (runs while until loss hits a local minimum) ---
function train(X, y, model, criterion)
	model:reset() -- reset the internal gradient accumulator

	learning_rate = 0.001
	for epoch = 1, 20 do 
	    nll = 0
	    model:zeroGradParameters()
	    for j = 1, X:size(1) do
	        -- Forward Pass
	        out = model:forward(X[j])
	        nll = nll + criterion:forward(out, y[j])

	        -- Backward Pass
	        deriv = criterion:backward(out, y[j])
	        model:backward(X[j], deriv)
	    end
	    -- Update the parameters
	    model:updateParameters(learning_rate)
	    print("Epoch:", epoch, nll)
	end
	return model 
end

--- Other training method ---
function trainer(X, y, model, criterion)
	dataset = {}
	function dataset:size() return X:size(1) end
	for j = 1, dataset:size() do
		dataset[j] = {X[j], y[j]}
	end
	trainer = nn.StochasticGradient(model, criterion)
	trainer.learningRate = 0.003
	trainer.maxIteration = 20
	trainer:train(dataset)

	return model
end

--- Evaluate results on a new chorale ---
function eval(X, y, model)
	print("Prediction", "Actual")
	i = 0
	for j = 1, X:size(1) do
		out = model:forward(X[j])
		top_val, index = out:max(1)
		print(index[1], y[j])
		if index[1] == y[j] then i = i + 1 end
	end
	print("Total", i, i / X:size(1))

	-- TODO: confusion matrix
end

function main()
	--- Learning constants ---
	embedding_size = 25

	f = hdf5.open('chorales.hdf5', 'r')
	X_train  = f:read('X_train'):all()
	y_train  = f:read('y_train'):all()
	X_test  = f:read('X_test'):all()
	y_test  = f:read('y_test'):all()
	indices = f:read('indices'):all()
	min_index = indices[1][1] -- should be 1
	max_index = indices[1][2] -- largest unique index for feature representation
	output_size = indices[1][3] -- should be the largest index for chords
	hidden_size = (embedding_size + output_size) / 2
	criterion = nn.ClassNLLCriterion()

	model = make_model(max_index, embedding_size, hidden_size, output_size)

	updated_model = trainer(X_train, y_train, model, criterion)

	-- Evaluate training --
	eval(X_test, y_test, updated_model)
end

main()


