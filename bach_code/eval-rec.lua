-- eval.lua

--[[
	Experiment 2: Construct an LSTM for learning the harmonization task.
	Adapted from harvardnlp/group/torch/rnn_lm/simple/main.lua.
--]]

-- Load the data class.
require 'load'

-- Other libraries
require 'nn'
require 'rnn'


--- Make the Model ---
-- nX: max input index
-- d: size of the hidden vectors
-- nY: max output index
-- nL: number of layers 
-- dropout_p: dropout probability
function make_model(nX, d, nY, nL, dropout_p)
   local model = nn.Sequential()
   model.lookups_zero = {}

   model:add(nn.LookupTable(nX, d)) -- d is the embedding vector size (doesn't have to be d)
   model:add(nn.Sum(2))
   model:add(nn.SplitTable(1))

   model:add(nn.Sequencer(nn.FastLSTM(d, d)))
   for j = 2, nL do
      model:add(nn.Sequencer(nn.Dropout(dropout_p)))
      model:add(nn.Sequencer(nn.FastLSTM(d, d)))
   end

   model:add(nn.Sequencer(nn.Dropout(dropout_p)))
   model:add(nn.Sequencer(nn.Linear(d, nY)))
   model:add(nn.Sequencer(nn.LogSoftMax()))
   -- model:add(nn.Sequencer(nn.Sum(1)))
   -- model:add(nn.JoinTable(1))

   model:remember('both') 
   criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())
   
   return model, criterion
end



--- Train the Model ---
function train(data_obj, model, criterion, epochs, param_init, max_grad_norm, learning_rate)
	local last_score = 1e9
	local params, grad_params = model:getParameters()
	params:uniform(param_init, param_init)
	for epoch = 1, epochs do
		model:training()
		for i = 1, 5 do
			model:zeroGradParameters()
			local chorale = data_obj.train[i]
			-- for j = 1, X:size(1) do
			input, goal = chorale.X, chorale.y
			-- Forward --
			local out = model:forward(input)
			print(out)
			local loss = criterion:forward(out, goal)
			-- Backward --
			deriv = criterion:backward(out, goal)
			model:backward(input, deriv)
			-- Grad Norm.
			local grad_norm = grad_params:norm()
			if grad_norm > max_grad_norm then
				grad_params:mul(max_grad_norm / grad_norm)
			end
    		-- Update params
    		params:add(grad_params:mul(-learning_rate))
         	-- end
        end
        -- Evaluate and update learning rate if necessary
		score = eval(data_obj, model)
		if score > last_score - 1 then
			opt.learning_rate = opt.learning_rate / 2
		end
		last_score = score
    end
end



-- Evaluation with NLL --
function eval(data_obj, model)
	model:evaluate()
	data_obj:reset_test_pointer()
	local nll = 0
	while true do
		input, goal = data_obj:next_test()
		out = model:forward(input)
	    nll = nll + criterion:forward(out, goal)
		if data_obj.test_pointer == 0 then 
			-- break
			for i = 1, input:size(1) do
				-- torch.max(torch.LongTensor(out), 1)
				print(nn.JoinTable(2):forward{out})
				max, argmax = torch.max(model:forward(input[i]), 1)
				answer = goal[i]
				print (argmax[1], answer)
			end
			break
		end
	end
	print("Total cost:", nll)
	return nll
end




function main() 
	-- Contants
	local d = 10
	local nL = 2
	local dropout_p = 0.0
	local param_init = 0.05
	local max_grad_norm = 5
	local epochs = 200
	local learning_rate = 0.00001

	-- Create the data loader class.
	local data_obj = data.new('data/')
	local X = data_obj.train.all.X
	local y = data_obj.train.all.y
	local nX = torch.max(data_obj.all.X)
	local nY = torch.max(data_obj.all.y)
	
	model, criterion = make_model(nX, d, nY, nL, dropout_p)

	-- Train
	train(data_obj, model, criterion, epochs, param_init, max_grad_norm, learning_rate)
	-- trainSGD(X, y, model, criterion, learning_rate, epochs, data_obj)
	-- eval(data, testing_model, criterion)
end

main()