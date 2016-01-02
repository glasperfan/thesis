--load.lua

--[[
	Create a system for loading data and accessing it in batches.
	Adapted from /group/torch/rnn_lm/data.lua
--]]

require 'hdf5'

-- Constants
num_chorales = 326
training_test_split = 0.1
split_index = math.floor(num_chorales * training_test_split) -- for 326 chorales, that's 32


-- Create the data class to be used in main
local data = torch.class("data")


-- Returns an indexed version and an aggregated version of the dataset
function data:__init(data_dir)
	self.train = {}
	self.test = {}
	self.test_pointer = 0
	-- Test data
	for i = 0, data:max_test_index() do
		local f = hdf5.open(data_dir .. 'test_' .. i .. '.hdf5', 'r')
		self.test[i] = {X = f:read('X'):all(), y = f:read('y'):all()}
		f:close()
		if i == 0 then
			self.test.all = {}
			self.test.all.X = self.test[i].X
			self.test.all.y = self.test[i].y
		else
			self.test.all.X = torch.cat(self.test.all.X, self.test[i].X, 1)
			self.test.all.y = torch.cat(self.test.all.y, self.test[i].y, 1)
		end
	end
	-- Training data
	for i = 0, data:max_train_index() do
		local f = hdf5.open(data_dir .. 'train_' .. i .. '.hdf5', 'r')
		self.train[i] = {X = f:read('X'):all(), y = f:read('y'):all()}
		f:close()
		if i == 0 then
			self.train.all = {}
			self.train.all.X = self.train[i].X
			self.train.all.y = self.train[i].y
		else
			self.train.all.X = torch.cat(self.train.all.X, self.train[i].X, 1)
			self.train.all.y = torch.cat(self.train.all.y, self.train[i].y, 1)
		end
	end
	-- All data
	self.all = {}
	self.all.X = torch.cat(self.train.all.X, self.test.all.X, 1)
	self.all.y = torch.cat(self.train.all.y, self.test.all.y, 1)
	-- Index information
	print("DATA:")
	print(self.all)
end

-- For training
function data:all_train_data()
	return self.train.all
end

function data:train_data()
	return self.train
end

function data:max_train_index()
	return math.floor(num_chorales * (1 - training_test_split))
end

-- For testing
function data:reset_test_pointer()
	self.test_pointer = 0
end

function data:max_test_index()
	return math.floor(num_chorales * training_test_split) - 1
end

function data:next_test()
	local test = self.test[self.test_pointer]
	if self.test_pointer == data:max_test_index() then
		self.test_pointer = 0
	else
		self.test_pointer = self.test_pointer + 1
	end
	return test
end

-- data:__init('data/')