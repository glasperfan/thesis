--load.lua

--[[
	Loads the new training data for Experiment 2.
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
	local f = hdf5.open("data/all_data_rec.hdf5", 'r')
	self.X_train = f:read('X_train'):all()
	self.X_test = f:read('X_test'):all()
	self.y_train = f:read('y_train'):all()
	self.y_test = f:read('y_test'):all()	
	self.X_all = torch.cat(self.X_train, self.X_test, 1)
	self.y_all = torch.cat(self.y_train, self.y_test, 1)
	-- Print to verify
	print("X_all size")
	print(self.X_all:size())
end

-- Helper functions
function data:all_data()
	return self.X_train, self.y_train, self.X_test, self.y_test
end

function data:max_indices()
	return torch.max(self.X_all), torch.max(self.y_all)
end

-- For testing
function data:reset_test_pointer()
	self.test_pointer = 0
end

function data:next_test()
	local test = self.test[self.test_pointer]
	if self.test_pointer == data:max_test_index() then
		self.test_pointer = 0
	else
		self.test_pointer = self.test_pointer + 1
	end
	return test.X, test.y
end

-- data:__init('data/')