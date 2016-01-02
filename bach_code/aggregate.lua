--aggregate.lua

--[[
	Preprocessing: consolidate training sets.
	Data is stored in the data/ folder.
--]]
require 'hdf5'

local clock = os.clock
function sleep(n)  -- seconds
  local t0 = clock()
  while clock() - t0 <= n do end
end


data_dir = "data/"
num_chorales = 326
training_test_split = 0.1
outfile_name = "train.hdf5"

local data_train = {}
for j = 0, math.floor(num_chorales * (1 - training_test_split)) do
	print(j)
	local f = hdf5.open(data_dir .. 'train_' .. j .. '.hdf5', 'r')
	local d = {X = f:read('X'):all(), y = f:read('y'):all()}
	f:close()
	if j == 0 then
		data_train.all = {}
		data_train.all.X = d.X
		data_train.all.y = d.y
	else
		data_train.all.X = torch.cat(data_train.all.X, d.X, 1)
		data_train.all.y = torch.cat(data_train.all.y, d.y, 1)
	end

end
print("DATA:")
-- local writeFile = hdf5.open(data_dir .. outfile_name, 'w')
-- writeFile:write('data', data)
-- writeFile:close()
print(data_train.all)
return data