require 'torch'
require 'math'
require 'image'
package.path = package.path .. ";'/home/user/cf/conv_frames/preproc/?.lua'"
require 'clean_action'

function decompose_action(output,action)
  local nframes=action:size()[1]
  local max=torch.max(action)
  local min=get_min_nonzero(action)
  for i=1,nframes do
    local filename=output .. "_" .. tonumber(i) ..".png"
    local frame=action[i]

    print(max)
    print(min)
    local scaled_frame=scale_frame(frame,min,max)
    image.save(filename, scaled_frame)
  end
end

function scale_frame(frame,min,max)
  frame=remove_zero(frame)
  frame=standarize_depth(frame,min,max)
  return image.scale(frame, 40, 80)
end

if table.getn(arg) > 1 then
  local input=arg[1]
  local output=arg[2]
  local action=torch.load(input)
  print(action:size())
  decompose_action(output,action)
end
