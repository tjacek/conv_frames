require 'torch'
require 'math'
require 'image'

function decompose_action(output,action)
  local nframes=action:size()[1]
  for i=1,nframes do
    local filename=output .. "_" .. tonumber(i) ..".png"
    local frame=action[i]
    local scaled_frame=scale_frame(frame)
    image.save(filename, scaled_frame)
  end
end

function scale_frame(frame)
  frame=remove_zero(frame)
  return image.scale(frame, 40, 80)
end

function remove_zero(frame)
  local min,max=find_extrema(frame)
  return frame:sub(min[1],max[1],min[2],max[2])
end

function find_extrema(frame)
  local dim=frame:size()
  local min={dim[1]+1,dim[2]+1}
  local max={0,0}
  for x_i=1,dim[1] do
    for y_i=1,dim[2] do
      if nonzero(frame,x_i,y_i) then
        if x_i<min[1] then
          min[1]=x_i
        end
        if y_i<min[2] then
          min[2]=y_i
        end
        if x_i>max[1] then
          max[1]=x_i
        end
        if y_i>max[2] then
          max[2]=y_i
        end
      end
    end
  end
  return min,max
end

function nonzero(frame,x,y)
  return not (frame[x][y]==0)
end

if table.getn(arg) > 1 then
  local input=arg[1]
  local output=arg[2]
  local action=torch.load(input)
  print(action:size())
  decompose_action(output,action)
end
