require 'torch'
require 'math'
require 'image'

function remove_zero(frame)
  local min,max=find_extrema(frame)
  return frame:sub(min[1],max[1],min[2],max[2])
end

function standarize_depth(frame,min,max)
  local diff=max-min
  frame:apply(function(x)  
    if(x==0) then
      return x
    else
      return (diff -(x-min))/diff
    end
  end)
  return frame
end

function get_min_nonzero(action)
  local min=99999
  local dim=action:size()
  for t_i=1,dim[1] do
    for x_i=1,dim[2] do
      for y_i=1,dim[3] do
        local frame=action[t_i]
        if nonzero(frame,x_i,y_i) and (frame[x_i][y_i] < min) then
          
           min=frame[x_i][y_i]
        end
      end
    end
  end
  return min
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

function show_nonzero(frame)
  local dim=frame:size()
  for x_i=1,dim[1] do
    for y_i=1,dim[2] do
      if nonzero(frame,x_i,y_i) then
         print(frame[x_i][y_i]) 
      end
    end
  end
end

