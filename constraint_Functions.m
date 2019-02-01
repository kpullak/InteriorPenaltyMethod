% Written all the constraint functions in a matrix by converting 
% them in to the form of <= 0
function y = constraint_Functions(x)
    y = [4*x(1)^2 + x(2)^2 - 16; 3*x(1) + 5*x(2) - 4; -x(1); -x(2)];
end