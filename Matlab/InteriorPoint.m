syms x1 x2 mu 

f_penalty = Objective_Function([x1,x2]);

constraints = constraint_Functions([x1,x2]);
for i = 1:4
    f_penalty = f_penalty + mu*log(-constraints(i));
end

x0 = [0.4,0.4]; % Initializing with a feasible point
x_old = x0;
lambda = 0.005; % Making it small so it won't move outside the feasible solution
mu_value = 0.1;

while mu_value < 10^8
    mu_value = mu_value*10;
end


 % Intialization value; increasing 0.1,1,10,100,1000...
f_penalty = subs(f_penalty,mu,mu_value);

grad_f = gradient(f_penalty);
dk(1) = -subs(grad_f(1),{x1,x2},{x_old(1),x_old(2)});
dk(2) = -subs(grad_f(2),{x1,x2},{x_old(1),x_old(2)});
searchDirection = [dk(1);dk(2)].';
x_new = x_old + lambda*searchDirection;
toggle = evaluateFeasibility(x_new);
isDecreasing = isDecreasingFunc(f_penalty,x_new,x_old);


while isDecreasing % && toggle -> this will be used in the bisection method instead
    mu_value = 10 * mu_value;
    if (f_new - f_old) < (alpha*lambda*dk*g_old) || (dk*g_new) > (neta*dk*g_old)
            break;
        else
            lambda = lambda/5;
            x_new = x_old + lambda*dk';
            %toggle = evaluateFeasibility(x_new);
            %if ~toggle
            %    continue;
            %end
            f_new = subs(f,{x1,x2},{x_new(1),x_new(2)});
            g_new = subs(grad_f,{x1,x2},{x_new(1),x_new(2)});
    end
end

function isDecreasing = isDecreasingFunc(f_penalty,x_new,x_old)
    syms x1 x2 
    epsilon_relativeFuncChange = 10^-8;
    f_new_value = subs(f_penalty,{x1,x2},{x_new(1),x_new(2)});
    f_old_value = subs(f_penalty,{x1,x2},{x_old(1),x_old(2)});
    isDecreasing = false;
    if abs((f_new_value - f_old_value)/(f_old_value)) < epsilon_relativeFuncChange
        isDecreasing = true;
    end
end


function toggle = evaluateFeasibility(x_new)
    constraints = constraint_Functions([x_new(1),x_new(2)]);
    toggle = true;
    for i=1:4
        if constraints(i) < 0
            toggle = false;
        end
    end

end