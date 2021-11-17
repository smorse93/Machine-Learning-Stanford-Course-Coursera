% Stripped down cost function
function [J, grad] = lrCostFunction(theta, X, y, lambda)

m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));

z = X*theta;

%get hypothesis function
h_theta = sigmoid(z)

%get cost from running hypothesis across each value of h_theta and y in
%Logistic regression manner
J = (1/m) * (-y' * log(h_theta) - (1-y)' * log(1-h_theta)) + (lambda/(2*m)) * (theta(2:length(theta)))' * theta(2:length(theta));

    
errors_vector = h_theta-y;
theta_zero = theta;
theta_zero(1)=0;

%find the grad from the errors 
 grad = (1/m).*(X')*errors_vector  +(lambda/m)*theta_zero;
 