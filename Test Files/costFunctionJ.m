function J = costFunctionJ(X,y,theta);

%X is the "design matrix" containing our traiuning examples
%Y is the class labels
m = size(X,1); % number of training examples)
predictions = X*theta; % Predictions of hypothesis on all m examples 
sqrErrors = (predictions-y).^2; %squared errors
J = 1/(2*m) * sum(sqrErrors);