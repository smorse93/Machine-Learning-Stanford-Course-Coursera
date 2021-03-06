function p = predictOneVsAll(all_theta, X, y)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
%m = 106 for this 

num_labels = size(all_theta, 1);
%num_labels = 5

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);
%p = 106 zeros
% Add ones to the X data matrix
X = [ones(m, 1) X];

predict = sigmoid(X*all_theta');
fprintf("this is predict");

v = zeros(106,2)
v(:,1) = y(:)

[~,p]=max(predict,[],2);
disp(max(predict,[],2));


for i = 1:m
    n = 0; 
    largest = 0;
    for j = 1:num_labels
        if (n < predict(i,j))
            n = predict(i,j);
            largest = j;
        end
    end
    
    v(i,2) = largest;
end
fprintf("these are the values vs the ones predicted");
disp(v);


% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       







% =========================================================================


end
