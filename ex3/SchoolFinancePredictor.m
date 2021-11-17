num_labels = 5; % 10 labels, from 1 to 10 
lambda = 0.1;
y = Answersvector;
X = SchoolFinanceMatrix(:,1:9);
[X_norm, mu, sigma] = featureNormalize(X);

fmin_iterations = 50;

[all_theta] = oneVsAll(X_norm, y, num_labels, lambda, fmin_iterations);

pred = predictOneVsAll(all_theta, X_norm, y);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);