
# Algorithms-GradientDescent

**Soft SVM:**

Stochastic Gradient Descent used to solve soft-SVM. 
Implemenation of SGD is done in two ways: 

**pratical**

a. in every epoch we sample a permutation which describes the order of sampling from train set.

b. calculate sub-gradient of function according to current observation.

c. update w,b vectors according to subgradients found in b, and according to update rule.

return w,b

**theoretical**

a. we run the algorithm num_samples*epochs time.

b. we sample a unique data point.

c. calculate sub-gradient of function according to current observation.

d. update w,b vectors according to subgradients found in b, and according to update rule.

return mean(w), mean(b)

After above runs, we plot train, test error of each method and compare the two.




**Logistic Regression**

1. Cross validation implemantion.

2. Run 5-folds CV on Logistic Regression model with different lambdas. 

3. plot bar plot for every lambda value and compare train, validation and test errors.
