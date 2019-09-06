# Linear_Regression_and_Logistic_Regression

This Project is an implementation of Logistic Regression model and Linear Regression model that is able to make such predictions for given data point. I used gradient Descent optimization techniques for doing so.

Let’s say we have n dimensional of input feature. A single training example will be represented as (x,y) where x is n dimensional feature vector and y is label (0/1, True/False etc.). m denotes the total number of training examples. So, X will have n rows and m columns. (n X m dimensional vector) and Y will have 1 row and m columns (1 X m dimensional vector).
The problem statement formulations turn out to be given X, we need to calculate ŷ = P( y=1 | X). What this means is that we need to calculate the probability of target variable to be 1 (or 0) given the training set X.
To solve the problem using logistic regression we take two parameters w, which is n dimensional vector and b which is a real number.

The logistic regression model to solve this is :
Equation for Logistic Regression

![image](https://miro.medium.com/max/520/1*xDjD0feFXCHkhgqMHYFvrg.png)

I applied sigmoid function so that I contain the result of ŷ between 0 and 1 (probability value). The sigmoid function definition is as follows:

![image](https://miro.medium.com/max/491/1*qJRi0QyZQAzcjRPI5zem-A.png)

Sigmoid function

When implementing logistic regression, our job is to learn parameters w and b so that ŷ is approximately equal to the test target . To learn the parameters w and b, we need to define a cost function which we would use to train the logistic regression model. A cost function is an estimator of how good or bad our model is in predicting the known output in general. But before that let us understand what a loss (error) function is. Simply putting it in a mathematical form, what we really want is:

![image](https://miro.medium.com/max/1078/1*WQsr-Mo1nQKysOjyh5T6Ug.png)

Loss Function could be defined as

![image](https://miro.medium.com/max/581/1*zoWobz6RAVkPrCG290-nyg.png)

But, historically it has been found that using the above loss function, optimization problem becomes non-convex. So, we end up with multiple local minima. Hence, we use the following loss function definition, which plays a similar role as squared error but turns the optimization problem convex.

Loss Function used in Logistic Regression
The important thing to note here is that loss function defines how well we are predicting in a single training example. To understand that, we define a cost function. The cost function is defined as:

![image](https://miro.medium.com/max/1877/1*9s9vtYRULW85fD0ol1H4jA.png)

Basically, the cost function measures how well our parameters w and b are doing on the training data set. So, it seems natural to minimize the cost function for minimal error across the training data set to find w and b. We would achieve the value of the parameters using gradient descent technique.

In real examples, w can be a much higher dimension. J(w,b) becomes a surface as shown above for various values of w and b. What we really want is to find w and b where the value of cost function is minimum (shown by the red arrow).
In the upcoming article (Part -2), we would go further to understand how gradient descent actually works and understand the mathematics to solve w and b.

--Credit [Here](https://medium.com/technology-nineleaps/logistic-regression-gradient-descent-optimization-part-1-ed320325a67e)
