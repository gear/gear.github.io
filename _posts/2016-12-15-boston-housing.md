
## Dataset: Boston housing

_First project of Udacity Machine Learning Nanodegree_

A description of the dataset can be found [here](https://archive.ics.uci.edu/ml/datasets/Housing). This dataset concerns housing values in suburbs of Boston. There is **506 instances** of **14 attributes** each in the dataset. Generally, this dataset is suitable for regression task. Attributes in the datasets:

- CRIM: Per capita crime rate by town.
- ZN: Proportion of residental land zoned for lots over 25,000 sq.ft.
- INDUS: Proportion of non-retail business acres per town.
- CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
- NOX: Nitric oxides concentration (parts per 10 million).
- RM: Average number of rooms per dwelling.
- AGE: Proportion of owner-occupied units built prior to 1940.
- DIS: Weighted distances to five Boston employment centres.
- RAD: Index of accessibility to radial highways.
- TAX: Full-value property-tax rate per \$10,000.
- PTRATIO: pupil-teacher ratio by town.
- B: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town.
- LSTAT: % lower status of the population.
- MEDV: Median value of owner-occupied homes in \$1000's (usually the target).

The Boston housing dataset is shipped with `scikit-learn`. The same description of the data as above can be obtained from `scikit-learn.datasets.load_boston().DESCR`.

## Requirement

This project requires `Python-3.5.2`, `jupyter-1.0.0`, `numpy-1.11.2`, `scikit-learn-0.18.1`, and `matplotlib-1.5.3` installed. I recommend to use Anaconda to manage Python virtual environments and packages.

First, we import necessary packages: 

- `numpy` for numeric computations.
- `matplotlib.pyplot` for visualization. (inline means the figures are shown in the notebook)
- `sklearn` for boston housing dataset and decision tree model.


{% highlight python %}
# Importing a few necessary libraries
import numpy as np
import matplotlib.pyplot as pl
import pandas as pd
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor

# Make matplotlib show plots inline
%matplotlib inline

# Create our client's feature set for 
# which we will be predicting a selling price
CLIENT_FEATURES = [[11.95, 0.00, 18.100, 0, 0.6590, 5.6090, 90.00, \
                    1.385, 24, 680.0, 20.20, 332.09, 12.13]]

# Load the Boston Housing dataset into the city_data variable
city_data = datasets.load_boston()

# Initialize the housing prices and housing features
housing_prices = city_data.target
housing_features = city_data.data

print("Boston Housing dataset loaded successfully!")
{% endhighlight %}

    Boston Housing dataset loaded successfully!



{% highlight python %}
city_data.feature_names
{% endhighlight %}




    array(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
           'TAX', 'PTRATIO', 'B', 'LSTAT'], 
          dtype='<U7')



I would like to see the data in a nice table format, so I load the data into a `pandas.DataFrame` and printed the first five rows with `.head()`.


{% highlight python %}
pdict = {'CRIM': city_data.data[:,0], 
         'ZN': city_data.data[:,1], 
         'INDUS': city_data.data[:,2], 
         'CHAS': city_data.data[:,3], 
         'NOX': city_data.data[:,4], 
         'RM': city_data.data[:,5],
         'AGE': city_data.data[:,6], 
         'DIS': city_data.data[:,7], 
         'RAD': city_data.data[:,8], 
         'TAX': city_data.data[:,9], 
         'PTRATIO': city_data.data[:,10], 
         'B': city_data.data[:,11], 
         'LSTAT': city_data.data[:,12], 
         'MEDV': city_data.target[:]}
ptable = pd.DataFrame(pdict)
{% endhighlight %}


{% highlight python %}
ptable.head()
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AGE</th>
      <th>B</th>
      <th>CHAS</th>
      <th>CRIM</th>
      <th>DIS</th>
      <th>INDUS</th>
      <th>LSTAT</th>
      <th>MEDV</th>
      <th>NOX</th>
      <th>PTRATIO</th>
      <th>RAD</th>
      <th>RM</th>
      <th>TAX</th>
      <th>ZN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>65.2</td>
      <td>396.90</td>
      <td>0.0</td>
      <td>0.00632</td>
      <td>4.0900</td>
      <td>2.31</td>
      <td>4.98</td>
      <td>24.0</td>
      <td>0.538</td>
      <td>15.3</td>
      <td>1.0</td>
      <td>6.575</td>
      <td>296.0</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>78.9</td>
      <td>396.90</td>
      <td>0.0</td>
      <td>0.02731</td>
      <td>4.9671</td>
      <td>7.07</td>
      <td>9.14</td>
      <td>21.6</td>
      <td>0.469</td>
      <td>17.8</td>
      <td>2.0</td>
      <td>6.421</td>
      <td>242.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>61.1</td>
      <td>392.83</td>
      <td>0.0</td>
      <td>0.02729</td>
      <td>4.9671</td>
      <td>7.07</td>
      <td>4.03</td>
      <td>34.7</td>
      <td>0.469</td>
      <td>17.8</td>
      <td>2.0</td>
      <td>7.185</td>
      <td>242.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>45.8</td>
      <td>394.63</td>
      <td>0.0</td>
      <td>0.03237</td>
      <td>6.0622</td>
      <td>2.18</td>
      <td>2.94</td>
      <td>33.4</td>
      <td>0.458</td>
      <td>18.7</td>
      <td>3.0</td>
      <td>6.998</td>
      <td>222.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>54.2</td>
      <td>396.90</td>
      <td>0.0</td>
      <td>0.06905</td>
      <td>6.0622</td>
      <td>2.18</td>
      <td>5.33</td>
      <td>36.2</td>
      <td>0.458</td>
      <td>18.7</td>
      <td>3.0</td>
      <td>7.147</td>
      <td>222.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



## Statistical Analysis and Data Exploration

Let's quickly investigate a few basic statistics about the dataset and look at the `CLIENT_FEATURES` to see how the data relates to it.


{% highlight python %}
# Number of houses and features in the dataset
total_houses, total_features = city_data.data.shape

# Minimum housing value in the dataset
minimum_price = housing_prices.min()

# Maximum housing value in the dataset
maximum_price = housing_prices.max()

# Mean house value of the dataset
mean_price = housing_prices.mean()

# Median house value of the dataset
median_price = np.median(housing_prices)

# Standard deviation of housing values of the dataset
std_dev = housing_prices.std()

# Show the calculated statistics
print("Boston Housing dataset statistics (in $1000's):\n")
print("Total number of houses:", total_houses)
print("Total number of features:", total_features)
print("Minimum house price:", minimum_price)
print("Maximum house price:", maximum_price)
print("Mean house price: {0:.3f}".format(mean_price))
print("Median house price:", median_price)
print("Standard deviation of house price: {0:.3f}".format(std_dev))
{% endhighlight %}

    Boston Housing dataset statistics (in $1000's):
    
    Total number of houses: 506
    Total number of features: 13
    Minimum house price: 5.0
    Maximum house price: 50.0
    Mean house price: 22.533
    Median house price: 21.2
    Standard deviation of house price: 9.188


By intuition, the top 3 deciding factors is crime rate (CRIM), proportion of blacks (B), and the accessibility to the highway (RAD).

- **CRIM**: Area with low crime rate must have higher security, income, insurrance, and better life in general. Hence the price of houses must be affected by this factor.
- **B**: Many people might think that area with many blacks will have be not so safe. Therefore the price might be higher for residence with smaller blacks porpotion. 
- **RAD**: The accessibility to the highway might also be desirable as it is more convenient to go to work.

Let's examine our client. There features we selected have the index `0` (CRIM), `8` (RAD), and `11` (B).


{% highlight python %}
print(CLIENT_FEATURES)
{% endhighlight %}

    [[11.95, 0.0, 18.1, 0, 0.659, 5.609, 90.0, 1.385, 24, 680.0, 20.2, 332.09, 12.13]]



{% highlight python %}
print('Client CRIM = ' + str(CLIENT_FEATURES[0][0]))
print('Client RAD = ' + str(CLIENT_FEATURES[0][8])) 
print('Client B = ' + str(CLIENT_FEATURES[0][11]))
{% endhighlight %}

    Client CRIM = 11.95
    Client RAD = 24
    Client B = 332.09


Our client's crime rate is quite high!

## Picking evaluation method

We first shuffle the data using `sklearn.utils.shuffle(*arrays, *options)`. This function will return new shuffled data and target arrays. Then we split data 70-30 to use for training and testing using `train_test_split(...)`.


{% highlight python %}
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def shuffle_split_data(X, y):
    """ 
    Shuffles and splits data into 70% training and 30% testing subsets,
    then returns the training and testing subsets. 
    """
    # Shuffled data
    X_s, y_s = shuffle(X, y, random_state=0)

    # Split the data into training (70%) and testing (30%)
    X_train, y_train, X_test, y_test = train_test_split(X_s, y_s,
                                                        test_size=0.3,
                                                        random_state=0)

    # Return the training and testing data subsets
    return X_train, y_train, X_test, y_test


# Test shuffle_split_data
try:
    X_train, y_train, X_test, y_test = shuffle_split_data(housing_features, 
                                                          housing_prices)
    print("Successfully shuffled and split the data!")
except:
    print("Something went wrong with shuffling and splitting the data.")
{% endhighlight %}

    Successfully shuffled and split the data!


Splitting the data for training and testing allows us to evaluate our model by looking at the performance on training and testing data. The learning curves for training and testing show us if the model is underfitting (bias) or overfitting (variation).

MSE or MAE are better choices for regression task. Metrics like accuracy, precision, recall, f1-score are often used for evaluating a classification problem.


{% highlight python %}
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE

def performance_metric(y_true, y_predict):
    """ 
    Calculates and returns the total error between true 
    and predicted values
    based on a performance metric chosen by the student. 
    """
    error = MSE(y_true, y_predict)
    return error

# Test performance_metric
try:
    total_error = performance_metric(y_train, y_train)
    print("Successfully performed a metric calculation!")
except:
    print("Something went wrong with performing a metric calculation.")
{% endhighlight %}

    Successfully performed a metric calculation!


As mentioned before, mean squared error (MSE) and mean absolute error (MAE) are both appropriate for predicting housing prices. MAE is robust to outlier but it is not always differentible for gradient methods. The advantage of MAE is the error output by this method is robust to outliers. However, the housing price target is the median value so it is not necessary to care much about outliers in our error function. Therefore, MAE is the most appropriate for predicting housing prices and analyzing the total error. 

`fit_model` performs grid search cross validation and return the best estimator. [GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html) is the object provided by `scikit-learn` to search for the best paratemeters using cross-validation and then return the best estimator. To use `GridSearchCV`, we need to pass the `estimator`, the dictionary containing the parameter grid `param_grid`, the [scrorer callable](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html) object `scoring`, and optionally the number of cross-validation fold `cv`.


{% highlight python %}
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

def fit_model(X, y, fold=10):
    """
    Tunes a decision tree regressor 
    model using GridSearchCV on the input data X 
    and target labels y and returns this optimal model.
    """

    # Create a decision tree regressor object
    regressor = DecisionTreeRegressor()

    # Set up the parameters we wish to tune
    parameters = {'max_depth':(1,2,3,4,5,6,7,8,9,10)}

    # Make an appropriate scoring function
    scoring_function = make_scorer(MSE)

    # Make the GridSearchCV object
    reg = GridSearchCV(regressor, parameters, 
                       scoring_function, cv=fold)

    # Fit the learner to the data to obtain the optimal 
    # model with tuned parameters
    reg.fit(X, y)

    # Return the optimal model
    return reg.best_estimator_

# Test fit_model on entire dataset
try:
    reg = fit_model(housing_features, housing_prices)
    print("Successfully fit a model!")
except:
    print("Something went wrong with fitting a model.")
{% endhighlight %}

    Successfully fit a model!


Grid search algorithm is a brute-force hyper-parameter search for the best estimator configuration. Grid search algorithm is applicable when we need to find the best parameters for our learning model. This algorithm searches through all possible hyper-parameter configurations, evaluates the error of each configuration, then returns the best one. The exhaustive search guarantee the best model configuration is returned. However, due to the nature of a brute force algorithm, grid search might not be suitable for models with a large number of hyper-parameters or the hyper-parameters have large search spaces. 

Cross-validation is a data reuse technique to maximize the usage of data for training and testing. Specifying an integer `k` in advanced, for each model, cross-validation scheme splits the given training data into k-fold, runs the training procedure k times with k-1 folds of data as training and the remaining 1 fold as testing data. The final error is then averaged for k folds. Based on this cross-validation error, we can evaluate our model for overfitting. In contrast, if we evaluate the error of our model on the training dataset, there is a high chance that the learning algorithm will overfit the data (it can just remember the exact input-output without generalizing the data). In grid search, each model's configuration might have different performance on the training dataset. Without cross-validation, the grid search algorithm might select the model configuration that best _overfits_ the data. On the other hand, with cross-validation, grid search can account for variation in the model's prediction and prevent overfitting.


{% highlight python %}
def learning_curves(X_train, y_train, X_test, y_test):
    """
    Calculates the performance of several models with 
    varying sizes of training data. The learning and testing 
    error rates for each model are then plotted. 
    """
    
    print("Creating learning curve graphs for max_depths \
           of 1, 3, 6, and 10. . .")
    
    # Create the figure window
    fig = pl.figure(figsize=(10,8))

    # We will vary the training set size so that 
    # we have 50 different sizes
    sizes = np.rint(np.linspace(1, len(X_train), 50)).astype(int)
    train_err = np.zeros(len(sizes))
    test_err = np.zeros(len(sizes))

    # Create four different models based on max_depth
    for k, depth in enumerate([1,3,6,10]):
        
        for i, s in enumerate(sizes):
            
            # Setup a decision tree regressor so that 
            # it learns a tree with max_depth = depth
            regressor = DecisionTreeRegressor(max_depth = depth)
            
            # Fit the learner to the training data
            regressor.fit(X_train[:s], y_train[:s])

            # Find the performance on the training set
            train_err[i] = performance_metric(y_train[:s], 
                                              regressor.predict(X_train[:s]))
            
            # Find the performance on the testing set
            test_err[i] = performance_metric(y_test, 
                                             regressor.predict(X_test))

        # Subplot the learning curve graph
        ax = fig.add_subplot(2, 2, k+1)
        ax.plot(sizes, test_err, lw = 2, label = 'Testing Error')
        ax.plot(sizes, train_err, lw = 2, label = 'Training Error')
        ax.legend()
        ax.set_title('max_depth = %s'%(depth))
        ax.set_xlabel('Number of Data Points in Training Set')
        ax.set_ylabel('Total Error')
        ax.set_xlim([0, len(X_train)])
    
    # Visual aesthetics
    fig.suptitle('Decision Tree Regressor Learning Performances', 
                 fontsize=18, y=1.03)
    fig.tight_layout()
    fig.show()
{% endhighlight %}


{% highlight python %}
def model_complexity(X_train, y_train, X_test, y_test):
    """ 
    Calculates the performance of the model 
    as model complexity increases. The learning and 
    testing errors rates are then plotted. 
    """
    
    print("Creating a model complexity graph. . . ")

    # We will vary the max_depth of a decision tree 
    # model from 1 to 14
    max_depth = np.arange(1, 14)
    train_err = np.zeros(len(max_depth))
    test_err = np.zeros(len(max_depth))

    for i, d in enumerate(max_depth):
        # Setup a Decision Tree Regressor so that it learns 
        # a tree with depth d
        regressor = DecisionTreeRegressor(max_depth = d)

        # Fit the learner to the training data
        regressor.fit(X_train, y_train)

        # Find the performance on the training set
        train_err[i] = performance_metric(y_train, 
                                          regressor.predict(X_train))

        # Find the performance on the testing set
        test_err[i] = performance_metric(y_test, 
                                         regressor.predict(X_test))

    # Plot the model complexity graph
    pl.figure(figsize=(7, 5))
    pl.title('Decision Tree Regressor Complexity Performance')
    pl.plot(max_depth, test_err, lw=2, label = 'Testing Error')
    pl.plot(max_depth, train_err, lw=2, label = 'Training Error')
    pl.legend()
    pl.xlabel('Maximum Depth')
    pl.ylabel('Total Error')
    pl.show()
{% endhighlight %}

## Analyzing Model Performance


{% highlight python %}
learning_curves(X_train, y_train, X_test, y_test)
{% endhighlight %}

    Creating learning curve graphs for max_depths            of 1, 3, 6, and 10. . .


    /home/hoangnt/anaconda3/envs/mlnano/lib/python3.5/site-packages/sklearn/utils/validation.py:395: DeprecationWarning: Passing 1d arrays as data is deprecated in 0.17 and will raise ValueError in 0.19. Reshape your data either using X.reshape(-1, 1) if your data has a single feature or X.reshape(1, -1) if it contains a single sample.
      DeprecationWarning)



    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-26-6279604a59b5> in <module>()
    ----> 1 learning_curves(X_train, y_train, X_test, y_test)
    

    <ipython-input-24-cedab86879a3> in learning_curves(X_train, y_train, X_test, y_test)
         35             # Find the performance on the testing set
         36             test_err[i] = performance_metric(y_test, 
    ---> 37                                              regressor.predict(X_test))
         38 
         39         # Subplot the learning curve graph


    /home/hoangnt/anaconda3/envs/mlnano/lib/python3.5/site-packages/sklearn/tree/tree.py in predict(self, X, check_input)
        402         """
        403 
    --> 404         X = self._validate_X_predict(X, check_input)
        405         proba = self.tree_.predict(X)
        406         n_samples = X.shape[0]


    /home/hoangnt/anaconda3/envs/mlnano/lib/python3.5/site-packages/sklearn/tree/tree.py in _validate_X_predict(self, X, check_input)
        374                              "match the input. Model n_features is %s and "
        375                              "input n_features is %s "
    --> 376                              % (self.n_features_, n_features))
        377 
        378         return X


    ValueError: Number of features of the model must match the input. Model n_features is 13 and input n_features is 354 



    <matplotlib.figure.Figure at 0x7f646ab42320>


## Question 7
*Choose one of the learning curve graphs that are created above. What is the max depth for the chosen model? As the size of the training set increases, what happens to the training error? What happens to the testing error?*

**Answer: **

## Question 8
*Look at the learning curve graphs for the model with a max depth of 1 and a max depth of 10. When the model is using the full training set, does it suffer from high bias or high variance when the max depth is 1? What about when the max depth is 10?*

**Answer: **


{% highlight python %}
model_complexity(X_train, y_train, X_test, y_test)
{% endhighlight %}

## Question 9
*From the model complexity graph above, describe the training and testing errors as the max depth increases. Based on your interpretation of the graph, which max depth results in a model that best generalizes the dataset? Why?*

**Answer: **

# Model Prediction
In this final section of the project, you will make a prediction on the client's feature set using an optimized model from `fit_model`. When applying grid search along with cross-validation to optimize your model, it would typically be performed and validated on a training set and subsequently evaluated on a **dedicated test set**. In this project, the optimization below is performed on the *entire dataset* (as opposed to the training set you made above) due to the many outliers in the data. Using the entire dataset for training provides for a less volatile prediction at the expense of not testing your model's performance. 

*To answer the following questions, it is recommended that you run the code blocks several times and use the median or mean value of the results.*

## Question 10
*Using grid search on the entire dataset, what is the optimal `max_depth` parameter for your model? How does this result compare to your intial intuition?*  
**Hint: ** Run the code block below to see the max depth produced by your optimized model.


{% highlight python %}
print "Final model has an optimal max_depth parameter of", reg.get_params()['max_depth']
{% endhighlight %}

**Answer: **

## Question 11
*With your parameter-tuned model, what is the best selling price for your client's home? How does this selling price compare to the basic statistics you calculated on the dataset?*  

**Hint: ** Run the code block below to have your parameter-tuned model make a prediction on the client's home.


{% highlight python %}
sale_price = reg.predict(CLIENT_FEATURES)
print "Predicted value of client's home: {0:.3f}".format(sale_price[0])
{% endhighlight %}

**Answer: **

## Question 12 (Final Question):
*In a few sentences, discuss whether you would use this model or not to predict the selling price of future clients' homes in the Greater Boston area.*

**Answer: **
