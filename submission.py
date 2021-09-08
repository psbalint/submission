# importing the pandas package for the data wrangling:

import pandas as pd

# reading the .csv files using pandas' read_csv() function:

X = pd.read_csv('md_raw_dataset.csv', sep = ';')
y = pd.read_csv('md_target_dataset.csv', sep = ';')

"""
with the help of the unique() function of pandas I found out
that in the 2 files the 'groups' variables represent the
same groups, and also that the index in the target file is
the same as the index in an unnamed column in the other file

considering all this now I could join the two files:
"""

df = pd.merge(X,
              y,
              how = 'left',
              left_on = ['Unnamed: 0', 'groups'],
              right_on = ['index', 'groups'])

"""
if the target variable is missing, there is no meaningful
way to predict it and compare the predicted value to the
actual value. I have to drop the rows where the target is missing:
"""

df = df[df['target'].notna()]

"""
in data imputation it makes little to no sense to impute if more
than 25%-30% of the values are missing

some even say imputing in case of 5% missing data is misleading.
so I have to drop the columns with many missing values.

also, I have to drop the columns that do not seem to
be measurements: they may be indices or time stamps or some
unique identifiers of measurements:
"""

df = df.drop(columns=['Unnamed: 0', 'tracking', 'when', 'expected_start',
       'start_subprocess1', 'start_critical_subprocess1', 'pure_seastone', 
       'predicted_process_end', 'process_end', 'subprocess1_end',
       'reported_on_tower', 'opened', 'index', 'etherium_before_start',
	   'raw_kryptonite', 'start_process'])

# just renaming some columns:

df = df.rename(columns={'Unnamed: 7': 'unnamed_7th_column',
                        'Unnamed: 17': 'unnamed_17th_column',
                        'Cycle': 'cycle'})

"""
now I have to look at the columns still containing missing values (one
could find them for example with the df.isna().sum() command)

what is a meaningful way to impute the values? turns out that all
of them seem to be measurements (that means no categorical values)

this means an imputation with the mean could make sense. another,
absolutely bad idea would be to delete the columns that have missing
values. a third, slightly better idea is to delete the rows that have
missing values. at this point, with this method we would delete
around 10% of the data points which is also something to avoid

so I impute the missing values with the mean of their columns:
"""

df = df.fillna(df.mean())

"""
one more thing to look for are the constant columns: they only contain
one number or string. we normally delete these before applying machine
learning methods. I checked this with pandas' .nunique() method,
and fortunately in this case we do not have any such columns

the next step is to make dummy variables out of the categorical
variables. in many cases (also in this dataset) the entry '5' in a
column called 'group' does not actually mean the number 5, it is
just a randomly assigned identifier for the group
"""

df = pd.get_dummies(df, columns = ['super_hero_group', 'crystal_supergroup',
                                    'cycle', 'groups', 'human_behavior_report',
                                    'unnamed_7th_column', 'place',
                                    'tracking_times', 'crystal_type'])

"""
now the dataset is almost ready for the application of
machine learning methods, we should think about scaling it.

one could use scikit-learn's preprocessing sub-package, and
the MinMaxScaler or the StandardScaler from it

I also used these as I tried out simpler methods like the
linear regression, its regularized versions Lasso and Ridge,
or support vector regression. in each of these cases
scaling is essential.

in this submission I only include the estimator that
delivered the best results. this is the Gradient Boosting
Regressor. it is a robust regressor that is not really
sensitive when it comes to the different scales of
the columns (or to constant columns for that matter)

now I import the packages from scikit-learn. I need
an error rate, which could be MAE but I prefer the MSE, so
I import it too.

also I import the GBR model and the train-test split:
"""

import sklearn

from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingRegressor

"""
first I separate the predictor variables from the target
variable and put them into a matrix & into a vector. I
convert them to numpy format so that scikit-learn can work
with them.

at first I only do only 1 train-test split and 1 model building
with the default parameters of the GBR model. I fit the
model on the training set (80% of the observations) and then
predict the target variable with the test set (20%)
"""

X_df = df.loc[:, df.columns != 'target']
y_df = df['target']

X = X_df.values
y = y_df.values

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.2,
                                                    random_state = 2) 

model = GradientBoostingRegressor()
model.fit(X_train, y_train) 

predictions = model.predict(X_test)

"""
at this point I could try out a lot of parameter combinations
in the model = GradientBoostingRegressor() line. but there is
also another problem:

if I do only 1 train-test split, I could overfit the model on
this 1 training set, that is why I will use cross-validation (CV)

there is a method is scikit-learn that not only uses CV but
can try out a wide set of paramaters and find the one set with
which the given model gives us the best performance. this is
called GridSearchCV:
"""

my_parameter_grid = {'max_depth': range(5, 10, 2),
                     'min_samples_split': range(200, 601, 200)}  

my_mse = make_scorer(mean_squared_error, greater_is_better = False)

grid = GridSearchCV(model, param_grid = my_parameter_grid, cv = 5, scoring = my_mse, refit = True, verbose = 3, n_jobs = -1) 
grid.fit(X_train, y_train)

grid_predictions = grid.predict(X_test)

"""
the command print(grid.best_params_) would print out the best
possible combination of parameters that this method finds.

as one can see in this submission I only take two parameters
and only a few (3) possible values for each. the reason for that
is that the 9 combinations combined with the 5-fold CV already
mean 45 model fittings, and by a model like GBR this could eat
up a lot of time on a normal computer like mine

I would like to note that even with the basic set of its
parameters the GBR works better than most other algorithms
(linear regression, SVR, etc.) I tried (compared with MSE & CV)

One could also fine-tune a solution like that with various
feature selection methods. My approach would be the FFS
(forward feature selection): I start with an empty model and
in each step I add the best variable to it (that causes the
biggest decrease in MSE)

This could be interpreted as a dimension reduction method since
I would not use all the hundreds of variables, only a few of
them (like 10-20). Another dimension reduction model would
be the PCA (principal component analysis, or PCR: principal
component regression) or the fairly similar PLS (partial least
squares regression). These may result in simpler and smaller
models, and also faster computational times.
"""