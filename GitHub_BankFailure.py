'''
"Modeling of bank failures by FDIC":
1. Used Logistic Regression from scikit-learn model
2. Constructed a Logistic Regression model using Tensorflow
3. Built and trained Neural Network for bank failure prediction
'''

# Libraries: pandas, numpy, sklearn, and tensorflow

import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn

import time
import os
import functools
import math
import random
import sys, getopt


try:
    import matplotlib.pyplot as plt
    get_ipython().magic('matplotlib inline')
except:
    pass
print('scikit-learn version:', sklearn.__version__)


# share this across notebook
state_cols = ['log_TA','NI_to_TA', 'Equity_to_TA', 'NPL_to_TL', 'REO_to_TA',
              'ALLL_to_TL', 'core_deposits_to_TA', 'brokered_deposits_to_TA',
              'liquid_assets_to_TA', 'loss_provision_to_TL', 'NIM', 'assets_growth']

all_MEVs = np.array(['term_spread',
                    'stock_mkt_growth',
                    'real_gdp_growth',
                    'unemployment_rate_change',
                    'treasury_yield_3m',
                    'bbb_spread',
                    'bbb_spread_change'])

MEV_cols = all_MEVs.tolist()

next_state_cols = ['log_TA_plus_1Q','NI_to_TA_plus_1Q', 'Equity_to_TA_plus_1Q', 'NPL_to_TL_plus_1Q', 'REO_to_TA_plus_1Q',
                   'ALLL_to_TL_plus_1Q', 'core_deposits_to_TA_plus_1Q', 'brokered_deposits_to_TA_plus_1Q',
                   'liquid_assets_to_TA_plus_1Q', 'loss_provision_to_TL_plus_1Q',
                   'ROA_plus_1Q',
                   'NIM_plus_1Q',
                   'assets_growth_plus_1Q',
                   'FDIC_assessment_base_plus_1Q_n']




df_train = pd.read_hdf('../readonly/df_train_FDIC_defaults_1Y.h5', key='df')
df_test = pd.read_hdf('../readonly/df_test_FDIC_defaults_1Y.h5', key='df')
df_data = pd.read_hdf('../readonly/data_adj_FDIC_small.h5', key='df')
df_closure_learn = pd.read_hdf('../readonly/df_FDIC_learn.h5',key='df')
print(df_closure_learn.index.names)


# LOGESTIC REGRESSION: Construct training and testing datasets

df_test.plot(x=state_cols[0], y='defaulter', kind='scatter')



# PLOT: 4 scatter plots together

first_indx = [0, 0, 0, 0]
second_indx = [1, 3, 2, 10]

X_train = df_train[state_cols].values
y_train = df_train.defaulter.values # .reshape(-1,1)

num_plots = 4
if num_plots % 2 == 0:
    f, axs = plt.subplots(num_plots // 2, 2)
else:
    f, axs = plt.subplots(num_plots// 2 + 1, 2)

f.subplots_adjust(hspace=.3)

f.set_figheight(10.0)
f.set_figwidth(10.0)

for i in range(num_plots):
    if i % 2 == 0:
        first_idx = i // 2
        second_idx = 0
    else:
        first_idx = i // 2
        second_idx = 1

    axs[first_idx,second_idx].plot(X_train[y_train == 1.0, first_indx[i]],
                                   X_train[y_train == 1.0, second_indx[i]], 'r^', label="Failed")
    axs[first_idx,second_idx].plot(X_train[y_train == 0.0, first_indx[i]],
                                   X_train[y_train == 0.0, second_indx[i]], 'go',label="Non-failed")

    axs[first_idx, second_idx].legend()
    axs[first_idx, second_idx].set_xlabel('%s' % state_cols[first_indx[i]])
    axs[first_idx, second_idx].set_ylabel('%s' % state_cols[second_indx[i]])
    axs[first_idx, second_idx].set_title('Failed banks vs non-failed banks')
    axs[first_idx, second_idx].grid(True)

if num_plots % 2 != 0:
    f.delaxes(axs[i // 2, 1])




def calc_metrics(model, df_test, y_true, threshold=0.5):
    """
    Arguments:
    model - trained model such as DecisionTreeClassifier
    df_test - Data Frame of predictors
    y_true - True binary labels in range {0, 1} or {-1, 1}.
             If labels are not binary, pos_label should be explicitly given.
    """
    if model is None:
        return 0., 0., 0.

    # prediction
    predicted_sm = model.predict(df_test, linear=False)
    predicted_binary = (predicted_sm > threshold).astype(int)

    # print(predicted_sm.shape, y_true.shape)
    fpr, tpr, _ = metrics.roc_curve(y_true, predicted_sm, pos_label=1)

    # compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores
    roc_auc = metrics.auc(fpr, tpr)
    ks = np.max(tpr - fpr) # Kolmogorov - Smirnov test

    # note that here teY[:,0] is the same as df_test.default_within_1Y
    accuracy_score = metrics.accuracy_score(y_true, predicted_binary)

    # equivalently, Area Under the ROC Curve could be computed as:
    # compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores
    # auc_score = metrics.roc_auc_score(y_true, predicted_sm)

    try:
        plt.title('Logistic Regression ROC curve')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0,1], [0,1], 'r--')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')


        plt.show()
    except:
        pass

    return roc_auc, accuracy_score, ks


# make the train and test datasets


def make_test_train(df_train, df_test, choice=0, predict_within_1Y=False):
    """
    Arguments:
    choice - an integer 0 or -1.
    predict_within_1Y - if True, predict defaults within one year
    Return:
        - training data set predictors
        - training data set : variable to predict
        - test data set : variable to predict
        - predictor variable names
    """

    if choice == -1: # only state cols
        predictors = state_cols
    elif choice == 0:  # original variables
        predictors = state_cols + MEV_cols

    trX = df_train[predictors].values
    teX = df_test[predictors].values
    num_features = len(predictors)
    num_classes = 2

    if predict_within_1Y == True:
        trY = df_train[['default_within_1Y','no_default_within_1Y']].values
        teY = df_test[['default_within_1Y','no_default_within_1Y']].values
    else:
        trY = df_train[['defaulter','non_defaulter']].values
        teY = df_test[['defaulter','non_defaulter']].values
    return trX, trY, teX, teY, predictors





# correlation Check

df_train[MEV_cols].corr()


'''
Logistic regression with statsmodels:
    cols_to_use: predictors
    df_train: training data set
    df_test: testing data set
perform prediction based on the already trained model with statsmodels.
'''

import statsmodels.api as sm
from sklearn import metrics

cols_to_use = state_cols + MEV_cols  + ['const']
model = None
df_train['const'] = 1

logit = sm.Logit(df_train.defaulter, df_train[cols_to_use])
model = logit.fit()

# prediction
predicted_sm = np.array([])

if model is not None:
    predicted_sm = model.predict(df_test[cols_to_use], linear=False)

threshold = 0.5
predicted_binary = (predicted_sm > threshold).astype(int)
auc_score, accuracy_score, ks = calc_metrics(model, df_test[cols_to_use], df_test.defaulter)

print('Accuracy score %f' % accuracy_score)
print('AUC score %f' % auc_score)
print('Kolmogorov-Smirnov statistic %f' % ks)

# teY[:,0] is the same as df_test.default_within_1Y


"""
Logistic Regression with sklearn:

    Penalty = "l1"
    Inverse of regularization strength = 1000.0
    Tolerance = 1e-6

"""


from sklearn import neighbors, linear_model

trX, trY, teX, teY, predictors = make_test_train(df_train, df_test)
lr_score = 0.
thisTrY = trY[:,0]
thisTeY = teY[:,0]

logistic = None # instantiate a model and reference it
result = None # result of fitting the model


logistic = linear_model.LogisticRegression(penalty='l1', tol=1e-6, C=1000.0)
result = logistic.fit(trX, thisTrY)
lr_score = result.score(teX, thisTeY)

print('LogisticRegression score: %f' % lr_score)



"""
LOGESTIC REGRESSION SKLEARN with smaller set of predictor variables based on P-values
    Predictors: cols_to_use
    Something to predict: defaulter
    Result of fitting the model to the training data set: result
"""


# Smaller set is based on the analysis of P-values for the logistic regression
cols_to_use = ['log_TA', 'NI_to_TA', 'Equity_to_TA', 'NPL_to_TL',
               'core_deposits_to_TA',
               'brokered_deposits_to_TA',
               'liquid_assets_to_TA'
              ] + ['term_spread', 'stock_mkt_growth']

lr_score = 0.
logistic = None
result = None


trX = df_train[cols_to_use].values
teX = df_test[cols_to_use].values
thisTrY = (df_train.defaulter.values)
thisTeY = (df_test.defaulter.values)
logistic = linear_model.LogisticRegression(penalty='l1', tol=1e-6, C=1000.0)
result = logistic.fit(trX, thisTrY)
lr_score = result.score(teX, thisTeY)
print('LogisticRegression score: %f' % lr_score)

# combine results of the Logistic Regression to a small dataframe df_coeffs_LR
df_coeffs_LR = pd.DataFrame({0: np.array([0.] * (len(cols_to_use) + 1), dtype=np.float32)})
if logistic is not None:
    model_params = np.hstack((logistic.coef_[0], logistic.intercept_))
    df_coeffs_LR = pd.DataFrame(data=model_params, index=cols_to_use + ['const'])
    df_coeffs_LR





'''Logistic Regression with Tensorflow'''


# Setup inputs and expeced outputs for Logistic Regression
cols = state_cols + MEV_cols
# inputs to Logistic Regression (via Tensorflow)
X_trainTf = df_train[cols].values
X_testTf = df_test[cols].values

# Add constant columns to both
X_trainTf = np.hstack((np.ones((X_trainTf.shape[0], 1)), X_trainTf))
X_testTf = np.hstack((np.ones((X_testTf.shape[0], 1)), X_testTf))

# Exepectd outputs:
y_trainTf = df_train.defaulter.astype('int').values.reshape(-1,1)
y_testTf = df_test.defaulter.astype('int').values.reshape(-1,1)

print('Unique values to predict:', np.unique(y_trainTf))
print('Number of samples to train on:', y_trainTf.shape[0])
print('Number of samples to test on:', y_testTf.shape[0])

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


def random_batch(X_train, y_train, batch_size):
    np.random.seed(42)
    rnd_indices = np.random.randint(0, len(X_train), batch_size)
    X_batch = X_train[rnd_indices]
    y_batch = y_train[rnd_indices]
    return X_batch, y_batch


"""
Build Logistic Regression TF model:

 - placeholder for inputs called 'X'
 - placeholder for outputs called 'y'
 - variable for model parameters called 'theta', initialized with theta_init

loss function: log loss
optimizer: Gradient Descent optimizer
"""


import tensorflow as tf

# define the model
reset_graph()
n_inputs = X_trainTf.shape[1]
learning_rate = 0.01
theta_init = tf.random_uniform([n_inputs, 1], -1.0, 1.0, seed=42)

# build Logistic Regression model using Tensorflow

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
theta = tf.Variable(theta_init, name="theta")


logits = tf.matmul(X, theta)
y_proba = tf.sigmoid(logits) # 1 / (1 + tf.exp(-logits))

# uses epsilon = 1e-7 by default to regularize the log function
loss = tf.losses.log_loss(y, y_proba, epsilon=1e-07)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


init = tf.global_variables_initializer()


"""
Train Logistic Regression TF model:

    - Use random_batch() function to grab batches from X_trainTf and y_trainTf.
    - Evaluate model based on X_testTf and y_testTf
    - y_proba_val: result of the evaluation on test dataset

"""



n_epochs = 1001
batch_size = 50
num_rec = X_trainTf.shape[0]
n_batches = int(np.ceil(num_rec / batch_size))

y_proba_val = np.array([], dtype=np.float32)

with tf.Session() as sess:

    for epoch in range(n_epochs):
        sess.run(init)
        X_trainTf_batch, y_trainTf_batch = random_batch(X_trainTf, y_trainTf, batch_size)
        sess.run([optimizer, loss], feed_dict={X: X_trainTf_batch, y: y_trainTf_batch})

    y_proba_val = sess.run(y_proba, feed_dict={X: X_testTf})



# predictions
threshold = 0.5
y_pred = (y_proba_val >= threshold)
print(np.sum(y_pred))


y_pred.squeeze()


# evaluate precision, recall, and AUC

auc_score = 0.
ks = 0.
roc_auc = 0.
recall = 0.
precision = 0.

from sklearn.metrics import precision_score, recall_score
if y_proba_val.shape == y_testTf.shape:
    precision = precision_score(y_testTf, y_pred)
    recall = recall_score(y_testTf, y_pred)
    auc_score = metrics.roc_auc_score(y_testTf, y_proba_val)
    fpr, tpr, threshold = metrics.roc_curve(y_testTf, y_proba_val, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    ks = np.max(tpr - fpr)

    print('precision: ', precision)
    print('recall: ', recall)
    print('AUC score = ', auc_score)
    print('roc_auc = ', roc_auc)
    print('KS_test = ', ks)

    try:
        plt.title('ROC_curve')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0,1], [0,1], 'r--')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.savefig('ROC_curve_TF.png')
        plt.show()
    except:
        pass



"""Neural Network with Tensorflow"""

cols = state_cols + MEV_cols
n_inputs = len(cols)

# inputs
X_trainTf = df_train[cols].values
X_testTf = df_test[cols].values

# outputs
y_trainTf = df_train['defaulter'].astype('int').values.reshape(-1,)
y_testTf = df_test['defaulter'].astype('int').values.reshape(-1,)


import numpy as np
def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        tf.set_random_seed(42)
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name="kernel")
        b = tf.Variable(tf.zeros([n_neurons]), name="bias")
        Z = tf.matmul(X, W) + b
        if activation is not None:
            return activation(Z)
        else:
            return Z

"""
Construct Neural Network:

    - Two hidden layers
    - Number of nodes in first and second hidden layers: n_hidden1 and n_hidden2
    - neuron_layer() function to construct neural network layers
    - ReLU activation function for hidden layers
    - Sparse softmax cross-entropy with logits as a loss function

"""



n_hidden1 = 20
n_hidden2 = 10
n_outputs = 2 # binary classification (defaulted, not defaulted bank)

reset_graph()

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

layer_1 = neuron_layer(X, n_hidden1, "layer_1", tf.nn.relu)
layer_2 = neuron_layer(layer_1, n_hidden2, "layer_2", tf.nn.relu)
logits = neuron_layer(layer_2, n_outputs, "logits")
entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
loss = tf.reduce_mean(entropy)

init = tf.global_variables_initializer()


"""
Train Neural Network:
    - Passing batches of inputs of size batch_size
    - Evaluate accuracy using X_testTf and y_testTf

"""

learning_rate = 0.05
n_epochs = 400
batch_size = 50
num_rec = X_trainTf.shape[0]
n_batches = int(np.ceil(num_rec / batch_size))
acc_test = 0. #  assign the result of accuracy testing to this variable

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    sess.run(init)
    for i in range(n_epochs):
        X_trainTf_batch, y_trainTf_batch = random_batch(X_trainTf, y_trainTf, batch_size)
        sess.run([optimizer, loss], feed_dict={X: X_trainTf_batch, y: y_trainTf_batch})

    _, loss, logits = sess.run([optimizer, loss, logits], feed_dict={X: X_testTf, y: y_testTf})
    preds = tf.nn.softmax(logits)
    correct_preds = tf.equal(tf.argmax(preds, 1), y_testTf)
    acc_test = sess.run(tf.reduce_sum(tf.cast(correct_preds, tf.float32))) / len(y_testTf)
