# This code is originally written by Reham El-kholy, Lecturer of Astronomy at Cairo University.
# The code was made available by author at : https://github.com/rehamelkholy/StellarAL
# We are grateful to the author for sharing this code publicly on GitHub.
# The code is licensed under the MIT License.
# If you use this code please cite the original author and the repository.
#
# Here, we have added two additional methods : load_pkl_data_for_AL() and preprocess_set_for_AL()
# to load and preprocess the data for active learning.
# The rest of the code is unchanged from the original version. 


######################
### Load Libraries ###
######################
import numpy as np
import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt 

from IPython.display import clear_output

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.metrics import matthews_corrcoef, roc_auc_score

from imblearn.metrics import sensitivity_score, specificity_score

from modAL.models import ActiveLearner, Committee
from modAL.uncertainty import entropy_sampling, margin_sampling, uncertainty_sampling
from modAL.disagreement import consensus_entropy_sampling, max_disagreement_sampling, vote_entropy_sampling


########################
### Define Functions ###
########################
def load_pkl_data(file_name):
    """
    Load data from a pickle file and return training and testing sets.

    This function reads a pickle file containing a dictionary with keys 
    'X_train', 'y_train', 'X_test', and 'y_test', and returns the 
    corresponding datasets. The data is assumed to be in the following format:
    - 'X_train': Features for the training set
    - 'y_train': Labels for the training set
    - 'X_test': Features for the testing set
    - 'y_test': Labels for the testing set

    Parameters
    ----------
    file_name : str
        The path to the pickle file to be loaded.

    Returns
    -------
    X_train : numpy.ndarray
        Features for the training set.
    
    X_test : numpy.ndarray
        Features for the testing set.
    
    y_train : numpy.ndarray
        Labels for the training set.
    
    y_test : numpy.ndarray
        Labels for the testing set.

    Examples
    --------
    >>> X_train, X_test, y_train, y_test = load_pkl_data('data.pkl')
    >>> print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    (500, 20) (100, 20) (500,) (100,)
    """
    with open(file_name, 'rb') as f:
        data = pkl.load(f)
    
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
    
    return X_train, X_test, y_train, y_test

def select_features(flux_arr, wl_arr, line_wl, bounds=8):
    """
    Select features from a flux array centered around a specified wavelength.

    This function identifies the index of the wavelength in `wl_arr` that is
    closest to the specified `line_wl`. It then extracts a subset of features 
    from `flux_arr`, centered around this wavelength index, within the 
    specified `bounds`.

    Parameters
    ----------
    flux_arr : numpy.ndarray
        A 2D array where each row corresponds to a spectrum and each column 
        corresponds to a flux value at a particular wavelength.
    
    wl_arr : numpy.ndarray
        A 1D array of wavelength values corresponding to the columns of 
        `flux_arr`.
    
    line_wl : float
        The target wavelength around which the features will be selected.
    
    bounds : int, optional
        The number of wavelengths to include on either side of `line_wl`. 
        The total number of features selected will be `2 * bounds + 1`. 
        Default is 8.

    Returns
    -------
    new_flux_arr : numpy.ndarray
        A 2D array containing the selected features, with the same number of 
        rows as `flux_arr` and `2 * bounds + 1` columns.

    Examples
    --------
    >>> flux_arr = np.random.random((100, 500))
    >>> wl_arr = np.linspace(4000, 5000, 500)
    >>> line_wl = 4500
    >>> selected_features = select_features(flux_arr, wl_arr, line_wl, bounds=8)
    >>> print(selected_features.shape)
    (100, 17)
    """
    # Find the index of the wavelength closest to the specified line_wl
    index = np.argmin(np.abs(wl_arr - line_wl))
 
    # Extract a subset of features centered around line_wl
    print(f"Wavelength bounds ({line_wl}): {wl_arr[index-bounds]:.2f} - {wl_arr[index+bounds]:.2f}")
    new_flux_arr = flux_arr[:,index-bounds:index+bounds+1]
    return new_flux_arr

def preprocess_set(X, y, param, test_size=0.1):
    """
    Preprocess a dataset by splitting, shuffling, scaling, and applying PCA, 
    then save the processed data to a file.

    This function performs the following steps:
    1. Splits the input dataset into training and testing sets with stratification.
    2. Shuffles the training and testing sets.
    3. Scales the features to a range [0, 1] using Min-Max Scaling.
    4. Applies Principal Component Analysis (PCA) to the scaled features.
    5. Saves the preprocessed data to a pickle file.

    Parameters
    ----------
    X : numpy.ndarray, shape (n_samples, n_features)
        The feature matrix of the dataset.
    
    y : numpy.ndarray, shape (n_samples,)
        The labels corresponding to the features.
    
    param : str
        A parameter to include in the output filename for identification.
    
    test_size : float, optional, default=0.1
        The proportion of the dataset to include in the test split.

    Returns
    -------
    None

    Examples
    --------
    >>> X = [[0, 1], [1, 0], [0, 0], [1, 1]]
    >>> y = [0, 1, 0, 1]
    >>> preprocess_set(X, y, 'example_param')
    Data saved in file sec3_1_data_output_example_param.pkl
    """
    # split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)
    # shuffle data after split
    X_train, y_train = shuffle(X_train, y_train)
    X_test, y_test = shuffle(X_test, y_test)
    
    # initialize a scaler
    scaler = MinMaxScaler()
    # fit the scaler using training set and transform all data
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # initialize a PCA object
    pca = PCA(n_components=X.shape[1])
    # fit PCA using training set and transform all data
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    
    # save data to file
    file_data = {'X_train': X_train, 'y_train': y_train,
             'X_test': X_test, 'y_test': y_test}
    file_name = 'sec3_1_data_output_'+ param + '.pkl'
    with open(file_name, 'wb') as f:
        pkl.dump(file_data, f, protocol=pkl.HIGHEST_PROTOCOL)
        
    print('Data saved in file ' + file_name)
    
    return

def modify_predict_output(y_unique, y_score, n_classes):
    """
    Modify prediction output to match the required number of classes.

    This function takes the unique class labels, their corresponding prediction 
    scores, and the total number of classes. It returns a modified array where 
    the scores are placed in the correct positions according to their class 
    labels, with the rest of the positions filled with zeros.

    Parameters
    ----------
    y_unique : numpy.ndarray
        The unique class labels for which predictions are available.
    
    y_score : numpy.ndarray
        The prediction scores corresponding to the unique class labels. 
        Should be of shape (n_samples, len(y_unique)).
    
    n_classes : int
        The total number of possible classes for the whole dataset.

    Returns
    -------
    new_array : numpy.ndarray
        A new array of shape (n_samples, n_classes) with prediction scores 
        placed in the appropriate positions according to their class labels.

    Examples
    --------
    >>> y_unique = [1, 3]
    >>> y_score = np.array([[0.8, 0.2], [0.4, 0.6]])
    >>> n_classes = 4
    >>> modify_predict_output(y_unique, y_score, n_classes)
    array([[0. , 0.8, 0. , 0.2],
           [0. , 0.4, 0. , 0.6]])

    Notes
    -----
    - The function assumes that `y_unique` contains valid class labels within 
      the range `[0, n_classes-1]`.
    - The length of `y_unique` should match the second dimension of `y_score`.
    """
    new_array = np.zeros((y_score.shape[0], n_classes))
    new_array[:,y_unique] = y_score
    return new_array

def random_sampling(clf, X_pool):
    """
    Select a random instance from a pool of unlabeled data points.

    This function randomly selects one sample from the provided pool of 
    unlabeled data points (X_pool) and returns the index of the selected 
    sample along with the sample itself. This method can be used in active 
    learning scenarios where a classifier (clf) is iteratively trained and 
    queried for labeling new data points.

    Parameters
    ----------
    clf : object
        A classifier object. This parameter is not used in the function but 
        is included to maintain a consistent interface for different sampling 
        strategies in an active learning framework.
    
    X_pool : numpy.ndarray, shape (n_samples, n_features)
        The pool of unlabeled data points from which a sample is to be 
        selected. Each row corresponds to a data point and each column 
        corresponds to a feature.

    Returns
    -------
    query_idx : int
        The index of the randomly selected sample from the pool.
    
    query_sample : numpy.ndarray, shape (n_features,)
        The randomly selected data sample from the pool.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> X_pool, _ = make_classification(n_samples=100, n_features=20)
    >>> clf = None  # Placeholder for classifier object
    >>> query_idx, query_sample = random_sampling(clf, X_pool)
    >>> print(query_idx)
    42
    >>> print(query_sample.shape)
    (20,)
    """
    n_samples = len(X_pool)
    query_idx = np.random.choice(range(n_samples))
    return query_idx, X_pool[query_idx]

def get_initial_sample_pool(X, y, n):
    """
    Generate an initial sample pool from the given dataset.

    This function randomly selects `n` samples from the dataset `X` and their 
    corresponding labels `y` to form an initial sample pool. The remaining 
    samples and labels are returned as the pool for further sampling.

    Parameters
    ----------
    X : numpy.ndarray, shape (n_samples, n_features)
        The feature matrix from which to sample.
    
    y : numpy.ndarray, shape (n_samples,)
        The corresponding labels for the feature matrix.
    
    n : int
        The number of samples to be included in the initial sample pool.

    Returns
    -------
    X_sample : numpy.ndarray, shape (n, n_features)
        The feature matrix of the initial sample pool.
    
    y_sample : numpy.ndarray, shape (n,)
        The corresponding labels of the initial sample pool.
    
    X_pool : numpy.ndarray, shape (n_samples - n, n_features)
        The feature matrix of the remaining pool.
    
    y_pool : numpy.ndarray, shape (n_samples - n,)
        The corresponding labels of the remaining pool.

    Examples
    --------
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    >>> y = np.array([0, 1, 0, 1, 0])
    >>> X_sample, y_sample, X_pool, y_pool = get_initial_sample_pool(X, y, 2)
    >>> print(X_sample)
    [[1 2]
     [9 10]]
    >>> print(y_sample)
    [0 0]
    >>> print(X_pool)
    [[ 3  4]
     [ 5  6]
     [ 7  8]]
    >>> print(y_pool)
    [1 0 1]
    """
    training_indices = np.random.randint(low=0, high=X.shape[0], size=n)
    
    X_sample = X[training_indices]
    y_sample = y[training_indices]
    
    X_pool = np.delete(X, training_indices, axis=0)
    y_pool = np.delete(y, training_indices)
    
    return X_sample, y_sample, X_pool, y_pool

def get_score(metric, y_test, y_pred=None, y_score=None):
    """
    Compute the specified evaluation metric for a classification model.

    This function calculates one of several evaluation metrics ('auc', 'mcc', 
    'sen', 'spec') based on the true labels (`y_test`) and the predicted labels
    (`y_pred`) or predicted scores (`y_score`).

    Parameters
    ----------
    metric : str
        The metric to be calculated. Valid options are:
        - 'auc': Area Under the Receiver Operating Characteristic Curve (AUC)
        - 'mcc': Matthews Correlation Coefficient (MCC)
        - 'sen': Sensitivity (Recall) score
        - 'spec': Specificity score
    y_test : numpy.ndarray
        True labels.
    y_pred : numpy.ndarray, optional
        Predicted labels. Required if the metric is 'mcc', 'sen', or 'spec'.
    y_score : numpy.ndarray, optional
        Predicted scores or probabilities. Required if the metric is 'auc'.

    Returns
    -------
    score : float
        The calculated score for the specified metric.

    Raises
    ------
    ValueError
        If an invalid metric is specified or if the required predicted values
        (`y_pred` or `y_score`) are not provided for the specified metric.

    Examples
    --------
    >>> y_test = [0, 1, 1, 0]
    >>> y_pred = [0, 1, 0, 0]
    >>> y_score = [0.1, 0.9, 0.4, 0.3]
    >>> get_score('auc', y_test, y_score=y_score)
    0.75
    >>> get_score('mcc', y_test, y_pred=y_pred)
    0.5773502691896258
    >>> get_score('sen', y_test, y_pred=y_pred)
    0.5
    >>> get_score('spec', y_test, y_pred=y_pred)
    0.6666666666666666
    """
    if metric=='auc':
        if y_score is None:
            raise ValueError("'y_score' must be set to use 'auc' as a metric.")
        else:
            score = roc_auc_score(y_test, y_score,
                                    average='macro',
                                    multi_class='ovo')
    elif metric == 'mcc':
        if y_pred is None:
            raise ValueError("'y_pred' must be set to use 'mcc' as a metric.")
        else:
            score = matthews_corrcoef(y_test, y_pred)
    elif metric == 'sen':
        if y_pred is None:
            raise ValueError("'y_pred' must be set to use 'sen' as a metric.")
        else:
            score = sensitivity_score(y_test, y_pred, average='macro')
    elif metric == 'spec':
        if y_pred is None:
            raise ValueError("'y_pred' must be set to use 'spec' as a metric.")
        else:
            score = specificity_score(y_test, y_pred, average='macro')
    else:
        raise ValueError("Invalid metric!\nValid metrics are 'auc', 'mcc', 'sen', and 'spec'.")
    
    return score

def test_sampling_method(X_init, y_init, X_pool_init, y_pool_init, model, strategy, X_test, y_test, n_classes, n_instances):
    """
    Evaluate an active learning strategy using a specified model and sampling method.

    This function simulates an active learning scenario where an initial labeled dataset is
    incrementally augmented by querying the most informative samples from a pool of unlabeled
    data. The selected model and sampling strategy are used to iteratively train the model
    and assess its performance on a test set.

    Parameters
    ----------
    X_init : numpy.ndarray, shape (n_samples_init, n_features)
        The initial set of training features.
    
    y_init : numpy.ndarray, shape (n_samples_init,)
        The initial set of training labels.
    
    X_pool_init : numpy.ndarray, shape (n_samples_pool, n_features)
        The initial pool of unlabeled data features.
    
    y_pool_init : numpy.ndarray, shape (n_samples_pool,)
        The initial pool of unlabeled data labels.
    
    model : str
        The type of model to use. If 'committee', a committee of models is used.
        Other valid options depend on the models available in the `get_model` function.
    
    strategy : str
        The sampling strategy to use. Valid strategies for 'committee' are
        'random', 'consensus', 'disagreement', and 'vote'. For single models,
        valid strategies are 'random', 'entropy', 'margin', and 'uncertainty'.
    
    X_test : numpy.ndarray, shape (n_samples_test, n_features)
        The test set features.
    
    y_test : numpy.ndarray, shape (n_samples_test,)
        The test set labels.
    
    n_classes : int
        The number of classes in the classification problem.
    
    n_instances : int
        The number of instances to query from the pool during active learning.
    
    Returns
    -------
    scores : numpy.ndarray, shape (n_metrics, n_queries // 5 + 1)
        The evaluation scores for the metrics 'auc', 'mcc', 'sen', and 'spec'
        at each step where evaluation is performed. Evaluation is done every 5 queries.

    Raises
    ------
    ValueError
        If an invalid strategy is specified for the given model.

    Examples
    --------
    >>> X_train, X_test, y_train, y_test = load_pkl_data('data.pkl')
    >>> X_init, y_init, X_pool_init, y_pool_init = get_initial_sample_pool(X_train, y_train, 3)
    >>> scores = test_sampling_method(X_init, y_init, X_pool_init, y_pool_init,
                                      model='committee', strategy='vote',
                                      X_test=X_test, y_test=y_test,
                                      n_classes=3, n_instances=50)
    >>> print(scores.shape)
    (4, 11)
    """
    X_sample, y_sample = X_init.copy(), y_init.copy()
    X_pool, y_pool = X_pool_init.copy(), y_pool_init.copy()
    
    if model == 'committee':
        learner_list = []
        for l in ['knn', 'rf', 'gb']:
            learner_list.append(ActiveLearner(
            estimator=get_model(l),
            X_training=X_sample, y_training=y_sample))
        
        if strategy == 'random':
            learner = Committee(learner_list=learner_list, query_strategy=random_sampling)
        elif strategy == 'consensus':
            learner = Committee(learner_list=learner_list, query_strategy=consensus_entropy_sampling)
        elif strategy == 'disagreement':
            learner = Committee(learner_list=learner_list, query_strategy=max_disagreement_sampling)
        elif strategy == 'vote':
            learner = Committee(learner_list=learner_list, query_strategy=vote_entropy_sampling)
        else:
            raise ValueError("Invalid strategy for model 'committee'!\nValid strategies are 'random', 'consensus', 'disagreement', and 'vote'.")
        
    else:
        clf = get_model(model)
        if strategy == 'random':
            learner = ActiveLearner(estimator=clf, query_strategy=random_sampling,
                                    X_training=X_sample, y_training=y_sample)
        elif strategy == 'entropy':
            learner = ActiveLearner(estimator=clf, query_strategy=entropy_sampling,
                                    X_training=X_sample, y_training=y_sample)
        elif strategy == 'margin':
            learner = ActiveLearner(estimator=clf, query_strategy=margin_sampling,
                                    X_training=X_sample, y_training=y_sample)
        elif strategy == 'uncertainty':
            learner = ActiveLearner(estimator=clf, query_strategy=uncertainty_sampling,
                                    X_training=X_sample, y_training=y_sample)
        else:
            raise ValueError("Invalid strategy for model" + model + "!\nValid strategies are 'random', 'entropy', 'margin', and 'uncertainty'.")
    
    y_pred = learner.predict(X_test)
    y_score = modify_predict_output(np.unique(y_sample), learner.predict_proba(X_test), n_classes)
    metrics = ['auc', 'mcc', 'sen', 'spec']
    scores = [[get_score(metric, y_test, y_pred, y_score)] for metric in metrics]
    
    for idx in range(n_instances):
        print('Query', idx+1, 'of', n_instances, end='\r')
        query_index, _ = learner.query(X_pool)
        
        X_inst = X_pool[query_index].reshape(1,-1)
        y_inst = y_pool[query_index].reshape(1,)
        learner.teach(X_inst, y_inst)
        
        X_sample = np.concatenate((X_sample, X_inst), axis=0)
        y_sample = np.concatenate((y_sample, y_inst))
        
        X_pool = np.delete(X_pool, query_index, axis=0)
        y_pool = np.delete(y_pool, query_index)
        
        if (y_sample.size - y_init.size) % 5 == 0:
            y_pred = learner.predict(X_test)
            y_score = modify_predict_output(np.unique(y_sample), learner.predict_proba(X_test), n_classes)
            for metric,score in zip(metrics,scores):
                score.append(get_score(metric, y_test, y_pred, y_score))
        
    return np.array(scores)

def get_model(model):
    """
    Return a machine learning classifier based on the specified model type.

    This function initializes and returns a classifier from Scikit-learn 
    based on the provided model type. The supported model types are:
    - 'knn': K-Nearest Neighbors classifier
    - 'rf': Random Forest classifier
    - 'gb': Histogram-based Gradient Boosting classifier
    - 'vote': A soft voting classifier combining KNN, Random Forest, 
              and Histogram-based Gradient Boosting classifiers with specified weights

    Parameters
    ----------
    model : str
        The type of model to return. Must be one of 'knn', 'rf', 'gb', or 'vote'.

    Returns
    -------
    clf : sklearn.base.BaseEstimator
        An initialized Scikit-learn classifier corresponding to the specified model type.

    Raises
    ------
    ValueError
        If the specified model type is not one of the supported values ('knn', 'rf', 'gb', 'vote').

    Examples
    --------
    >>> clf = get_model('knn')
    >>> print(type(clf))
    <class 'sklearn.neighbors._classification.KNeighborsClassifier'>

    >>> clf = get_model('vote')
    >>> print(type(clf))
    <class 'sklearn.ensemble._voting.VotingClassifier'>
    """
    # Choose the appropriate classifier based on the specified model type
    if model == 'knn':
        clf = KNeighborsClassifier()
    
    elif model == 'rf':
        clf = RandomForestClassifier()
    
    elif model == 'gb':
        clf = HistGradientBoostingClassifier()
        
    elif model == 'vote':
        # Voting classifier combining KNN, RF, and XGB
        clf = VotingClassifier(estimators=[
            ('gb', HistGradientBoostingClassifier()),
            ('rf', RandomForestClassifier()),
            ('knn', KNeighborsClassifier())],
                               voting='soft',
                               weights=[2,2,1],
                               flatten_transform=True)
    
    else:
        raise ValueError("Invalid model!\nValid models are 'knn', 'rf', 'gb', and 'vote'.")
    
    return clf

def get_baseline_model(X_train, y_train, X_test, y_test):
    """
    Train and evaluate baseline machine learning models on the given dataset.

    This function trains a series of baseline models ('knn', 'rf', 'gb', 'vote') on the 
    provided training data and evaluates them on the testing data using specified metrics.
    The results are returned in a DataFrame containing the performance scores for each model.

    Parameters
    ----------
    X_train : numpy.ndarray
        Features for the training set.
    
    y_train : numpy.ndarray
        Labels for the training set.
    
    X_test : numpy.ndarray
        Features for the testing set.
    
    y_test : numpy.ndarray
        Labels for the testing set.

    Returns
    -------
    scores : pandas.DataFrame
        A DataFrame containing the performance scores of each model. The DataFrame has columns:
        - 'model': Name of the model.
        - 'auc': Area under the ROC curve.
        - 'mcc': Matthews correlation coefficient.
        - 'sen': Sensitivity (recall).
        - 'spec': Specificity.

    Notes
    -----
    The function uses the following models:
    - 'knn': K-Nearest Neighbors
    - 'rf': Random Forest
    - 'gb': Histogram-based Gradient Boosting
    - 'vote': Voting Classifier

    The following metrics are used for evaluation:
    - 'auc': Area Under the Curve
    - 'mcc': Matthews Correlation Coefficient
    - 'sen': Sensitivity (Recall)
    - 'spec': Specificity

    The `get_model` function is assumed to return the model instance based on the model name.
    The `get_score` function is assumed to calculate the specified metric.

    Raises
    ------
    KeyError
        If a specified metric or model is not recognized by `get_model` or `get_score`.

    Examples
    --------
    >>> X_train, X_test, y_train, y_test = load_pkl_data('data.pkl')
    >>> scores = get_baseline_model(X_train, y_train, X_test, y_test)
    >>> print(scores)
         model       auc       mcc       sen      spec
    0      knn  0.843210  0.750000  0.812500  0.875000
    1       rf  0.927340  0.810000  0.850000  0.890000
    2       gb  0.912345  0.789000  0.820000  0.860000
    3     vote  0.934210  0.820000  0.860000  0.900000
    """
    models = ['knn', 'rf', 'gb', 'vote']
    metrics = ['auc', 'mcc', 'sen', 'spec']

    scores = pd.DataFrame(columns=['model']+metrics)
    for model in models:
        clear_output()
        
        scores.loc[len(scores)] = [model,0.,0.,0.,0.]
        
        print("Training '" + model + "'..")
        clf = get_model(model)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_score = clf.predict_proba(X_test)
        
        print("Testing '" + model + "'..")
        for metric in metrics:
            scores.at[len(scores)-1, metric] = get_score(metric, y_test, y_pred=y_pred, y_score=y_score)
    
    clear_output()
    print('Done!')
    return scores
        
def run_methods(X_train, y_train, X_test, y_test, n_classes, n_initial, n_instances, n_runs, model, strategies):
    """
    Evaluate multiple sampling strategies for a given model on provided datasets.

    This function runs multiple sampling strategies on the training and testing 
    datasets, records their performance metrics, and returns a DataFrame 
    summarizing the results. Each strategy is tested multiple times to assess 
    its mean performance.

    Parameters
    ----------
    X_train : numpy.ndarray
        Features of the training dataset.
    
    y_train : numpy.ndarray
        Labels of the training dataset.
    
    X_test : numpy.ndarray
        Features of the testing dataset.
    
    y_test : numpy.ndarray
        Labels of the testing dataset.
    
    n_classes : int
        The number of classes in the dataset.
    
    n_initial : int
        The initial number of samples to start with.
    
    n_instances : int
        The number of instances to sample in each run.
    
    n_runs : int
        The number of runs to perform for each strategy.
    
    model : object
        The machine learning model to be used.
    
    strategies : list of str
        A list of sampling strategies to be evaluated.

    Returns
    -------
    scores_df : pandas.DataFrame
        A DataFrame containing the mean scores for each strategy and each metric.
        The columns include 'strategy', 'auc', 'mcc', 'sen', and 'spec'.

    Raises
    ------
    ValueError
        If any of the input parameters are invalid or if the strategies list is empty.

    Notes
    -----
    - This function uses an external method `get_initial_sample_pool` to split the initial 
      sample and pool from the training dataset.
    - It also uses an external method `test_sampling_method` to test each sampling strategy.
    - The function assumes the presence of the `clear_output` function from IPython.display 
      for output clearing during iterations.
    """
    scores = {}
    for strategy in strategies: scores[strategy] = []
    
    for run in range(n_runs):
        X_initial, y_initial, X_pool, y_pool = get_initial_sample_pool(X_train, y_train, n_initial)
        for strategy in strategies:
            clear_output()
            print('Initial batch size:', n_initial)
            print(strategy.upper() + ': Run', run+1, 'of', n_runs)
            scores[strategy].append(test_sampling_method(X_initial, y_initial, X_pool, y_pool, model, strategy, X_test, y_test, n_classes, n_instances))
    clear_output()
    print('Calculating scores..')
    
    # List to store all the results as rows for DataFrame
    results = []
    metrics = ['auc', 'mcc', 'sen', 'spec']
    # scores_df = pd.DataFrame(columns=['strategy']+metrics)
    # arr = np.array([])
    # for strategy in strategies:
    #     scores_df.loc[len(scores_df)] = [strategy,arr,arr,arr,arr]
    #     for metric,idx in zip(metrics,range(len(metrics))):
    #         score = np.empty((n_runs, int(n_instances/5)+1))
    #         for run in range(n_runs):
    #             score[run,:] = scores[strategy][run][idx,:]
            
    #         scores_df.at[len(scores_df)-1, metric] = np.mean(score, axis=0)
    
    # Populate the results list with individual run scores
    for strategy in strategies:
        for run in range(n_runs):
            run_scores = scores[strategy][run]
            for metric_idx, metric in enumerate(metrics):
                metric_scores = run_scores[metric_idx, :]  # Scores for each metric across instances
                for instance_idx, instance_score in enumerate(metric_scores):
                    results.append({
                        'strategy': strategy,
                        'run': run + 1,  # 1-based run index
                        'instance': instance_idx * 5,  # 1-based instance index
                        metric: instance_score
                    })
    
    # Convert results to DataFrame
    scores_df = pd.DataFrame(results)
    # Group by `strategy`, `run`, and `instance` and aggregate with `first` to combine rows
    scores_df = scores_df.groupby(['strategy', 'run', 'instance'], as_index=False).agg({
        'auc': 'first',
        'mcc': 'first',
        'sen': 'first',
        'spec': 'first'
    })
    
    print('Done!')
    
    return scores_df
        
def plot_scores(scores, n_initial, n_instances, metric, ax, add_title=False, add_xlabel=False, add_ylabel=False, add_legend=False):
    """
    Plot performance scores for different strategies over additional instances.

    This function plots the performance of various strategies as a function of 
    the number of additional instances added during the evaluation. The performance 
    is measured using a specified metric and plotted on the provided Axes object.

    Parameters
    ----------
    scores : pandas.DataFrame
        A DataFrame containing the performance scores. It should have a column 
        named 'strategy' to distinguish between different strategies and columns
        for each metric to be plotted.
    
    n_initial : int
        The initial batch size.
    
    n_instances : int
        The number of additional instances over which the performance is evaluated.
    
    metric : str
        The performance metric to be plotted. Must be one of 'auc', 'mcc', 'sen', or 'spec'.
    
    ax : matplotlib.axes.Axes
        The axes object on which to plot the scores.
    
    add_title : bool, optional
        If True, adds a title to the plot indicating the initial batch size (default is False).
    
    add_xlabel : bool, optional
        If True, adds an x-axis label to the plot (default is False).
    
    add_ylabel : bool, optional
        If True, adds a y-axis label to the plot (default is False).
    
    add_legend : bool, optional
        If True, adds a legend to the plot (default is False).

    Raises
    ------
    ValueError
        If an invalid metric is provided. Valid metrics are 'auc', 'mcc', 'sen', and 'spec'.

    Returns
    -------
    None

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> plot_scores(scores, n_initial=10, n_instances=50, metric='auc', ax=ax, add_title=True, add_xlabel=True, add_ylabel=True, add_legend=True)
    """
    n_list = range(0,n_instances+1,5)
    markers = ['o', 's', '^', 'P']
    strategies = scores['strategy'].unique()
    for idx in range(len(strategies)):
        ax.plot(n_list, scores[metric][idx], label=strategies[idx], marker=markers[idx], markeredgecolor='white', markersize=7)
    
    if metric == 'auc': m = 'AUC'
    elif metric == 'mcc': m = 'MCC'
    elif metric == 'sen': m = 'Sensitivity'
    elif metric == 'spec': m ='Specificity'
    else: raise ValueError("Invalid metric!\nValid metrics are 'auc', 'mcc', 'sen', and 'spec'.")
    
    if add_title: ax.set_title('Initial batch size = ' + str(n_initial))
    if add_xlabel: ax.set_xlabel('Number of additional instances')
    if add_ylabel: ax.set_ylabel(m + ' on test set')
    
    #ax.grid(True)
    
    if add_legend: ax.legend();
    
    return

def create_scores_figure(scores, y_min, y_max, fig_title, pdf_name):
    """
    Create and save a figure with subplots of score metrics across different initial sample sizes.

    This function generates a 4x3 grid of subplots, each displaying a specific metric 
    (AUC, MCC, Sensitivity, Specificity) across different initial sample sizes 
    (20, 100, 500). Each subplot is customized with titles, labels, and legends as needed. 
    The y-axis limits for each row of subplots are set according to the provided y_min and y_max values. 
    The figure is saved as a PDF with the specified filename.

    Parameters
    ----------
    scores : list of list
        A nested list where each inner list contains scores for the different metrics.
        The outer list should have a length of 3, corresponding to the different initial sample sizes.
        
    y_min : list of float
        A list of minimum y-axis values for each row of subplots.
        The list should have a length of 4, corresponding to the 4 metrics.
        
    y_max : list of float
        A list of maximum y-axis values for each row of subplots.
        The list should have a length of 4, corresponding to the 4 metrics.
        
    fig_title : str
        The title of the entire figure.
        
    pdf_name : str
        The base name for the output PDF file (without the '.pdf' extension).

    Returns
    -------
    None

    Notes
    -----
    - The function assumes that the `plot_scores` function is defined elsewhere and 
      takes the following parameters: scores, n_initial, n_instances, metric, ax, and 
      optional keyword arguments: add_title, add_xlabel, add_ylabel, add_legend.
      
    - The metrics displayed in the subplots are AUC, MCC, Sensitivity, and Specificity.

    Example
    -------
    >>> scores = [
    ...     [auc_scores_20, auc_scores_100, auc_scores_500],
    ...     [mcc_scores_20, mcc_scores_100, mcc_scores_500],
    ...     [sen_scores_20, sen_scores_100, sen_scores_500],
    ...     [spec_scores_20, spec_scores_100, spec_scores_500]
    ... ]
    >>> y_min = [0.0, 0.0, 0.0, 0.0]
    >>> y_max = [1.0, 1.0, 1.0, 1.0]
    >>> fig_title = "Score Metrics for Different Initial Sample Sizes"
    >>> pdf_name = "score_metrics_figure"
    >>> create_scores_figure(scores, y_min, y_max, fig_title, pdf_name)
    """
    n_initials = [20, 100, 500]
    metrics = ['auc', 'mcc', 'sen', 'spec']
    
    # Create a figure and an array of subplots
    fig, axs = plt.subplots(4, 3, figsize=(12, 12))
    
    # Loop through each subplot and call your function
    for i in range(4):
        for j in range(3):
            ax = axs[i, j]  # Get the current axis
            
            if i == 0 and j == 0:
                plot_scores(scores[j], n_initials[j], 50, metrics[i], ax, add_title=True, add_ylabel=True, add_legend=True)
            elif i == 3 and j == 0:
                plot_scores(scores[j], n_initials[j], 50, metrics[i], ax, add_xlabel=True, add_ylabel=True)
            elif i == 0:
                plot_scores(scores[j], n_initials[j], 50, metrics[i], ax, add_title=True)
            elif i == 3:
                plot_scores(scores[j], n_initials[j], 50, metrics[i], ax, add_xlabel=True)
            elif j == 0:
                plot_scores(scores[j], n_initials[j], 50, metrics[i], ax, add_ylabel=True)
            else:
                plot_scores(scores[j], n_initials[j], 50, metrics[i], ax)
    
    # Loop through each subplot again to set the y-axis limits
    for i in range(4):
        for j in range(3):
            ax = axs[i, j]  # Get the current axis
            ax.set_ylim(y_min[i], y_max[i])  # Set the y-axis limits
    
    plt.suptitle(fig_title, fontsize=16, fontweight='medium')
    
    # Adjust layout to prevent overlap of titles
    plt.tight_layout()
    
    plt.savefig(pdf_name+'.pdf')
    
    # Show the plot
    plt.show(); 

def preprocess_set_for_AL(X, y, param,train_size=200,test_size=0.2):
    """
    Preprocess a dataset by splitting, shuffling, scaling, and applying PCA, 
    then save the processed data to a file.

    This function performs the following steps:
    1. Splits the input dataset into training, pool and testing sets with stratification.
    2. Shuffles the training, pool and testing sets.
    3. Scales the features to a range [0, 1] using Min-Max Scaling.
    4. Applies Principal Component Analysis (PCA) to the scaled features.
    5. Saves the preprocessed data to a pickle file.

    Parameters
    ----------
    X : numpy.ndarray, shape (n_samples, n_features)
        The feature matrix of the dataset.
    
    y : numpy.ndarray, shape (n_samples,)
        The labels corresponding to the features.
    
    param : str
        A parameter to include in the output filename for identification.
    
    test_size : float, optional, default=0.1
        The proportion of the dataset to include in the test split.

    Returns
    -------
    None

    Examples
    --------
    >>> X = [[0, 1], [1, 0], [0, 0], [1, 1]]
    >>> y = [0, 1, 0, 1]
    >>> preprocess_set(X, y, 'example_param')
    Data saved in file sec3_1_data_output_for_AL_example_param.pkl
    """
    #Splitting the dataset

    X_train, X_pool, y_train, y_pool = train_test_split(X, y, train_size=train_size, stratify=y) #Here, train_size = 20 => 20 samples for training
    X_pool, X_test, y_pool, y_test = train_test_split(X_pool, y_pool, test_size=test_size) #Here, test_size = 0.2 => 20% of the pool set for testing

    # shuffle data after split
    X_train, y_train = shuffle(X_train, y_train)
    X_pool, y_pool = shuffle(X_pool, y_pool)
    X_test, y_test = shuffle(X_test, y_test)
    
    # initialize a scaler
    scaler = MinMaxScaler()
    # fit the scaler using training set and transform all data
    X_train = scaler.fit_transform(X_train)
    X_pool = scaler.transform(X_pool)
    X_test = scaler.transform(X_test)
    
    # initialize a PCA object
    pca = PCA(n_components=X.shape[1])
    # fit PCA using training set and transform all data
    X_train = pca.fit_transform(X_train)
    X_pool = pca.transform(X_pool)
    X_test = pca.transform(X_test)
    
    # save data to file
    file_data = {'X_train': X_train, 'y_train': y_train,'X_pool': X_pool, 'y_pool': y_pool,
             'X_test': X_test, 'y_test': y_test}
    file_name = 'sec3_1_data_output_for_AL_'+ param + '.pkl'
    with open(file_name, 'wb') as f:
        pkl.dump(file_data, f, protocol=pkl.HIGHEST_PROTOCOL)
        
    print('Data saved in file ' + file_name)
    
    return

def load_pkl_data_for_AL(file_name):
    """
    Load data from a pickle file and return training and testing sets.

    This function reads a pickle file containing a dictionary with keys 
    'X_train', 'y_train','X_pool','y_pool', 'X_test', and 'y_test', and returns the 
    corresponding datasets. The data is assumed to be in the following format:
    - 'X_train': Features for the training set
    - 'y_train': Labels for the training set
    - 'X_pool': Features for the pool set
    - 'y_pool': Labels for the pool set
    - 'X_test': Features for the testing set
    - 'y_test': Labels for the testing set

    Parameters
    ----------
    file_name : str
        The path to the pickle file to be loaded.

    Returns
    -------
    X_train : numpy.ndarray
        Features for the training set.
    
    X_test : numpy.ndarray
        Features for the testing set.

    X_pool : numpy.ndarray
        Features for the pool set.

    y_pool : numpy.ndarray
        Labels for the pool set.
    
    y_train : numpy.ndarray
        Labels for the training set.
    
    y_test : numpy.ndarray
        Labels for the testing set.

    Examples
    --------
    >>> X_train, X_test, X_pool, y_pool, y_train, y_test = load_pkl_data_for_AL('data.pkl')
    >>> print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    (500, 20) (100, 20) (500,) (100,)
    """
    with open(file_name, 'rb') as f:
        data = pkl.load(f)
    
    X_train, y_train = data['X_train'], data['y_train']
    X_pool, y_pool = data['X_pool'], data['y_pool']
    X_test, y_test = data['X_test'], data['y_test']
    
    return X_train, X_pool, X_test, y_train, y_pool, y_test