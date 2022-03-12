import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os
import copy
from sklearn import metrics as s_mtr, tree, ensemble, cluster, decomposition as sdec, feature_extraction as sfe

# alternatively: from seaborn.utils import np, pd, plt, os    
    

def check_correlations(df, title):
    """plot correlation heatmap from dataframe"""
    fig = plt.figure(figsize=(18, 8), dpi=200)
    ax = sns.heatmap(np.round(df.corr(), 2), annot_kws={'fontsize':6},
                     annot=True, 
                     square=True,)
                     
    plt.title(title, weight='bold', x=0.5, y=1.05)
    
    ax.set_xticklabels(ax.get_xticklabels(), size=6)
    ax.set_yticklabels(ax.get_yticklabels(), size=6)
    
    return fig
    

def K_search(X, k_max=3):
    """SSD is the sum of squared distance of 
    each data point from its closest cluster center.
    Returns
    dataframe: a report"""
    
    result = {"K": [],
             "SSD": []}
    
    for k in range(2, k_max):
        kmeans = cluster.KMeans(n_clusters=k, random_state=101)
        kmeans.fit_predict(X)
        result['K'].append(k)
        result['SSD'].append(kmeans.inertia_)
        
    return pd.DataFrame(result)
    

def check_multi_colinearity(df, x_colnames: list, y_col):
    """check for multi-colinearity among features in a dataframe"""
    
    if not isinstance(y_col, (str, pd.Series)):
        raise ValueError("y_col must be either str or series")
        
    if isinstance(y_col, str):
        y_col = df[y_col]
        
    x = pd.DataFrame(df[x_colnames])
    
    return x.corrwith(y_col)


def calc_days_between(historic_date: str, later_date: str):
    """compute, in days, the difference between two given dates.
    Date difference is later_date - historic_date"""
    
    if not isinstance(historic_date, (np.datetime64, str)) or not isinstance(later_date, (np.datetime64, str)):
        raise TypeError("date given must be in str or np.datetime64 types")
    
    historic_date, later_date = np.datetime64(historic_date), np.datetime64(later_date)
    
    return (later_date - historic_date).astype(int)


def visualize_binaryclf_report(trained_model, test_input, test_label, is_ANN=False, ANN_cutoff=0.5):
    """performance report for trained model"""
    
    test_pred = trained_model.predict(test_input)
    if is_ANN:
        test_pred = (test_pred >= ANN_cutoff).astype(int)
    
    print(s_mtr.classification_report(test_label, test_pred))
    
    sns.set_style('white')
    plt.figure(figsize=(6, 5), dpi=200)
    s_mtr.ConfusionMatrixDisplay.from_predictions(test_label, test_pred)
    
    
def cast_appropriate_dtypes(df):
    """convert each dataframe column to an appropriate datatype.
    Firstly, try to cast to integer. If unsuccessful, try to 
    cast to float. If still unsuccessful, leave column as string.
    
    Returns:
    dataframe with appropriate datatypes."""
    
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas dataframe object")
        
    cols = df.columns
    result = pd.DataFrame()
    for i in range(len(cols)):
        try:  # firstly, try to convert to integer
            result[cols[i]] = df[cols[i]].astype(int)
        except ValueError as err:
            try:  # secondly, try to convert to float
                result[cols[i]] = df[cols[i]].astype(float)
            except ValueError as err:  # lastly, leave as str
                result[cols[i]] = df[cols[i]]
    
    return result


def complete_rules(freq_A, freq_B, freq_AB, Total):
    def support(freq_AB, Total):
        try:
            result = np.round(freq_AB/Total, 4)
        except ZeroDivisionError as err:
#             print('result is inconclusive')
            result = np.nan
        return result
    
    def confidence(freq_AB, freq_A):
        try:
            result = np.round(freq_AB/freq_A, 4)
        except ZeroDivisionError as err:
#             print('result is inconclusive')
            result = np.nan
        return result
    
    def completedness(freq_AB, freq_B):
        try:
            result = np.round(freq_AB/freq_B, 4)
        except ZeroDivisionError as err:
#             print('result is inconclusive')
            result = np.nan
        return result
    
    def lift(supportAB, supportA, supportB):
        try:
            result =  np.round((supportAB/(supportA * supportB)), 4)
        except ZeroDivisionError as err:
#             print('result is inconclusive')
            result = np.nan
        return result

    def conviction(supportB, confidenceAB):
        try:
            result = np.round((1 - supportB)/ (1 - confidenceAB), 4)
        except ZeroDivisionError as err:
#             print('result is inconclusive')
            result = np.nan
        return result
    
    def interestingness(freq_AB, freq_A, freq_B, Total):
        try:
            result = np.round((freq_AB - ((freq_A * freq_B)/Total)), 4)
        except ZeroDivisionError as err:
#             print('result is inconclusive')
            result = np.nan
        return result
    
    return pd.Series({'support': support(freq_AB, Total),
           'confidence': confidence(freq_AB, freq_A),
           'completedness': completedness(freq_AB, freq_B),
           'lift': lift(supportAB, supportA, supportB),
           'conviction': conviction(supportB, confidenceAB),
           'interestingness': interestingness(freq_AB, freq_A, freq_B, Total)})
           

def association_rules(A:str, B: str, df, report_as_df: bool=True):
    """compute the support, confidence, completedness, lift, conviction, interestingness
    for columns A and B."""
    
    unqA = df[A].unique()
    unqB = df[B].unique()
    total = len(df)
    result = dict()
    
    for a in unqA:
        freqA = len(df.loc[df[A] == a])
        for b in unqB:
            freqB = len(df.loc[df[B] == b])
            freqAB = len(df.loc[(df[A] == a) &
                                (df[B] == b)])
            result[f'{a} & {b}'] = complete_rules(freqA, freqB, freqAB, total)
    if report_as_df:
        return pd.DataFrame(data=result.values(), index=result.keys()).sort_index()
    return result


def select_top_five_threshold(association_rules_df, support_thresh=0.5, confidence_thresh=0.5):
    """pick the top 5 above threshold value."""
    
    return association_rules_df.loc[(association_rules_df['support'] >= support_thresh) &
                                    (association_rules_df['confidence'] >= confidence_thresh)].sort_values(['support', 'confidence'], ascending=False).iloc[:5]


def association_rules_many(df, A: list, B: str):
    """compute association rules for two columns (A) and
    single column (B)"""
    
    unq_B = df[B].unique()

    for main_col in A:
        unq_main_vals = df[main_col].unique()
        others = list(A)
        others.remove(main_col)
        unq_other_vals = {}

        for other_col in others:
            unq_other_vals[other_col] = df[other_col].unique()
        for b in unq_B:   
            for main_unq in unq_main_vals:
                for ocol, other_unq in unq_other_vals.items():
                    for val in other_unq:
                        freqA = len(df.loc[(df[main_col] == main_unq) &
                                               (df[ocol] == val)])
                        freqB = len(df.loc[df[B] == b])
                        freqAB = len(df.loc[(df[main_col] == main_unq) &
                                               (df[ocol] == val) &
                                               (df[B] == b)])
                        result[f"{main_unq} + {val} & {b}"] = complete_rules(freqA, freqB, freqAB, total)

    return pd.DataFrame(data=result.values(), index=result.keys()).sort_index()
    
    
def gen_word_freq_from_docs(filenames: 'str_or_array', add_bias: int=0, report_as_dataframe: bool=False):
    """Generate word frequency count for one or more text file(s)"""
    
    def read_one(filenames: str):
        result = dict()
        try:
            with open(filenames, mode='r', encoding='utf8') as f:
                result = [w.strip() for w in f.read().lower().split()]
        except err:
            print(f"Error has occurred: {err}")
            
        return result
    
    def read_many(filenames: list):
        return [read_one(filenames[i]) for i in range(len(filenames))]
        
    def create_all_unq_words(files: list):
        unq_words = dict()
        final_dict = set()
        # dictionary for each file
        for i in range(len(files)):
            unq_words[i] = set(files[i])
        # set of unique words from each file
        for file_ix, unq_word in unq_words.items():
            final_dict = final_dict.union(unq_word)
            
        return sorted( [[w, c] for w, c in zip(tuple(final_dict), tuple([0]*len(final_dict)))] )
    
    def value_extractor(word_vector, extract_count=True):
        if extract_count:
            return [c for w, c in word_vector]
        return [w for w, c in word_vector]
    
    def each_file_count(files: 'list of words'):
        """count occurrences"""
        
        if not isinstance(files[0], list):
            files = [files] 
            
        result = dict()
        all_word_count = create_all_unq_words(files)
    
        for f_ix, f in enumerate(files):  # for each file
            this_count = [[w, c+add_bias] for w,c in copy.deepcopy(all_word_count)]
            
            for i in range(len(f)):  # for each word index in current file
                # go thru each word in the general dictionary
                for n in range(len(all_word_count)):
                    # if current word is found in general dictionary
                    if this_count[n][0] == f[i]:
                        # increment that word's count by 1
                        this_count[n][1] += 1
            # store word count for file in result
            result[f"doc_{f_ix}"] = this_count

        return value_extractor(all_word_count, False), result
    
    def report_as_df(dict_output):
        """convert the final output from dict format to a pandas dataframe"""
        
        data_header = dict_output['words_column']
        data_vals = [val for label, val in dict_output.items() if 'doc' in label]
        data_ind = [label for label, val in dict_output.items() if 'doc' in label]
        
        return pd.DataFrame(data=data_vals, index=data_ind, columns=data_header)
    
    # read file content
    if isinstance(filenames, str):
        file_out = read_one(filenames)
    elif isinstance(filenames, (list, tuple)):
        file_out = read_many(filenames)
    
    all_words, result = each_file_count(file_out)
    final_output = {'words_column': all_words}
    
    for fname, count in result.items():
        final_output[fname] = value_extractor(count, True)
    
    if report_as_dataframe:
        return report_as_df(final_output)
    
    return final_output
    

def classification_metrics(y_true, y_pred, for_both_classes=False):
    """Calculate the accuracy, precision, and recall, f1_score values
    considering the expected values and predicted values.
    if for_both_classes is True, return each performance metric 
    for class 0 and 1, otherwise return for only class 1
    Returns
    performance metrics dict"""
    result = dict()
    using = 'binary'
    if for_both_classes:
        using = None
    result['recall'] = np.round(s_mtr.recall_score(y_true, y_pred, average=using), 4)
    result['precision'] = np.round(s_mtr.precision_score(y_true, y_pred, average=using), 4)
    result['f1_score'] = np.round(s_mtr.f1_score(y_true, y_pred, average=using), 4)
    result['accuracy'] = np.round(s_mtr.accuracy_score(y_true, y_pred), 4)
    
    return result
    

def tree_feat_weights(model, col_names):
    """present the individual weights assigned to each columns
    by decision tree model"""
    
    return pd.Series(data=model.feature_importances_, index=col_names, name='feature_weights').sort_values(ascending=False)
    

def np_value_counts(arr):
    """count the number of unique values in a 1D numpy array.
    """
    unq_ele = np.unique(arr)
    return {unq_ele[i]: len(np.where(arr == unq_ele[i])[0]) for i in range(len(unq_ele))}
    

def label_from_binary_vectors(arr: 'numpy.ndarray', label_guide: dict):
    """convert a 2D array of columns containing the probabilities (in floats)
     of belonging to each class (column) into their respective labels (string),
      as indicated by label_guide dict
    """

    def index_from_probs(arr):
        """grab the index position of the highest value along each row.
        Now, each row is transformed from a vector into an integer"""
        return np.argmax(arr, axis=1)

    label_indices = index_from_probs(arr)

    return np.vectorize(lambda x: label_guide[x])(label_indices)


def float_to_binary(val: float):
    """
    transform decimal to 0 or 1 if val >= 0.5
    :param val:
    :return:
    """
    standard = 0.4444444444444445
    return 1 if val >= standard else 0


def categ_to_binary(ser: 'pd.Series', guide: dict):
    """
    convert categorical value to binary value
    :param val:
    :param guide:
    :return: 0 or 1
    """
    if not isinstance(ser, pd.Series) or not isinstance(guide, dict):
        raise TypeError("ser and guide must be a pandas series and dictionary instances respectively")

    def to_binary(val: str):
        return guide[val]

    return ser.apply(func=lambda val: to_binary(val))


def convert_month(val: str, to_words: str = False):
    """change datatype of month from words to integer (vice versa)
    """

    guide = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
             'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}

    def convert_to_words(val):
        return dict([(v, k) for k, v in guide.items()])[val]

    def convert_to_int(val):
        for k, v in guide.items():
            if val.lower() in k.lower():
                return v

    if to_words:
        return convert_to_words(val)
    return convert_to_int(val)


def impute_null_values(df: 'pd.DataFrame', pivot_cols: list, target_col: str, with_mean=True):
    """
    Impute the null values in a target feature using aggregated values
    (eg mean, median, mode) based on pivot features
    :param df:
    :param pivot_cols:
    :param target_col:
    :param with_mean:
    :return: impute_guide: 'a pandas dataframe'
    """
    
    if ('object' in str(df[target_col].dtypes)) or ('bool' in str(df[target_col].dtypes)):
        with_mean = False
        
    def pivot_mode(df: 'pd.DataFrame', pivot_cols: list, target_col: str):
        """rank the occurrences of target_col values based on pivot_cols,
        and return only the mode (i.e highest occurring) target_col
        value per combination of pivot_cols.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Object given is not a pandas dataframe")
        if not isinstance(pivot_cols, list):
            raise TypeError("pivot columns should be in list")
        if not isinstance(target_col, str):
            raise TypeError("Target column name must be a string")

        freq_df = df.loc[df[target_col].notnull(), pivot_cols + [target_col]].value_counts().reset_index()
        return freq_df.drop_duplicates(subset=pivot_cols).drop(labels=[0], axis=1)

    def pivot_mean(df: 'pd.DataFrame', pivot_cols: list, target_col: str):
        """rank the occurrences of target_col values based on pivot_cols,
        and return only the highest occurring target_col value per combination
        of pivot_cols.
        """
        
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Object given is not a pandas dataframe")
        if not isinstance(pivot_cols, list):
            raise TypeError("pivot columns should be in list")
        if not isinstance(target_col, str):
            raise TypeError("Target column name must be a string")
        
        dec_places = 4
        targ_dtype = str(df[target_col].dtypes)
        
        if ('int' in targ_dtype):
            dec_places = 0
            targ_dtype = str(df[target_col].dtypes)[:-2]
            
        elif ('float' in targ_dtype):    
            sample = str(df.loc[df[target_col].notnull(), target_col].iloc[0])
            dec_places = len(sample.split('.')[-1])
            targ_dtype = str(df[target_col].dtypes)[:-2]
        
        freq_df = np.round(df.loc[df[target_col].notnull(), pivot_cols + [target_col]].groupby(by=pivot_cols).mean().reset_index(), dec_places)
        return freq_df.drop_duplicates(subset=pivot_cols)

    if with_mean:
        impute_guide = pivot_mean(df, pivot_cols, target_col)
    else:
        impute_guide = pivot_mode(df, pivot_cols, target_col)

    # fill null values with means
    null_ind = df.loc[df[target_col].isnull()].index

    piv_rec = pd.merge(left=df.loc[null_ind, pivot_cols],
                       right=impute_guide,
                       how='left', on=pivot_cols)
    # fill all lingering null values with general mean
    if piv_rec[target_col].isnull().sum():
        piv_rec.loc[piv_rec[target_col].isnull(), target_col] = impute_guide[target_col].mode().values[0]

    piv_rec.index = null_ind

    return pd.concat([df.loc[df[target_col].notnull(), target_col],
                      piv_rec[target_col]],
                     join='inner', axis=0).sort_index()


def dataset_split(x_array: 'np.ndarray', y_array: 'np.ndarray', perc_test, perc_val=0):
    """create train, validation, test sets from x and y dataframes
    Returns:
    if  val_len != 0:  # perc_val is not 0
        (x_train, x_val, x_test, y_train, y_val, y_test)
    else:  # perc_val is 0 (no validation set)
        (x_train, x_test, y_train, y_test)"""

    if not (isinstance(x_array, np.ndarray) and isinstance(y_array, np.ndarray)):
        raise TypeError("x_array/y_array is not np.ndarray")

    nx, ny = len(x_array), len(y_array)

    if nx != ny:
        raise ValueError("x_array and y_array have unequal number of samples")

    # number of samples for each set
    test_len = int(nx * perc_test)
    val_len = int(nx * perc_val)
    train_len = int(nx - sum([val_len, test_len]))

    print(f"Training: {train_len} samples")
    print(f"Validation: {val_len} samples")
    print(f"Test: {test_len} samples")

    if sum([train_len, test_len, val_len]) != nx:
        print("Error in computation")
        return None

    # indexes for x_array/y_array
    inds = np.arange(nx)
    np.random.shuffle(inds)

    # random indexes of test, val and training sets
    test_ind = inds[:test_len]
    val_ind = inds[test_len: test_len + val_len]
    train_ind = inds[test_len + val_len:]

    x_test, y_test = x_array[test_ind], y_array[test_ind]
    x_val, y_val = x_array[val_ind], y_array[val_ind]
    x_train, y_train = x_array[train_ind], y_array[train_ind]

    return (x_train, x_val, x_test, y_train, y_val, y_test) if val_len else (x_train, x_test, y_train, y_test)


def nn_weights_biases(model_instance: 'Keras_model'):
    """
    get weights and biases of a neural network
    :param model_instance: 
    :return: weights, biases
    """
    print("Weights and biases are given below:\n"+
          f"{model_instance.weights}")
    params = model_instance.get_weights()
    weights = [params[i] for i in range(len(params)) if i % 2 == 0]
    biases = [params[i] for i in range(len(params)) if i % 2 != 0]
    return weights, biases


def one_hot_encode_array_integers(arr: np.array):
    '''transform an integer into a list of binary values
    with 1 at the index position of the original integer
    an 0s at the remaining index positions
    '''
    guide = {0: np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
             1: np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
             2: np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
             3: np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
             4: np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
             5: np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
             6: np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
             7: np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
             8: np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
             9: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])}

    def transform_int(val: int):
        return guide[val]

    return np.array(list(map(transform_int, arr)))


def classify_age(row):
    age_guide = {(0, 9): '0 to 9', (10, 19): '10 to 19',
                 (20, 29): '20 to 29', (30, 39): '30 to 39',
                 (40, 49): '40 to 49', (50, 59): '50 to 59',
                 (60, 69): '60 to 69', (70, 79): '70 to 79',
                 (80, 89): '80 to 89'}

    for bracket, age_cls in age_guide.items():
        if (row >= bracket[0]) and (row <= bracket[1]):
            return age_cls


def normalize_cols(df, col_names=[]):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("please insert a dataframe")

    cols = list(df.columns)
    new_df = pd.DataFrame()
    if len(col_names):
        if not isinstance(col_names, (tuple, list)):
            raise TypeError("Please input a list of columns")
        for col in col_names:
            if col not in cols:
                raise ValueError(f"{col} not in DataFrame")
            if df[col].dtypes == 'object':
                raise TypeError(f"{col} does not contain numeric values.")

            new_df[f"Norm_{col}"] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    else:
        num_cols = list(df.select_dtypes(exclude='object').columns)
        for col in num_cols:
            new_df[f"Norm_{col}"] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return new_df


def null_checker(df: pd.DataFrame, in_perc: bool=False, only_nulls=False):
        """return quantity of missing data per dataframe column."""
        
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Argument you pass as df is not a pandas dataframe")
            
        null_df = df.isnull().sum()
            
        if in_perc:
            null_df = (null_df*100)/len(df)
            if len(null_df):
                print("Result given in percentage")
        
        if only_nulls:
            null_df = null_df.loc[null_df > 0]
            if len(null_df):
                print("Only columns with null values are shown below:")
            
        return np.round(null_df, 2)
        

def check_for_empty_str(df: sns.categorical.pd.DataFrame):
    """Return True for column containing '' or ' '.
    Output is a dictionary with column name as key and True/False as value."""

    if not isinstance(df, sns.categorical.pd.DataFrame):
        raise TypeError("Argument you pass as df is not a pandas dataframe")
        
    cols = list(df.columns)  # list of columns
    result = dict()
    for i in range(len(cols)):
        # True for columns having empty strings
        result[cols[i]] = df.loc[(df[cols[i]] == ' ') |
                                (df[cols[i]] == '')].shape[0] > 0
    return result


def round_up_num(decimal: str):
    """round up to nearest whole number if preceding number is 5 or greater."""

    whole, dec = str(float(decimal)).split('.')
    return str(int(whole) + 1) if int(dec[0]) >= 5 or (int(dec[0]) + 1 >= 5 and int(dec[1]) >= 5) else whole


def transform_val(ser: sns.categorical.pd.Series, guide: dict):
    """Change values in a series from one format to new format specified in the guide dictionary."""

    return ser.apply(lambda val: str(guide[val]) if val in guide.keys() else val)


def unique_categs(df: sns.categorical.pd.DataFrame):
    """return unique values per dataframe column."""

    cols = tuple(df.columns)
    uniq_vals = dict()
    for i in range(len(cols)):
        uniq_vals[cols[i]] = list(df[cols[i]].unique())
    return uniq_vals


def train_test_RMSE(deg_lim: int, X: 'array-like', y: 'array-like'):

    "decide the best polynomial order for transformation of features or predictors"
    "by calculating the root mean squared errors"
    "for both train and test sets of given dataset"
    "at incremental polynomial order of features"

    "return dicts: train_rmse, test_rmse"

    from sklearn.preprocessing import PolynomialFeatures

    train_rmse, test_rmse = {}, {}  # to store each rmse at different polynomial order

    for n in range(1, deg_lim):
        converter = PolynomialFeatures(degree=n, include_bias=False)  #  transform to n degrees
        X_polynomial = converter.fit_transform(X)  # X features has been transformed

        # calculate the separate RMSEs for transformed train and test sets of features
        from sklearn.model_selection import train_test_split
        # split polynomial features, y label into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_polynomial, y, test_size=0.3, random_state=101)

        from sklearn.linear_model import LinearRegression
        estimator = LinearRegression()  # instance of linear model to serve as estimator
        estimator.fit(X_train, y_train)  # teach the estimator about dataset perculiarities with train set

        train_pred = estimator.predict(X_train)  # produce label estimates based on input from polynomial features train set
        test_pred = estimator.predict(X_test)  # produce label estimates based on input from polynomial features test set

        from sklearn.metrics import mean_squared_error
        import numpy as np
        # calculate root mean squared error for each estimate/predictions
        tr_rmse, te_rmse = np.sqrt(mean_squared_error(y_train, train_pred)), np.sqrt(mean_squared_error(y_test, test_pred))

        # separately store rmse
        train_rmse.setdefault(n, tr_rmse), test_rmse.setdefault(n, te_rmse)

    return train_rmse, test_rmse


def ds_modules_importer():
    '''
    import the following data science modules:
    pandas as pd, numpy as np, seaborn as sns
    matplotlib.pyplot as plt
    :return pd, np, plt, sns, train_test_split, PolynomialFeatures, StandardScaler, SCORERS, mean_absolute_error, mean_squared_error
    '''

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.metrics import SCORERS, mean_absolute_error, mean_squared_error

    print('\n\nData Science Modules imported!\n'
          'numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns\n')
    return pd, np, plt, sns, train_test_split, PolynomialFeatures, StandardScaler, SCORERS, mean_absolute_error, mean_squared_error


def percentage_missing_data(df: 'DataFrame or Series', in_percentage: 'bool'=True, draw_graph: 'bool'=False):
    """
    Return missing data per column (in percentage, if True)
    input is a pandas dataframe or series
    output is the number of missing rows (in percentage, if specified)
    if draw_graph is false and in_percentage is false, return missing_data
    if draw_graph is false and in_percentage is true, return missing_data_perc
    if draw_graph is true and in_percentage is false, plot graph and return missing_data
    if draw_graph is true and in_percentage is false, plot graph and return missing_data_perc"""

    missing_data = df.isna().sum()[df.isna().sum() > 0].sort_values(ascending=False)
    missing_data_perc = np.round(missing_data/df.shape[0] * 100, 2)

    if not draw_graph and not in_percentage:  # do not plot graph and output number of missing rows
        print(f'Missing rows found in: {tuple(missing_data.index)}')
        return missing_data
    if not draw_graph and in_percentage:  # don't plot graph and output percent of missing rows
        print(f'Percentage number of missing rows for: {tuple(missing_data.index)}')
        return missing_data_perc

    #  plot graph to display info
    sns.set_theme(style='darkgrid')
    fig = plt.figure(figsize=(6, 3), dpi=150)
    ax = fig.add_axes(rect=[0, 0, 1, 1])
    row_lab = "Missing Rows"
    titl = "Missing Rows per Column"

    if not in_percentage:
        sns.barplot(y=missing_data.index, x=missing_data, ax=ax)

    else:
        row_lab = f"{row_lab} (in %)"
        titl = f"{titl} (in %)"
        sns.barplot(y=missing_data_perc.index, x=missing_data_perc, ax=ax)

    ax.set_ylabel("Feature"), ax.set_xlabel(row_lab)
    ax.set_title(titl)

    #  save graph as image
    graph_folder = f'{os.getcwd()}\\DisplayGraphs'
    joblib.os.makedirs(graph_folder, exist_ok=True)  # folder created
    graph_file = f'{graph_folder}\\plot1.png'
    fig.savefig(fname=graph_file, pad_inches=0.25,
                dpi=200, bbox_inches='tight')

    # display saved figure image
    from PIL import Image, ImageShow
    img = Image.open(graph_file)
    ImageShow.show(img)

    if not in_percentage:  # output number of missing rows
        print('Number of missing rows:')
        return missing_data
    print('Percentage number of missing rows:')
    return missing_data_perc  # output percent of missing rows


def cleanup_xy_suffixes(df: 'DataFrame', delete_suffix: 'str'):
    '''
    delete column with name containing the given suffix
    then get rid of _x or _y suffixes from name of remaining column

    :param df: dataframe having the unwanted suffixes,
    delete_suffix: the suffix alphabet that you wish to delete

    :return: df
    '''
    # select the columns having _y suffix
    y_cols = df.columns[df.columns.str.endswith('_y')]

    # select the columns having _x suffix
    x_cols = df.columns[df.columns.str.endswith('_x')]

    # select the suffix for deletion
    dropper = df.columns[df.columns.str.endswith(f"_{delete_suffix.strip('_').lower()}")]

    # drop all columns having the _y sufix
    df = df.drop(labels=dropper, axis=1)
    # remove _x from column names
    df.columns = df.columns.str.replace('_x', '')
    df.columns = df.columns.str.replace('_y', '')

    return df


def calc_age_from_dob(dob_col: 'Series'):
    '''
    calculate ages from date_of_birth values
    NOTE: all null values in the dob_col series have been removed
    in this result
    :param dob_col: a series of datetime type
    :return: age_yrs
    '''
    #  current date
    today_date = np.datetime64(datetime.date.today())
    print(today_date)
    #  exclude null date_of_birth values
    age_days = today_date - dob_col[dob_col.notnull()]
    age_yrs = np.int64(age_days.dt.days/365)
    print(age_yrs)

    return age_yrs
