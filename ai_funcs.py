import random
import numpy as np
import pickle
import pandas as pd
from sklearn import metrics as s_mtr, tree, ensemble, cluster, decomposition as s_dec
from sklearn import utils as s_utils, feature_extraction as s_fex
from tensorflow.keras import backend as K, models, layers, callbacks, preprocessing as k_prep
import nltk
from  nltk import word_tokenize, pos_tag, corpus, stem
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
nltk.download('wordnet')

from ds_funcs import DsUtils


class AiUtils:

    @staticmethod
    def get_coeff_importance(trained_model: 'dataframe', plot_title=None):
        """determine feature importance from coefficient values (beta) of predictors
        based on contribution to model prediction.
        trained_model: instance of trained model having coefficients
        Return
        feature_importance: series ranking of features"""
        
        cls = AiUtils()
        
        weights = pd.DataFrame(trained_model.coef_, columns=trained_model.feature_names_in_)
        weights.loc[:, 'intercept'] = trained_model.intercept_
        print('\nModel Coefficients per Target Outcome:')
        display(weights)

        feat_importance = weights.drop('intercept', axis=1).apply(lambda row: np.e**row).round(decimals=2).mean().sort_values(ascending=False)
    #     display(feat_importance)
        if not plot_title:
            plot_title = 'Feature Contribution to Model Prediction'
        cls.plot_column(feat_importance.index, feat_importance.round(2),
                        plot_title=plot_title,
                        rotate_xticklabe=90, color='brown', reduce_barw_by=1.5)
        return feat_importance

    @staticmethod
    def reduce_train_test_dims(xtrain, xtest, pca_pct=0.95):
        """reduced an already split and rescaled dataset train and test dataset
        pca_pct: percentage of explained variance for pca
        Return
        reduced_xtrain, reduced_xtest"""
        
        cls = AiUtils()
        # engineer a row indicator to distinguish between train and test data
        xtrain.loc[:, 'train_set'] = 1
        xtest.loc[:, 'train_set'] = 0

        # join train set to test set before PCA
        train_test_rows = pd.concat([xtrain, xtest], ignore_index=True)
        # record test data index
        test_ix = train_test_rows.loc[train_test_rows['train_set'] == 0].index
        # reduce train_test data
        xreduced, pca_comps = cls.train_PCA(train_test_rows, perc_components=pca_pct)
        # select train and test set data rows
        reduced_xtrain = xreduced.loc[~xreduced.index.isin(test_ix)].reset_index(drop=True)
        reduced_xtest = xreduced.loc[test_ix].reset_index(drop=True)
        
        return reduced_xtrain, reduced_xtest, pca_comps

    @staticmethod
    def outlier_detection_kmeans(df, n_clusters=3, top_n=10):
        """detect outliers using KMeans"""
        
        X = copy.deepcopy(df)
        kmeans = cluster.KMeans(n_clusters=n_clusters)
        kmeans.fit(X)
        
        labels = kmeans.predict(X)
        centroids = kmeans.cluster_centers_
        
        # determine the distance from each data point in df to the closest cluster centroid:
        distance = [(min(distance), i) for i,distance in enumerate(kmeans.transform(X))]
        
        # determine the top_n most distant outliers:
        indices_of_outliers = [row[1] for row in sorted(distance, key=lambda row: row[0], reverse=True)][:top_n]
        return X.loc[indices_of_outliers]
        
    @staticmethod
    def outlier_detection_iqr(arr, n_iqr=1.5, precision=2):
        """Detect Outliers in arr (1-D array) based on n_iqr * IQR.
        Where IQR (Inter-quartile range) is the middle 50% of sorted 
        IQR of arr = Q3 (75th percentile) - Q1 (25th percentile)
        Outliers are data points below Q1 - (n_iqr * IQR )
        or above Q3 + (n_iqr * IQR)
        ie., where arr < Q1 - (n_iqr*IQR) or arr > Q3 + (n_iqr*IQR)"""
        
        def detect_outliers(num, lower, upper):
            if (num < lower) or (num > upper):
                return num
            
        x = np.sort(arr)
        q1, q3 = np.percentile(x, q=[25, 75], interpolation='midpoint')
        iqr = q3 - q1
        coef = n_iqr*iqr

        llim, ulim = np.round(q1 - coef, precision), np.round(q3 + coef, precision)
        outl_ser = pd.Series(x).apply(lambda n: detect_outliers(n, llim, ulim))
        outliers = outl_ser.loc[outl_ser.notnull()]
        return {'lower_limit':llim, 'upper_limit':ulim, 'n_outliers':len(outliers), 'outliers':outliers}

    @staticmethod
    def extract_sentence(text):
        return text.split('\n')

    @staticmethod
    def extract_nwords(text, topic_word, n_words=1):
        
        words_list = text.split()
        matches = []
        for i in range(len(words_list)):
            if words_list[i].lower() == topic_word.lower():
                start = i - n_words
                stop = i + n_words
                if start < 0:
                    start = 0
                extract = words_list[start:stop+1]
                matches.append(' '.join(extract))
        return matches if len(matches) > 1 else matches[0] if len(matches) == 1 else None

    @staticmethod
    def extract_nsentences(text, topic_word, n_sentence=1):
        sentences_list = text.split('\n')
        matches = []
        for i in range(len(sentences_list)):
            if topic_word.lower() in sentences_list[i].lower():
                topic_sentence = sentences_list[i]
                for i in range(len(sentences_list)):
                    if topic_sentence.lower() == sentences_list[i].lower():
                        start = i - n_sentence
                        stop = i + n_sentence
                        if start < 0:
                            start = 0
                        extract = sentences_list[start: stop+1]
                        matches.append(' '.join(extract))
        return matches if len(matches) > 1 else matches[0] if len(matches) == 1 else None

    @staticmethod
    def extract_each_kw_from_review(review, kwords, n_sentences=1):
        kw_list = kwords.split()
        sentence_list = review.split('\n')
        for kw in kw_list:
            pass
    
    @staticmethod
    def get_maximum_len(train_encoded, test_encoded):
        return  max([max(list(map(len, train_encoded))), max(list(map(len, test_encoded)))])

    @staticmethod
    def convert_to_integer(data, vocab_to_int):
        all_sentences = []
        for sentence in data:
    #         print(sentence)
            all_sentences.append([vocab_to_int[word] if word in vocab_to_int else vocab_to_int["<UNK>"] for word in sentence.split()])
        return all_sentences
        
    @staticmethod
    def convert_to_vocab(encoded_data, int_to_vocab):
        all_sentences = []
        for sentence in encoded_data:
            all_sentences.append([int_to_vocab[word_int] for word_int in sentence])
        return all_sentences

    @staticmethod
    def create_lookup(text_list: list):
        """create a lookup dictionary.
        Return:
        (vocab_to_integer, integer_to_vocab)"""
        
        vocab = set(text_list)
        vocab_to_integer = {word:i for i, word in enumerate(vocab)}
        integer_to_vocab = {v:k for k, v in vocab_to_integer.items()}
        return vocab_to_integer, integer_to_vocab

    @staticmethod
    def text_cleaner(text_body: str, remove_stop_words=False, added_stopwords=None):
        """remove punctuation or [and stop words] from text_body.
        Return:
        sentence: str"""
        
        import string
        
        original_stwords = set(s_fex.text.ENGLISH_STOP_WORDS)
        stwrds = [str.lower(wrd) for wrd in set(added_stopwords)]
        
        nopunc = [char.lower() for char in str(text_body) if char not in string.punctuation]
        nopunc_sentence = ''.join(nopunc)
        if remove_stop_words:
            return ' '.join([word for word in str(nopunc_sentence).split() if word.lower() not in stwrds])
        return nopunc_sentence
    
    @staticmethod
    def remove_puncs_lcase(sentence):
        sw1 = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        return ''.join([str.lower(char) for char in sentence if char not in sw1])
    
    @staticmethod
    def build_gru():
        """input_dim: total number of unique words (int)
        output_dim: shape of the embedding output vector 
        input_length: fixed length of padded input sequence (list of ints)"""
        
        K.clear_session()

        model = models.Sequential()
        model.add(layers.Embedding(input_dim=vocab_size, output_dim=embedding_vector_len, input_length=max_sentence_len))
        model.add(layers.GRU(256, dropout=dropout, recurrent_dropout=dropout, return_sequences=True))
        model.add(layers.GRU(256, dropout=dropout, recurrent_dropout=0.0))

        model.add(layers.Dense(len(labels), activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
        print(model.summary())
        return model

    @staticmethod
    def save_python_obj(fname, py_obj):
        """save python object to file system.
        NOTE: always add .pkl to fname"""
        
        with open(fname, 'wb') as f:
            pickle.dump(py_obj, f)
        print("Python object has been saved")
        
    @staticmethod
    def load_python_obj(fname):
        """load python object from filesystem
        NOTE: always include filename extension"""
        
        with open(fname, 'rb') as f:
            obj = pickle.load(f)
        print('Loading complete')
        return obj
            
    @staticmethod
    def remove_entry_where(df, where_val_is=-1):
        """drop entries containing specified val
        Return:
        df: dataframe without entries containing the specified values"""
        
        col_names = df.columns
        df2 = copy.deepcopy(df)
        del_inds = []
        
        for col in col_names:
            del_inds.extend(df2.loc[df2[col] == where_val_is].index)
        return df2.loc[~df2.index.isin(del_inds)]
        
    @staticmethod
    def split_sequence(X, y=None, test_size=0.2):
        """Split sequentially.
        Return
        (X_train, X_test, y_train, y_test)"""
        
        if (not isinstance(X, (pd.Series, pd.DataFrame, np.ndarray))):
            raise TypeError("X must be an array/dataframe/series")
            
        if (y is not None) and not isinstance(y, (pd.Series, pd.DataFrame, np.ndarray)):
            raise TypeError("y must be an array/dataframe/series")
            
        train_size = 1 - test_size
        train_ind = int(round(len(X)*train_size))
        
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X_train = X.iloc[: train_ind]
            X_test = X.iloc[train_ind:]
            
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y_train = y.iloc[: train_ind]
            y_test = y.iloc[train_ind:]
            
        if isinstance(X, np.ndarray):
            X_train = X[:train_ind]
            X_test = X[train_ind:]
            
        if isinstance(y, np.ndarray):
            y_train = y[:train_ind]
            y_test = y[train_ind:]
        
        if y is not None:    
            return (X_train, X_test, y_train, y_test)
        return (X_train, X_test)
        
    @staticmethod
    def basic_regressor_score(X_train, y_train, X_val, y_val, X_test, y_test):
        """determine rmse of a trained
        basic linear regression model
        Return:
        trained_nnet, history"""
        
        def build_nnet(n_feats):
            K.clear_session()
            est = models.Sequential()
            est.add(layers.Dense(n_feats*4, input_shape=[n_feats], activation='relu'))
            est.add(layers.Dropout(0.2))
            est.add(layers.Dense(n_feats*2, activation='relu'))
            est.add(layers.Dropout(0.5))
            est.add(layers.Dense(2, activation='relu'))
            est.add(layers.Dense(1))
            
            est.compile(optimizer='adam',
                       loss='mean_squared_error')
            return est
        
        scaler = s_prep.MinMaxScaler()
        scaler.fit(X_train)
        sc_Xtrain = scaler.transform(X_train)
        sc_Xval = scaler.transform(X_val)
        sc_Xtest = scaler.transform(X_test)
        
        es = callbacks.EarlyStopping(monitor='val_loss', patience=3)
        nnet = build_nnet(X_train.shape[1])
        history = nnet.fit(sc_Xtrain, y_train, epochs=50,
                           validation_data=[sc_Xval, y_val],)

        preds = nnet.predict(sc_Xtest)
        print('Root Mean Squared Error\n'+
              f'{s_mtr.mean_squared_error(y_test, preds, squared=False)}')
        
        return nnet, pd.DataFrame(history)
    
    @staticmethod
    def test_stat(sample_mean, pop_mean, sample_std, sample_size):
        return round(((sample_mean - pop_mean)/sample_std) * sample_size**0.5, 4)
        
    @staticmethod
    def compute_mse(y_true, y_pred, root=False):
        """compute the mean squared error or root mse if root is True
        Return
        """
        
        if len(y_true) != len(y_pred):
            raise ValueError('mismatch between length of y_true and y_pred')
            
        error = (np.array(y_pred) - np.array(y_true))
        sumsq_err = np.sum(error**2)
        mse = sumsq_err/len(y_pred)
        if root:
            return np.sqrt(mse)
        
        return mse
    
    @staticmethod
    def compute_balanced_weights(y: 'array', as_samp_weights=True):
        """compute balanced sample weights for unbalanced classes.
        idea is from sklearn:
        balanced_weight = total_samples / (no of classes * count_per_class)
        WHERE:
        total_samples = len(y)
        no of classes = len(np.unique(y))
        no of samples per class = pd.Series(y).value_counts().sort_index().values
        unique_weights = no of classes * no of samples per class
        samp_weights = {labe:weight for labe, weight in zip(np.unique(y), unique_weights)}
        
        weight_per_labe = np.vectorize(lambda l, weight_dict: weight_dict[l])(y, samp_weights)
        Return:
        weight_per_labe: if samp_weights is True
        class_weights: if samp_weights is False"""
        
        y = np.array(y)
        n_samples = len(y)
        n_classes = len(np.unique(y))
        samples_per_class = pd.Series(y).value_counts().sort_index(ascending=True).values
        denom = samples_per_class * n_classes
        unique_weights = n_samples/denom
        cls_weights = {l:w for l, w in zip(np.unique(y), unique_weights)}
        
        if as_samp_weights:
            return np.vectorize(lambda l, weight_dict: weight_dict[l])(y, cls_weights)
        return cls_weights

    @staticmethod
    def compute_accuracy(y_true, y_pred, verbose=False):
        """Percentage of correctly classified samples.
        Ratio of (True Positives + True Negatives) to Total Number of Samples.
        Args:
        TP (True Positive) = Correctly Predicted class 1 labels
        TN (True Negative) = Correctly Predicted class 0 labels
        FP (False Positive) = Wrongly Predicted Class 1 labels
        FN (False Negative) = Wrongly Predicted Class 0 labels
        Correct = TP + TN
        Wrong = FP + FN
        Total Samples = Correct + Wrong
        Return:
        Correct/Total Samples"""
        
        n_true, n_pred = len(y_true), len(y_pred)
        if n_pred != n_true:
            raise ValueError('y_true must contain the same number as y_pred elements')
            
        # TP is when predicted is 1/positive, and actual is equal to predicted
        tp = sum([1 if (p == 1) and (p == t) else 0 for t, p in zip(y_true, y_pred)])
        
        # TN is when predicted is 0/negative, and actual is equal to predicted
        tn = sum([1 if (p == 0) and (p == t) else 0 for t, p in zip(y_true, y_pred)])
        
        # FP is when predicted is 1/positive, and actual is not equal to predicted
        fp = sum([1 if (p == 1) and (p != t) else 0 for t, p in zip(y_true, y_pred)])
        
        # FN is when predicted is 0/negative, and actual is not equal to predicted
        fn = sum([1 if (p == 0) and (p != t) else 0 for t, p in zip(y_true, y_pred)])
        
        # correct is TP + TN
        correct = sum([tp, tn])
        # wrong is FP + FN
        wrong = sum([fp, fn])
        
        # total = correct + wrong
        total = sum([correct, wrong])
        
        if verbose:
            print(f'\n\nTrue Positives = {tp}\n' +
                  f'True Negatives = {tn}\n' +
                  f'False Positives = {fp}\n' +
                  f'False Negatives = {fn}\n' +
                 f'Correctly classified = {correct}\n' +
                 f'Wrongly classified = {wrong}\n\n')
        
        return round(correct/total, 4)

    @staticmethod
    def compute_recall(y_true, y_pred, focus_on_label=1, verbose=False):
        """Ability not to classify as positive a sample that is negative.
        Here, the focus is on the proportion of FALSE NEGATIVES.
        Recall score < 50% = more false negatives than true positives
        And Recall score > 50% = more true positives than false negatives.
        Ratio of True Positives to Total Number of Actual Positive labels
        TP (True Positive) = Correctly Predicted class 1 labels
        TN (True Negative) = Correctly Predicted class 0 labels
        FP (False Positive) = Wrongly Predicted Class 1 labels
        FN (False Negative) = Wrongly Predicted Class 0 labels
        Actual Positives = TP + FN
        Args:
        focus_on_label is class to be used as positive
        Return:
        TP/Actual Positives"""
        
        n_true, n_pred = len(y_true), len(y_pred)
        if n_pred != n_true:
            raise ValueError('y_true must contain the same number as y_pred elements')
            
        # TP is when predicted is 1/positive, and actual is equal to predicted
        tp = sum([1 if (p == focus_on_label) and (p == t) else 0 for t, p in zip(y_true, y_pred)])
        
        # FN is when predicted is 0/negative, and actual is not equal to predicted
        fn = sum([1 if (p != focus_on_label) and (p != t) else 0 for t, p in zip(y_true, y_pred)])
        
        # actual_positives is TP + FN
        actual_positives = sum([tp, fn])
        
        if verbose:
            print(f'\n\nWith {focus_on_label} as the Positive Label,\n\n' +
                  f'True Positives = {tp}\n' +
                  f'False Negatives = {fn}\n\n')
        
        return round(tp/actual_positives, 4)

    @staticmethod
    def compute_precision(y_true, y_pred, focus_on_label=1, verbose=False):
        """Ability to find all the positive samples.
        Here, the focus is on the proportion of FALSE POSITIVES.
        A precision score below 50% means more false positives than true positives.
        And precision score > 50% = more true positives than false positives
        Ratio of True Positives to Total Number of Predicted Positive labels
        TP (True Positive) = Correctly Predicted class 1 labels
        TN (True Negative) = Correctly Predicted class 0 labels
        FP (False Positive) = Wrongly Predicted Class 1 labels
        FN (False Negative) = Wrongly Predicted Class 0 labels
        Predicted Positives = TP + FP
        Args:
        focus_on_label is class to be used as positive
        Return:
        TP/Predicted Positives"""
        
        n_true, n_pred = len(y_true), len(y_pred)
        if n_pred != n_true:
            raise ValueError('y_true must contain the same number as y_pred elements')
            
        # TP is when predicted is 1/positive, and actual is equal to predicted
        tp = sum([1 if (p == focus_on_label) and (p == t) else 0 for t, p in zip(y_true, y_pred)])
        
        # FP is when predicted is 1/positive, and actual is not equal to predicted
        fp = sum([1 if (p == focus_on_label) and (p != t) else 0 for t, p in zip(y_true, y_pred)])
        
        # pred_positives is TP + FP
        pred_positives = sum([tp, fp])
        
        if verbose:
            print(f'\n\nWith {focus_on_label} as the Positive Label,\n\n' +
                  f'True Positives = {tp}\n' +
                  f'False Positives = {fp}\n\n')
        
        return round(tp/pred_positives, 4)

    @staticmethod
    def compute_f1_score(y_true, y_pred, focus_on_label=1, verbose=False):
        """Harmonic Average of Recall and Precision scores. Note that this is
        different from the arithmetic mean.
        Two times the product of recall and precision scores,
        divided by sum of recall and precision scores
        Args:
        focus_on_label is class to be used as positive
        Return:
        2*(recall_score*precision_score)/(recall_score + precision_score)"""
        
        cls = AiUtils()
        
        n_true, n_pred = len(y_true), len(y_pred)
        if n_pred != n_true:
            raise ValueError('y_true must contain the same number as y_pred elements')
            
        recall_score = cls.compute_recall(y_true, y_pred, focus_on_label)
        precision_score = cls.compute_precision(y_true, y_pred, focus_on_label)
        f1_score = 2*((recall_score*precision_score)/(recall_score + precision_score))
        
        if verbose:
            print(f'\n\nWith {focus_on_label} as the Positive Label,\n\n' +
                  f'Recall score = {recall_score}\n' +
                  f'Precision score = {precision_score}\n\n')
        
        return round(f1_score, 4)
        
    @staticmethod
    def random_upsampling(imb_df: pd.DataFrame, target_col: str):
        """oversample the minority class by creating random copies.
        1. compute difference between both classes.
        2. get minority class name
        3. generate new instances of minority labels 
        just enough to balance the dataset.
        4. use indexes of existing minority entries to create
        copies of corresponding features.
        Return
        new_df: containing additional entries of minority labels and features"""
        
        if not isinstance(imb_df, (pd.DataFrame)):
            raise TypeError("imbalanced_y must be a dataframe or series")

        X = imb_df.drop(target_col, axis=1)
        X_features = X.columns
        imbalanced_y = imb_df[target_col]
        
        label_freq = imbalanced_y.value_counts()
        unq_vals = tuple(imbalanced_y.unique())

        # difference between number of entries for majority and minority labels
        label_diff = label_freq.max() - label_freq.min()
        
        # minority class name
        min_label = label_freq.loc[label_freq == label_freq.min()].index[0]
        # generate balancing number of minority labels
        generated_min_labes = pd.Series([min_label]*label_diff,
                                       name=target_col)
        
        # min_label indexes
        min_labe_inds = imbalanced_y.loc[imbalanced_y == min_label].index
        
        rand_new_X = pd.DataFrame(np.hstack([X.loc[X.index == random.choice(min_labe_inds)].values[0]
                                             for i in range(label_diff)]),
                                  columns=X_features)
        
        copies = pd.concat([rand_new_X, generated_min_labes], axis=1)
        return pd.concat([imb_df, copies]).reset_index(drop=True)
        
    @staticmethod
    def upsample_imbalanced_data(X: pd.DataFrame, y: pd.Series, random_state=1):
        """oversample minority class instances.
        Return:
        X_upsampled, y_upsampled."""
        
        ds = DsUtils()
        from imblearn import over_sampling
        
        catg_cols = ds.get_features_with_dtypes(X, str)
        num_cols = ds.get_features_with_dtypes(X)
        
        if len(catg_cols):  # presence of categorical features
            # get column indexes for categorical columns
            col_ind = {v:i for i, v in list(enumerate(X.columns.astype(str)))}

            catg_ind = [col_ind[col] for col in catg_cols]
            if len(catg_ind) == 1:
                catg_ind = catg_ind[0]
            #print(catg_ind)
        
            if len(num_cols):
                # numeric and categorical upsampler
                over_sampler = over_sampling.SMOTENC(categorical_features=(catg_ind,), random_state=1)
            else:
                over_sampler = over_sampling.SMOTEN(random_state=random_state)
        else:
            over_sampler = over_sampling.SMOTE(random_state=random_state)
            
        return over_sampler.fit_resample(X, y)

    @staticmethod
    def downsample_imbalanced_data(y, with_replacement=False):
        """randomly select equal number of entries from labels that
        have more than the minimal count.
        Return: 
        index_list: sorted list of indexes containing equal instances of classes"""
        
        if not isinstance(y, (pd.Series, pd.DataFrame,)):
            raise TypeError("y must be a dataframe or series")
            
        label_freq = y.value_counts()
        unq_vals = tuple(y.unique())
    #     print(unq_vals)
        min_count = label_freq.min()
    #     print(min_label)
        # get the indexes of each label
        label_indexes = {labe:tuple(y.loc[y == labe].index) for labe in unq_vals}
        result = {labe: random.sample(inds, k=min_count) if len(inds) != min_count else inds 
                  for labe, inds in label_indexes.items()}
        
        collated_indexes = []
        for k, v in result.items():
            collated_indexes.extend(v)
        return sorted(collated_indexes)
        
    @staticmethod
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
        
    @staticmethod
    def update_vocab(old_vocab: dict, new_sentences: list):
        """update a BOW vocabulary with new terms from new_sentences
        old_vocab: old bow vocabulary
        new_sentences: list of new sentences (str)
        Return: updated_dict"""
        
        def check_keys(old_vocab, token):
            return str.lower(token) in old_vocab.keys()
        
        cvect = s_fex.text.CountVectorizer(ngram_range=(1, 3)).fit(new_sentences)
        new_vocab = cvect.vocabulary_
        vocab = dict(old_vocab)
        last_count = max(old_vocab.values())
        for w in new_vocab.keys():
            if check_keys(old_vocab, w):
                continue
            last_count += 1
            vocab[w] = last_count
        return vocab
        
    @staticmethod
    def get_part_of_speech(sentence: str):
        """get root words of words in a sentence.
        Returns
        tagged_words: list of tuples of word, part of speech pairs"""
        
        # split up sentences into components
        sentence_components = word_tokenize(sentence)
        # get part of speech for each word
        tagged_words = pos_tag(sentence_components, tagset='universal')
        return tagged_words

    @staticmethod
    def split_sequence(X, y=None, test_size=0.2):
        """Split sequentially.
        Return
        (X_train, X_test, y_train, y_test)"""
        
        if (not isinstance(X, (pd.Series, pd.DataFrame, np.ndarray))):
            raise TypeError("X must be an array/dataframe/series")
            
        if (y is not None) and not isinstance(y, (pd.Series, pd.DataFrame, np.ndarray)):
            raise TypeError("y must be an array/dataframe/series")
            
        train_size = 1 - test_size
        train_ind = int(round(len(X)*train_size))
        
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X_train = X.iloc[: train_ind]
            X_test = X.iloc[train_ind:]
            
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y_train = y.iloc[: train_ind]
            y_test = y.iloc[train_ind:]
            
        if isinstance(X, np.ndarray):
            X_train = X[:train_ind]
            X_test = X[train_ind:]
            
        if isinstance(y, np.ndarray):
            y_train = y[:train_ind]
            y_test = y[train_ind:]
        
        if y is not None:    
            return (X_train, X_test, y_train, y_test)
        return (X_train, X_test)

    @staticmethod
    def tf_sparsity_density(term_frequency: 'sparse matrix'):
        """ Density = nonzero occurrences/total count
        Sparsity = zero occurrences/total count"""
        
        total = term_frequency.shape[0] * term_frequency.shape[1]
        sparse_count = total - term_frequency.getnnz()
        dense_count = term_frequency.getnnz()
        print(f'Sparse count: {sparse_count}')
        print(f'Dense count: {dense_count}')
        print(f'Sparsity: {np.round((total - term_frequency.getnnz())/total, 4)}')
        print(f'Density: {np.round(term_frequency.getnnz()/total, 4)}')
        
    @staticmethod
    def train_PCA(df, perc_components=0.95):
        """Reduce the dimension a dataframe using a PCA model.
        principal components explain a proportion of the variance
        of the given input
        perc_components: fraction of the df variance you want explained 
        by the output
        
        NOTE: Rescale the input to between (0, 1) BEFORE 
        using PCA
        
        Return 
        (compressed_df, PCA_comps)"""
        
        if not isinstance(perc_components, float):
            raise ValueError('please input the percentage of the input variance you want explained by the components')
        
        if (perc_components > 1.0) or (perc_components < 0):
            raise ValueError('please input a percentage of within range (0, 1)')
        
        np.random.seed(1)
        pca = s_dec.PCA(perc_components, random_state=1)
        df_pca = pd.DataFrame(np.round(pca.fit_transform(df), 2))
        cc = {i: f'Component_{i+1}' for i in range(len(pca.components_))}
        df_pca.columns = cc.values()
        
        pca_comp = pd.DataFrame(np.round(pca.components_, 2), columns=pca.feature_names_in_)
        expl_var =  pd.DataFrame({'explained_variance_perc': np.round(pca.explained_variance_ratio_, 2)})
        pca_comp = pd.concat([pca_comp, expl_var], axis=1)
        
        print(f"These {len(pca.components_)} components explains {np.round(100*pca_comp['explained_variance_perc'].sum(), 2)}% of input variance\n")
        print(f'The variance of each component is given as:\n{100*np.cumsum(pca.explained_variance_ratio_).round(2)}')
        
        return df_pca, pca_comp
        
    @staticmethod
    def train_kmeans(k, df):
        """develop  centroids and labels from KMeans model
        Returns
        kmeans, centroids, labes"""
        
        kmeans = cluster.KMeans(n_clusters=k)
        kmeans.fit(df)
        
        centroids = pd.DataFrame(np.round(kmeans.cluster_centers_, 2), columns=kmeans.feature_names_in_)
        
        guide = {i: f'C{i+1}' for i in range(len(np.unique(kmeans.labels_)))}
        labes = np.vectorize(lambda x: guide[x])(kmeans.labels_)
        
        return kmeans, centroids, labes
        
    @staticmethod
    def show_layer_shapes(nn_model):
        for i in range(len(nn_model.layers)):

            print(f'Layer {i}: \nInput_shape: {nn_model.layers[i].input_shape}' +
                 f'\nOutput shape: {nn_model.layers[i].output_shape}\n\n')
        
    @staticmethod
    def build_autoencoder(n_features, encoder_dim):
        """build a standard autoencoder network, where
        input and output layers have same number
        of units.
        
        Returns:
        autoencoder, encoder, decoder
        
        output dimension is defined by number of units,
        and input dimension by n_features arg.
        encoder layer is the first hidden layer.
        decoder layer is the output layer.
        autoencoder is simply the stacking of both
        encoder and decoder layers."""
        
        K.clear_session()
        
        encoder = models.Sequential(name='Encoder')
        encoder.add(layers.Dense(units=encoder_dim, input_shape=[n_features], activation='relu'))
        
        decoder = models.Sequential(name='Decoder')
        decoder.add(layers.Dense(units=n_features, input_shape=[encoder_dim], activation='relu'))
        
        autoencoder = models.Sequential([encoder, decoder])
        
        autoencoder.compile(optimizer=optimizers.SGD(learning_rate=1.5),
                           loss='binary_crossentropy')
        
        autoencoder.summary()
        
        return autoencoder, encoder, decoder
        
    @staticmethod
    def build_nn(inp_shape, 
                    olayer_units=1,
                    olayer_activation='sigmoid', 
                    hlayer_activation='relu', 
                    optimizer='adam',
                    loss='binary_crossentropy',
                    metrics='accuracy'):
        """build an ANN, use default lr=1.8e-4"""
        
        if optimizer == 'adam':
            optmizer = optimizers.Adam(learning_rate=lr)
        
        hl_units = np.array(inp_shape).flatten()[0] + 16
        backend.clear_session()

        model = models.Sequential()    
        model.add(layers.Dense(hl_units, activation=hlayer_activation, input_shape=inp_shape))
        
        model.add(layers.Dense(hl_units, activation=hlayer_activation))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(hl_units, activation=hlayer_activation))
        
        model.add(layers.Dense(olayer_units, activation=olayer_activation))
        
        model.compile(optimizer=optimizer, 
                  loss=loss, 
                  metrics=metrics)
        
        model.summary()
        return model