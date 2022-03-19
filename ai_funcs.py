from seaborn.utils import np, pd, plt, os
import seaborn as sns
from sklearn import metrics as s_mtr, tree, ensemble, cluster, decomposition as sdec, feature_extraction as sfex
from tensorflow.keras import backend as K, models, layers, callbacks, preprocessing as k_prep
import nltk
from  nltk import word_tokenize, pos_tag, corpus, stem
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
nltk.download('wordnet')


def text_cleaner(text_body: str, remove_stop_words=False):
    """remove punctuation or [and stop words] from text_body."""
    
    import string
    
    nopunc = [char.lower() for char in text_body if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    if remove_stop_words:
        return [word for word in nopunc.split() if word.lower() not in sfex.text.ENGLISH_STOP_WORDS]
    return nopunc
    
    
def update_vocab(old_vocab: dict, new_sentences: list):
    """update a BOW vocabulary with new terms from new_sentences
    old_vocab: old bow vocabulary
    new_sentences: list of new sentences (str)
    Return: updated_dict"""
    
    def check_keys(old_vocab, token):
        return str.lower(token) in old_vocab.keys()
    
    cvect = sfex.text.CountVectorizer(ngram_range=(1, 3)).fit(new_sentences)
    new_vocab = cvect.vocabulary_
    vocab = dict(old_vocab)
    last_count = max(old_vocab.values())
    for w in new_vocab.keys():
        if check_keys(old_vocab, w):
            continue
        last_count += 1
        vocab[w] = last_count
    return vocab
    
    

def get_part_of_speech(sentence: str):
    """get root words of words in a sentence.
    Returns
    tagged_words: list of tuples of word, part of speech pairs"""
    
    # split up sentences into components
    sentence_components = word_tokenize(sentence)
    # get part of speech for each word
    tagged_words = pos_tag(sentence_components, tagset='universal')
    return tagged_words

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
    

def train_PCA(num_components, df):
    """Reduce the dimension a dataframe using a PCA model
    Return 
    (compressed_df, PCA_comp)"""
    
    cc = {i: f'Component_{i+1}' for i in range(num_components)}
    pca = sdec.PCA(num_components, random_state=1)
    df_pca = pd.DataFrame(np.round(pca.fit_transform(df), 2), columns=cc.values())
    
    pca_comp = pd.DataFrame(np.round(pca.components_, 2), columns=pca.feature_names_in_)
    expl_var =  pd.DataFrame({'explained_variance_perc': np.round(pca.explained_variance_ratio_, 2)})
    pca_comp = pd.concat([pca_comp, expl_var], axis=1)
    
    print(f"These {num_components} components explains {np.round(100*pca_comp['explained_variance_perc'].sum(), 2)}% of data variance")
    
    return df_pca, pca_comp
    

def train_kmeans(k, df):
    """develop  centroids and labels from KMeans model
    Returns
    (centroids_df, labes)"""
    
    kmeans = cluster.KMeans(n_clusters=k)
    kmeans.fit(df)
    
    centroids = pd.DataFrame(np.round(kmeans.cluster_centers_, 2), columns=kmeans.feature_names_in_)
    
    guide = {i: f'C{i+1}' for i in range(len(np.unique(kmeans.labels_)))}
    labes = np.vectorize(lambda x: guide[x])(kmeans.labels_)
    
    return centroids, labes
    
    
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
    
    
def build_nn(inp_shape, 
                olayer_units=1,
                olayer_activation='sigmoid', 
                hlayer_activation='relu', 
                optimizer='adam',
                loss='binary_crossentropy',
                metrics='accuracy'):
    """build an ANN"""
    hl_units = np.array(inp_shape).flatten()[0] + 16
    backend.clear_session()

    model = models.Sequential()    
    model.add(layers.Dense(hl_units, activation=hlayer_activation, input_shape=inp_shape))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(hl_units, activation=hlayer_activation))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(hl_units, activation=hlayer_activation))

    model.add(layers.Dense(olayer_units, activation=olayer_activation))
    
    model.compile(optimizer=optimizer, 
              loss=loss, 
              metrics=metrics)
    
    model.summary()
    return model