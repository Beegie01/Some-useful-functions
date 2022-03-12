from seaborn.utils import np, pd, plt, os
import seaborn as sns


def plot_hist(x=None, y=None, condition_on=None, title="A Histogram", bins=None, interval_per_bin: int=None, bin_range=None, color=None, layer_type='default', axis=None):
    """A histogram plot on an axis.
    layer_type: {"layer", "dodge", "stack", "fill"}
    Return 
    axis"""
    
    if (not isinstance(x, (pd.Series, np.ndarray)) and x is not None):
        raise TypeError("x must be a pandas series or numpy array")
    elif (not isinstance(y, (pd.Series, np.ndarray)) and y is not None):
        raise TypeError("y must be a pandas series or numpy array")
        
    if bins is None:
        bins = 'auto'
    if str.lower(layer_type) == 'default':
        layer_type = 'layer'
    if axis is None:
        axis = sns.histplot(x=x, y=y, hue=condition_on, bins=bins, binrange=bin_range, 
                            binwidth=interval_per_bin, color=color, multiple=layer_type)
    else:
        sns.histplot(x=x, y=y, hue=condition_on, bins=bins, binrange=bin_range,
                     binwidth=interval_per_bin, color=color, multiple=layer_type, 
                     ax=axis)
    axis.set_title(title, x=0.5, y=1.025)
    
    return axis


def plot_box(x=None, y=None, condition_on=None, title="A Boxplot", orientation='horizontal', axis=None):
    """A box distribution plot on an axis.
    Return 
    axis"""
    if (not isinstance(x, (pd.Series, np.ndarray)) and x is not None):
        raise TypeError("x must be a pandas series or numpy array")
    elif (not isinstance(y, (pd.Series, np.ndarray)) and y is not None):
        raise TypeError("y must be a pandas series or numpy array")
        
    if axis is None:
        axis = sns.boxplot(x=x, y=y, hue=condition_on, orient=orientation)
    else:
        sns.boxplot(x=x, y=y, hue=condition_on, orient=orientation, ax=axis)
    axis.set_title(title, x=0.5, y=1.025)
    
    return axis
    

def plot_line(x, y, condition_on=None, marker=None, color=None, axis=None):
    """plot line graph on an axis.
    Return
    axis """
    
    if axis is None:
        return sns.lineplot(x=x, y=y, hue=condition_on, marker=marker, palette='viridis', color=color)
    
    return sns.lineplot(x=x, y=y, hue=condition_on, marker=marker,
                          ax=axis, palette='viridis', color=color)
                          
def plot_scatter(x, y, condition_on=None, marker=None, color=None, axis=None):
    """plot scatter graph on an axis.
    Return
    axis """
    
    if axis is None:
        return sns.scatterplot(x=x, y=y, hue=condition_on, marker=marker, palette='viridis', color=color)
    
    return sns.scatterplot(x=x, y=y, hue=condition_on, marker=marker,
                          ax=axis, palette='viridis', color=color)
                          
def plot_bar(x, y, condition_on=None, color=None, axis=None):
    """plot bar graph on an axis.
    Return
    axis """
    
    if axis is None:
        return sns.barplot(x=x, y=y, hue=condition_on, palette='viridis', color=color)
    
    return sns.barplot(x=x, y=y, hue=condition_on,
                          ax=axis, palette='viridis', color=color)
                          

def plot_freq(data: 'DataFrame or Series', freq_col_name: str=None,  plot_title: str='Bar Chart', include_perc=False, top_labe_gap=10, bot_labe_gap=10, h_labe_shift=-0.4, index_order: bool=True, fig_h: int=6, fig_w=4, dpi=150, ax=None):
    """plot bar chart on an axis using a frequecy table
    Return
    axis"""
    
    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise TypeError("Data must be a dataframe or series")
        
    if isinstance(data, pd.Series):
        freq_col = data.value_counts().sort_index()
    
    elif isinstance(data, pd.DataFrame):
        freq_col = data[freq_col_name].value_counts().sort_index()

    if not index_order:
        freq_col = freq_col.sort_values()
        
    if include_perc:
        perc_freq = np.round(100 * freq_col/len(data), 2)
    
    plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    if ax is None:
        ax = sns.barplot(x=freq_col.index, y=freq_col, palette='viridis')
    else:
        sns.barplot(x=freq_col.index, y=freq_col, palette='viridis', ax=ax)

    for i in range(len(freq_col)):
        labe = freq_col.iloc[i]
        ax.text(i+h_labe_shift, freq_col.iloc[i]+bot_labe_gap, labe,
               size=6, weight='bold')
        if include_perc:
            labe = f'({perc_freq.iloc[i]}%)'
            ax.text(i+h_labe_shift, freq_col.iloc[i]+top_labe_gap, labe,
                   color='blue', size=6, weight='bold')
     
    ax.set_xlabel(freq_col.name), ax.set_ylabel('Count')
    ax.set_title(plot_title, weight='bold', size=15)
    return ax    

def plot_3D(X, y=None, title='A 3-D scatter plot'):
    """Create 3 dimensional plots.
    X is an array with the 3 features
    y is a categorical feature"""
    
    if not isinstance(X, (pd.DataFrame, np.ndarray)):
        raise TypeError('X must be a dataframe or 3D array')
    
    from mpl_toolkits.mplot3d import Axes3D
    
    #%matplotlib notebook
    
    fig = plt.figure(figsize=(5, 5), dpi=200)
    ax = fig.add_subplot(111, projection='3d')
    if isinstance(X, np.ndarray):
        ax.scatter(X[:, 0], X[:, 1], X[:, 2],c=y)
    elif isinstance(X, pd.DataFrame):
        ax.scatter(X.iloc[:, 0], X.iloc[:, 1], X.iloc[:, 2],c=y, palette='viridis')
    ax.set_title(title, weight='bold')
    
    return fig
    

def fig_writer(fname: str, plotter: plt.figure=None, dpi: int=200, file_type='png'):
    """save an image of the given figure plot in the filesystem."""
    
    plotter.get_figure().savefig(fname, dpi=dpi, format=file_type,
                                     bbox_inches='tight', pad_inches=0.25)
    return fname
    
    
def plot_graph(X_ax: 'array-like', y_ax:'array-like', fig_loc: str, size: tuple=[6, 3], clarity: int=150, ncols=2, plot_type: tuple=['line', 'scatter']):
    '''
    plot a graph for X against y
    consisting of num_sections of sections
    along with corresponding plots
    returns figure storage location - fig_loc

    NOTE: length of plot_type and number of columns must match
    return: fig_loc
    '''
    import matplotlib.pyplot as plt, seaborn as sns

    fig = plt.figure(figsize=size, dpi=clarity)

    # create a dict of section: plot_type
    sects, x = {}, 0
    for j in range(ncols):
        w = 0.6
        axis = fig.add_axes([x, 0, w, 1])
        x += 0.2 + w
        plotter = f"sns.{plot_type[j]}plot(x=X_ax, y=y_ax, ax=axis)"
        eval(plotter)

    fig.savefig(fname=fig_loc, bbox_inches='tight')

    return fig_loc