from seaborn.utils import np, pd, plt, os
import seaborn as sns


def plot_correl_heatmap(df, plot_title, axis=None):
    """plot correlation heatmap from dataframe"""
    
    if not axis:
        axis = sns.heatmap(np.round(df.corr(), 2), 
                           annot_kws={'fontsize':6}, annot=True, square=True,)
    else:
        sns.heatmap(np.round(df.corr(), 2), ax=axis,
                           annot_kws={'fontsize':6}, annot=True, square=True,)
                     
    axis.set_title(plot_title, weight='bold', x=0.5, y=1.05)
    
    axis.set_xticklabels(axis.get_xticklabels(), size=6)
    axis.set_yticklabels(axis.get_yticklabels(), size=6)
    
    return axis
    
    
def plot_hist(x=None, y=None, condition_on=None, plot_title="A Histogram", bins=None, interval_per_bin: int=None, 
              bin_range=None, color=None, layer_type='default', x_labe=None, y_labe=None, axis=None):
    """A histogram plot on an axis.
    layer_type: {"layer", "dodge", "stack", "fill"}
    Return 
    axis"""
    
    if (not isinstance(x, (pd.Series, np.ndarray)) and x is not None):
        raise TypeError("x must be a pandas series or numpy array")
    elif (not isinstance(y, (pd.Series, np.ndarray)) and y is not None):
        raise TypeError("y must be a pandas series or numpy array")
        
    if not bins:
        bins = 'auto'
    if str.lower(layer_type) == 'default':
        layer_type = 'layer'
    if not axis:
        axis = sns.histplot(x=x, y=y, hue=condition_on, bins=bins, binrange=bin_range, 
                            binwidth=interval_per_bin, color=color, multiple=layer_type)
    else:
        sns.histplot(x=x, y=y, hue=condition_on, bins=bins, binrange=bin_range,
                     binwidth=interval_per_bin, color=color, multiple=layer_type, 
                     ax=axis)
    axis.set_title(plot_title, x=0.5, y=1.025)
    if x_labe:
        axis.set_xlabel(x_labe, weight='bold')
    if y_labe:
        axis.set_ylabel(y_labe, weight='bold')
        
    
    return axis


def plot_box(x=None, y=None, condition_on=None, title="A Boxplot", orientation='horizontal', 
             x_labe=None, y_labe=None, axis=None):
    """A box distribution plot on an axis.
    Return: axis"""
    
    if (not isinstance(x, (pd.Series, np.ndarray)) and x is not None):
        raise TypeError("x must be a pandas series or numpy array")
    elif (not isinstance(y, (pd.Series, np.ndarray)) and y is not None):
        raise TypeError("y must be a pandas series or numpy array")
        
    if not axis:
        axis = sns.boxplot(x=x, y=y, hue=condition_on, orient=orientation)
    else:
        sns.boxplot(x=x, y=y, hue=condition_on, orient=orientation, ax=axis)
    axis.set_title(title, x=0.5, y=1.025)
    if x_labe:
            axis.set_xlabel(x_labe, weight='bold')
    if y_labe:
        axis.set_ylabel(y_labe, weight='bold')
    
    return axis
    

def plot_line(x, y, condition_on=None, plot_title='Line Plot', line_size=None, 
              legend_labe=None, show_legend=False, marker=None, color=None, 
              x_labe=None, y_labe=None, axis=None):
    """plot line graph on an axis.
    Return: axis """
    
    if not axis:
        axis = sns.lineplot(x=x, y=y, hue=condition_on, marker=marker, size=line_size, label=legend_labe, 
                           legend=show_legend, palette='viridis', color=color)
    else:
        sns.lineplot(x=x, y=y, hue=condition_on, marker=marker, size=line_size, legend=show_legend,
                              ax=axis, palette='viridis', color=color, label=legend_labe)
    
    axis.set_title(plot_title, weight='bold')
    if x_labe:
        axis.set_xlabel(x_labe, weight='bold')
    if y_labe:
        axis.set_ylabel(y_labe, weight='bold')
    return axis
                          
def plot_scatter(x, y, condition_on=None, plot_title='Scatter Plot', marker=None, color=None, 
                 x_labe=None, y_labe=None, axis=None):
    """plot scatter graph on an axis.
    Return: axis """
    
    if not axis:
        axis = sns.scatterplot(x=x, y=y, hue=condition_on, marker=marker, palette='viridis', color=color)
    else:
        sns.scatterplot(x=x, y=y, hue=condition_on, marker=marker,
                              ax=axis, palette='viridis', color=color)
    axis.set_title(plot_title, weight='bold')
    if x_labe:
        axis.set_xlabel(x_labe, weight='bold')
    if y_labe:
        axis.set_ylabel(y_labe, weight='bold')
    return axis

                          
def plot_column(x, y, condition_on=None, plot_title='A Column Chart', x_labe=None, y_labe=None,
             color=None, paletter='viridis', conf_intvl=None, include_perc=False, perc_freq=None, top_labe_gap=10, 
             bot_labe_gap=10, h_labe_shift=-0.4, top_labe_color='black', bot_labe_color='blue', 
             index_order: bool=True, rotate_xticklabe=False, axis=None):
    """plot bar graph on an axis.
    If include_perc is True, then perc_freq must be provided.
    :Return: axis """
    
    freq_col = y
    
    if color:
        paletter = None
    if not axis:
        axis = sns.barplot(x=x, y=y, hue=condition_on, 
                          palette=paletter, color=color, ci=conf_intvl)
    else:
        sns.barplot(x=x, y=y, hue=condition_on, ci=conf_intvl, 
                    palette=paletter, color=color, ax=axis)
    
    axis.set_title(plot_title, weight='bold', size=15, x=0.5, y=1.05)
    
    if rotate_xticklabe:
            axis.set_xticklabels(axis.get_xticklabels(), rotation=90)
    if x_labe:
        axis.set_xlabel(x_labe, weight='bold')
    if y_labe:
        axis.set_ylabel(y_labe, weight='bold')
    
    for i in range(len(freq_col)):
            labe = freq_col.iloc[i]
            axis.text(i+h_labe_shift, freq_col.iloc[i]+bot_labe_gap, labe,
                      color=bot_labe_color, size=6, weight='bold')
            if include_perc and perc_freq is not None:
                labe = f'({perc_freq.iloc[i]}%)'
                axis.text(i+h_labe_shift, freq_col.iloc[i]+top_labe_gap, labe,
                          color=top_labe_color, size=6, weight='bold')

    return axis
             
def plot_bar(x, y, condition_on=None, plot_title='A Bar Chart', x_labe=None, y_labe=None,
             color=None, paletter='viridis', conf_intvl=None, include_perc=False, perc_freq=None, top_labe_gap=10, 
             bot_labe_gap=10, v_labe_shift=-0.4, top_labe_color='black', bot_labe_color='blue', 
             index_order: bool=True, rotate_yticklabe=False, annot=False, axis=None):
    """plot bar graph on an axis.
    If include_perc is True, then perc_freq must be provided.
    :Return: axis """
    
    freq_col = x
        
    
    if color:
        paletter = None
    if not axis:
        axis = sns.barplot(x=x, y=y, hue=condition_on, orient='h',
                          palette=paletter, color=color, ci=conf_intvl)
    else:
        sns.barplot(x=x, y=y, hue=condition_on, ci=conf_intvl, 
                    palette=paletter, color=color, orient='h', ax=axis)
    
    axis.set_title(plot_title, weight='bold', size=15, x=0.5, y=1.05)
    
    if rotate_yticklabe:
        axis.set_xticklabels(axis.get_xticklabels(), rotation=90)
    if x_labe:
        axis.set_xlabel(x_labe, weight='bold')
    if y_labe:
        axis.set_ylabel(y_labe, weight='bold')
    if annot:
        for i in range(len(freq_col)):
            labe = freq_col.iloc[i]
            axis.text(freq_col.iloc[i]+bot_labe_gap, i+v_labe_shift, labe,
                      color=bot_labe_color, size=6, weight='bold')
            if include_perc and perc_freq is not None:
                labe = f'({perc_freq.iloc[i]}%)'
                axis.text(freq_col.iloc[i]+top_labe_gap, i+v_labe_shift, labe,
                          color=top_labe_color, size=6, weight='bold')

    return axis             

def plot_freq(data: 'DataFrame or Series', freq_col_name: str=None,  plot_title: str='Bar Chart', 
              include_perc=False, top_labe_gap=10, bot_labe_gap=10, h_labe_shift=-0.4, index_order: bool=True, 
              fig_h: int=6, fig_w=4, dpi=150, x_labe=None, y_labe=None, color=None, 
              rotate_xticklabe=False, axis=None):
    """plot bar chart on an axis using a frequecy table
    :Return: axis"""
    
    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise TypeError("Data must be a dataframe or series")
        
    if isinstance(data, pd.Series):
        freq_col = data.value_counts().sort_index()
    
    elif isinstance(data, pd.DataFrame):
        freq_col = data[freq_col_name].value_counts().sort_index()

    paletter = 'viridis'
    if color:
        paletter = None
    
    if not index_order:
        freq_col = freq_col.sort_values()
        
    if include_perc:
        perc_freq = np.round(100 * freq_col/len(data), 2)
    
    plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    if not axis:
        axis = sns.barplot(x=freq_col.index, y=freq_col, palette=paletter, color=color)
    else:
        sns.barplot(x=freq_col.index, y=freq_col, palette=paletter, color=color, ax=axis)

    for i in range(len(freq_col)):
        labe = freq_col.iloc[i]
        axis.text(i+h_labe_shift, freq_col.iloc[i]+bot_labe_gap, labe,
               size=6, weight='bold')
        if include_perc:
            labe = f'({perc_freq.iloc[i]}%)'
            axis.text(i+h_labe_shift, freq_col.iloc[i]+top_labe_gap, labe,
                   color='blue', size=6, weight='bold')
    if x_labe:
        axis.set_xlabel(x_labe, weight='bold')
    else:
        axis.set_xlabel(freq_col.name)
    if y_labe:
        axis.set_ylabel(y_labe, weight='bold') 
    else:
        axis.set_ylabel('Count')
    
    if rotate_xticklabe:
        axis.set_xticklabels(axis.get_xticklabels(), rotation=90)
    axis.set_title(plot_title, weight='bold', size=15)
    return axis 
    

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