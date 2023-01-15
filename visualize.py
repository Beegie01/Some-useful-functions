from seaborn.utils import np, pd, plt, os
import seaborn as sns
from matplotlib.transforms import Affine2D
from ds_funcs import DsUtils as ds


class VizUtils:
    dayname_cmap = {'Sunday': 'green', 'Monday': 'blue', 'Tuesday': 'yellow', 'Wednesday': 'darkorange', 'Thursday': 'red', 'Friday': 'black', 'Saturday': 'gray'}
    
    @classmethod  # class methods do not need class instantiation
    def adjust_axis(self, axis, plot_title='Plot Title', title_size=12, rotate_xticklabe=0, rotate_yticklabe=0, x_labe=None, precision=2,
                    y_labe=None, xy_labe_size=8, xlim=None, ylim=None, xy_ticksize=5, annotate=False, annot_size=6,  
                    reduce_barw_by=1, bot_labe_color='blue', include_perc=False, perc_total=None, perc_labe_color='black', 
                    show_legend_at:tuple=None, legend_size=7, perc_labe_gap=0, h_labe_shift=0, savefig=False, fig_filename='figplot.png'):
        """edit the setting of statistical plot diagram"""
        
        if x_labe:
            axis.set_xlabel(x_labe, weight='bold', size=xy_labe_size)
        if y_labe:
            axis.set_ylabel(y_labe, weight='bold', size=xy_labe_size)

        if xlim:
            axis.set_xlim(left=xlim[0], right=xlim[1])

        if ylim:
            axis.set_ylim(bottom=ylim[0], top=ylim[1])
            
        if not show_legend_at:
            axis.legend().remove()
        else:
            axis.legend(loc=show_legend_at, prop={'size':legend_size})

        axis.set_title(plot_title, weight='bold', size=title_size)
        axis.set_yticklabels(np.round(axis.get_yticks(), precision), rotation=rotate_yticklabe, fontdict={'fontsize':xy_ticksize})
        axis.set_xticklabels(axis.get_xticklabels(), rotation=rotate_xticklabe, fontdict={'fontsize':xy_ticksize})

        # labels on columns
        if annotate: 
            cont = axis.containers
            for i in range(len(cont)):
    #                 print(len(cont))
                axis.bar_label(container=axis.containers[i], color=bot_labe_color, size=annot_size,
                                  weight='bold')
        
        y_freq = list()
        # position of columns
        for p in axis.patches:
            x = p.get_x()
            w, center = p.get_width()/reduce_barw_by, x+p.get_width()/2
            p.set_width(w)
            p.set_x(center-w/2)
            y_freq.append(p.get_height())

        # percentage labels
        if include_perc:
            if not perc_total:
                perc_total = np.sum(y_freq)

            for p in axis.patches:
                x, y = p.get_xy()
                bot_labe_pos = p.get_height()
                perc = round(100 * p.get_height()/perc_total, precision)
                labe = f'{perc}%'
                perc_labe_pos = bot_labe_pos+perc_labe_gap
                axis.text(x-h_labe_shift, perc_labe_pos, labe, color=perc_labe_color, 
                          size=annot_size, weight='bold')

        if savefig:
            print(self.fig_writer(fig_filename, fig))
        return axis
    
    @classmethod
    def plot_intervals(self, x, n_groups=5, include_right=True, precision=2, plot_title='A Histogram', x_labe=None, 
                      y_labe='count', annotate=True, color=None, paletter='viridis', conf_intvl=None, include_perc=False, 
                      xy_labe_size=8, annot_size=6, use_bar=False, rotate_xticklabe=0, rotate_yticklabe=0, theme='darkgrid',
                       title_size=15, xlim: tuple=None, ylim: tuple=None, axis=None, xy_ticksize=7, perc_labe_gap=None, 
                       perc_total=None, h_labe_shift=0.1, v_labe_shift=0, perc_labe_color='black', bot_labe_color='blue',
                       reduce_barw_by=1, figsize=(8, 4), dpi=200, savefig=False, fig_filename='intervalplot.png'):
            """plot histogram on an axis.
            If include_perc is True, then perc_freq must be provided.
            :Return: axis """
            
            sns.set_style(theme)
            bins = ds.get_interval_freq(x, n_groups, precision)
            freq = ds.count_occurrences(bins).sort_values(freq.columns[0])
            x, y = freq[freq.columns[0]], freq['total_count']
            display(freq)
            
            if color:
                paletter = None
            if not axis:
                fig = plt.figure(figsize=figsize, dpi=dpi)
                if not use_bar:
                    axis = sns.barplot(x=x, y=y, palette=paletter, color=color, ci=conf_intvl)
                else:
                    axis = sns.barplot(x=y, y=x, palette=paletter, color=color, ci=conf_intvl, orient='h')
            else:
                if not use_bar:
                    sns.barplot(x=x, y=y, ci=conf_intvl, 
                                palette=paletter, color=color, ax=axis)
                else:
                    sns.barplot(x=y, y=x, ci=conf_intvl, orient='h',
                            palette=paletter, color=color, ax=axis)
            
            axis.set_title(plot_title, weight='bold', size=title_size)
            
            # axis.set_xticklabels(axis.get_xticklabels(), rotation=rotate_xticklabe)
            # axis.set_yticklabels(axis.get_yticklabels(), rotation=rotate_yticklabe)
            
            if x_labe:
                axis.set_xlabel(x_labe, weight='bold', size=xy_labe_size)
            if y_labe:
                axis.set_ylabel(y_labe, weight='bold', size=xy_labe_size)
            
            if xlim:
                axis.set_xlim(left=xlim[0], right=xlim[1])
            
            if ylim:
                axis.set_ylim(bottom=ylim[0], top=ylim[1])
                
            axis.set_title(plot_title, weight='bold', size=title_size)
            if not use_bar:
                axis.set_yticklabels(axis.get_yticks(), rotation=rotate_yticklabe, fontdict={'fontsize':xy_ticksize})
                axis.set_xticklabels(axis.get_xticklabels(), rotation=rotate_xticklabe, fontdict={'fontsize':xy_ticksize})
            else:
                axis.set_xticklabels(axis.get_xticks(), rotation=rotate_xticklabe, fontdict={'fontsize':xy_ticksize})
                axis.set_yticklabels(axis.get_yticklabels(), rotation=rotate_yticklabe, fontdict={'fontsize':xy_ticksize})
            
            if annotate: 
                cont = axis.containers
                for i in range(len(cont)):
    #                 print(len(cont))
                    axis.bar_label(container=axis.containers[i], color=bot_labe_color, size=annot_size,
                                      weight='bold')
            for p in axis.patches:
                if not use_bar:
                    x = p.get_x()
                    w, center = p.get_width()/reduce_barw_by, x+p.get_width()/2
                    p.set_width(w)
                    p.set_x(center-w/2)
                else:
                    y = p.get_y()
                    bar_width, center = p.get_height()/reduce_barw_by, y+p.get_height()/2
                    p.set_height(bar_width)
                    p.set_y(center-bar_width/2)
                    
            if include_perc:
                if not perc_total:
                    perc_total = y.sum()
                
                for p in axis.patches:
                    x, y = p.get_xy()
                    if not use_bar:
                        bot_labe_pos = p.get_height()
                        perc = round(100 * p.get_height()/perc_total, precision)
                        labe = f'{perc}%'
                        perc_labe_pos = bot_labe_pos+perc_labe_gap
                        axis.text(x-h_labe_shift, perc_labe_pos, labe, color=perc_labe_color, 
                                  size=annot_size, weight='bold')
                    else:
                        y_range = y.max() - y.min()
                        if not perc_labe_gap:
                            perc_labe_gap=y_range/1000 
                        perc = round(100 * p.get_width()/perc_total, precision)
                        labe = f'{perc}%'
                        perc_labe_pos = p.get_width()+perc_labe_gap
                        axis.text(perc_labe_pos, y-v_labe_shift, labe, color=perc_labe_color, 
                                  size=annot_size, weight='bold')
                        
            if savefig:
                print(self.fig_writer(fig_filename, fig))
            return axis
    
    @classmethod
    def visualize_nulls(self, df, plot_title="Missing Entries per Variable", annotate=True, annot_size=6, include_perc=True, 
                        perc_total=None, use_bar=False, perc_labe_gap=0.01, h_labe_shift=0.1, perc_labe_color='black', theme='whitegrid',
                        color=None, reduce_barw_by=1, fig_size=(8, 6), dpi=200, savefig=False, fig_filename='missing_data.png'):
                """plot count plot for null values in df."""
                
                null_cols = ds.null_checker(df, only_nulls=True)
                y_range = null_cols.max() - null_cols.min()
                
                if include_perc:
                    perc_nulls = ds.null_checker(df, only_nulls=True, in_perc=True)
                
                
                if not len(null_cols):
                    return 'No null values in the dataframe'

                fig = plt.figure(figsize=fig_size, dpi=dpi)
                ax = fig.add_axes([0, 0, 1, 1])
                
                if not color:
                    color = 'brown'
                    
                if use_bar:  #use bar chart
                    self.plot_bar(null_cols, null_cols.index, plot_title=plot_title, rotate_yticklabe=False, theme=theme,
                                  y_labe='Column Names', x_labe='Number of Missing Values', include_perc=include_perc,
                                  perc_total=perc_total, annotate=annotate, annot_size=annot_size, perc_labe_gap=perc_labe_gap, 
                                  v_labe_shift=h_labe_shift, perc_labe_color=perc_labe_color, color=color, reduce_barw_by=reduce_barw_by,
                                  figsize=fig_size, dpi=dpi, axis=ax)
                    
                    plt.xlim(right=null_cols.max()+y_range/2)
                else:  #use column chart
                    self.plot_column(x=null_cols.index, y=null_cols, plot_title=plot_title, rotate_xticklabe=True, theme=theme,
                                    x_labe='Column Names', y_labe='Number of Missing Values', include_perc=include_perc, 
                                    annotate=annotate, annot_size=annot_size, perc_labe_gap=perc_labe_gap, h_labe_shift=h_labe_shift,
                                    color=color, perc_labe_color=perc_labe_color, reduce_barw_by=reduce_barw_by, figsize=fig_size,
                                    dpi=dpi, axis=ax)
                    
                    plt.ylim(top=null_cols.max()+y_range/2)
                plt.show()
                if savefig:
                    print(self.fig_writer(fig_filename, fig))

    @classmethod
    def plot_pyramid(self, left_side, right_side, common_catg, left_legend='Left', right_legend='right', xlim=None, ylim=None,
                 plot_title='Pyramid Plot', title_size=15, left_legend_color='white', right_legend_color='white', theme='darkgrid',
                 x_labe='total_count', y_labe=None, left_labe_shift=0, right_labe_shift=0, rv_labe_shift=0, lv_labe_shift=0,
                 left_side_color='orange', right_side_color='blue', fig_w=6, fig_h=8, savefig=False, fig_filename='pyramid_plot.png', 
                 dpi=200):
        """Pyramid view of negative values vs positive values."""
        
        negative_side = -left_side
        positive_side = right_side
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

        self.plot_bar(x=negative_side, y=common_catg,  theme=theme,
                   y_labe=y_labe, x_labe=x_labe, bot_labe_color='black',
                   color=left_side_color, annotate=True, v_labe_shift=0.05, axis=ax)

        self.plot_bar(x=positive_side, y=common_catg, plot_title=plot_title,  theme=theme,
                   title_size=title_size, y_labe=y_labe, x_labe=x_labe, bot_labe_color='black',
                   color=right_side_color, annotate=True, v_labe_shift=0.05, axis=ax)

        neg_range = (abs(negative_side.min()) - abs(negative_side.max()))
        pos_range = positive_side.max() - positive_side.min()
        left_pos = negative_side.min() - neg_range/2
        right_pos = positive_side.max() + pos_range/2
        
        if not xlim:
            xlim = (left_pos, right_pos)
        ax.set_xlim(xlim[0], xlim[1])

        labe = left_legend
        min_ind = negative_side.loc[negative_side == negative_side.max()].index[0]
        x_pos = (abs(negative_side.min()) - abs(negative_side.max()))/2
        ax.text(-x_pos-left_labe_shift, min_ind-lv_labe_shift, labe, color=left_legend_color, 
                weight='bold', bbox={'facecolor':left_side_color})

        labe = right_legend
        max_ind = positive_side.loc[positive_side == positive_side.min()].index[0]
        x_pos = (positive_side.max() - positive_side.min())/2
        ax.text(x_pos+right_labe_shift, max_ind-rv_labe_shift, labe, color=right_legend_color, 
                weight='bold', bbox={'facecolor':right_side_color})
        
        ax.set_xticklabels(ax.get_xticks())
        plt.show()
        
        # save figure to filesystem as png
        if savefig:
            fname = fig_filename
            print(self.fig_writer(fname, fig))

    @classmethod
    def plot_diff(self, left_side, right_side, common_catgs, left_legend='Left', right_legend='right', xlim=None, ylim=None,
              plot_title='Comparison Plot', title_size=15, left_legend_color='white', right_legend_color='white', theme='darkgrid',
              precision=2, y_labe=None, x_labe='total_count', left_labe_shift=0, right_labe_shift=0, lv_labe_shift=0,
              rv_labe_shift=0, left_side_color='orange', right_side_color='blue', fig_w=6, fig_h=8, savefig=False, 
              fig_filename='comparison_plot.png', dpi=200):
        """Comparison view of left values vs right values."""
            
        diff = np.round(right_side - left_side, precision)
        color_mapping = {i: right_side_color if v >= 0 else left_side_color for i, v in zip(common_catgs, diff)}
        #         print(color_mapping)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

        self.plot_bar(x=diff, y=common_catgs, y_labe=y_labe, plot_title=plot_title, title_size=title_size,  theme=theme,
                    x_labe=x_labe, bot_labe_color='black', annotate=True, v_labe_shift=0.05, paletter=color_mapping, 
                    precision=precision, axis=ax)

        left_pos = diff.min() + diff.min()/2
        right_pos = diff.max() + diff.max()/2
        
        if not xlim:
            xlim = (left_pos, right_pos)
        ax.set_xlim(xlim[0], xlim[1])

        labe = left_legend
        min_ind = left_side.loc[left_side == left_side.min()].index[0]
        x_pos = diff.min()/2
        ax.text(x_pos-left_labe_shift, min_ind-lv_labe_shift, labe, color=left_legend_color, 
                weight='bold', bbox={'facecolor':left_side_color})

        labe = right_legend
        min_ind = right_side.loc[right_side == right_side.min()].index[0]
        x_pos = diff.max()/2
        ax.text(x_pos+right_labe_shift, min_ind-rv_labe_shift, labe, color=right_legend_color, 
                weight='bold', bbox={'facecolor':right_side_color})
        ax.set_xticklabels(ax.get_xticks())
        plt.show()
        
        # save figure to filesystem as png
        if savefig:
            fname = fig_filename
            print(self.fig_writer(fname, fig))

    @classmethod
    def visualize_distributions(self, df, condition_on=None, theme='darkgrid', savefig=False):
        """display distributions of each numeric column by plotting a
        scatterplot for columns > 15 categories, and columnplot for
        columns <=15 categories."""

        for col in list(df.select_dtypes(include='number').columns):
            freq = df[col].value_counts().round(2).sort_index()
            if len(freq.index) > 15:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), dpi=150)
                self.plot_scatter(freq.index, freq, condition_on=condition_on, plot_title=f'Distribution of {col.capitalize()}', 
                                   theme=theme, x_labe=col, y_labe='Count', axis=ax1)
                self.plot_box(df[col], plot_title=f'{col.capitalize()} Boxplot', axis=ax2)
                plt.show()
            else:
                fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=150)
                self.plot_column(freq.index, freq, condition_on=condition_on, theme=theme,
                                  plot_title=f'Distribution of {col.capitalize()}', x_labe=col, y_labe='Count',
                                  axis=ax)
                plt.show()
            if savefig:
                fig_filename = f"distr_{str.lower(col)}.png"
                print(self.fig_writer(fig_filename, fig))

    @classmethod
    def plot_correl_heatmap(self, df, plot_title="A Heatmap", title_size=10, annot_size=6, xy_ticklabe_size=6, theme='darkgrid',
                            xlabe=None, ylabe=None, xy_labe_size=8, run_correlation=True, axis=None, figsize=(8, 4), dpi=150, 
                            precision=2, show_cbar=True, cbar_orient='vertical', cbar_size=1, savefig=False, fig_filename='heatmap.png'):
        """plot heatmap for correlation of dataframe.
        If run_correlation is True, execute df.corr() and plot
        else, plot df"""
        
        if run_correlation:
            corr_vals = np.round(df.corr(), precision)
        else:
            corr_vals = df
        
        sns.set_style(theme)
        if not axis:
            fig = plt.figure(figsize=figsize, dpi=dpi)
            axis = sns.heatmap(corr_vals,  cbar=show_cbar, fmt='0.2f',#f'0.{precision}f',
                               annot_kws={'fontsize':annot_size}, annot=True, cbar_kws={'orientation':cbar_orient,
                                                                                       'shrink':cbar_size})#, square=True,)
        else:
            sns.heatmap(corr_vals, ax=axis, cbar=show_cbar, fmt=f'0.{precision}f',
                       annot_kws={'fontsize':annot_size}, annot=True)#, square=True,)
                         
        axis.set_title(plot_title, size=title_size, weight='bold',)# x=0.5, y=1.05)
        
        axis.set_xticklabels(axis.get_xticklabels(), size=xy_ticklabe_size)
        axis.set_yticklabels(axis.get_yticklabels(), size=xy_ticklabe_size)
        
        if xlabe:
            axis.set_xlabel(xlabe, weight='bold', size=xy_labe_size)
        if ylabe:
            axis.set_ylabel(ylabe, weight='bold', size=xy_labe_size)
            
        if savefig:
            print(self.fig_writer(fig_filename, fig))
        
        return axis

    @classmethod
    def plot_hist(self, x=None, y=None, condition_on=None, plot_title="A Histogram", bins=None, interval_per_bin: int=None, 
              bin_range=None, color=None, layer_type='default', x_labe=None, y_labe=None, axis=None, figsize=(8, 4), dpi=150,
              theme='darkgrid', stat='count', include_kde=False, savefig=False, fig_filename='histogram.png'):
        """A histogram plot on an axis.
        layer_type: {"layer", "dodge", "stack", "fill"}
        stat: {'count', 'frequency', 'probability', 'percent', 'density'}
        Return: axis"""
        
        if (not isinstance(x, (pd.Series, np.ndarray)) and x is not None):
            raise TypeError("x must be a pandas series or numpy array")
        elif (not isinstance(y, (pd.Series, np.ndarray)) and y is not None):
            raise TypeError("y must be a pandas series or numpy array")
            
        if not bins:
            bins = 'auto'
        if str.lower(layer_type) == 'default':
            layer_type = 'layer'
        
        sns.set_style(theme)
        if not axis:
            fig = plt.figure(figsize=figsize, dpi=dpi)
            axis = sns.histplot(x=x, y=y, hue=condition_on, bins=bins, binrange=bin_range, 
                                binwidth=interval_per_bin, color=color, multiple=layer_type,
                               stat=stat, kde=include_kde)
        else:
            sns.histplot(x=x, y=y, hue=condition_on, bins=bins, binrange=bin_range,
                         binwidth=interval_per_bin, color=color, multiple=layer_type, 
                         stat=stat, kde=include_kde, ax=axis)
        axis.set_title(plot_title, weight='bold')
        if x_labe:
            axis.set_xlabel(x_labe, weight='bold')
        if y_labe:
            axis.set_ylabel(y_labe, weight='bold')
            
        if savefig:
            print(self.fig_writer(fig_filename, fig))
            
        return axis

    @classmethod
    def plot_box(self, x=None, y=None, condition_on=None, condition_order=None, plot_title="A Boxplot", title_size=12, 
             orientation='vertical', x_labe_order=None, x_labe=None, y_labe=None, axis=None, paletter='viridis', 
             whiskers=1.5, color=None, figsize=(8, 4), dpi=150, show_legend_at=None, legend_size=8, y_lim=None,
             xy_labe_size=6, rotate_xticklabe=0, xy_ticksize=7, box_width=0.8, 
             theme='darkgrid', savefig=False, fig_filename='boxplot.png'):
            """Draw a box distribution plot on an axis.
            Return: axis"""
            
            if (not isinstance(x, (pd.Series, np.ndarray)) and x is not None):
                raise TypeError("x must be a pandas series or numpy array")
            elif (not isinstance(y, (pd.Series, np.ndarray)) and y is not None):
                raise TypeError("y must be a pandas series or numpy array")

            sns.set_style(theme)
            if not axis:
                fig = plt.figure(figsize=figsize, dpi=dpi)
                axis = sns.boxplot(x=x, y=y, hue=condition_on, hue_order=condition_order, order=x_labe_order,
                                   whis=whiskers, orient=orientation, width=box_width, color=color, palette=paletter)
            else:
                sns.boxplot(x=x, y=y, hue=condition_on, hue_order=condition_order, order=x_labe_order, 
                            whis=whiskers, orient=orientation, color=color, width=box_width, palette=paletter, ax=axis)

            if y_lim:
                axis.set_ylim(ymin=y_lim[0], ymax=y_lim[-1])
            axis.set_title(plot_title, size=title_size, weight='bold')
            axis.set_yticklabels(axis.get_yticks(), fontdict={'fontsize':xy_ticksize})
            axis.set_xticklabels(axis.get_xticklabels(), rotation=rotate_xticklabe, fontdict={'fontsize':xy_ticksize})

#             if x_labe:
            axis.set_xlabel(x_labe, weight='bold', size=xy_labe_size)
#             if y_labe:
            axis.set_ylabel(y_labe, weight='bold', size=xy_labe_size)

            if not show_legend_at:
                axis.legend().remove()
            else:
                axis.legend(loc=show_legend_at, prop={'size':legend_size})

            if savefig:
                print(self.fig_writer(fig_filename, fig))
            return axis

    @classmethod
    def plot_line(self, x, y, condition_on=None, plot_title='Line Plot', title_size=15, line_size=None, err_bar=None,
              paletter='viridis', legend_labe=None, show_legend_at=None, legend_size=7, marker=None, color=None, 
              xy_labe_size=8, x_labe=None, y_labe=None, rotate_xticklabe=0, axis=None, xlim=None, ylim=None, 
              xy_ticksize=7, theme='darkgrid', figsize=(8, 4), dpi=200, savefig=False, fig_filename='linegraph.png'):
        """plot line graph on an axis.
        Return: axis """
        
        sns.set_style(theme)
        
        if not axis:
            fig = plt.figure(figsize=figsize, dpi=dpi)
            axis = sns.lineplot(x=x, y=y, hue=condition_on, marker=marker, size=line_size, #legend=show_legend,
                               label=legend_labe,  palette=paletter, color=color, errorbar=err_bar)
        else:
            sns.lineplot(x=x, y=y, hue=condition_on, marker=marker, size=line_size, #legend=show_legend,
                        ax=axis, palette=paletter, color=color, label=legend_labe, errorbar=err_bar)
            
#         if not show_legend_at:
#             axis.legend().remove()
#         else:
#             axis.legend(loc=show_legend_at, prop={'size':legend_size})
            
        if x_labe:
            axis.set_xlabel(x_labe, weight='bold', size=xy_labe_size)
        if y_labe:
            axis.set_ylabel(y_labe, weight='bold', size=xy_labe_size)
        
        axis.set_title(plot_title, weight='bold', size=title_size)
        plt.xticks(ticks=axis.get_xticks(), fontsize=xy_ticksize, rotation=rotate_xticklabe, )
            
        if savefig:
            print(self.fig_writer(fig_filename, fig))
        return axis

    @classmethod
    def plot_strip(self, x, y, condition_on=None, x_labe_order=None, condition_order=None, plot_title='Strip Plot', title_size=14, 
               marker_size=5, color=None, paletter='viridis', x_labe=None, y_labe=None, xy_labe_size=8, axis=None, theme='darkgrid',
               figsize=(8, 4), dpi=200, orientation='v', rotate_xticklabe=None, rotate_yticklabe=None, alpha=None, 
               xy_ticklabe_size=6, savefig=False, fig_filename='stripplot.png'):
            """plot scatter graph for categorical vs numeric variables.
            Return: axis """
            
            sns.set_style(theme)
            if not axis:
                fig = plt.figure(figsize=figsize, dpi=dpi)
                axis = sns.stripplot(x=x, y=y, hue=condition_on, order=x_labe_order, hue_order=condition_order, 
                                     size=marker_size, alpha=alpha, palette=paletter, color=color, orient=orientation)
            else:
                sns.stripplot(x=x, y=y, hue=condition_on, order=x_labe_order, hue_order=condition_order, 
                              size=marker_size, alpha=alpha, ax=axis, palette=paletter, orient=orientation, color=color)
                
            axis.set_title(plot_title, weight='bold', size=title_size)
            if orientation.lower() in 'vertical':
                axis.set_xticklabels(axis.get_xticklabels(), size=xy_ticklabe_size, rotation=rotate_xticklabe)
                axis.set_yticklabels(axis.get_yticks(), size=xy_ticklabe_size, rotation=rotate_yticklabe)
            elif orientation.lower() in 'horizontal':
                axis.set_xticklabels(axis.get_xticks(), size=xy_ticklabe_size, rotation=rotate_xticklabe)
                axis.set_yticklabels(axis.get_yticklabels(), size=xy_ticklabe_size, rotation=rotate_yticklabe)
            if x_labe:
                axis.set_xlabel(x_labe, weight='bold', size=xy_labe_size)
            if y_labe:
                axis.set_ylabel(y_labe, weight='bold', size=xy_labe_size)
                
            if savefig:
                print(self.fig_writer(fig_filename, fig))
            return axis

    @classmethod
    def plot_scatter(self, x, y, condition_on=None, plot_title='Scatter Plot', title_size=14, marker=None, color=None, 
                     paletter='viridis', x_labe=None, y_labe=None, xy_labe_size=8, axis=None, figsize=(8, 4), dpi=200,
                     theme='darkgrid', rotate_xticklabe=False, alpha=None, savefig=False, fig_filename='scatterplot.png'):
            """plot scatter graph on an axis.
            Return: axis """
            
            sns.set_style(theme)
            if not axis:
                fig = plt.figure(figsize=figsize, dpi=dpi)
                axis = sns.scatterplot(x=x, y=y, hue=condition_on, marker=marker, 
                                       alpha=alpha, palette=paletter, color=color)
            else:
                sns.scatterplot(x=x, y=y, hue=condition_on, marker=marker,
                                alpha=alpha, ax=axis, palette=paletter, color=color)
                                
            axis.set_title(plot_title, weight='bold', size=title_size)
            if x_labe:
                axis.set_xlabel(x_labe, weight='bold', size=xy_labe_size)
            if y_labe:
                axis.set_ylabel(y_labe, weight='bold', size=xy_labe_size)
                
            if savefig:
                print(self.fig_writer(fig_filename, fig))
            return axis
    
    @classmethod
    def plot_column(self, x, y, condition_on=None, condition_order=None, x_labe_order=None, plot_title='A Column Chart', title_size=15, x_labe=None, y_labe=None,
                    annotate=True, color=None, paletter='viridis', err_bar=None, include_perc=False, xy_labe_size=8, precision=2, theme='darkgrid',
                    annot_size=6, perc_labe_gap=None, perc_total=None, h_labe_shift=0.1, perc_labe_color='black', bot_labe_color='blue', 
                    show_legend_at: tuple=None, legend_size=7, index_order: bool=True, rotate_xticklabe=0, xlim: tuple=None, ylim: tuple=None, 
                    axis=None, xy_ticksize=7, reduce_barw_by=1, figsize=(8, 4), dpi=200, savefig=False, fig_filename='columnplot.png'):
        """plot bar graph on an axis.
        If include_perc is True, then perc_total must be provided.
        :Return: axis """
        
        freq_col = pd.Series(y)
        
        if color:
            paletter = None
        
        sns.set_style(theme)
        if not axis:
            fig = plt.figure(figsize=figsize, dpi=dpi)
            axis = sns.barplot(x=x, y=y, hue=condition_on, order=x_labe_order, hue_order=condition_order,
                              palette=paletter, color=color, errorbar=err_bar)
        else:
            sns.barplot(x=x, y=y, hue=condition_on, order=x_labe_order, hue_order=condition_order, errorbar=err_bar, 
                        palette=paletter, color=color, ax=axis)
        
        axis.set_title(plot_title, weight='bold', size=title_size)
        
        if x_labe:
            axis.set_xlabel(x_labe, weight='bold', size=xy_labe_size)
        if y_labe:
            axis.set_ylabel(y_labe, weight='bold', size=xy_labe_size)
        
        if xlim:
            axis.set_xlim(left=xlim[0], right=xlim[1])
        
        if ylim:
            axis.set_ylim(bottom=ylim[0], top=ylim[1])
            
        if not show_legend_at:
            axis.legend().remove()
        else:
            axis.legend(loc=show_legend_at, prop={'size':legend_size})
            
        axis.set_title(plot_title, weight='bold', size=title_size)
        axis.set_yticklabels(np.round(axis.get_yticks(), precision), fontdict={'fontsize':xy_ticksize})
        axis.set_xticklabels(axis.get_xticklabels(), rotation=rotate_xticklabe, fontdict={'fontsize':xy_ticksize})
        
        y_range = y.max() - y.min()
        if not perc_labe_gap:
            perc_labe_gap=y_range/1000
        
        if annotate: 
            cont = axis.containers
            for i in range(len(cont)):
#                 print(len(cont))
                axis.bar_label(container=axis.containers[i], color=bot_labe_color, size=annot_size,
                                  weight='bold')
        for p in axis.patches:
            x = p.get_x()
            w, center = p.get_width()/reduce_barw_by, x+p.get_width()/2
            p.set_width(w)
            p.set_x(center-w/2)
                    
        if include_perc:
            if not perc_total:
                perc_total = freq_col.sum()
                
            for p in axis.patches:
                x, y = p.get_xy()
                bot_labe_pos = p.get_height()
                perc = round(100 * p.get_height()/perc_total, precision)
                labe = f'{perc}%'
                perc_labe_pos = bot_labe_pos+perc_labe_gap
                axis.text(x-h_labe_shift, perc_labe_pos, labe, color=perc_labe_color, 
                          size=annot_size, weight='bold')
                    
        if savefig:
            print(self.fig_writer(fig_filename, fig))
        return axis
    
    @classmethod
    def plot_bar(self, x, y, condition_on=None, x_labe_order=None, condition_order=None, plot_title='A Bar Chart', title_size=15,
                 x_labe=None, y_labe=None, xy_labe_size=8, color=None, paletter='viridis', err_bar=None, include_perc=False, 
                 perc_total=None, annot_size=6, perc_labe_gap=10, v_labe_shift=0, perc_labe_color='black', bot_labe_color='blue',
                 precision=2, index_order: bool=True, rotate_yticklabe=0, annotate=False, axis=None, figsize=(8, 4), dpi=200,
                 xlim=None, xy_ticksize=7, show_legend_at:tuple=None, legend_size=7, reduce_barw_by=1, 
                 theme='darkgrid', savefig=False, fig_filename='barplot.png'):
        """plot bar graph on an axis.
        If include_perc is True, then perc_freq must be provided.
        :Return: axis """
        
        
        freq_col = x

        if color:
            paletter = None
        sns.set_style(theme)
        if not axis:
            fig = plt.figure(figsize=figsize, dpi=dpi)
            axis = sns.barplot(x=x, y=y, hue=condition_on, order=x_labe_order, hue_order=condition_order, orient='h',
                              palette=paletter, color=color, errorbar=err_bar)
        else:
            sns.barplot(x=x, y=y, hue=condition_on, order=x_labe_order, hue_order=condition_order, errorbar=err_bar, 
                        palette=paletter, color=color, orient='h', ax=axis)

        if x_labe:
            axis.set_xlabel(x_labe, weight='bold', size=xy_labe_size)
        if y_labe:
            axis.set_ylabel(y_labe, weight='bold', size=xy_labe_size)
        if xlim:
            axis.set_xlim(left=xlim[0], right=xlim[1])
            
        if not show_legend_at:
            axis.legend().remove()
        else:
            axis.legend(loc=show_legend_at, prop={'size':legend_size})
            
        axis.set_title(plot_title, weight='bold', size=title_size)
        axis.set_yticklabels(axis.get_yticklabels(), rotation=rotate_yticklabe, fontdict={'fontsize':xy_ticksize})
        axis.set_xticklabels(np.round(axis.get_xticks(), precision), fontdict={'fontsize':xy_ticksize})
        
        if annotate: 
            cont = axis.containers
            for i in range(len(cont)):
#                 print(len(cont))
                axis.bar_label(container=axis.containers[i], color=bot_labe_color, size=annot_size,
                                  weight='bold')
            if include_perc and perc_total is not None:
                for p in axis.patches:
                    x, y = p.get_xy()
                    perc = round(100 * p.get_width()/perc_total, precision)
                    labe = f'{perc}%'
                    perc_labe_pos = p.get_width()+perc_labe_gap
                    axis.text(perc_labe_pos, y-v_labe_shift, labe, color=perc_labe_color, 
                              size=annot_size, weight='bold')
        
        for p in axis.patches:
            y = p.get_y()
            bar_width, center = p.get_height()/reduce_barw_by, y+p.get_height()/2
            p.set_height(bar_width)
            p.set_y(center-bar_width/2)
                              
        if savefig:
            print(self.fig_writer(fig_filename, fig))
        return axis             
        
    @staticmethod
    def view_image_file(fname: str):
        from PIL import Image
        
        img_obj = Image.open(fname)
        Image._show(img_obj)
        print(f'{fname} now on display')
        
    @staticmethod  # standalone function inside a class
    def plot_errorbars(x, y, yerror, axis, plot_title='Line Plot', title_size=15, capsize=2,
                  err_linewidth=0.5, errbar_color='black', shift=False):
        """plot errorbars on an axis.
        yerror can be int or array
        especially where there are multiple collection of means.
        use shift to push errorbar horizontally
        Return
        axis"""
        
        # move errorbars sideways
        trans1 = Affine2D().translate(-0.1, 0.0) + axis.transData
        trans2 = Affine2D().translate(+0.1, 0.0) + axis.transData
        
        # plot errorbars on the background graph
        if not shift:
            axis.errorbar(x, y, yerr=yerror, fmt='none', capsize=capsize, elinewidth=err_linewidth,
                         ecolor=errbar_color, transform=trans1)
        else: # shift the errorbar horizontally
            axis.errorbar(x, y, yerr=yerror, fmt='none', capsize=capsize, elinewidth=err_linewidth,
                          ecolor=errbar_color, transform=trans2)
        return axis
        
    @staticmethod
    def plot_3D(X, y=None, plot_title='A 3-D scatter plot'):
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
        ax.set_title(plot_title, weight='bold')
        return fig
        
    @staticmethod  # independent function within a class, thus does not need the 'self' argument
    def fig_writer(fname: str, plotter: plt.figure=None, dpi: int=200, file_type='png'):
        """Save the figure plot as an image in the filesystem with name as fname.
        Default file_type is PNG.
        Return: fname"""
        
        plotter.get_figure().savefig(fname, dpi=dpi, format=file_type,
                                     bbox_inches='tight', pad_inches=0.25)
        return fname