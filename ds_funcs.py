import pickle
import os
import copy
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import random
from sklearn import metrics as s_mtr, tree, ensemble, cluster, decomposition as s_dec, feature_extraction as s_fex
from sklearn import preprocessing as s_prep, feature_selection as s_fs, utils as s_utils
from scipy import stats
from scipy.stats import ttest_ind_from_stats, ttest_ind
import statsmodels as stat
import statsmodels.api as sm
import statsmodels.formula.api as smf
from mlxtend import frequent_patterns


class DsUtils:


    @staticmethod
    def get_correlcoeff_with_pvalues(X, y, show_pval_thresh=1):
        """compute pearson's correlation coefficient (and pvalues) between
        X variables and y
        Return
        result_df: if y is 1D, dataframe containing pearson coefficient (r) and pvalues
        else, dictionary containing y categories as keys and their r and pvalues for each
        variable"""
        
        
        def get_pearsonr_pval(X, y):
            """compute pearson's correlation coefficient (and pvalues) between
            X variables and y
            Return
            result_df: dataframe containing pearson coefficient (r) and pvalues"""
            r_dict, pval_dict = dict(), dict()

            X = pd.DataFrame(X)
            cols = X.columns

            for x in cols:
                r, pval = stats.pearsonr(X[x], y)
                r_dict[f'{x}'] = r
                pval_dict[f'{x}'] = pval
            return pd.DataFrame([pd.Series(r_dict,name='r'),
                                   pd.Series(pval_dict, name='pval')]).T.sort_values(by='pval')
        
        if (len(y.shape) > 1) and (y.shape[-1] > 1):
            y_dum = pd.get_dummies(pd.DataFrame(y))
        
            y_catg = dict()
            for y_col in y_dum:
                y = y_dum[y_col]
                result = get_pearsonr_pval(X, y)
                y_catg[f'{y_col}'] = result.loc[result['pval'] < show_pval_thresh]
            
            return y_catg
        
        result = get_pearsonr_pval(X, y)
        return result.loc[result['pval'] < show_pval_thresh]
        
        return get_pearsonr_pval(X, y)

    @staticmethod
    def get_unique_combinations(unq_elements: 'array of unique elements', n_combo_elements):
        """create combination of unique elements where each combination is n_combo_elements"""
        
        return [c  for c in combinations(list(unq_elements), n_combo_elements)]

    @staticmethod
    def compute_median_std_sterr(arr, group_name='group', precision=2):
        """compute median, standard deviation and standard error for arr
        Return:
        result_df: dataframe reporting median, std_from_median, and sterror"""
        
        grp_mdn = pd.Series(np.median(arr).round(precision), name=f'{group_name}_median')
        
        # standard deviation from median
        dev = arr - arr.median()
        sq_dev = dev**2
        sumsq_dev = sq_dev.sum()
        deg_freedom = len(arr) -1
        variance = sumsq_dev/deg_freedom
        mdn_std = variance**0.5
        grp_std = pd.Series(np.round(mdn_std, precision), name=f'{group_name}_std')
        
        # standard error from median
        mdn_sterr = np.round(mdn_std/len(arr)**0.5, precision)
        grp_sterr = pd.Series(mdn_sterr, name=f'{group_name}_sterr')

        return pd.concat([grp_mdn, grp_std, grp_sterr], axis=1)

    @staticmethod
    def compute_mean_std_sterr(arr, group_name='group', precision=2):
        """compute mean, standard deviation and standard error for arr
        Return:
        result_df: dataframe reporting mean, std, and sterror"""
        
        grp_avg = pd.Series(np.mean(arr).round(precision), name=f'{group_name}_avg')
        grp_std = pd.Series(np.std(arr, ddof=1).round(precision), name=f'{group_name}_std')
        grp_sterr = pd.Series(np.round(np.std(arr, ddof=1)/np.sqrt(len(arr)), precision), 
                              name=f'{group_name}_sterr')
        
        return pd.concat([grp_avg, grp_std, grp_sterr], axis=1)

    @staticmethod
    def run_glm_predictor(X, y, binary_class=True, precision=2):
        """run a generalised linear model (logisitic regression) on the relationship between
        the dataframe's x_cols and y_col
        Return 
        model_summary: glm model summary object"""
        
        def generate_formula(X, y):
            """generate formula in string format"""
            cols, formula = X.columns, ""
            for n in range(len(cols)):
                if n == 0:
                    formula += cols[n]
                    continue
                formula += f"+{cols[n]}"
            return f"{y.name} ~ {formula}"
        
        # normalise X
        scaler = s_prep.MinMaxScaler()
        scx = scaler.fit_transform(X)
        scX = pd.DataFrame(scx, columns=X.columns)
        Xc = sm.add_constant(scX)
        
        # create dataframe comprising only normalised X and y
        rlike_df = pd.merge(Xc, y, left_index=True, right_index=True)
        
        # create formula string consisting of dependent variable and independent variables
        formula = generate_formula(scX, y)
        
        # run glm model
        if binary_class:
            model = smf.glm(formula=formula, data=rlike_df, family=sm.families.Binomial())
            tmodel = model.fit()
        else:
            model = smf.mnlogit(formula=formula, data=rlike_df, )#optimizer='powell')#, family=sm.families.Poisson())
#             model = smf.glm(formula=formula, data=rlike_df)
#             model = stat.discrete.discrete_model.MultinomialModel(endog=y, exog=Xc)
#             model = sm.MNLogit(endog=y, exog=Xc)

            tmodel = model.fit_regularized()
        
        print(f'Dependent Variable:\n{model.endog_names}, \n\nPValues:\n{tmodel.pvalues}, \n\nCoefficients:\n{tmodel.params}')
        return tmodel
    
    @staticmethod
    def combine_multiple_columns(df, *multi_cols:str, output_name='combo_variable'):
        """collapse multiple columns into one column"""
        
        def collapsed(row, mult_vars):
            result = ""
            for i in range(len(mult_vars)):
                if i == 0:
                    result += f"{row[mult_vars[i]]}"
                else:
                    result += f"_{row[mult_vars[i]]}"
            return result
        
        if not multi_cols:
            multi_cols = df.columns
        multi_cols = list(multi_cols)
        collapsed_var = pd.Series(df.apply(lambda row: collapsed(row, multi_cols), axis=1),
                                  name=output_name)
        return pd.concat([collapsed_var, df.drop(multi_cols, axis=1)], axis=1)
    
    @staticmethod
    def convert_multilevel_columns_to_single(df, uniq_id):
        """convert a multi-level column dataframe to a single-level dataframe"""
        
        new_colnames = [f"{c1}_{c2}" if c1.lower() != uniq_id.lower() else c1 for c1, c2 in df.columns]
        new_df = df.droplevel(level=1, axis=1)
        new_df.columns = new_colnames
        return new_df
    
    @staticmethod
    def convert_rows_to_columns(df, unique_identifier:str, shared_identifier:str, var_prefix:str=None, output_multi_lvl=False):
            """generate columns from row values per unique_identifier
            unique_identifier: primary identifier column
            shared_identifier: its unique values will each form a new header"""
            
            cls = DsUtils()
            unq = df[shared_identifier].unique()
            col_rename = {u: f"{var_prefix}_{u}" if var_prefix else u for u in unq}
            # print(col_rename)
            new_df = pd.DataFrame.pivot(df, index=unique_identifier, columns=shared_identifier).reset_index().rename(columns=col_rename)
            if output_multi_lvl:
                return new_df
            return cls.convert_multilevel_columns_to_single(new_df, unique_identifier)

    @staticmethod
    def convert_rows_to_columns(df, primary_index_col:str, secondary_index_col:str, target_col:str, var_prefix:str):
        """generate columns from row values per primary_index_col
        primary_index_col: primary identifier column
        secondary_index_col: its unique values will each form a new header
        target_col: determines the values in each of the new header"""
        
        cols = [primary_index_col, secondary_index_col, target_col]
        unq = df[secondary_index_col].unique()
        col_rename = {u: f"{var_prefix}_{u}" for u in unq}
        return pd.DataFrame.pivot(df[cols], index=primary_index_col, columns=secondary_index_col, values=target_col).reset_index().rename(columns=col_rename)
        
    @staticmethod
    def convert_column_to_row(df, primary_id:str, combo_column_name:str, value_name:str):
        """generate rows from columns"""
        
        return pd.DataFrame.melt(df, id_vars=primary_id, var_name=combo_column_name, value_name=value_name)
    
    @staticmethod
    def save_python_obj(fname, py_obj):
        """save python object to file system.
        NOTE: always add .pkl to fname"""
        
        with open(fname, 'wb') as f:
            pickle.dump(txt, f)
        print("Python object has been saved")
        
    @staticmethod
    def load_python_obj(fname):
        """"""
        with open(fname, 'rb') as f:
            txt = pickle.load(f)
        print('Loading complete')

    @staticmethod
    def run_apriori(df, use_cols=None, use_colnames=True, min_support=0.5):
        
        sel_cols = df.select_dtypes(exclude='number')
        if use_cols:
            sel_cols = sel_cols[use_cols]
            
        result = frequent_patterns.apriori(pd.get_dummies(sel_cols),
                                       use_colnames=use_colnames, min_support=min_support)
        result['num_sets'] = result['itemsets'].apply(lambda x: len(x))
        return result.sort_values('support', ascending=False)
        
    @staticmethod
    def replace_value_with(df, replacement_guide:dict=None, colnames:list=None):
        """replace all occurrences of old value (key) in colnames with new value (value).
            if colnames is None, replace all occurrences of old value across entire
            df with new value.
            Return 
            new_df: dataframe with new values in place of old values"""

        def replace_value(df, old_value, new_value, colnames=None):
            """replace all occurrences of old value in colnames with new value.
            if colnames is None, replace all occurrences of old value across entire
            df with new value.
            Return 
            new_df: dataframe with new values in place of old values"""

            if not colnames:
                cols = tuple(df.columns)
            else:
                cols = tuple(colnames)

            new_df = pd.DataFrame(df)

            for c in cols:
                new_df[c] = new_df[c].apply(lambda x: new_value if x == old_value else  x)
            return new_df
        
        for i, (old_val, new_val) in enumerate(replacement_guide.items()):
            if i == 0:
                new_df = replace_value(df, old_val, new_val, colnames)
                continue
            new_df = replace_value(new_df, old_val, new_val, colnames)
        return new_df
    
    @staticmethod
    def count_occurrences(df, count_colname:str =None, output_name='total_count', attach_to_df=False):
        """count occurrences per unique count_colname or unique combined categs"""
        
        if not count_colname:
            freq = df.value_counts().reset_index()
        else:
            freq = df.groupby(count_colname).size().reset_index()
        freq = freq.rename(columns={0:output_name})
        if attach_to_df and count_colname:
            return pd.merge(freq, df, on=count_colname)
        return freq
    
    @staticmethod
    def give_percentage(arr, perc_total=None, precision=2):
        """output the percentage of each element in array
        arr: dataframe or series
        perc_total: total for calculating percentage of each value
        if perc_total is None, len(arr) is used instead
        Return
        perc_arr"""
        
        if not perc_total:
            return np.round(100*arr/len(arr), precision)
        return np.round(100*arr/perc_total, precision)
        
    @staticmethod
    def percentage_per_row(df, grouper_col:str=None, precision=2):
        """compute percentage per row of each unique grouper_col value
        Return
        result: df in percentage"""
        
        if grouper_col:
            new_df = df.set_index(grouper_col)
        else:
            new_df = df
        denom = new_df.sum(axis=1)
        return np.round(100 * new_df.div(denom, axis=0), precision)
        
    @staticmethod
    def get_subset_percentage(df, freq_col:str, sum_by_col:str, precision=2):
        """compute percentage per group of each unique sum_by_col value
        Return
        result: df including percentage"""
        
        # create aggregate sum for percentage
        cols = [sum_by_col, freq_col]
        summed = df[cols].groupby(cols[0]).sum().rename(columns={freq_col:'summed'}).reset_index()
        
        #attach aggregate value to each row
        new_df = pd.merge(df, summed, on=cols[0])
        new_df.loc[:, f'%{freq_col}'] = 100 * new_df.apply(lambda row: row[freq_col] / row['summed'], axis=1).round(precision)
        return new_df.drop('summed', axis=1)
        
    @staticmethod
    def compare_variables(var1, var2, var1_name=None, var2_name=None, where_equal=True):
        """compare two variables and return where equal/unequal (if True/False)
        Return
        result: dataframe where series are equal/unequal"""
        
        if not var1_name:
            var1_name = var1.name
        if not var2_name:
            var2_name = var2.name
            
        df = pd.concat([pd.Series(var1, name=var1_name),
                        pd.Series(var2, name=var2_name)], axis=1)
        cond = (var1 == var2).astype(int)
        if where_equal:
            return df.loc[cond == 1]
        return df.loc[cond == 0]
    
    @staticmethod
    def generate_interval_labels(min_val, max_val, n_groups=5, bin_width=None, precision=2):
        """generate labels from lower and upper bounds of interval values.
        NOTE: categories/n_labels = num_intervals - 1
        Return
        label_dict: dictionary containing integer index: label"""
        
        def push_boundary(upper_lim, interval):
            """to make sure that the upper boundary is included"""
            
            remainder = upper_lim%interval
            if remainder != 0:
                return (interval - remainder) + upper_lim
            return upper_lim
        
        if not bin_width:
            num_range = np.round(np.linspace(min_val, max_val+0.1, n_groups-1), precision)
        else:
            if max_val < 0:  # upper boundary is a negative value
                pmax = -1*push_boundary(np.abs(max_val), bin_width) + (bin_width/10)
                num_range = np.arange(start=min_val, stop=pmax, step=bin_width)
            elif min_val < 0:  # lower boundary is a negative value
                # include negative lower boundary
                nmax = push_boundary(np.abs(min_val), bin_width) + (bin_width/10)
                # negative range of values, excluding the first value (-0)
                nmax_range = -np.arange(start=0, stop=nmax, step=bin_width)[1:]
                # include positive upper boundary
                pmax = push_boundary(max_val, bin_width) + (bin_width/10)
                # positive range of values
                pmax_range = np.arange(start=0, stop=pmax, step=bin_width)
                # merge negative and positive range of values
                num_range = np.sort(np.append(nmax_range, pmax_range))
            
        return {i:f'[{num_range[i-1]} to {num_range[i]})' for i in range(len(num_range)) if i > 0}

    @staticmethod
    def get_interval_freq(continuous_ser, n_groups=5, bin_width=None, precision=2, output_colnames=None):
            """generate interval frequencies for a continuous series.
            NOTE: categories/n_labels = num_intervals - 1
            Return
            bmi_change_interval: dataframe containing (intervals_as_int, intervals_as_str)"""
            
            cls = DsUtils()
            def classify_val(val:'number', interval_guide:dict):
                for ix, invls in interval_guide.items():
                    lower_bound , upper_bound = float(invls.split()[0][1:]), float(invls.split()[-1][:-1])
                    if (val >= lower_bound and val < upper_bound):
                        return ix 
            
            x = pd.Series(continuous_ser)
            min_val, max_val = np.percentile(x, [0, 100])
            labes_dict = cls.generate_interval_labels(min_val, max_val, n_groups, bin_width, precision)
            invl_ix = x.apply(lambda x: classify_val(x, labes_dict))
            invl_labe = invl_ix.map(labes_dict)
            if not output_colnames:
                output_colnames = ['bmi_diff_class', 'bmi_diff_band']
            return pd.concat([invl_ix, invl_labe], axis=1, keys=output_colnames)
    
    @staticmethod
    def equalize_categories(left, right, grouper_colname:str, freq_colname:str):
        """include zero frequencies for missing categories between two dfs:
        Return
        result: keys are (left, right) containing df with 0 in place of missing categories"""
        
        def merge_categories(catg1: 'pd.Series', catg2: 'pd.Series'):
            """create a common list of categories"""
            
            unq1, unq2 = list(catg1.unique()), list(catg2.unique())
            merg = pd.Series(sorted(set(unq1 + unq2)))
            return merg
        
        result = dict()
        common_categs = merge_categories(left[grouper_colname], right[grouper_colname])
        common_categs.name = grouper_colname
        common_categs = pd.DataFrame(common_categs)
        result['left'] = pd.merge(common_categs, left, on=grouper_colname, how='left')
        result['left'][freq_colname] = result['left'][freq_colname].fillna(int(0))
        result['right'] = pd.merge(common_categs, right, on=grouper_colname, how='left')
        result['right'][freq_colname] = result['right'][freq_colname].fillna(int(0))
        return result
    
    @staticmethod
    def get_from_df(df, col1, col1_is, col2=None, col2_is=None, col3=None, col3_is=None):
        """return filtered view of dataframe"""
        
        def show_with_one_cond(df, col1, col1_is):
            """return filtered view of dataframe"""
            if df is None:
                if isinstance(col1, pd.Series):
                    if isinstance(col1_is, (tuple, list)):
                        cond = (col1.isin(col1_is))
                    else:
                        cond = (col1 == col1_is)
                return col1.loc[cond]
            if isinstance(col1_is, (tuple, list)):
                cond = (df[col1].isin(col1_is))
            else:
                cond = (df[col1] == col1_is)
            return df.loc[cond]

        def show_with_two_cond(df, col1, col1_is, col2, col2_is):
            """return filtered view of dataframe"""
            
            result = show_with_one_cond(df, col1, col1_is)
            if isinstance(col2_is, (tuple, list)):
                cond = (result[col2].isin(col2_is))
            else:
                cond = (result[col2] == col2_is)
            return result.loc[cond]

        def show_with_three_cond(df, col1, col1_is, col2, col2_is, col3, col3_is):
            """return filtered view of dataframe"""
            
            result = show_with_two_cond(df, col1, col1_is, col2, col2_is)
            if isinstance(col3_is, (tuple, list)):
                cond = (result[col3].isin(col3_is))
            else:
                cond = (result[col3] == col3_is)
            return result.loc[cond]
        
        if col2 is not None and col2_is is not None:
            
            if col3 is not None and col3_is is not None:
                return show_with_three_cond(df, col1, col1_is, col2, col2_is, col3, col3_is)
            
            return show_with_two_cond(df, col1, col1_is, col2, col2_is)
        
        return show_with_one_cond(df, col1, col1_is)
            
    @staticmethod
    def generate_multi_index_from_df(df, sort_by_col_num=0):
        """generate a multi-level index series from dataframe unque categories"""
        
        return pd.MultiIndex.from_frame(df, sortorder=sort_by_col_num)
        
    @staticmethod
    def get_interval_freq(continuous_ser, n_groups=5, precision=2, output_colnames=None):
        """generate interval frequencies for a continuous series.
        NOTE: categories/n_labels = num_intervals - 1
        Return
        bmi_change_interval: dataframe containing (intervals_as_int, intervals_as_str)"""
        
        def classify_val(val, interval_guide):
            for ix, invls in interval_guide.items():
                lower_bound , upper_bound = float(invls.split()[0][1:]), float(invls.split()[-1][:-1])
                if (val >= lower_bound and val < upper_bound):
                    return ix 
        
        x = pd.Series(continuous_ser)
        min_val, max_val = np.percentile(x, [0, 100])
        labes_dict = cls.generate_interval_labels(min_val, max_val, n_groups, precision)
        invl_ix = x.apply(lambda x: classify_val(x, labes_dict))
        invl_labe = invl_ix.map(labes_dict)
        return pd.concat([invl_ix, invl_labe], axis=1, keys=output_colnames)
    
    @staticmethod
    def generate_interval_labels(min_val, max_val, n_groups=5, bin_width=None, precision=2):
        """generate labels from lower and upper bounds of interval values.
        NOTE: categories/n_labels = num_intervals - 1
        Return
        label_dict: dictionary containing integer index: label"""
        
        def push_boundary(upper_lim, interval):
            """to make sure that the upper boundary is included"""
            
            remainder = upper_lim%interval
            if remainder != 0:
                return (interval - remainder) + upper_lim
            return upper_lim
        
        num_range = np.round(np.linspace(min_val, max_val+0.1, n_groups-1), precision)
        
        if bin_width:
            if max_val < 0:  # upper boundary is a negative value
                pmax = -1*push_boundary(np.abs(max_val), bin_width) + (bin_width/10)
                num_range = np.arange(start=min_val, stop=pmax, step=bin_width)
            elif min_val < 0:  # lower boundary is a negative value
                # include negative lower boundary
                nmax = push_boundary(np.abs(min_val), bin_width) + (bin_width/10)
                # negative range of values, excluding the first value (-0)
                nmax_range = -np.arange(start=0, stop=nmax, step=bin_width)[1:]
                # include positive upper boundary
                pmax = push_boundary(max_val, bin_width) + (bin_width/10)
                # positive range of values
                pmax_range = np.arange(start=0, stop=pmax, step=bin_width)
                # merge negative and positive range of values
                num_range = np.sort(np.append(nmax_range, pmax_range))
                
            else:
                num_range = np.round(np.arange(min_val, max_val+0.1, bin_width), precision)
            
        return {i:f'[{num_range[i-1]} to {num_range[i]})' for i in range(1, len(num_range))}

    @staticmethod
    def rank_top_occurrences(df, ranking_col=None, top_n=3, min_count_allowed=1):
        """rank top n occurrences per given ranking column"""
        
        if not ranking_col:
            ranking_col = list(df.columns)[0]
        counts = df.value_counts().sort_values(ascending=False).reset_index()
        counts = counts.groupby(ranking_col).head(top_n)
        counts = counts.rename(columns={0:'total_count'})
        if min_count_allowed:
            counts = counts.loc[counts['total_count'] >= min_count_allowed]
        return counts.sort_values('total_count', ascending=False).reset_index(drop=True)
        
    @staticmethod
    def create_label_from_ranking(df, exclude_last_col=True):
    
        if 'total_count' in df.columns:
            df = df.drop('total_count', axis=1)
        ranking_cols = list(df.columns)
        if exclude_last_col:
            ranking_cols = list(df.iloc[:, :-1].columns)
        labe, n_cols = [], len(ranking_cols)
        for i in range(len(df)):
            row_labe = ''
            for j in range(len(ranking_cols)):
                if j == n_cols - 1:
                    row_labe += f"{ranking_cols[j]}: {df.iloc[i, j]}"
                    continue
                row_labe += f"{ranking_cols[j]}: {df.iloc[i, j]} &\n"
                
            labe.append(row_labe)
        return pd.Series(labe, name='combined_variables', index=df.index)

    @staticmethod
    def generate_aggregated_lookup(df, using_cols: list=None):
        """generate a lookup table using a subset of variables
        comprising a total accident count per category.
        Return:
        freq_lookup_df"""
        
        if not using_cols:
            using_cols = ('month', 'week_num', 'day_of_week', 'day_name', 'hour', 'day_num')
                                                     
        aggregate_df = df[using_cols].value_counts().sort_index().reset_index()
        aggregate_df.columns = aggregate_df.columns.astype(str).str.replace('0', 'total_count')
        return aggregate_df

    @staticmethod
    def get_correl_with_threshold(df: pd.DataFrame, thresh: float=0.5, report_only_colnames=False):
        """select only variables with correlation equal to or above the threshold value.
        Return:
        df_corr"""
        
        def count_thresh_corrs(row, thresh):
            """to count how many values in each row >= thresh"""
            
            count = 0
            for val in row:
                if abs(val) >= abs(thresh):
                    count += 1
            return count
        
        df = pd.DataFrame(df)
        if not isinstance(df, pd.DataFrame):
            raise TypeError('df must be a dataframe')
            
        print(f'Features with correlation of {round(100*thresh, 2)}% and above:\n')
        # number of values >= threshold value per row
        all_corrs = df.corr().round(4)
        selected_corr = all_corrs.apply(lambda val: count_thresh_corrs(val, thresh))
        correlated_vars = selected_corr[selected_corr > 1]
        if report_only_colnames:
            return list(correlated_vars.index)
        return all_corrs.loc[all_corrs.index.isin(list(correlated_vars.index)), list(correlated_vars.index)]
    
    @staticmethod
    def get_columns_with_pattern(df, search_pattern:'str or list', find_exact_match=False):
        """select from dataframe all columns containing the search pattern(s)."""
        
        if isinstance(search_pattern, str): # when search pattern is str
            if not find_exact_match:
                cols = [c for c in df.columns if str.lower(search_pattern) in c.lower()]
            else:
                cols = [c for c in df.columns if str.lower(search_pattern) == c.lower()]
        elif isinstance(search_pattern, list): # when list of search pattern is given
            cols = list()
            for c in df.columns:
                for search_for in search_pattern:
                    if not find_exact_match and str(search_for).lower() in c.lower():
                        cols.append(c)
                    elif find_exact_match and (str(search_for).lower() == c.lower()):
                        cols.append(c)
        return df[cols]
    
    @staticmethod
    def get_percentage(arr: pd.Series):
        """output the percentage of each element in array"""
        
        ser = pd.Series(arr)
        return np.round(100*ser/ser.sum(), 2)
        
    @staticmethod
    def percentage_per_class(df, freq_colname:str, catg_colname:str, precision=2):
        """compute percentage of freq_colname per catg_colname's category"""
        
        cols = [catg_colname, freq_colname]
        denom = df[cols].groupby(cols[0]).sum().rename(columns={cols[-1]:'denom'})
        calc = pd.merge(df[cols], denom, on=cols[0])
    #     display(calc)
        return pd.concat([df, np.round(calc.apply(lambda row: row[freq_colname]/row['denom'], axis=1), precision)],
                         axis=1).rename(columns={0:'perc'}).drop(freq_colname, axis=1)
        
    @staticmethod
    def median_absolute_deviation(x: 'array'):
      """compute the median absolute deviation.
      Return:
      MAD: integer"""

      return np.median(np.abs(x - np.median(x)))
    
    @staticmethod
    def hampel_filtering(timeseries: pd.Series, window_flank: int=5, n: int=3):
      """determine the median absolute deviation (MAD) outlier
      in timeseries data.
      :param timeseries: a pandas Series object containing time series data.
      :param window_flank: total window size will be 2*window_flank + 1
      :param n: threshold value with default 3 (Pearson's rule).
      Return:
      timeseries: corrected timeseries"""
      
      cls = DsUtils()
      
      if not isinstance(timeseries, pd.Series):
        raise TypeError('timeseries must be a pandas series object')
      
      if not isinstance(window_flank, int):
        raise TypeError('window_flank must be an integer')
      elif window_flank <= 0:
        raise ValueError('window_flank must be a positive number')
      
      if not isinstance(n, int):
        raise TypeError('n must be an integer')
      elif window_flank <= 0:
        raise ValueError('n must be a positive number')

      k = 1.4826
      ts_cleaned = timeseries.copy()

      ts_rolling = ts_cleaned.rolling(window=window_flank*2, center=True)
      ts_rolling_median = ts_rolling.median().fillna(method='bfill').fillna(method='ffill')
      ts_rolling_sigma = k*ts_rolling.apply(cls.median_absolute_deviation).fillna(method='bfill').fillna(method='ffill')

      outlier_indices = list(
            np.array(np.where(np.abs(ts_cleaned - ts_rolling_median) >= (n * ts_rolling_sigma))).flatten())
      ts_cleaned[outlier_indices] = ts_rolling_median[outlier_indices]

      return ts_cleaned
      
      def make_cols_lowercase(df):
        """convert column names to lowercase.
        Return: list of df columns in lowercase."""
        
        return list(map(str.lower, list(df.columns)))
        
    @staticmethod
    def placeholder_to_nan(df, current_nan_char: str=-1, new_placeholder=np.nan):
        """convert placeholders values to np.nan
        Return:
        df: dataframe with placeholder replaced by -1."""
        
        col_names = df.columns
        df2 = copy.deepcopy(df)
        for col in col_names:
    #         print(col)
            df2.loc[df2[col] == current_nan_char, col] = new_placeholder
        return df2
    
    @staticmethod
    def cast_categs_to_int(ser: pd.Series):
        """cast categories to integers
        ser: categorical series
        Return: integer-encoded series"""
        
        int_encoder = s_prep.LabelEncoder()
        return pd.Series(int_encoder.fit_transform(ser))
        
    @staticmethod
    def get_features_with_dtypes(df, feat_datatype: str='number'):
        """get a list of features with specified datatype in the dataframe.
        default output is numeric features.
        feat_datatype: options are number/int/float, object/str, bool, datetime/date/time
        Return: feat_list"""
        
        if not isinstance(df, pd.DataFrame):
            raise TypeError('df must be a dataframe')
            
        if not isinstance(feat_datatype, str):
            feat_datatype = eval(str(feat_datatype).split()[1][:-1])
        
        guide = {'str': 'object', 'int': 'number', 'float': 'number', 'date': 'datetime', 'time': 'datetime'}
        if feat_datatype in guide:
            use_datatype = guide[feat_datatype]
            #print(use_datatype)
            cols_selected = df.select_dtypes(include=use_datatype)
        else:
            cols_selected = df.select_dtypes(include=feat_datatype)
        
        
        if feat_datatype in ['int', 'float']:
            
            print(feat_datatype)
            col_dtypes = cols_selected.dtypes.astype(str).str.lower()
            return list(col_dtypes[col_dtypes.str.contains(feat_datatype)].index)
        #print('yes')
        return list(cols_selected.columns)
    
    @staticmethod
    def get_ttest_pvalue_from_array(X1_set, X2_set, 
                                alternative_hypothesis='two-sided', precision=None):
            """compute tstatistics and pvalue of two arrays X1 and X2.
            alternative_hypothesis: {'two-sided', 'less', 'greater'}
            Return
            result_dict: {'t_statistic':t, 'pvalue':p}"""
            
            def check_equal_var(var1, var2):
                """determine whether to use student (equal variance) 
                or Welch (unequal variance) t-test.
                equal variance = (larger variance/smaller variance < 4) or (larger std/smaller std < 2)
                unequal variance = (larger var/smaller var >= 4)"""
                
                top, bot = max([var1, var2]), min([var1, var2])
                return ((top+1)/(bot+1)) < (4+1) # add 1 to cancel out ZeroDiv
            
            var1, var2 = np.var(X1_set, ddof=1), np.var(X2_set, ddof=1)

            if len(np.shape(X1_set)) > 1: # multiple features
                ev = [check_equal_var(ix1, ix2) for ix1, ix2 in zip(var1, var2)]
            else:
                ev = check_equal_var(var1, var2)
                if ev:
                    print("\nEqual Variance Detected! -> Student T-Test\n")
                else:
                    print("\nUnequal Variance Detected! -> Welch Test:\n")
            if not precision:
                t, p = ttest_ind(X1_set, X2_set, equal_var=ev, alternative=alternative_hypothesis)
            else:
                t, p = np.round(ttest_ind(X1_set, X2_set, equal_var=ev, 
                                          alternative=alternative_hypothesis), precision)
            return {'t_statistic':t, 'pvalue':p}

    @staticmethod
    def get_ttest_pvalue_from_stats(X1_mean, X1_std, X1_count, X2_mean, X2_std, X2_count,
                                    alternative_hypothesis='two-sided', precision=None):
            """compute tstatistics and pvalue of two arrays X1 and X2 from the mean,
            std, and sample size.
            alternative_hypothesis: {'two-sided', 'less', 'greater'}
            Return
            result_dict: {'t_statistic':t, 'pvalue':p}"""
            
            def check_equal_var(var1, var2):
                """determine whether to use student (equal variance) 
                or Welch (unequal variance) t-test.
                equal variance = (larger variance/smaller variance < 4) or (larger std/smaller std < 2)
                unequal variance = (larger var/smaller var >= 4)"""
                
                top, bot = max([var1, var2]), min([var1, var2])
                return (top/bot) < 4
            
            ev = check_equal_var(X1_std, X2_std)
            if ev:
                print("\nEqual Variance Detected! -> Student T-Test\n")
            else:
                print("\nUnequal Variance Detected! -> Welch Test:\n")
            if not precision:
                t, p = ttest_ind_from_stats(mean1=X1_mean, std1=X1_std, nobs1=X1_count,
                                            mean2=X2_mean, std2=X2_std, nobs2=X2_count,
                                            equal_var=ev, alternative=alternative_hypothesis)
            else:
                t, p = np.round(ttest_ind_from_stats(mean1=X1_mean, std1=X1_std, nobs1=X1_count,
                                            mean2=X2_mean, std2=X2_std, nobs2=X2_count,
                                            equal_var=ev, alternative=alternative_hypothesis), altprecision)
            return {'t_statistic':t, 'pvalue':p}
       
    @staticmethod
    def remove_val_ind(X, y, predictors: list=None, remove_val=np.nan):
        """remove indexes where there is a remove_val character"""
        
        null_inds = []
        if predictors is None:
            predictors = list(X.columns)
        for col in predictors:
            null_inds.extend(X.loc[X[col] == remove_val].index)
        null_inds = set(null_inds)
        X_new = X.loc[~X.index.isin(null_inds)]
        y_new = y.loc[~y.index.isin(null_inds)]
        return X_new, y_new
    
    @staticmethod
    def test_hypotheses(selected_data, agg_cols=None, focus_col=None, bigger_set_name='X1', smaller_set_name='X2',
                        bigger_set_vals=None, smaller_set_vals=None, second_condition_col=None, second_condition_val=None,
                        third_condition_col=None, third_condition_val=None, balance_unequal=True):
        """run hypothesis test on selected data for focus_col's bigger_set_name > focus_col's smaller_set_name"""
        
        cls = DsUtils()
        if not agg_cols:
            agg_cols = ['hour']
        a_cols = [str.lower(c) for c in agg_cols if c != 'total_count']
        agg_df = cls.generate_aggregated_lookup(selected_data, include_cols=a_cols)
        x_cols = a_cols + ['total_count']
        display(x_cols)
        # total df
        total_acc = agg_df[x_cols].groupby(a_cols).sum().reset_index()
        display(total_acc)
        bigger_set = cls.get_from_df(total_acc, col1=focus_col, col1_is=bigger_set_vals, 
                                             col2=second_condition_col, col2_is=second_condition_val,
                                            col3=third_condition_col, col3_is=third_condition_val)
        print(f"{bigger_set_name}:")

        display(bigger_set)
        smaller_set = cls.get_from_df(total_acc, col1=focus_col, col1_is=smaller_set_vals,
                                             col2=second_condition_col, col2_is=second_condition_val,
                                            col3=third_condition_col, col3_is=third_condition_val)
        print("\n>\n")
        print(f"{smaller_set_name}:")
        display(smaller_set)
        
        cls.report_a_significance(bigger_set['total_count'], smaller_set['total_count'],
                                   X1_name=bigger_set_name, X2_name=smaller_set_name,
                                  balance=balance_unequal)
        
    @staticmethod
    def report_a_significance(X1_set, X2_set, n_deg_freedom=1, X1_name='X1', X2_name='X2', seed=None, balance=True):
        """Test for statistical significant difference between X1_set and X2_set
        at 99% and 95% Confidence.
        X1_set: 1D array of observations
        X2_set: 1D array of observations."""
        
        cls = DsUtils()
        
        def get_min_denom(n1, n2):
            return min([n1, n2])
        
        def detect_unequal_sizes(n1, n2):
            """check if both lengths are not the same"""
            return n1 != n2
        
        # to ensure reproducibility
        if not seed:
            seed = 1
        np.random.seed(seed)
        
        samp_sizes = {X1_name: pd.Series(X1_set), 
                      X2_name: pd.Series(X2_set)}
        
        print(f'\n\nHYPOTHESIS TEST FOR:\n{X1_name} > {X2_name}\n')
        
        # use to compare single values
        if len(samp_sizes[X1_name]) == 1:
            total_X1, X1_mean, X1_std = samp_sizes[X1_name].iloc[0], samp_sizes[X1_name].iloc[0], samp_sizes[X1_name].iloc[0]**0.5
            total_X2, X2_mean, X2_std = samp_sizes[X2_name].iloc[0], samp_sizes[X2_name].iloc[0], samp_sizes[X2_name].iloc[0]**0.5
        else:
            X1_size, X2_size = len(X1_set), len(X2_set)
            print(f'ORIGINAL SAMPLE SIZE: \n{X1_name}: {X1_size}\n' +
                  f'{X2_name}: {X2_size}\n\n')
            
            # check if sample sizes are unequal
            if detect_unequal_sizes(X1_size, X2_size):
                print("Unequal Sample Sizes Detected!!\n")
                if balance:
                    print("\n....DOWNSAMPLING RANDOMLY....\n")
                    min_size = get_min_denom(X1_size, X2_size)
                    max_samp_name = [name for name, val in samp_sizes.items() if len(val) != min_size][0]
                    # downsampling: 
                    # randomly generate min_size indexes for max_samp_name
                    rand_indexes = cls.index_generator(len(samp_sizes[max_samp_name]), min_size, random_state=seed)
                    # select only random min_size indexes for max_samp_name set
                    samp_sizes[max_samp_name] = samp_sizes[max_samp_name].iloc[rand_indexes]
                    X1_size, X2_size = len(samp_sizes[X1_name]), len(samp_sizes[X2_name])
                    print(f'ADJUSTED SAMPLE SIZE: \n{X1_name}: {X1_size}\n' +
                          f'{X2_name}: {X2_size}\n\n')
            total_X1, X1_mean, X1_std = cls.compute_mean_std(samp_sizes[X1_name], X1_size, n_deg_freedom)
            total_X2, X2_mean, X2_std = cls.compute_mean_std(samp_sizes[X2_name], X2_size, n_deg_freedom)
        
        null_hypo = np.round(X1_mean - X2_mean, 4)
        pooled_std = cls.compute_pstd(X1_std, X2_std)

        print(f'{X1_name}:\n Total = {total_X1}\n Average = {X1_mean}\n Standard deviation = {X1_std}\n\n' +
              f'{X2_name}:\n Total = {total_X2}\n Average = {X2_mean}\n Standard deviation = {X2_std}\n\n' +
              f'MEAN DIFFERENCE = {null_hypo}\n' +
              f'POOLED STD = {pooled_std}\n\n')
        
        print(f'HYPOTHESIS TEST:\nIs {X1_mean} significantly HIGHER THAN {X2_mean}?\n' +
             f'BASED ON the chosen level of significance\nIs the difference {null_hypo} > 0?\n')

        # check for both 99% and 95% confidence level
        # Meaning, is the difference between both figures greater than 
        # 3 pooled std and 2 pooled std respectively

        alpha = 0.01
        test_result = cls.compute_test(pooled_std, alpha)
        if null_hypo > test_result[1]:
            return print(f'At {test_result[0]}, REJECT the null hypothesis!\n {null_hypo} is greater than {test_result[1]}\n')
        else:
            test_result = cls.compute_test(pooled_std)
            if null_hypo > test_result[1]:
                return print(f'At {test_result[0]}, REJECT the null hypothesis!\n{null_hypo} is greater than {test_result[1]}\n')
        print(f'Do NOT reject the null hypothesis\n{null_hypo} is less than or equal to {test_result[1]}')
        
    @staticmethod
    def calc_deg_freedom(denom, n_deg):
        """compute degrees of freedom."""
        
        return denom - n_deg
        
    @staticmethod
    def compute_mean_std(arr, denom, n_deg):
        """compute sum, mean, stdev of array using 
        the given denominator and degrees of freedom"""
        
        cls = DsUtils()
        
        total = np.sum(arr)
        avg = np.round(total/denom, 4)
        deg_freedom = cls.calc_deg_freedom(denom, n_deg)
        sumsq = np.sum((arr - avg)**2)
        stdv = np.sqrt(sumsq/deg_freedom).round(4)
        return (total, avg, stdv)
        
    @staticmethod
    def compute_pstd(stdv_1, stdv_2):
        """Compute pooled standard devs from two stdevs"""
        
        return round(np.sum([stdv_1**2, stdv_2**2])**0.5, 4)
    
    @staticmethod
    def compute_test(pooled_stdv, at_alpha=0.05):
        """Compute test sigma at specified significance with pooled_std.
        at_alpha: significance
        pooled_std: pooled standard deviation
        Return: (confidence, test_sigma)"""
        
        sig_to_conf = {0.05: (2, '95% confidence'),
                      0.01: (3, '99% confidence')}
        test_sigma = round(sig_to_conf[at_alpha][0] * pooled_stdv, 4)
        return (sig_to_conf[at_alpha][1], test_sigma)
    
    @staticmethod
    def index_generator(sample_size: 'array', n_index=1, random_state=1):
        """Randomly generate n indexes.
        :Return: random_indexes"""

        import random

        def select_from_array(sample_array, n_select=1, random_state=1):
            np.random.seed(random_state)
            return random.choices(population=sample_array, k=n_select, random_state=random_state)
        
        indexes = range(0, sample_size, 1)

        return select_from_array(indexes, n_index, random_state)
        
    @staticmethod
    def compute_pvalue(x1, x2, h1, seed):
        """compute test_statistic and pvalue of null hypothesis.
        h1: {'greater', 'less', 'two-sided'}
        Returns:
        statistic : float or array
            The calculated t-statistic.
        pvalue : float or array
            The two-tailed p-value."""
    
    #     print(help(stats.ttest_ind))
        np.random.seed(seed)
        rng = np.random.default_rng
        
        tstat, pval = np.round(stats.ttest_ind(x1, x2, equal_var=False, alternative=h1, random_state=rng), 4)
        return tstat, pval
        
    @staticmethod
    def compute_stat_pval(x1, x2, h1='greater',  X1_name='X1', X2_name='X2', deg_freedom=1, seed=1):
        """report statistical significance using pvalue"""
    
        
        def get_min_denom(n1, n2):
                return min([n1, n2])
            
        def detect_unequal_sizes(n1, n2):
            """check if both lengths are not the same"""
            return n1 != n2
        
        cls = DsUtils()

        np.random.seed(seed)
        X1_size, X2_size = len(x1), len(x2)
        samp_sizes = {X1_name: pd.Series(x1),
                      X2_name: pd.Series(x2)}
        print(f'ORIGINAL SAMPLE SIZE: \n{X1_name}: {X1_size}\n' +
              f'{X2_name}: {X2_size}\n\n')

        # check if sample sizes are unequal
        if detect_unequal_sizes(X1_size, X2_size):
            print("Unequal Sample Sizes Detected!!")
            min_size = get_min_denom(X1_size, X2_size)
            max_samp_name = [name for name, val in samp_sizes.items() if len(val) != min_size][0]
            # downsampling: 
            # randomly generate min_size indexes for max_samp_name
            rand_indexes = cls.index_generator(len(samp_sizes[max_samp_name]), min_size)
            # select only random min_size indexes for max_samp_name set
            samp_sizes[max_samp_name] = samp_sizes[max_samp_name].iloc[rand_indexes]
            X1_size, X2_size = len(samp_sizes[X1_name]), len(samp_sizes[X2_name])
            print(f'ADJUSTED SAMPLE SIZE: \n{X1_name}: {X1_size}\n' +
                  f'{X2_name}: {X2_size}\n\n')
            
        total_X1, X1_mean, X1_std = cls.compute_mean_std(samp_sizes[X1_name], X1_size, deg_freedom)
        total_X2, X2_mean, X2_std = cls.compute_mean_std(samp_sizes[X2_name], X2_size, deg_freedom)

        null_hypo = np.round(X1_mean - X2_mean, 4)
        pooled_std = cls.compute_pstd(X1_std, X2_std)

        print(f'{X1_name}:\n Total = {total_X1}\n Average = {X1_mean}\n Standard deviation = {X1_std}\n\n' +
              f'{X2_name}:\n Total = {total_X2}\n Average = {X2_mean}\n Standard deviation = {X2_std}\n\n' +
              f'MEAN DIFFERENCE = {null_hypo}\n' +
              f'POOLED STD = {pooled_std}\n\n')
        
        tstat, pval = cls.compute_pvalue(samp_sizes[X1_name], samp_sizes[X2_name], h1, seed)
        test_result = (['99%', 0.01], 
                       ['95%', 0.05])
        if pval <= test_result[0][1]:
            return print(f'At {test_result[0][0]}, REJECT the null hypothesis!\n {pval} is less than {test_result[0][1]}\n')
        elif pval <= test_result[1][1]:
                return print(f'At {test_result[1][0]}, REJECT the null hypothesis!\n{pval} is less than or equal to {test_result[1][1]}\n')
        print(f'Do NOT reject the null hypothesis\n{pval} is greater than {test_result[1][1]}')
    
    @staticmethod
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
        
    @staticmethod
    def get_correlations(df: pd.DataFrame, y_col: 'str or Series'=None, x_colnames: list=None, precision=2):
        """get %correlations between df features and y_col.
        And return ABSOLUTE values of correlations.
        Return: xy_corrwith (sorted absolute values)"""
        
        if not isinstance(df, (pd.Series, pd.DataFrame)):
            raise ValueError("df must be either a dataframe or series")
            
        if not isinstance(y_col, (str, pd.Series)):
            raise ValueError("y_col must be either str or series")
            
        if isinstance(y_col, str):
            y_col = df[y_col]
        if isinstance(df, pd.Series):
            df = pd.DataFrame(df)
        
        if x_colnames:    
            x = pd.DataFrame(df[x_colnames])
        else:
            x = df
        if not y_col:
            return x.corr().apply(lambda col: round(100*col, precision))
        return x.corrwith(y_col).abs().sort_values(ascending=False).apply(lambda x: 100*x).round(precision)

    @staticmethod
    def corr_with_pearson(X, y, include_direction=False, scale_up=100, precision=2, top_n=None):
        """compute Pearson correlation for outcome y with X variables"""
        
        X, y = pd.DataFrame(X), pd.Series(y)
        if include_direction:
            return np.round(scale_up * X.corrwith(y), precision).sort_values(ascending=False).iloc[:top_n]
        return np.round(scale_up * X.corrwith(y).abs(), precision).sort_values(ascending=False).iloc[:top_n]
    
    @staticmethod
    def corr_with_kbest(X, y):
        """using sklearn.preprocessing.SelectKBest, quantify correlations
        between features in X and y.
        Return: correlation series"""
        
        X = pd.DataFrame(X)
        selector = s_fs.SelectKBest(k='all').fit(X, y)
        return pd.Series(selector.scores_, index=selector.feature_names_in_).sort_values(ascending=False)

    @staticmethod
    def calc_days_between(historic_date: str, later_date: str):
        """compute, in days, the difference between two given dates.
        Date difference is later_date - historic_date"""
        
        if not isinstance(historic_date, (np.datetime64, str)) or not isinstance(later_date, (np.datetime64, str)):
            raise TypeError("date given must be in str or np.datetime64 types")
        
        historic_date, later_date = np.datetime64(historic_date), np.datetime64(later_date)
        
        return (later_date - historic_date).astype(int)
        
        
    @staticmethod
    def split_time_series(time_col: 'datetime series', sep=':'):
        """split series containing time data in str format into
        dataframe of three columns (hour, minute, seconds)
        Return:
        Dataframe"""
        
        time_col = time_col.astype(np.datetime64)
        hr, minute, secs = time_col.dt.hour, time_col.dt.minute, time_col.dt.second
        return pd.DataFrame({'Hour': hr,
                            'Minute': minute,
                            'Seconds': secs})
    @staticmethod
    def split_date_series(date_col, sep='/', year_first=True):
        """split series containing date data in str format into
        dataframe of tdayee columns (year, month, day)
        Return:
        Dataframe"""
        
        date_col = date_col.str.split(sep, expand=True)
        if year_first:
            day = date_col[2]
            mon = date_col[1]
            yr = date_col[0]
        else:
            day = date_col[0]
            mon = date_col[1]
            yr = date_col[2]
        
        return pd.DataFrame({'Year': yr,
                            'Month': mon,
                            'Day': day})
        
    @staticmethod
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

    @staticmethod
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
               
    @staticmethod
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

    @staticmethod
    def select_top_five_threshold(association_rules_df, support_thresh=0.5, confidence_thresh=0.5):
        """pick the top 5 above threshold value."""
        
        return association_rules_df.loc[(association_rules_df['support'] >= support_thresh) &
                                        (association_rules_df['confidence'] >= confidence_thresh)].sort_values(['support', 'confidence'], ascending=False).iloc[:5]

    @staticmethod
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
        
    @staticmethod
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
        
    @staticmethod
    def classification_metrics(y_true, y_pred, target_label=1, for_both_classes=False):
        """Calculate the accuracy, precision, and recall, f1_score values
        considering the expected values and predicted values.
        if for_both_classes is True, return each performance metric 
        for class 0 and 1, otherwise return for only class 1
        target_label is label of focus
        Returns
        performance metrics dict"""
        
        result = dict()
        using = 'binary'
        if for_both_classes:
            using = None
        result['recall'] = np.round(s_mtr.recall_score(y_true, y_pred, pos_label=target_label, average=using), 4)
        result['precision'] = np.round(s_mtr.precision_score(y_true, y_pred, pos_label=target_label, average=using), 4)
        result['f1_score'] = np.round(s_mtr.f1_score(y_true, y_pred, pos_label=target_label, average=using), 4)
        result['accuracy'] = np.round(s_mtr.accuracy_score(y_true, y_pred), 4)
        
        return result
        
    @staticmethod
    def report_with_conf_matrix(y_true, pred):
        """print classification report and return the corresponding
        confusion matrix.
        Return: confusion_matrix_plot"""
        
        print(s_mtr.classification_report(y_true, pred))
        
        sns.set_style('white')
        ax1 = s_mtr.ConfusionMatrixDisplay.from_predictions(y_true, pred)
        plt.title("Confusion Matrix", weight='bold')
        return ax1
        
    @staticmethod
    def visualize_report_binclf(trained_model, test_input, test_label, is_ANN=False, ANN_cutoff=0.5):
        """performance report for trained model"""
        
        test_pred = trained_model.predict(test_input)
        if is_ANN:
            test_pred = (test_pred >= ANN_cutoff).astype(int)
        
        print(s_mtr.classification_report(test_label, test_pred))
        
        sns.set_style('white')
        plt.figure(figsize=(6, 5), dpi=200)
        s_mtr.ConfusionMatrixDisplay.from_predictions(test_label, test_pred)
        
    @staticmethod
    def tree_feat_weights(model, col_names):
        """present the individual weights assigned to each columns
        by decision tree model"""
        
        return pd.Series(data=model.feature_importances_, index=col_names, name='feature_weights').sort_values(ascending=False)
        
    @staticmethod
    def np_value_counts(arr):
        """count the number of unique values in a 1D numpy array."""
        
        unq_ele = np.unique(arr)
        return {unq_ele[i]: len(np.where(arr == unq_ele[i])[0]) for i in range(len(unq_ele))}
        
    @staticmethod
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

    @staticmethod
    def float_to_binary(val: float):
        """
        transform decimal to 0 or 1 if val >= 0.5
        :param val:
        :return:
        """
        standard = 0.4444444444444445
        return 1 if val >= standard else 0

    @staticmethod
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

    @staticmethod
    def get_month(val: str, in_words: str = False):
        """change datatype of month from words to integer (vice versa)
        """
        
        if not isinstance(val, (int, str)):
            raise TypeError('val must be int or str')
        
        if isinstance(val, int):
            if (0 > val) or (val > 12):
                raise ValueError('val must be between 1 and 12')
        
        guide = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                 'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
        
        if isinstance(val, str):
            if val.capitalize() not in guide.keys():
                raise ValueError('val is wrongly spelt')

        def convert_to_words(val):
            return dict([(v, k) for k, v in guide.items()])[val]

        def convert_to_int(val):
            for k, v in guide.items():
                if val.lower() in k.lower():
                    return v

        if in_words:
            return convert_to_words(val)
        return convert_to_int(val)

    @staticmethod
    def impute_null_values(df: 'pd.DataFrame', pivot_cols: list, target_col: str, stat_used: str='mean'):
        """Impute the null values in a target feature using aggregated values
        (eg mean, median, mode) based on pivot features
        :param df:
        :param pivot_cols:
        :param target_col:
        :param stat_used: {'mean', 'mode', 'median'}
        :return: impute_guide: 'a pandas dataframe'"""
        
        cls = DsUtils()
        
        if ('object' in str(df[target_col].dtypes)) or ('bool' in str(df[target_col].dtypes)):
            stat_used = 'mode'

        if str.lower(stat_used) == 'mean':
            impute_guide = cls.pivot_mean(df, pivot_cols, target_col)
        elif str.lower(stat_used) == 'mode':
            impute_guide = cls.pivot_mode(df, pivot_cols, target_col)
        elif str.lower(stat_used) == 'median':
            impute_guide = cls.pivot_median(df, pivot_cols, target_col)
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
    
    @staticmethod
    def pivot_mode(df: 'pd.DataFrame', pivot_cols: list, target_col: str):
        """rank the occurrences of target_col values based on pivot_cols,
        and return the mode (i.e the highest occurring) target_col
        value per combination of pivot_cols.
        Return:
        impute_guide: lookup table"""
        
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Object given is not a pandas dataframe")
        if not isinstance(pivot_cols, list):
            raise TypeError("pivot columns should be in list")
        if not isinstance(target_col, str):
            raise TypeError("Target column name must be a string")

        freq_df = df.loc[df[target_col].notnull(), pivot_cols + [target_col]].value_counts().reset_index()
        return freq_df.drop_duplicates(subset=pivot_cols).drop(labels=[0], axis=1)
        
    @staticmethod
    def pivot_mean(df: 'pd.DataFrame', pivot_cols: list, target_col: str):
        """rank the occurrences of target_col values based on pivot_cols,
        and return the average target_col value per combination
        of pivot_cols.
        Return:
        impute_guide: lookup table"""
        
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
        
    @staticmethod
    def pivot_median(df: 'pd.DataFrame', pivot_cols: list, target_col: str):
        """rank the occurrences of target_col values based on pivot_cols,
        and return the median target_col value per combination
        of pivot_cols.
        Return:
        impute_guide: lookup table"""
        
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
        
        freq_df = np.round(df.loc[df[target_col].notnull(), pivot_cols + [target_col]].groupby(by=pivot_cols).median().reset_index(), dec_places)
        return freq_df.drop_duplicates(subset=pivot_cols)

    @staticmethod
    def dataset_split(x_array: 'np.ndarray', y_array: 'np.ndarray', perc_test=0.25, perc_val=None):
        """create train, validation, test sets from x and y dataframes
        Returns:
        if  val_len != 0:  # perc_val is not 0
            (x_train, x_val, x_test, y_train, y_val, y_test)
        else:  # perc_val is 0 (no validation set)
            (x_train, x_test, y_train, y_test)"""

        if not (isinstance(x_array, np.ndarray) or not isinstance(y_array, np.ndarray)):
            raise TypeError("x_array/y_array is not np.ndarray")

        nx, ny = len(x_array), len(y_array)

        if nx != ny:
            raise ValueError("x_array and y_array have unequal number of samples")
            
        if perc_val is None:
            perc_val = 0

        # number of samples for each set
        combo_len = int(nx * perc_test)
        train_len = int(nx - combo_len)
        val_len = int(combo_len * perc_val)
        test_len = combo_len - val_len

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
        train_ind = inds[: train_len]
        combo_ind = inds[train_len:]
        
        test_ind = combo_ind[: test_len]
        val_ind = combo_ind[test_len: ]
            
        x_test, y_test = x_array[test_ind], y_array[test_ind]
        x_val, y_val = x_array[val_ind], y_array[val_ind]
        x_train, y_train = x_array[train_ind], y_array[train_ind]

        return (x_train, x_val, x_test, y_train, y_val, y_test) if val_len else (x_train, x_test, y_train, y_test)

    @staticmethod
    def classify_age(row):
        age_guide = {(0, 9): '0 to 9', (10, 19): '10 to 19',
                     (20, 29): '20 to 29', (30, 39): '30 to 39',
                     (40, 49): '40 to 49', (50, 59): '50 to 59',
                     (60, 69): '60 to 69', (70, 79): '70 to 79',
                     (80, 89): '80 to 89'}

        for bracket, age_cls in age_guide.items():
            if (row >= bracket[0]) and (row <= bracket[1]):
                return age_cls

    @staticmethod
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

    @staticmethod
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
            
    @staticmethod
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

    @staticmethod
    def round_up_num(decimal: str):
        """round up to nearest whole number if preceding number is 5 or greater."""

        whole, dec = str(float(decimal)).split('.')
        return str(int(whole) + 1) if int(dec[0]) >= 5 or (int(dec[0]) + 1 >= 5 and int(dec[1]) >= 5) else whole

    @staticmethod
    def transform_val(ser: sns.categorical.pd.Series, guide: dict):
        """Change values in a series from one format to new format specified in the guide dictionary."""

        return ser.apply(lambda val: str(guide[val]) if val in guide.keys() else val)

    @staticmethod
    def unique_categs(df: sns.categorical.pd.DataFrame):
        """return unique values per dataframe column.
        Return:
        unique_vals_per_col: dictionary containing each column name and unique values."""

        cols = tuple(df.columns)
        uniq_vals = dict()
        for i in range(len(cols)):
            uniq_vals[cols[i]] = list(df[cols[i]].unique())
        return uniq_vals
        
    @staticmethod
    def make_cols_lowercase(df):
        """convert column names to lowercase.
        Return: 
        col_list: list of df columns in lowercase."""
        
        return list(map(str.lower, list(df.columns)))

    @staticmethod
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

    @staticmethod
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
        os.makedirs(graph_folder, exist_ok=True)  # folder created
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

    @staticmethod
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

    @staticmethod
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