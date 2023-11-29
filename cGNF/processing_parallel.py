import pandas as pd
import numpy as np
import collections.abc
collections.Iterable = collections.abc.Iterable
import torch  # Importing the PyTorch library, which provides tools for deep learning.
import pickle  # Importing the pickle module, which allows to serialize and deserialize Python object structures.
import networkx as nx
from causalgraphicalmodels import CausalGraphicalModel


def process(path="", dataset_name="", dag_name='DAG', sens_corr= None, test_size=0.2, cat_var=None, seed=None):

    # Read the data file.
    df = pd.read_csv(path + dataset_name + '.csv')

    ordered_columns = df.columns.tolist()  # Get the column names from the DataFrame

    # Try to read the DAG CSV file.
    df_cDAG = pd.read_csv(path + dag_name + '.csv', index_col=0)

    df_cDAG = df_cDAG.reindex(index=ordered_columns, columns=ordered_columns)  # Reorder both rows and columns

    print(df_cDAG)


    num_vars = len(df.columns)
    corr_matrix = np.eye(num_vars)

    # If edge strengths are provided, update the strength matrix.
    if sens_corr:
        # Create a mapping of column name to index to use in the matrix.
        col_to_index = {col: idx for idx, col in enumerate(df.columns)}

        # Update the corr_matrix with the specified strengths.
        for (source, target), strength in sens_corr.items():
            if source in col_to_index and target in col_to_index:
                idx_source = col_to_index[source]
                idx_target = col_to_index[target]

                # Set the corresponding values in the matrix, and its symmetric counterpart.
                corr_matrix[idx_source][idx_target] = strength
                corr_matrix[idx_target][idx_source] = strength

        # Save the updated correlation matrix
        pd.DataFrame(corr_matrix, index=df.columns, columns=df.columns).to_csv(
            path + 'sens_corr_matrix_name.csv')

    # Imports the function 'train_test_split' from sklearn's model selection module.
    from sklearn.model_selection import train_test_split
    # Splits the DataFrame 'df' into a training set and a validation set. This function returns two dataframes: the training set and the validation set.
    df_train, df_val = train_test_split(df, test_size= test_size, random_state=seed)

    # Converts the training and validation datasets from pandas DataFrame to numpy arrays. These arrays will be used for further data processing.
    df_trn, df_val = df_train.to_numpy(), df_val.to_numpy()

    # Vertically stacks the training and validation arrays into one array. This combined array is used for calculating the mean and standard deviation for data standardization.
    data = np.vstack((df_trn, df_val))

    mu = data.mean(axis=0) # Calculates the mean of 'data' along the column axis. This mean will be used for data standardization.
    sig = data.std(axis=0) # Calculates the standard deviation of 'data' along the column axis. This standard deviation will be used for data standardization.

    all_df_columns = list(df)  # get the list of all the variable names in the dataset
    dict_cat_dims = {}  # Initialize an empty dictionary 'dict_cat_dims' to store the maximum value of each categorical column + 1, which essentially reflects the number of unique categories.

    # Prepare to dequantize
    if cat_var:
        # loop to change each column to category type to get the dimension/position of the categorical variable in the dataframe and respective unique categories.
        dict_unique_cats = {}  # Initialize an empty dictionary 'dict_unique_cats' to store unique categories for each categorical column.

        # The enumerate function is used when you want to iterate over an iterable and also want to have an index attached to each element.
        for i, col in enumerate(all_df_columns):  # Loop over each column in the DataFrame. 'i' is the index and 'col' is the column name.
            if col in cat_var:  # Check if the column 'col' is in the list of categorical column names 'cat_col_names'.
                df[col] = df[col].astype('category', copy=False)  # If so, convert that column to 'category' type. This is often used to save memory or to perform some pandas operations faster.
                dict_unique_cats[col] = list(df[col].unique())  # Add the unique categories of the column 'col' to the dictionary 'dict_unique_cats'.
                # dict_cat_dims[i] = len(dict_unique_cats[col])
                dict_cat_dims[i] = max(dict_unique_cats[col]) + 1  # Instead, it sets the value in 'dict_cat_dims' for the key 'i' to one more than the maximum category of column 'col'. This assumes that the categories are numerical and can be ordered.


    # A dictionary 'pickle_objects' is created to hold the necessary preprocessed variables.
    # These will later be saved into a pickle file for easy reloading in future sessions.
    pickle_objects = {}

    # Adding various data and information to the dictionary.
    pickle_objects['df'] = df  # The entire DataFrame.
    pickle_objects['trn'] = df_trn  # The training data.
    pickle_objects['val'] = df_val  # The validation data.
    pickle_objects['mu'] = mu  # The column-wise mean of the data.
    pickle_objects['sig'] = sig  # The column-wise standard deviation of the data.
    pickle_objects['df_all_columns'] = all_df_columns  # All column names of the DataFrame.
    pickle_objects['df_cat_columns'] = cat_var  # The categorical column names.
    pickle_objects['cat_dims'] = dict_cat_dims  # The dimensions of the categorical columns.
    pickle_objects['seed'] = seed  # The random seed.
    pickle_objects['dataset_filepath'] = path + dataset_name  # The file path of the dataset.
    pickle_objects['A'] = torch.from_numpy(df_cDAG.to_numpy().transpose()).float()  # The adjacency matrix of the causal graph, converted to a PyTorch tensor.
    pickle_objects['Z_Sigma'] = torch.from_numpy(corr_matrix).float()

    # The context manager 'with open' is used to open a file in write-binary mode ('wb').
    # The pickle.dump function is then used to write the 'pickle_objects' dictionary to this file.
    with open(path + dataset_name + '.pkl', "wb") as f:
        pickle.dump(pickle_objects, f)

