# causal-Graphical Normalizing Flows (cGNF)

## About c-GNF

The causal-Graphical Normalizing Flows (c-GNF) project primarily focuses on causal questions addressed through the use of normalizing flows. This project incorporates deep learning components from Graphical Normalizing Flows (GNFs) and Unconstrained Monotonic Neural Networks (UMNNs). 

The method assumes a causal structure for the data in the form of a Directed Acyclic Graph (DAG). It then learns a normalizing flow for the joint distribution of the data based on this causal structure. This flow is inverted to perform hypothetical interventions and then simulate potential outcomes.

---

### Work Flow

1. **Assume a causal structure**: specify a DAG for the dataset.
   
2. **Learn a normalizing flow**: fit a deep neural net to model the joint distribution of the data, given the assumed causal structure.
   
3. **Invert the normalizing flow**: for counterfactual inference.
   
4. **Intervene on an exposure**: set a parent variable to a new value to study its effect on a child variable.
   
5. **Simulate potential outcomes**: sample from a standard normal distribution and then pipe these samples through the inverted flow to simulate counterfactuals.

### Deep Learning Models

For a more comprehensive understanding of the deep learning models used by cGNF, please refer to the original works on GNF and UMNN linked below:

- [GNF](https://github.com/AWehenkel/Graphical-Normalizing-Flows) for graphical normalizing flows
- [UMNN](https://github.com/AWehenkel/UMNN) for unconstrained monotonic neural networks

---

## User Guide

This guide walks you through setting up the environment and utilizing `cGNF` model to analyze your own dataset. For users who are new to Python, it is recommended to follow the instructions step by step. Experienced Python users can directly go to [Installing Necessary Packages](#installing-necessary-packages) section to download the `cGNF` modules, and then skip to [Setting up Dataset](#setting-up-dataset).

### Tutorial Contents

1. [Setting up a Python Environment](#setting-up-a-python-environment)
2. [Installing Necessary Packages](#installing-necessary-packages)
3. [Setting up Jupyter Lab](#setting-up-jupyter-lab)
4. [Setting up Dataset](#setting-up-dataset)
5. [Running the Model](#running-the-model)
6. [Simulation Tutorials](#simulation-tutorials)

---

## Setting up a Python Environment

1. **Install Conda**

    Choose between **Anaconda** and **Miniconda**:
    
    - **Anaconda**: A comprehensive distribution of Python and R for scientific computing and data science. Comes with many libraries pre-installed.
      - [Download Anaconda](https://www.anaconda.com/products/distribution)
      
    - **Miniconda**: A minimal installer for Conda. Only install what you need, making it lightweight and suitable for servers or minimal environments.
      - [Download Miniconda](https://docs.conda.io/en/latest/miniconda.html)

2. **Initialize Conda for Shell**

    Ensure Conda is initialized and available in your shell :
    
    ```bash
    conda init
    ```
   _Note_: _all commands before the **Setting up Dataset** section are in shell_

3. **Navigate to Your Workspace**

    Change directory to where you want to set up the environment:
    
    ```bash
    cd /your_working_directory/your_folder/
    ```

4. **Create a New Conda Environment**

    Set up a new Conda environment in your chosen directory:
    
    ```bash
    conda create --prefix=/your_working_directory/your_folder/your_env_name
    ```

5. **Activate the Environment**

    Always activate the environment before installing any packages or running any Python code:
    
    ```bash
    conda activate /your_working_directory/your_folder/your_env_name
    ```

---

## Installing Necessary Packages

1. **Install `cGNF` Base Modules**

   Install the baseline modules for `cGNF` from the specified GitHub repository:

   ```bash
   pip install git+https://github.com/JesseZhou-1/cGNF.git
   ```

2. **Install PyTorch and Related Libraries**

   If you need CUDA support, make sure to specify the correct version:

   ```bash
   conda install pytorch torchvision torchaudio cudatoolkit=<desired_version, e.g., 11.2> -c pytorch
   ```

3. **Setup Jupyter Lab**

   Install Jupyter Lab and other related tools:

   ```bash
   pip install jupyterlab ipykernel
   ```

---

## Setting up Jupyter Lab

1. **Activate the Environment**

   Always activate the environment before launching Jupyter Lab:

   ```bash
   conda activate /your_working_directory/your_folder/your_env_name
   ```

2. **Launch Jupyter Lab**

   ```bash
   jupyter-lab
   ```

   For users with Jupyter Lab installed in a different location, add the environment's Python kernel to Jupyter:

   ```bash
   python -m ipykernel install --user --name=your_env_name
   ```

3. **Switch to the Desired Kernel**

   In Jupyter Lab, open a new notebook. Click on the top-right corner of the window and select `your_env_name`.

---

## Setting up Dataset

1. **Prepare the Data Frame**

    Ensure your data frame is stored in CSV format with the first row set as variable names and subsequent rows as values. An example structure:
    
    | X         | Y          | Z         |
    |-----------|------------|-----------|
    | -0.673503 | 0.86791503 | -0.673503 |
    | 0.7082311 | -0.8327477 | 0.7082311 |
    | ...       | ...        | ...       |

    *Note*: any row with at least one missing value will be automatically removed during the data preprocessing stage (see [Running the Model](#running-the-model)).

2. **Specify a Directed Acyclic Graph (DAG)**

    `cGNF` utilizes an adjacency matrix in CSV format to recognize a DAG. Use the following steps in Jupyter Notebook or Python to generate an adjacency matrix:
    
    #### a. **Import Required Libraries**:
    
    ```python
    import collections.abc
    collections.Iterable = collections.abc.Iterable 
    import networkx as nx 
    from causalgraphicalmodels import CausalGraphicalModel     
    ```
    
    #### b. **Draw the DAG**:
    
   Define your DAG structure using the `CausalGraphicalModel`:
    
    ```python
    Your_DAG_name = CausalGraphicalModel(
        nodes=["var1", "var2", ...],
        edges=[("parent", "child"), ...]
    )
    ```
    
   For example, for a simple DAG X &rarr; Y &rarr; Z, the argument will be as follows:
    
   ```python
   Simple_DAG = CausalGraphicalModel(
       nodes=["X", "Y", "Z"],
       edges=[("X", "Y"), ("Y", "Z")]
   )
   ```
    
    #### c. **Convert the DAG to an Adjacency Matrix**:
    
   ```python
   your_adj_mat_name = nx.to_pandas_adjacency(Your_DAG_name.dag, dtype=int)
   ```
    
   Save the matrix as a CSV file:
    
   ```python
   your_adj_mat_name.to_csv('/path_to_data_directory/' + 'your_adj_mat_name' + '.csv')
   ```
    
    #### d. **Manually Create an Adjacency Matrix**:
    
    Alternatively, you can manually create an adjacency matrix in a CSV file by listing variables in both the first row and the first column. Here's how you interpret the matrix:
    
    - The row represents the parent node, and the column represents the child node.
      
    - If the cell at row X and column Y (i.e., position (X, Y)) contains a 1, it means X leads to Y.
      
    - If it contains a 0, it means X does not lead to Y.
  
    - Remember, since this is a directed graph, a 0 at position (Y, X) doesn't imply a 0 at position (X, Y).
    
    For example, the below adjacency matrix describes a DAG where X &rarr; Y &rarr; Z. 
    
    |   | X | Y | Z |
    |---|---|---|---|
    | X | 0 | 1 | 0 |
    | Y | 0 | 0 | 1 |
    | Z | 0 | 0 | 0 |

    _Note_: Ensure you save the adjacency matrix in the same directory as your dataframe.
   
---

## Running the Model

### Essential Functions

`cGNF` operates through three core stages, defined in separate Python functions:

1. **`process`**: Prepares the dataset and adjacency matrix.
   
2. **`train`**: Trains the model.
   
3. **`sim`**: Conducts counterfactual inference.

Refer to the provided code snippets for details on function parameters and usage.

---

1. **Data Preprocessing (defined in `processing.py`)**:

    ```python
    from cGNF import process
    process(
        path='/path_to_data_directory/',  # File path where the dataset and DAG are located
        dataset_name='your_dataset_name',  # Name of the dataset
        dag_name= 'you_adj_mat_name',  # Name of the adjacency matrix (DAG) to be used
        test_size=0.2,  # Proportion of data for the validation set
        cat_var=['X', 'Y'],  # List of categorical variables
        seed=None  # Seed for reproducibility
    )
    ```

   *Notes*:
   - `cat_var`: If the dataset has no categorical variables, set `cat_var=None`.
   
   - The function will automatically remove any row that contains at least one missing value.

   - The function converts the dataset and the adjacency matrix into tensors. These tensors are then packaged into a PKL file named after `dataset_name` and saved within the `path` directory. This PKL file is later utilized for model training.

---

2. **Training (defined in `training.py`)**:
   
    ```python
    from cGNF import train
    train(
        path='/path_to_data_directory/',  # File path where the PKL file are located
        dataset_name='your_dataset_name',  # Name of the dataset
        path_save='/your_directory/' + 'your_trained_model',  # Save path for the model
        trn_batch_size=1024,  # Training batch size
        val_batch_size=1000,  # Validation batch size
        learning_rate=3e-4,  # Learning rate
        seed=None,  # Seed for reproducibility
        nb_epoch=50000,  # Number of total epochs
        emb_net=[100, 90, 80, 60],  # Architecture of the embedding network
        int_net=[60,50,40,30],  # Architecture of the internal network
        nb_estop=50,  # Number of epochs for early stopping
        val_freq=1  # Frequency of validation check
    )
    ```

   *Notes*:
   - `path_save`: If the specified directory doesn't exist, the function will create it and save the model there.
     
   - Regularization parameters to enhance the neural network's performance include increasing the layers and nodes of `emb_net` & `int_net`, increasing `nb_estops`, or raising `val_freq`. However, be mindful of potential overfitting and increased computation time.

---

3. **Counterfactual Inference (defined in `simulation.py`)**:

    ```python
    from cGNF import sim
    sim(
        path='/path_to_data_directory/',  # File path where the PKL file are located
        dataset_name='your_dataset_name',  # Name of the dataset
        path_save='/your_directory/' + 'your_trained_model',  # Path where the trained model are located
        n_mce_samples=100000,  #  Number of draws for Monte Carlo estimation
        treatment='X',  # Treatment variable
        cat_list=[0, 1],  # Treatment values for counterfactual outcomes
        moderator=['C'],  # Specify to conduct moderation analysis
        quant_mod=4,  # Quantile divisions for continuous moderators
        mediator=['M1', 'M2'],  # List mediators for mediation analysis; 
        outcome='Y',   # Outcome variable
        inv_datafile_name='your_counterfactual_dataset'  # Name of the file for counterfactual data
    )
    ```

   *Notes*:
   - Increasing `n_mce_samples` helps reduce sampling error during the inference stage but may increase computation time.

   - `cat_list`: Multiple treatment values are permitted. If a mediator is specified, only two values are allowed, where the first value represents the control condition and the second represents the treated condition.

   - `moderator`: If the moderator is categorical and has fewer than 10 categories, the function will display potential outcomes based on different moderator values.

     For continuous moderators or those with over ten categories, the outcomes are displayed based on quantiles, determined by `quant_mod`. By default, with `quant_mod=4`, the moderator values are divided on **quartiles**.

     When conditional treatment effects are not of interest, or the dataset has no moderators, set `moderator=None`.

   - `mediator`: Multiple mediators are permitted. When specifying several mediators, ensure the causal order for a generalized path-specific analysis.

     When path-specific (including direct and indirect) effects are not of interest, or the dataset has no mediators, set `mediator=None`.

     Moderated mediation analysis is available by specifying the `moderator` and `mediator` parameters simultaneously

   - `inv_datafile_name`: In the absence of specified mediators, he function will produce a CSV file with counterfactual results based on the values provided in `cat_list`, saved within the `path` directory.

     With specified mediators, additional counterfactual data files will be produced for each mediator's path, suffixed with m*n*_0 and m*n*_1.

     '_0' denotes the path where treatment and all mediators after the *n*th mediator are set to the control condition, while all mediators before and including the *n*th mediator are set to the treated condition;

     '_1' denotes the path where treatment and all mediators after the *n*th mediator are set to the treated condition, while all mediators before and including the *n*th mediator are set to the control condition.

#### Remember to adjust paths, environment names, and other placeholders as per your actual setup.

---
