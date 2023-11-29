import pandas as pd
import os
from cGNF.processing_parallel import process
from cGNF.training_parallel import train
from cGNF.simulation_parallel import sim
from joblib import Parallel, delayed
import time

def bootstrap(n_iterations=None, num_cores_reserve=None, base_path=None, folder_name=None, dataset_name=None, dag_name=None,process_args=None, train_args=None, skip_train=False, sim_args_list=None):
    def run_simulation(i, process_args, train_args, sim_args_list):
        print(f"Running simulation {i} on Process ID: {os.getpid()}")
        start_time = time.time()

        # Load and sample data
        data = pd.read_csv(os.path.join(base_path, dataset_name + f'.csv'))
        dag =  pd.read_csv(os.path.join(base_path, dag_name + f'.csv'), index_col=0)
        df = data.sample(n=len(data), replace=True)
        folder = f'{folder_name}_{i}'
        path = os.path.join(base_path, folder, '')
        if not os.path.isdir(path):
            os.makedirs(path)

        df_filename = os.path.join(path, dataset_name + f'.csv')
        dag_filename = os.path.join(path, dag_name + f'.csv')
        df.to_csv(df_filename, index=False)
        dag.to_csv(dag_filename, index=False)

        # Run process and train functions, if arguments are provided
        if process_args:
            updated_process_args = {**process_args, 'path': path, 'dataset_name': dataset_name, 'dag_name': dataset_name}
            process(**updated_process_args)

        if not skip_train:
            updated_train_args = {**train_args, 'path': path, 'dataset_name': dataset_name}
            train(**updated_train_args)

        # Initialize results list
        values_list = []
        outcome_list = []

        sim_results = pd.DataFrame()
        # Run sim functions and collect results
        if sim_args_list:
            for sim_args in sim_args_list:
                updated_sim_args = {**sim_args, 'path': path, 'dataset_name': dataset_name}
                sim_result = sim(**updated_sim_args)
                sim_results = pd.concat([sim_results, sim_result], ignore_index=True)

            # Process results
            values_list = sim_results['Value'].tolist()
            outcome_list = sim_results['Potential Outcome'].tolist()

        end_time = time.time()
        time_taken = end_time - start_time
        values_list.append(time_taken)
        outcome_list.append('Time Taken')

        return values_list, outcome_list

    num_cores = os.cpu_count()

    print(
        f'Running bootstrap with {n_iterations} iterations on {num_cores - num_cores_reserve} out of {num_cores} CPU cores ({num_cores_reserve} cores reserved)')

    # Run simulations in parallel
    results = Parallel(n_jobs=num_cores - num_cores_reserve)(
        delayed(run_simulation)(i, process_args, train_args, sim_args_list) for i in range(n_iterations)
    )

    # Extract values and outcome_list from the results
    all_values = [result[0] for result in results]
    outcome_list = results[0][1] if results else []

    # Create final results DataFrame
    final_results_df = pd.DataFrame(all_values, columns=outcome_list)
    final_results_df.to_csv(os.path.join(base_path, dataset_name + f'_bootstrap_result.csv'), index=False)

    return final_results_df
