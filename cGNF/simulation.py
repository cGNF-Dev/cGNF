import torch
import pickle
import pandas as pd
import os
import numpy as np
import random


def sim(path="", dataset_name="", model_name="models", n_mce_samples=10000, seed=None, treatment='', cat_list=[0, 1],
        moderator=None, quant_mod=4, mediator=None, outcome=None, inv_datafile_name='potential_outcome'):

    results_df = pd.DataFrame(columns=["Potential Outcome", "Value"])

    # Identify whether the system has a GPU, if yes it sets the device to "cuda:0" else "cpu"
    device = "cpu" if not (torch.cuda.is_available()) else "cuda:0"

    path_save = os.path.join(path, model_name)

    # Load the previously saved PyTorch model from the disk
    model = torch.load(path_save + '/_best_model.pt', map_location=device)

    # Load original dataset
    with open(path + dataset_name + '.pkl', 'rb') as f:
        data = pickle.load(f)

    # extract the list of variables
    df = data['df']
    Z_Sigma = data['Z_Sigma']
    variable_list = df.columns.tolist()
    # identify the index of variables in the variable list
    loc_treatment = variable_list.index(treatment)
    loc_outcome = variable_list.index(outcome)

    # Move the model to the appropriate device (GPU if available or CPU)
    model = model.to(device)

    # Extract the adjacency matrix from the model's conditioner (after applying soft-thresholding) and move it to CPU. Convert it to numpy array.
    # Soft_threshold_A?: squares A, multiplies it by 2, passes the result through a sigmoid function to squash the values into the range (0, 1), and then subtracts 0.5 to center the values around 0. This operation retains all the connections in the graph but reduces the weight of weaker connections.
    A_mat = model.getConditioners()[0].soft_thresholded_A().detach().cpu().numpy()

    # Define the dimensions of the adjacency matrix (number of nodes/variables)
    dim = A_mat.shape[0]

    # Set the model to evaluation mode
    model.eval()

    # Import multivariate normal distribution from PyTorch distributions module
    from torch.distributions.multivariate_normal import MultivariateNormal

    # Set the seed for random number generation. If not provided, select a random seed between 1 and 20000
    if seed is None:
        seed = random.randint(1, 20000)
    print(f"Running simulation with seed {seed}")

    # setting the seed for random number generators in python, numpy and pytorch for reproducible results
    random.seed(seed)
    np.random.seed(seed=seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # ensures that the CUDA backend for PyTorch (CuDNN) uses deterministic algorithms.
        torch.backends.cudnn.benchmark = True  # enables the CuDNN autotuner, which selects the best algorithm for CuDNN operations given the current hardware setup.

    # Disable gradient calculation to save memory and computation during inference
    with torch.no_grad():
        # Define a multivariate normal distribution with mean 0 (torch.zeros) and identity covariance matrix (torch.eye)
        Z_do = MultivariateNormal(torch.zeros(dim), Z_Sigma)
        # Sample from the defined multivariate normal distribution
        # z_do =
        # [
        #   [a1, a2],
        #   [b1, b2],
        #   [c1, c2],
        #   [d1, d2],
        #     ...
        # ]
        z_do = Z_do.sample(torch.Size([n_mce_samples])).to(device)

        # Create a tensor that contains all possible categories for our treatments.
        all_a = torch.tensor(cat_list).unsqueeze(1).float()

        # Add an extra dimension to z_do tensor at position 1 (making it a 3D tensor), and replicate the original dimension to the new one. The number of times it is replicated corresponds to the number of treatment categories (i.e., the size of all_a along the 0th dimension). The resulting tensor, z_do_n, has dimensions  (n_mce_samples, number of treatment categories, dim).
        # z_do_n =
        # [
        #     [[a1, a2], [a1, a2]],
        #     [[b1, b2], [b1, b2]],
        #     [[c1, c2], [c1, c2]],
        #     [[d1, d2], [d1, d2]]
        #     ...
        # ]
        z_do_n = z_do.unsqueeze(1).expand(-1, all_a.shape[0], -1).clone().to(device)

        # Add an extra dimension to all_a tensor at position 0 (making it a 3D tensor), expand it so that the first dimension matches the number of MCE samples. The resulting tensor, all_a_n, has dimensions (n_mce_samples, number of treatment categories, 1).
        # all_a_n =
        # [
        #     [[0], [1]],
        #     [[0], [1]],
        #     [[0], [1]],
        #     [[0], [1]],
        #     ...
        # ]
        all_a_n = all_a.unsqueeze(0).expand(n_mce_samples, -1, -1).to(device)

        # Substitute treatment values (all_a_n) to the 'l_do_cats_dim'th element of z_do_n, so that z_do_n becomes a tensor that contains all possible combinations of the treatment and a sample from the multivariate normal distribution.
        # z_do_n =
        # [
        #     [[0, a2], [1, a2]],
        #     [[0, b2], [1, b2]],
        #     [[0, c2], [1, c2]],
        #     [[0, d2], [1, d2]]
        #     ...
        # ]
        z_do_n[:, :, list([loc_treatment])] = all_a_n

        # Reshape z_do_n for processing (prepare it for 'model.invert') and move to appropriate device (GPU if available or CPU)
        z_do_n = z_do_n.transpose_(1, 0).reshape(-1, dim).to(device)  # .view(-1,n_samples,dim)

    # Counterfactual inference block
    with torch.no_grad():
        # Perform counterfactual inference by applying all treatments for all units through a invertible flow of the model (model.intvert)
        cur_x_do_inv = model.invert(z_do_n, do_idx=list([loc_treatment]),
                                    do_val=torch.narrow(z_do_n, 1, min(list([loc_treatment])),
                                                        len(list([loc_treatment]))))
        # Reshape the results. The final shape of cur_x_do_inv after the view operation is [all_a.shape[0], n_mce_samples, dim].
        cur_x_do_inv = cur_x_do_inv.view(-1, n_mce_samples, dim)

        # Reshape the tensor to 2D
        cur_x_do_inv_2d = cur_x_do_inv.reshape(-1, cur_x_do_inv.shape[-1])
        # Convert the tensor to a numpy array and then to a pandas DataFrame
        inv_output = pd.DataFrame(cur_x_do_inv_2d.cpu().detach().numpy())
        # Set the column names
        inv_output.columns = variable_list
        # Save to a CSV file
        inv_output.to_csv(path + inv_datafile_name + f'.csv')

        # Convert the tensor to a numpy array and then to a pandas DataFrame
        inv_output = pd.DataFrame(cur_x_do_inv_2d.cpu().detach().numpy())
        # Set the column names
        inv_output.columns = variable_list
        # Save to a CSV file
        inv_output.to_csv(path + inv_datafile_name + f'.csv')

        if mediator:

            n_mediators = len(mediator)

            # Check if mediator values are specified
            specified_values = [m.split('=') if '=' in m else None for m in mediator]
            mediator_names = [s[0] if s else m for s, m in zip(specified_values, mediator)]
            mediator_values = [float(s[1]) if s else None for s in specified_values]

            # Get the location of mediators in the variable list
            loc_mediator = [variable_list.index(m) for m in mediator_names]

            # Retrieve the control and treatment values for each mediator
            x_control = cur_x_do_inv[0, :, loc_treatment]
            x_treatment = cur_x_do_inv[1, :, loc_treatment]

            # Duplicate a Tensor with Z noise for mediation analysis
            z_do_n_med = z_do.unsqueeze(1).expand(-1, 1, -1).clone().to(device)
            z_do_n_med = z_do_n_med.transpose_(1, 0).reshape(-1, dim).to(device)

            cur_x_med_dict = {}
            inv_output_med_dict = {}

            for i in range(n_mediators):
                # For each mediator, generate a CSV file for control and treatment values
                for val in ['0', '1']:  # Control and treatment
                    do_values = [x_treatment if val == '1' else x_control]

                    for j in range(i + 1):  # Include all mediators up to i
                        if mediator_values[j] is not None:  # If a specific value is provided
                            m_value = torch.tensor(mediator_values[j]).expand(n_mce_samples).to(device)
                            do_values.append(m_value)
                        else:
                            m_control = cur_x_do_inv[0, :, loc_mediator[j]]
                            m_treatment = cur_x_do_inv[1, :, loc_mediator[j]]
                            do_values.append(m_treatment if val == '0' else m_control)

                    # 'do' operation and inversion
                    do_val = torch.stack(do_values, -1)
                    cur_x_med = model.invert(z_do_n_med, do_idx=[loc_treatment] + loc_mediator[:i + 1], do_val=do_val)
                    cur_x_med = cur_x_med.view(-1, n_mce_samples, dim)

                    # Reshape and convert to DataFrame
                    cur_x_med_2d = cur_x_med.reshape(-1, cur_x_do_inv.shape[-1])
                    inv_output_med = pd.DataFrame(cur_x_med_2d.cpu().detach().numpy())

                    # Store the cur_x_med and inv_output in dictionaries
                    cur_x_med_dict[f"m{i + 1}_{val}"] = cur_x_med
                    inv_output_med_dict[f"m{i + 1}_{val}"] = inv_output_med

                    # Set the column names and save to CSV
                    inv_output_med.columns = variable_list
                    inv_output_med.to_csv(f"{path}{inv_datafile_name}_m{i + 1}_{val}.csv")

            if moderator:

                # Create the potential outcome Dataframe prior to intervention
                z_do_n_mod = z_do.unsqueeze(1).expand(-1, all_a.shape[0], -1).clone().to(device)
                z_do_n_mod = z_do_n_mod.transpose_(1, 0).reshape(-1, dim).to(device)
                cur_x_mod = model.invert(z_do_n_mod)

                # Prepare for the update of inv_output
                cur_x_mod_med = cur_x_mod[:cur_x_mod.shape[0] // all_a.shape[0]]
                cur_x_mod_med_2d = cur_x_mod_med.reshape(-1, cur_x_mod_med.shape[-1])
                inv_output_mod_med = pd.DataFrame(cur_x_mod_med_2d.cpu().detach().numpy())
                inv_output_mod_med.columns = variable_list

                # Reshape the tensor to 2D
                cur_x_mod = cur_x_mod.view(-1, n_mce_samples, dim)
                cur_x_mod_2d = cur_x_mod.reshape(-1, cur_x_mod.shape[-1])
                # Convert the tensor to a numpy array and then to a pandas DataFrame
                inv_output_mod = pd.DataFrame(cur_x_mod_2d.cpu().detach().numpy())

                # Set the column names
                inv_output_mod.columns = variable_list

                # Update the inv_output
                inv_output[f"Observed_{moderator}"] = inv_output_mod[moderator].values
                # Get unique values of the moderator variable
                unique_moderator_values = inv_output[f"Observed_{moderator}"].unique()

                # Update the CSV file
                inv_output.to_csv(path + inv_datafile_name + f'.csv')

                if len(unique_moderator_values) > 10:
                    quartile_labels, quartile_intervals = pd.qcut(inv_output[f"Observed_{moderator}"], q=quant_mod, retbins=True)
                    inv_output[f"Observed_{moderator}"] = quartile_labels
                    quartiles = inv_output[f"Observed_{moderator}"].cat.categories

                    for idx, q in enumerate(quartiles, start=1):
                        print(f"---- {moderator} (Quartile {idx}) ----")

                        # Main DataFrame subset where the moderator equals the current unique value
                        subset_df = inv_output[inv_output[f"Observed_{moderator}"] == q]

                        for t_val in cat_list:
                            sub_subset_df = subset_df[subset_df[treatment] == t_val]
                            conditional_mean = sub_subset_df[outcome].mean()
                            new_row = pd.DataFrame(
                                {"Potential Outcome": f"E[{outcome}({treatment}={t_val} | {moderator}={q})]",
                                 "Value": conditional_mean}, index=[0])
                            results_df = pd.concat([results_df, new_row], ignore_index=True)
                            print(
                                f"E[{outcome}({treatment}={t_val} | {moderator}={q})] = {conditional_mean}")

                        # Loop through each mediator to print the expected outcomes for different combinations
                        for i in range(len(mediator)):
                            for val in ['0', '1']:  # Control and treatment

                                # Get the boolean series for the current value of the moderator from the DataFrame
                                moderator_series = inv_output[f"Observed_{moderator}"] == q

                                # Insert moderator column into the final output Dataframe
                                inv_output_med_dict[f"m{i + 1}_{val}"][f"Observed_{moderator}"] = inv_output_mod_med[
                                    moderator].values

                                # Update all mediation CSV file
                                inv_output_med.to_csv(f"{path}{inv_datafile_name}_m{i + 1}_{val}.csv")

                                # Subset the DataFrame for the current mediator and treatment level using the moderator condition
                                subset_df_mediator = inv_output_med_dict[f"m{i + 1}_{val}"][
                                    moderator_series.reindex(inv_output_med_dict[f"m{i + 1}_{val}"].index,
                                                             fill_value=False)]

                                # Perform calculations for mean counterfactual outcome under different moderator values
                                m_moderated_mean = subset_df_mediator[outcome].mean()

                                for j in range(i + 1):  # Include all mediators up to i
                                    mediator_conditions = ", ".join([f"{mediator[j]}" if mediator_values[
                                                                                             j] is not None else f"{mediator[j]}({treatment}={cat_list[1 if int(val) == 0 else 0]})"
                                                                     for j in range(i + 1)])
                                new_row = pd.DataFrame(
                                    {
                                        "Potential Outcome": f"E[{outcome}({treatment}={cat_list[int(val)]}, {mediator_conditions} | {moderator}={q})]",
                                        "Value": m_moderated_mean}, index=[0])
                                results_df = pd.concat([results_df, new_row], ignore_index=True)
                                print(
                                    f"E[{outcome}({treatment}={cat_list[int(val)]}, {mediator_conditions} | {moderator}={q})] = {m_moderated_mean}")

                else:
                    for mod_val in unique_moderator_values:
                        print(f'---- {moderator} = {mod_val} ----')

                        # Main DataFrame subset where the moderator equals the current unique value
                        subset_df = inv_output[inv_output[f"Observed_{moderator}"] == mod_val]

                        for t_val in cat_list:
                            sub_subset_df = subset_df[subset_df[treatment] == t_val]
                            conditional_mean = sub_subset_df[outcome].mean()
                            new_row = pd.DataFrame(
                                {"Potential Outcome": f"E[{outcome}({treatment}={t_val} | {moderator}={mod_val})]",
                                 "Value": conditional_mean}, index=[0])
                            results_df = pd.concat([results_df, new_row], ignore_index=True)
                            print(
                                f"E[{outcome}({treatment}={t_val} | {moderator}={mod_val})] = {conditional_mean}")

                        # Loop through each mediator to print the expected outcomes for different combinations
                        for i in range(len(mediator)):
                            for val in ['0', '1']:  # Control and treatment

                                # Get the boolean series for the current value of the moderator from the DataFrame
                                moderator_series = inv_output[f"Observed_{moderator}"] == mod_val

                                # Insert moderator column into the final output Dataframe
                                inv_output_med_dict[f"m{i + 1}_{val}"][f"Observed_{moderator}"] = inv_output_mod_med[
                                    moderator].values

                                # Update all mediation CSV file
                                inv_output_med.to_csv(f"{path}{inv_datafile_name}_m{i + 1}_{val}.csv")

                                # Subset the DataFrame for the current mediator and treatment level using the moderator condition
                                subset_df_mediator = inv_output_med_dict[f"m{i + 1}_{val}"][
                                    moderator_series.reindex(inv_output_med_dict[f"m{i + 1}_{val}"].index,
                                                             fill_value=False)]

                                # Perform calculations for mean counterfactual outcome under different moderator values
                                m_moderated_mean = subset_df_mediator[outcome].mean()

                                for j in range(i + 1):  # Include all mediators up to i
                                    mediator_conditions = ", ".join([f"{mediator[j]}" if mediator_values[
                                                                                             j] is not None else f"{mediator[j]}({treatment}={cat_list[1 if int(val) == 0 else 0]})"
                                                                     for j in range(i + 1)])
                                new_row = pd.DataFrame(
                                    {
                                        "Potential Outcome": f"E[{outcome}({treatment}={cat_list[int(val)]}, {mediator_conditions} | {moderator}={mod_val})]",
                                        "Value": m_moderated_mean}, index=[0])
                                results_df = pd.concat([results_df, new_row], ignore_index=True)
                                print(
                                    f"E[{outcome}({treatment}={cat_list[int(val)]}, {mediator_conditions} | {moderator}={mod_val})] = {m_moderated_mean}")

            else:

                # Calculate the mean counterfactual outcome over all MCE samples
                cur_x_do_n_inv_mean = cur_x_do_inv.mean(1).cpu().numpy()

                E_Y_1_1 = cur_x_do_n_inv_mean[1]  # Treated treatment
                E_Y_0_0 = cur_x_do_n_inv_mean[0]  # Control treatment

                # Initialize a dictionary to hold the expected outcomes for each combination of treatment and mediator values
                E_Y = {}

                # Print the mean outcomes under both the control and treated conditions
                new_row = pd.DataFrame(
                    {"Potential Outcome": f"E[{outcome}({treatment}={cat_list[0]})]",
                     "Value": E_Y_0_0[loc_outcome]}, index=[0])
                results_df = pd.concat([results_df, new_row], ignore_index=True)
                print(f"E[{outcome}({treatment}={cat_list[0]})] = {E_Y_0_0[loc_outcome]}")
                new_row = pd.DataFrame(
                    {"Potential Outcome": f"E[{outcome}({treatment}={cat_list[1]})]",
                     "Value": E_Y_1_1[loc_outcome]}, index=[0])
                results_df = pd.concat([results_df, new_row], ignore_index=True)
                print(f"E[{outcome}({treatment}={cat_list[1]})] = {E_Y_1_1[loc_outcome]}")

                # Loop through each mediator to print the expected outcomes for different combinations
                for i in range(len(mediator)):
                    for val in ['0', '1']:  # Control and treatment
                        cur_x_med_mean = cur_x_med_dict[f"m{i + 1}_{val}"].mean(1).cpu().numpy().squeeze()
                        E_Y[f"m{i + 1}_{val}"] = cur_x_med_mean[loc_outcome]

                        for j in range(i + 1):  # Include all mediators up to i
                            mediator_conditions = ", ".join([f"{mediator[j]}" if mediator_values[
                                                                                     j] is not None else f"{mediator[j]}({treatment}={cat_list[1 if int(val) == 0 else 0]})"
                                                             for j in range(i + 1)])
                        new_row = pd.DataFrame(
                            {
                                "Potential Outcome": f"E[{outcome}({treatment}={cat_list[int(val)]}, {mediator_conditions})]",
                                "Value": E_Y[f'm{i + 1}_{val}']}, index=[0])
                        results_df = pd.concat([results_df, new_row], ignore_index=True)
                        print(
                            f"E[{outcome}({treatment}={cat_list[int(val)]}, {mediator_conditions})] = {E_Y[f'm{i + 1}_{val}']}")


        else:
            # If the moderator variable is specified
            if moderator:

                # Create the potential outcome Dataframe prior to intervention
                z_do_n_mod = z_do.unsqueeze(1).expand(-1, all_a.shape[0], -1).clone().to(device)
                z_do_n_mod = z_do_n_mod.transpose_(1, 0).reshape(-1, dim).to(device)
                cur_x_mod = model.invert(z_do_n_mod)
                cur_x_mod = cur_x_mod.view(-1, n_mce_samples, dim)

                # Reshape the tensor to 2D
                cur_x_mod_2d = cur_x_mod.reshape(-1, cur_x_mod.shape[-1])
                # Convert the tensor to a numpy array and then to a pandas DataFrame
                inv_output_mod = pd.DataFrame(cur_x_mod_2d.cpu().detach().numpy())

                # Set the column names
                inv_output_mod.columns = variable_list

                # Update the inv_output
                inv_output[f"Observed_{moderator}"] = inv_output_mod[moderator].values
                # Get unique values of the moderator variable
                unique_moderator_values = inv_output[f"Observed_{moderator}"].unique()

                # Update the CSV file
                inv_output.to_csv(path + inv_datafile_name + f'.csv')

                if len(unique_moderator_values) > 10:
                    quartile_labels, quartile_intervals = pd.qcut(inv_output[f"Observed_{moderator}"], q=quant_mod,
                                                                  retbins=True)
                    inv_output[f"Observed_{moderator}"] = quartile_labels
                    quartiles = inv_output[f"Observed_{moderator}"].cat.categories

                    for idx, q in enumerate(quartiles, start=1):
                        print(f"---- {moderator} (Quartile {idx}) ----")
                        subset_df = inv_output[inv_output[f"Observed_{moderator}"] == q]
                        for t_val in cat_list:
                            sub_subset_df = subset_df[subset_df[treatment] == t_val]
                            conditional_mean = sub_subset_df[outcome].mean()
                            new_row = pd.DataFrame(
                                {"Potential Outcome": f"E[{outcome}({treatment}={t_val} | {moderator}={q})]",
                                 "Value": conditional_mean}, index=[0])
                            results_df = pd.concat([results_df, new_row], ignore_index=True)
                            print(f"E[{outcome}({treatment}={t_val} | {moderator}={q})] = {conditional_mean}")

                else:
                    # For each unique value of the moderator variable
                    for val in unique_moderator_values:
                        print(f'---- {moderator} = {val} ----')
                        # Subset the DataFrame where the moderator equals the current unique value
                        subset_df = inv_output[inv_output[f"Observed_{moderator}"] == val]
                        # For each unique value of the treatment variable
                        for t_val in cat_list:
                            # Further subset the DataFrame where the treatment equals the current unique value
                            sub_subset_df = subset_df[subset_df[treatment] == t_val]
                            # Calculate the conditional mean counterfactual outcome
                            conditional_mean = sub_subset_df[outcome].mean()
                            new_row = pd.DataFrame(
                                {"Potential Outcome": f"E[{outcome}({treatment}={t_val} | {moderator}={val})]",
                                 "Value": conditional_mean}, index=[0])
                            results_df = pd.concat([results_df, new_row], ignore_index=True)
                            # Print the results DataFrame
                            print(f"E[{outcome}({treatment}={t_val} | {moderator}={val})] = {conditional_mean}")

            else:
                for t_val in cat_list:
                    # Further subset the DataFrame where the treatment equals the current unique value
                    subset_df = inv_output[inv_output[treatment] == t_val]
                    # Calculate the conditional mean counterfactual outcome
                    counter_mean = subset_df[outcome].mean()
                    new_row = pd.DataFrame(
                        {"Potential Outcome": f"E[{outcome}({treatment}={t_val})]",
                         "Value": counter_mean}, index=[0])
                    results_df = pd.concat([results_df, new_row], ignore_index=True)
                    # Print the results DataFrame
                    print(f"E[{outcome}({treatment}={t_val})] = {counter_mean}")

    results_df.to_csv(f"{path}{inv_datafile_name}_results.csv", index=False)
    return results_df

