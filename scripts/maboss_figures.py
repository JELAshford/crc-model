from itertools import product
from tqdm import tqdm
import pandas as pd
import numpy as np

import matplotlib.pylab as plt
import plotly_express as px

import maboss


def apply_iterative_mutations(model, mutation_states):
    this_model = model.copy()
    trajectories = []
    save_result = None
    for step in range(len(mutation_states["kRAS"])):
        # Apply step mutation
        for gene, states in mutation_states.items():
            this_model.mutate(gene, states[step])
        # Run
        if save_result is not None:
            this_model.continue_from_result(save_result)
        result = this_model.run()
        # Save state for continuation
        save_result = result
        # Save trajectory
        trajectory = result.get_nodes_probtraj()
        trajectories.append(trajectory)
    return trajectories


def sequential_trajectories_lineplot(trajectory_list, mutation_order=["None", "kRAS", "APC", "SMAD24", "p53"], save_path=None, show=True):
    trajectory_df = pd.concat(trajectory_list).reset_index(drop=True)
    run_steps = trajectory_list[0].shape[0] - 1
    _, ax = plt.subplots(1, 1, figsize=(10, 5))
    _ = trajectory_df.plot(ax=ax)
    for ind, mutation in enumerate(mutation_order):
        start, end = (ind * run_steps), ((ind+1)*run_steps)
        mid = start + (end-start)/3
        plt.axvspan(start, end, color='red' if ind %
                    2 == 0 else 'blue', alpha=0.05)
        plt.annotate(mutation, xy=(mid, 0.6))
    plt.legend(loc='upper center', bbox_to_anchor=(
        0.5, 1.15), ncol=6, fancybox=True, shadow=True)
    plt.xlabel('Time')
    plt.ylabel('Probability')
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()


def sequential_trajectories_image(trajectory_list, mutation_order=["None", "kRAS", "APC", "SMAD24", "p53"], save_path=None, show=True):
    trajectory_df = pd.concat(trajectory_list).reset_index(drop=True)
    run_steps = trajectory_list[0].shape[0]
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    plt.imshow(trajectory_df.values.T, aspect="auto", interpolation="nearest")
    plt.xlabel('Time')
    plt.ylabel('Cell State')
    plt.xticks(ticks=np.arange(len(mutation_order)) * run_steps + (run_steps/2),
               labels=mutation_order)
    plt.yticks(ticks=list(range(len(trajectory_df.columns))),
               labels=trajectory_df.columns)
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()


gene_names = ["kRAS", "APC", "SMAD24", "p53"]
output_names = ["Apoptosis", "Carcinoma", "Invasion",
                "LargeAdenoma", "Proliferating", "SmallAdenoma"]

model_name = "vogelstein_wnt"  # "updated_vogelstein"  # "complex_vogelstein_updated"
base_model = maboss.loadSBML(
    f"./rsc/{model_name}.sbml",
    f"./rsc/{model_name}.cfg",
    use_sbml_names=True
)

# Add in simulated mutations
baseline_model = base_model.copy()
baseline_model.mutate("APC", "ON")
baseline_model.mutate("kRAS", "OFF")
baseline_model.mutate("SMAD24", "OFF")
baseline_model.mutate("p53", "OFF")
baseline_res = baseline_model.run()
baseline_res.plot_piechart()
baseline_res.plot_trajectory()

# # FULL MUTANT
full_mutant_model = base_model.copy()
full_mutant_model.mutate("APC", "OFF")
full_mutant_model.mutate("kRAS", "ON")
full_mutant_model.mutate("SMAD24", "ON")
full_mutant_model.mutate("p53", "ON")
mutant_res = full_mutant_model.run()
mutant_res.plot_piechart()
mutant_res.plot_trajectory()


# Try all combinations of ON/OFF for 4 genes
all_names = gene_names + output_names

result_list = []
combinations = list(product(["ON", "OFF"], repeat=4))
for mutation_profile in tqdm(combinations):
    apc, kras, smad24, p53 = mutation_profile
    # Copy and mutate baseline
    mutant_model = base_model.copy()
    mutant_model.mutate("APC", apc)
    mutant_model.mutate("kRAS", kras)
    mutant_model.mutate("SMAD24", smad24)
    mutant_model.mutate("p53", p53)
    # Run model
    mutant_res = mutant_model.run()
    # Save reults as DF
    run_probs = mutant_res.get_nodes_probtraj().iloc[39, ]
    mut_profile_df = pd.DataFrame(
        {gene: [mutation_profile[ind]] for ind, gene in enumerate(gene_names)})
    run_prob_df = pd.DataFrame(
        {ind: [val] for ind, val in zip(run_probs.index, run_probs.values)})
    result_list.append(pd.concat([mut_profile_df, run_prob_df], axis=1))

result_df = pd.concat(result_list, ignore_index=True)
result_df.to_csv("out/all_mutation_combinations.csv")


#### MUTATION ANALYSIS ####
# Try all combinations of sequentially mutating the BaseEpithelial cells to
# the full cancer geneotype.
# combined_trajectories = pd.concat(trajectories).reset_index(drop=True)
# px.scatter(combined_trajectories)

typical_mutation_order = {
    "kRAS": ["ON", "OFF", "OFF", "OFF", "OFF"],
    "APC": ["OFF", "OFF", "ON", "ON", "ON"],
    "SMAD24": ["OFF", "OFF", "OFF", "ON", "ON"],
    "p53": ["OFF", "OFF", "OFF", "OFF", "ON"]
}
trajectories = apply_iterative_mutations(base_model, typical_mutation_order)
sequential_trajectories_lineplot(trajectories, show=True)
sequential_trajectories_image(trajectories, show=True)

#  Generate that mutation dictionary from a reverse mutation order
mutation_order_rev = ["None", "p53", "SMAD24", "APC", "kRAS"]
reverse_mutation_order = {
    "kRAS": ["ON", "ON", "ON", "ON", "OFF"],
    "APC": ["OFF", "OFF", "OFF", "ON", "ON"],
    "SMAD24": ["OFF", "OFF", "ON", "ON", "ON"],
    "p53": ["OFF", "ON", "ON", "ON", "ON"]
}
rev_trajectories = apply_iterative_mutations(
    base_model, reverse_mutation_order)
sequential_trajectories_lineplot(
    rev_trajectories, mutation_order=mutation_order_rev, show=True)
sequential_trajectories_image(
    rev_trajectories, mutation_order=mutation_order_rev, show=True)
