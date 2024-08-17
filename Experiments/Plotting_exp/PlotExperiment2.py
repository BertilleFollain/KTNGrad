import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle

# Load results
results_data = pickle.load(open('../../Experiments_results/Results/Experiment2.pkl', 'rb'))
results = results_data['results']
n_values = results_data['n_values']
d_values = results_data['d_values']
repetitions = results_data['repetitions']

# Setup for plotting
sns.set(style="whitegrid", context="talk")
methods = ['KTNGrad', 'PyMave', 'KTNGrad, retrained', 'KRR']
palette = sns.color_palette()
color_palette = [palette[0], palette[3], palette[2], palette[1]]

# Plot results for varying n (fixed d)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot Test Scores
for method_idx, method in enumerate(methods):
    avg_scores = np.mean(results[method]['scores'][0:len(n_values), :], axis=1)  # Average across repetitions
    for rep in range(repetitions):
        axes[0].plot(n_values, results[method]['scores'][0:len(n_values), rep],
                     color=color_palette[method_idx], alpha=0.1)
    axes[0].plot(n_values, avg_scores, label=f'{method} - Average', color=color_palette[method_idx], linewidth=2.5)

axes[0].set_title('Test score with fixed $d$', fontsize=18)
axes[0].set_xlabel('Sample size $n$', fontsize=16)
axes[0].set_ylabel('$R^2$ score on test set', fontsize=16)
axes[0].legend(loc='best', fontsize=10)
axes[0].set_ylim(ymin=-1, ymax=1.05)
axes[0].tick_params(axis='both', which='major', labelsize=14)
axes[0].grid(True)

# Plot Feature Learning Score and Dimension Score on the same subplot
for method_idx, method in enumerate(['KTNGrad', 'PyMave']):
    avg_scores_feat = np.mean(results[method]['features'][0:len(n_values), :], axis=1)  # Average across repetitions
    avg_scores_dim = np.mean(results[method]['dimension'][0:len(n_values), :], axis=1)  # Average across repetitions
    for rep in range(repetitions):
        axes[1].plot(n_values, results[method]['features'][0:len(n_values), rep],
                     color=color_palette[method_idx], alpha=0.1)
        axes[1].plot(n_values, results[method]['dimension'][0:len(n_values), rep],
                     color=color_palette[method_idx], alpha=0.1, linestyle='--')
    axes[1].plot(n_values, avg_scores_feat, label=f'{method} - Feature - Average', color=color_palette[method_idx],
                 linewidth=2.5)
    axes[1].plot(n_values, avg_scores_dim, label=f'{method} - Dimension - Average', color=color_palette[method_idx],
                 linewidth=2.5, linestyle='--')

axes[1].set_title('Feature & dimension scores with fixed $d$', fontsize=18)
axes[1].set_xlabel('Sample size $n$', fontsize=16)
axes[1].set_ylabel('Scores', fontsize=16)
axes[1].set_ylim(-0.05, 1.05)
axes[1].legend(loc='best', fontsize=10)
axes[1].tick_params(axis='both', which='major', labelsize=14)
axes[1].grid(True)

plt.tight_layout()
plt.savefig('../../Experiments_results/Plots/Experiment2_fixed_d.pdf', dpi=300, bbox_inches='tight', format='pdf')
plt.show()

# Plot results for varying d (fixed n)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot Test Scores
for method_idx, method in enumerate(methods):
    avg_scores = np.mean(results[method]['scores'][len(n_values):, :], axis=1)  # Average across repetitions
    for rep in range(repetitions):
        axes[0].plot(d_values, results[method]['scores'][len(n_values):, rep],
                     color=color_palette[method_idx], alpha=0.1)
    axes[0].plot(d_values, avg_scores, label=f'{method} - Average', color=color_palette[method_idx], linewidth=2.5)

axes[0].set_title('Test score with fixed $n$', fontsize=18)
axes[0].set_xlabel('Data dimension $d$', fontsize=16)
axes[0].set_ylabel('$R^2$ score on test set', fontsize=16)
axes[0].legend(loc='best', fontsize=10)
axes[0].set_ylim(ymin=-0.25, ymax=1.05)
axes[0].tick_params(axis='both', which='major', labelsize=14)
axes[0].grid(True)

# Plot Feature Learning Score and Dimension Score on the same subplot
for method_idx, method in enumerate(['KTNGrad', 'PyMave']):
    avg_scores_feat = np.mean(results[method]['features'][len(n_values):, :], axis=1)  # Average across repetitions
    avg_scores_dim = np.mean(results[method]['dimension'][len(n_values):, :], axis=1)  # Average across repetitions
    for rep in range(repetitions):
        axes[1].plot(d_values, results[method]['features'][len(n_values):, rep],
                     color=color_palette[method_idx], alpha=0.1)
        axes[1].plot(d_values, results[method]['dimension'][len(n_values):, rep],
                     color=color_palette[method_idx], alpha=0.1, linestyle='--')
    axes[1].plot(d_values, avg_scores_feat, label=f'{method} - Feature - Average', color=color_palette[method_idx],
                 linewidth=2.5)
    axes[1].plot(d_values, avg_scores_dim, label=f'{method} - Dimension - Average', color=color_palette[method_idx],
                 linewidth=2.5, linestyle='--')

axes[1].set_title('Feature & dimension scores with fixed $n$', fontsize=18)
axes[1].set_xlabel('Data dimension $d$', fontsize=16)
axes[1].set_ylabel('Scores', fontsize=16)
axes[1].set_ylim(-0.05, 1.05)
axes[1].legend(loc='best', fontsize=10)
axes[1].tick_params(axis='both', which='major', labelsize=14)
axes[1].grid(True)

plt.tight_layout()
plt.savefig('../../Experiments_results/Plots/Experiment2_fixed_n.pdf', dpi=300, bbox_inches='tight', format='pdf')
plt.show()

print('Plotting Experiment2 over')
