import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load results
results = pickle.load(open('../../Experiments_results/Results/Experiment1.pkl', 'rb'))

# Set plot style
sns.set(style="whitegrid", context="talk")

# Create a figure with three subplots arranged horizontally
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: Primal, Dual, and Dual Gap over iterations
axes[0].plot(results['primal_values'], label="Primal value", marker='o')
axes[0].plot(results['dual_values'], label="Dual value", marker='o')
axes[0].plot(results['dual_gaps'], label="Dual gap", marker='o', linestyle='--')
axes[0].set_xlabel("Iterations")
axes[0].set_ylabel("Objective value")
axes[0].set_title("Primal, dual, and dual gap across training")
axes[0].legend()
axes[0].grid(True)

# Subplot 2: Training and Test Scores over iterations
axes[1].plot(results['train_scores'], label="Training score", marker='o')
axes[1].plot(results['test_scores'], label="Test score", marker='o')
axes[1].set_xlabel("Iterations")
axes[1].set_ylabel("$R^2$ Score")
axes[1].set_ylim(0, 1)
axes[1].set_title("Training and test scores across training")
axes[1].legend()
axes[1].grid(True)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig('../../Experiments_results/Plots/Experiment1.pdf', dpi=300, bbox_inches='tight')
plt.show()

print('Plotting Experiment1 over')
