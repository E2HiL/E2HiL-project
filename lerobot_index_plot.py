import pandas as pd
import matplotlib.pyplot as plt

# File path
base_path = "outputs/HIL-SERL-so101/"
files = [
    ("entropy_change_predic.csv", "Entropy Change Pred", "tab:blue", 10),
    ("policy_entropy.csv", "Policy Entropy", "tab:green", 10),
    ("success_rate.csv", "Episodic Reward", "tab:red", 30)
]

plt.figure(figsize=(10, 12))

for i, (filename, title, color, window) in enumerate(files):
    df = pd.read_csv(base_path + filename)
    y_col = df.columns[1]
    # Moving average
    smooth = df[y_col].rolling(window=window, min_periods=1).mean()
    plt.subplot(3, 1, i+1)
    plt.plot(df["Step"], df[y_col], marker='o', linestyle=':', alpha=0.1, color=color, label='Raw')
    plt.plot(df["Step"], smooth, linewidth=2, color=color, label=f'Smoothed (window={window})')
    plt.title(title, fontsize=14)
    plt.xlabel("Step", fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

plt.tight_layout()
plt.savefig(base_path + "lerobot_index_plot.png", dpi=200)
plt.show()
