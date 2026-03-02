import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

input_file = "data/enriched_project_dataset.csv"
output_dir = "visuals"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Loading data for visualization...")
df = pd.read_csv(input_file, low_memory=False)

# Setup the visual style
sns.set_theme(style="whitegrid")

# --- PLOT 1: Sentiment vs. Priority ---
print("Generating Chart 1: Sentiment by Priority...")
plt.figure(figsize=(10, 6))
priority_order = ['Blocker', 'Critical', 'Major', 'Minor', 'Trivial']
plot_df1 = df[df['priority'].isin(priority_order)]

sns.boxplot(x='priority', y='behavior_score', data=plot_df1, order=priority_order, palette="coolwarm_r")
plt.title('Developer Sentiment Distribution by Task Priority', fontsize=14)
plt.ylabel('Behavior Score (-1.0 = Max Stress, +1.0 = Max Positive)')
plt.xlabel('JIRA Ticket Priority')
plt.axhline(0, color='grey', linestyle='--', linewidth=1)
plt.savefig(f"{output_dir}/1_sentiment_by_priority.png", dpi=300, bbox_inches='tight')
plt.close()

# --- PLOT 2: Stalled vs Active Sentiment ---
print("Generating Chart 2: Stalled vs Active Tasks...")
plt.figure(figsize=(8, 5))
df['Task Status'] = df['is_stalled'].map({1: 'Stalled / Open', 0: 'Resolved / Closed'})

sns.violinplot(x='Task Status', y='behavior_score', data=df, palette="muted", inner="quartile")
plt.title('Does Project Delay Impact Developer Emotion?', fontsize=14)
plt.ylabel('Behavior Score')
plt.axhline(0, color='red', linestyle=':', linewidth=2)
plt.savefig(f"{output_dir}/2_stalled_vs_active.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Samuel: Visuals saved successfully to the '{output_dir}' folder!")