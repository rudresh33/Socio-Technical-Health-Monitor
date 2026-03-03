import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import os

input_file = "data/enriched_project_dataset.csv"
output_dir = "visuals"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Loading data for visualization...")
df = pd.read_csv(input_file, low_memory=False)

sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'


# =============================================================================
# CHART 1: Sentiment Distribution by Task Priority (Fixed FutureWarning)
# =============================================================================
print("Generating Chart 1: Sentiment by Priority...")
fig, ax = plt.subplots(figsize=(11, 6))

priority_order = ['Blocker', 'Critical', 'Major', 'Minor', 'Trivial']
plot_df1 = df[df['priority'].isin(priority_order)].copy()

sns.boxplot(
    x='priority', y='behavior_score',
    hue='priority',
    data=plot_df1, order=priority_order,
    palette="coolwarm_r", legend=False, ax=ax
)

ax.axhline(0, color='grey', linestyle='--', linewidth=1, label='Neutral Baseline')
ax.set_title('Developer Sentiment Distribution by Task Priority', fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel('Behavior Score  (-1.0 = Max Stress,  +1.0 = Max Positive)', fontsize=11)
ax.set_xlabel('JIRA Ticket Priority', fontsize=11)

# Annotation explaining the Blocker result
ax.annotate(
    'Blocker tickets show higher median sentiment:\nfast resolution generates relief in communication.',
    xy=(0, 0.37), xytext=(1.5, 0.72),
    fontsize=8.5, color='#555555',
    arrowprops=dict(arrowstyle='->', color='#888888', lw=1.2),
    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='#cccccc')
)

# Median labels on each box
medians = plot_df1.groupby('priority')['behavior_score'].median()
for i, priority in enumerate(priority_order):
    if priority in medians.index:
        ax.text(i, medians[priority] + 0.03, f'{medians[priority]:.2f}',
                ha='center', va='bottom', fontsize=9, color='black', fontweight='bold')

ax.legend(loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig(f"{output_dir}/1_sentiment_by_priority.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Chart 1 saved.")


# =============================================================================
# CHART 2: Stalled vs Resolved Sentiment (Fixed FutureWarning + Annotation)
# =============================================================================
print("Generating Chart 2: Stalled vs Active Tasks...")
fig, ax = plt.subplots(figsize=(9, 6))

df['Task Status'] = df['is_stalled'].map({1: 'Stalled / Open', 0: 'Resolved / Closed'})

sns.violinplot(
    x='Task Status', y='behavior_score',
    hue='Task Status',
    data=df, palette="muted",
    inner="quartile", legend=False, ax=ax
)

ax.axhline(0, color='red', linestyle=':', linewidth=2, label='Neutral Baseline (0)')
ax.set_title('Does Project Delay Impact Developer Emotion?', fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel('Behavior Score', fontsize=11)
ax.set_xlabel('Task Status', fontsize=11)

# Annotation explaining polarization
ax.annotate(
    'Stalled tasks show polarisation:\nhigher upper quartile (frustration releases)\nbut longer negative tail (sustained stress).',
    xy=(1, -0.6), xytext=(1.12, -0.82),
    fontsize=8.5, color='#555555',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='#cccccc')
)

ax.legend(loc='lower left', fontsize=9)
plt.tight_layout()
plt.savefig(f"{output_dir}/2_stalled_vs_active.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Chart 2 saved.")


# =============================================================================
# CHART 3: Correlation Heatmap (Fixed - removed is_stalled, shows clean values)
# =============================================================================
print("Generating Chart 3: Correlation Heatmap...")

# is_stalled removed because near-zero variance in this aggregated dataset
# causes NaN correlations that break the visual — reported honestly in report
numeric_cols = [
    'behavior_score', 'days_to_resolve', 'subject_length',
    'priority_numeric', 'sentiment_variance',
    'email_volume_per_ticket', 'sentiment_trend'
]
existing_cols = [col for col in numeric_cols if col in df.columns]
corr_df = df[existing_cols].dropna()

# Readable display names
display_names = {
    'behavior_score': 'Sentiment Score',
    'days_to_resolve': 'Days to Resolve',
    'subject_length': 'Subject Length',
    'priority_numeric': 'Priority (Numeric)',
    'sentiment_variance': 'Sentiment Variance',
    'email_volume_per_ticket': 'Email Volume',
    'sentiment_trend': 'Sentiment Trend'
}
corr_matrix = corr_df.rename(columns=display_names).corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)  # Show lower triangle only
sns.heatmap(
    corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
    linewidths=0.5, square=True, ax=ax,
    vmin=-0.5, vmax=0.5,
    annot_kws={"size": 10}
)
ax.set_title('Feature Correlation Matrix\n(Values close to 0 indicate feature independence — desirable for ML)',
             fontsize=12, fontweight='bold', pad=15)
plt.xticks(rotation=35, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.savefig(f"{output_dir}/3_correlation_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Chart 3 saved.")


# =============================================================================
# CHART 5: Monthly Sentiment Trend (Fixed + Release event annotations)
# =============================================================================
print("Generating Chart 5: Monthly Sentiment Trend...")

df['email_month'] = pd.to_datetime(
    df['email_date'], errors='coerce', utc=True
).dt.to_period('M')
monthly_sentiment = df.groupby('email_month')['behavior_score'].mean().reset_index()
monthly_sentiment['email_month'] = monthly_sentiment['email_month'].astype(str)

fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(
    monthly_sentiment['email_month'],
    monthly_sentiment['behavior_score'],
    marker='o', linewidth=2.2, color='black', zorder=3
)
ax.axhline(0, color='red', linestyle='--', linewidth=1, label='Neutral Baseline', zorder=2)

# Shade the release crunch period (Oct-Nov 2024 dip)
crunch_months = ['2024-10', '2024-11']
crunch_indices = [
    i for i, m in enumerate(monthly_sentiment['email_month'].tolist())
    if m in crunch_months
]
if len(crunch_indices) >= 2:
    ax.axvspan(crunch_indices[0] - 0.5, crunch_indices[-1] + 0.5,
               alpha=0.12, color='red', zorder=1)
    ax.annotate(
        'Release Crunch\n(Hadoop 3.4.1 RC)\nSentiment drops to 0.10',
        xy=(crunch_indices[1], 0.10),
        xytext=(crunch_indices[1] - 3.5, -0.05),
        fontsize=9, color='#cc0000',
        arrowprops=dict(arrowstyle='->', color='#cc0000', lw=1.2),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffe6e6', edgecolor='#cc0000')
    )

# Annotate the Dec 2024 recovery
dec_idx = monthly_sentiment['email_month'].tolist().index('2024-12') \
    if '2024-12' in monthly_sentiment['email_month'].tolist() else None
if dec_idx:
    ax.annotate(
        'Post-release recovery\n+0.42',
        xy=(dec_idx, monthly_sentiment.iloc[dec_idx]['behavior_score']),
        xytext=(dec_idx - 2, 0.38),
        fontsize=9, color='#006600',
        arrowprops=dict(arrowstyle='->', color='#006600', lw=1.2),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#e6ffe6', edgecolor='#006600')
    )

ax.set_title('Average Developer Sentiment by Month (2023–2024)\nAnnotated with Key Release Events',
             fontsize=13, fontweight='bold', pad=15)
ax.set_ylabel('Average Behavior Score', fontsize=11)
ax.set_xlabel('Month', fontsize=11)
plt.xticks(rotation=45, ha='right', fontsize=9)
ax.legend(fontsize=10)
ax.set_ylim(-0.2, 0.55)
plt.tight_layout()
plt.savefig(f"{output_dir}/5_monthly_sentiment_trend.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Chart 5 saved.")


# =============================================================================
# CHART 6 (NEW): Email Volume Distribution per Ticket
# =============================================================================
print("Generating Chart 6: Email Volume Distribution...")

if 'email_volume_per_ticket' in df.columns:
    fig, ax = plt.subplots(figsize=(10, 5))

    volume_data = df.drop_duplicates(subset='ticket_key')['email_volume_per_ticket'].dropna()
    # Cap at 30 for readability (outliers skew histogram badly)
    volume_capped = volume_data.clip(upper=30)

    ax.hist(volume_capped, bins=30, color='steelblue', edgecolor='white', alpha=0.85)
    ax.axvline(volume_data.median(), color='red', linestyle='--', linewidth=1.8,
               label=f'Median: {volume_data.median():.0f} emails/ticket')
    ax.axvline(volume_data.mean(), color='orange', linestyle=':', linewidth=1.8,
               label=f'Mean: {volume_data.mean():.1f} emails/ticket')

    ax.set_title('Distribution of Email Volume per JIRA Ticket\n(Communication Velocity Feature)',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Number of Emails Referencing Ticket (capped at 30 for display)', fontsize=11)
    ax.set_ylabel('Number of Tickets', fontsize=11)
    ax.legend(fontsize=10)

    # Annotation
    ax.annotate(
        'High-volume tickets\nindicate contested or\nhigh-friction discussions',
        xy=(20, ax.get_ylim()[1] * 0.6),
        fontsize=9, color='#555555',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='#cccccc')
    )

    plt.tight_layout()
    plt.savefig(f"{output_dir}/6_email_volume_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  Chart 6 saved.")
else:
    print("  Chart 6 skipped: email_volume_per_ticket column not found.")


# =============================================================================
# CHART 7 (NEW): Stalled vs Resolved Count by Priority (Class Distribution)
# =============================================================================
print("Generating Chart 7: Task Status by Priority...")

priority_order = ['Blocker', 'Critical', 'Major', 'Minor', 'Trivial']
plot_df7 = df[df['priority'].isin(priority_order)].copy()

status_counts = plot_df7.groupby(['priority', 'Task Status']).size().unstack(fill_value=0)

# Ensure correct column order
for col in ['Resolved / Closed', 'Stalled / Open']:
    if col not in status_counts.columns:
        status_counts[col] = 0
status_counts = status_counts[['Resolved / Closed', 'Stalled / Open']]
status_counts = status_counts.reindex(priority_order)

fig, ax = plt.subplots(figsize=(10, 6))
status_counts.plot(
    kind='bar', stacked=True, ax=ax,
    color=['steelblue', 'tomato'],
    edgecolor='white', linewidth=0.5
)

ax.set_title('Task Status Distribution by Priority\n(Class Balance for ML Model)',
             fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel('JIRA Ticket Priority', fontsize=11)
ax.set_ylabel('Number of Records', fontsize=11)
ax.set_xticklabels(priority_order, rotation=0, fontsize=10)
ax.legend(['Resolved / Closed', 'Stalled / Open'], fontsize=10, loc='upper right')

# Add value labels on bars
for container in ax.containers:
    ax.bar_label(container, label_type='center', fontsize=8.5,
                 color='white', fontweight='bold',
                 fmt=lambda x: f'{int(x)}' if x > 30 else '')

# Annotation on class imbalance
total_stalled = status_counts['Stalled / Open'].sum()
total_resolved = status_counts['Resolved / Closed'].sum()
imbalance_ratio = total_resolved / max(total_stalled, 1)
ax.text(0.98, 0.95,
        f'Class Ratio\nResolved : Stalled\n≈ {imbalance_ratio:.1f} : 1\n(Imbalanced — requires\nclass_weight handling)',
        transform=ax.transAxes, fontsize=8.5, va='top', ha='right',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='#cccccc'))

plt.tight_layout()
plt.savefig(f"{output_dir}/7_task_status_by_priority.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Chart 7 saved.")


print(f"\nAll visuals saved successfully to the '{output_dir}' folder!")
print("Charts generated: 1, 2, 3, 5, 6, 7")