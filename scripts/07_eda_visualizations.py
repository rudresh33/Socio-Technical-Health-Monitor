"""
=============================================================================
EDA VISUALISATIONS 
=============================================================================
Operates on the AGGREGATED ticket-level dataset (916 unique tickets).

Produces six charts for Appendix B (descriptive statistics) and Appendix D
(feature correlation):
  B1 — Email volume distribution
  B2 — Unique senders distribution
  B3 — Sentiment by ticket status
  B4 — Class balance across the four subprojects
  B5 — Ticket count by year
  D1 — Pairwise feature correlation heatmap

Output: charts/*.png at 150 DPI.
=============================================================================
"""

import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
TICKET_FILE = "data/processed/final_ticket_level.csv"
EMAIL_FILE = "data/processed/isa3_enriched_dataset.csv"  # only for ticket → year mapping
OUTPUT_DIR = "charts"
DPI = 150

COLOR_RESOLVED = "#3b7dd8"   # blue
COLOR_STALLED = "#e74c3c"    # red
COLOR_NEUTRAL = "#7f8c8d"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "DejaVu Sans"


# ---------------------------------------------------------------------------
# AUDIT REPORT — what changed from the previous version of this script
# ---------------------------------------------------------------------------
print("=" * 70)
print("AUDIT REPORT — Visualisation Script vs Current Dataset")
print("=" * 70)

audit_messages = [
    ("Input file",
     "Old: data/processed/isa3_enriched_dataset.csv (email-level, ~7,051 rows)",
     "New: data/processed/final_ticket_level.csv (916 unique tickets — correct unit of analysis)"),
    ("Sentiment column",
     "Old: 'behavior_score' (per-email VADER score, not present in ticket-level dataset)",
     "New: 'avg_sentiment' (ticket-level mean, the variable used by the model)"),
    ("Class labels",
     "Old: 'Stalled / Open' vs 'Resolved / Closed'",
     "New: 'Stalled' vs 'Resolved/Active' (matches dissertation terminology)"),
    ("Output directory",
     "Old: visuals/eda_plots/ (300 DPI)",
     "New: charts/ (150 DPI per appendix spec)"),
    ("Removed: Chart 1 'Sentiment by Priority' (boxplot)",
     "Reason: not in dissertation appendix; superseded by B3 + B4.",
     ""),
    ("Removed: Chart 5 'Monthly Sentiment Trend'",
     "Reason: relied on hard-coded Hadoop 3.4.1 release annotations and email_date,",
     "        which is no longer present at the ticket level. Replaced by B5 (yearly counts)."),
    ("Removed: Chart 7 'Task Status by Priority'",
     "Reason: overlaps B4. Project-level class balance is the appendix-mandated cut.",
     ""),
    ("Replaced: Chart 6 'Email Volume Distribution' → B1 (using ticket-level data, not email-level dedupe)",
     "", ""),
    ("Replaced: Chart 3 'Correlation Heatmap' → D1 (now includes is_stalled and unique_senders)",
     "", ""),
    ("Added: B2 (unique senders distribution), B4 (class balance by project), B5 (tickets by year)",
     "", ""),
]
for title, line1, line2 in audit_messages:
    print(f"\n  • {title}")
    if line1: print(f"    {line1}")
    if line2: print(f"    {line2}")
print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------------
print("\nLoading ticket-level dataset...")
df = pd.read_csv(TICKET_FILE, low_memory=False)
print(f"  Shape: {df.shape}")

stalled_n = int(df["is_stalled"].sum())
resolved_n = int((df["is_stalled"] == 0).sum())
print(f"  Stalled : {stalled_n}")
print(f"  Resolved/Active : {resolved_n}")
print(f"  Class ratio : {resolved_n / max(stalled_n, 1):.1f} : 1\n")

# Status text column for plotting
df["Ticket Status"] = df["is_stalled"].map({0: "Resolved/Active", 1: "Stalled"})


# ---------------------------------------------------------------------------
# CHART B1 — Email Volume Distribution
# ---------------------------------------------------------------------------
print("Generating B1: Email Volume Distribution...")
fig, ax = plt.subplots(figsize=(10, 5.5))

vol = df["email_count_per_ticket"].dropna()
vol_capped = vol.clip(upper=30)

ax.hist(vol_capped, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
ax.axvline(vol.median(), color="red", linestyle="--", linewidth=1.8,
           label=f"Median = {vol.median():.0f}")
ax.axvline(vol.mean(), color="orange", linestyle=":", linewidth=1.8,
           label=f"Mean = {vol.mean():.1f}")

ax.set_title("B1 — Distribution of Email Volume per Ticket\n(916 unique tickets; long tail capped at 30 for display)",
             fontsize=12, fontweight="bold", pad=12)
ax.set_xlabel("Number of Emails Referencing the Ticket", fontsize=11)
ax.set_ylabel("Number of Tickets", fontsize=11)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/B1_email_volume_distribution.png", dpi=DPI, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTPUT_DIR}/B1_email_volume_distribution.png")


# ---------------------------------------------------------------------------
# CHART B2 — Unique Senders Distribution
# ---------------------------------------------------------------------------
print("Generating B2: Unique Senders Distribution...")

if "unique_senders" not in df.columns:
    print("  SKIPPED — unique_senders column not present in dataset.")
else:
    fig, ax = plt.subplots(figsize=(10, 5.5))

    senders = df["unique_senders"].dropna()
    senders_capped = senders.clip(upper=15)

    ax.hist(senders_capped, bins=range(0, 17), color="#5b8def", edgecolor="white", alpha=0.85)
    ax.axvline(senders.mean(), color="orange", linestyle=":", linewidth=1.8,
               label=f"Mean = {senders.mean():.1f}")

    ax.set_title("B2 — Distribution of Unique Senders per Ticket\n(916 unique tickets; capped at 15 for display)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Distinct Senders", fontsize=11)
    ax.set_ylabel("Number of Tickets", fontsize=11)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/B2_unique_senders_distribution.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/B2_unique_senders_distribution.png")


# ---------------------------------------------------------------------------
# CHART B3 — Sentiment by Ticket Status
# ---------------------------------------------------------------------------
print("Generating B3: Sentiment by Ticket Status...")
fig, ax = plt.subplots(figsize=(8.5, 5.5))

order = ["Resolved/Active", "Stalled"]
palette = {"Resolved/Active": COLOR_RESOLVED, "Stalled": COLOR_STALLED}

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    sns.violinplot(
        x="Ticket Status", y="avg_sentiment", hue="Ticket Status",
        data=df, order=order, palette=palette, inner="quartile",
        legend=False, ax=ax,
    )

ax.axhline(0, color="black", linestyle=":", linewidth=1.2, label="Neutral baseline (0)")
ax.set_title("B3 — Sentiment Distribution by Ticket Status\n(916 unique tickets)",
             fontsize=12, fontweight="bold", pad=12)
ax.set_ylabel("Average Sentiment Score (−1 = negative, +1 = positive)", fontsize=11)
ax.set_xlabel("Ticket Status", fontsize=11)
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/B3_sentiment_by_status.png", dpi=DPI, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTPUT_DIR}/B3_sentiment_by_status.png")


# ---------------------------------------------------------------------------
# CHART B4 — Class Balance Across Subprojects
# ---------------------------------------------------------------------------
print("Generating B4: Class Balance by Project...")

project_order = ["HADOOP", "HDFS", "YARN", "MAPREDUCE"]
plot_df = df[df["project.key"].isin(project_order)].copy()

counts = (plot_df.groupby(["project.key", "Ticket Status"])
                 .size().unstack(fill_value=0))
for c in ["Resolved/Active", "Stalled"]:
    if c not in counts.columns:
        counts[c] = 0
counts = counts[["Resolved/Active", "Stalled"]].reindex(project_order, fill_value=0)

fig, ax = plt.subplots(figsize=(10, 6))
counts.plot(kind="bar", stacked=True, ax=ax,
            color=[COLOR_RESOLVED, COLOR_STALLED],
            edgecolor="white", linewidth=0.8)

ax.set_title("B4 — Class Balance Across the Four Subprojects",
             fontsize=12, fontweight="bold", pad=12)
ax.set_xlabel("Subproject", fontsize=11)
ax.set_ylabel("Number of Tickets", fontsize=11)
ax.set_xticklabels(project_order, rotation=0, fontsize=10)
ax.legend(["Resolved/Active", "Stalled"], fontsize=10, loc="upper right")

# Per-bar totals + class ratio above each bar
for i, project in enumerate(project_order):
    total = int(counts.loc[project].sum())
    stalled = int(counts.loc[project, "Stalled"])
    resolved = int(counts.loc[project, "Resolved/Active"])
    ratio = resolved / max(stalled, 1)
    ax.text(i, total + max(counts.values.sum(axis=1)) * 0.02,
            f"n={total}\n{ratio:.1f}:1",
            ha="center", va="bottom", fontsize=8.5, color="#333")

# Stacked-segment value labels
for container in ax.containers:
    ax.bar_label(container, label_type="center", fontsize=8.5,
                 color="white", fontweight="bold",
                 fmt=lambda x: f"{int(x)}" if x > 8 else "")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/B4_class_balance_by_project.png", dpi=DPI, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTPUT_DIR}/B4_class_balance_by_project.png")


# ---------------------------------------------------------------------------
# CHART B5 — Ticket Count by Year (2018–2024)
# ---------------------------------------------------------------------------
print("Generating B5: Tickets by Year...")

year_df = None
if os.path.exists(EMAIL_FILE):
    try:
        email_min = pd.read_csv(EMAIL_FILE, low_memory=False, usecols=["ticket_key", "created"])
        email_min["created_dt"] = pd.to_datetime(email_min["created"], errors="coerce", utc=True)
        ticket_year = (email_min.dropna(subset=["created_dt"])
                                .groupby("ticket_key")["created_dt"].min()
                                .dt.year.astype("Int64"))
        year_df = df.merge(ticket_year.rename("year"),
                           left_on="ticket_key", right_index=True, how="left")
    except Exception as exc:
        print(f"  WARNING: could not derive year from email-level dataset ({exc}). Skipping B5.")

if year_df is None or year_df["year"].isna().all():
    print("  SKIPPED — no usable 'created' year information available.")
else:
    yearly = (year_df.dropna(subset=["year"])
                     .groupby(["year", "Ticket Status"]).size()
                     .unstack(fill_value=0))
    for c in ["Resolved/Active", "Stalled"]:
        if c not in yearly.columns:
            yearly[c] = 0
    yearly = yearly[["Resolved/Active", "Stalled"]].sort_index()

    fig, ax = plt.subplots(figsize=(11, 5.5))
    yearly.plot(kind="bar", stacked=True, ax=ax,
                color=[COLOR_RESOLVED, COLOR_STALLED],
                edgecolor="white", linewidth=0.6)

    ax.set_title("B5 — Ticket Count by Year (Project Coverage 2018–2024)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Year (Earliest Mention in Mailing Lists)", fontsize=11)
    ax.set_ylabel("Number of Unique Tickets", fontsize=11)
    ax.set_xticklabels([int(y) for y in yearly.index], rotation=0, fontsize=10)
    ax.legend(["Resolved/Active", "Stalled"], fontsize=10, loc="upper left")

    # Total above each bar
    totals = yearly.sum(axis=1)
    for i, total in enumerate(totals.values):
        ax.text(i, total + totals.max() * 0.02, f"{int(total)}",
                ha="center", va="bottom", fontsize=8.5, color="#333")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/B5_tickets_by_year.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/B5_tickets_by_year.png")


# ---------------------------------------------------------------------------
# CHART D1 — Correlation Heatmap
# ---------------------------------------------------------------------------
print("Generating D1: Correlation Heatmap...")

corr_features = [
    "email_count_per_ticket", "subject_length",
    "avg_sentiment", "sentiment_variance", "sentiment_trend",
    "priority_numeric", "unique_senders",
    "description_length", "is_stalled",
]
existing = [c for c in corr_features if c in df.columns]
missing = [c for c in corr_features if c not in df.columns]
if missing:
    print(f"  Note: missing columns excluded from heatmap — {missing}")

display_names = {
    "email_count_per_ticket": "Email Volume",
    "subject_length": "Subject Length",
    "avg_sentiment": "Avg Sentiment",
    "sentiment_variance": "Sentiment Variance",
    "sentiment_trend": "Sentiment Trend",
    "priority_numeric": "Priority (numeric)",
    "unique_senders": "Unique Senders",
    "description_length": "Description Length",
    "is_stalled": "Stalled (target)",
}
corr_matrix = df[existing].corr().rename(columns=display_names, index=display_names)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            linewidths=0.5, square=True,
            annot_kws={"size": 9}, ax=ax)
ax.set_title("D1 — Pairwise Pearson Correlation Across Features\n(916 unique tickets)",
             fontsize=12, fontweight="bold", pad=12)
plt.xticks(rotation=35, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/D1_correlation_heatmap.png", dpi=DPI, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTPUT_DIR}/D1_correlation_heatmap.png")


# ---------------------------------------------------------------------------
# VERIFICATION SUMMARY
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)

generated = sorted(
    f for f in os.listdir(OUTPUT_DIR)
    if f.endswith(".png") and (f.startswith("B") or f.startswith("D"))
)
print(f"  Charts generated : {len(generated)}")
for f in generated:
    print(f"    - {OUTPUT_DIR}/{f}")
print(f"  Dataset shape    : {df.shape} (expected ~916 rows)")
print(f"  Stalled / Active : {stalled_n} / {resolved_n}")
print(f"  Class ratio      : {resolved_n / max(stalled_n, 1):.1f} : 1")
print("=" * 70)
print("Done.")
