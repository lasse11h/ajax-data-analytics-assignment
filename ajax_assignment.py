### Imports ###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from math import pi

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

### 1. Analysis ###

# Load the data
event_data = pd.read_parquet("event_data.parquet")

# Explore the data 

event_data.info()
event_data.describe(include='all')  # include object/categorical too
print(event_data.head())

## a. Finishing Efficiency ##

# First, I filter the data to only include shots
shots = event_data[event_data['action_type'] == 'shot'].copy()

# I turn the 'goal' column into a 0/1 flag because it's NaN when no goal was scored
shots['goal_flag'] = shots['goal'].notna() * 1

# Now I calculate total goals, xG, and number of shots for each player
finishing_stats = shots.groupby('player_id').agg(
    total_goals=('goal_flag', 'sum'),
    total_xg=('shot_xg', 'sum'),
    total_shots=('goal_flag', 'count')  # one row = one shot
).reset_index()

# To reduce noise from players with only a few shots, I filter for those with at least 10 attempts — this threshold can be adjusted if needed
min_shots = 10
finishing_stats = finishing_stats[finishing_stats['total_shots'] >= min_shots]

# This gives me a finishing efficiency score by comparing goals scored to total xG
finishing_stats['finishing_efficiency'] = finishing_stats['total_goals'] / finishing_stats['total_xg']

# Finally, I sort the players by this metric to find the top overperformers
top_finishers = finishing_stats.sort_values(by='finishing_efficiency', ascending=False)

top_finishers.head(20)

# Visualization 
# Barchar Chart of Top 10 Overperforming Finishers

top10 = top_finishers.head(10)
plt.barh(top10['player_id'].astype(str), top10['finishing_efficiency'])
plt.xlabel("Finishing Efficiency (Goals / xG)")
plt.title("Top 10 Overperforming Finishers")
plt.gca().invert_yaxis()
plt.show()

# Scatter Plot of Goals vs xG
plt.figure(figsize=(8,6))
plt.scatter(finishing_stats['total_xg'], finishing_stats['total_goals'], alpha=0.7)
plt.plot([0, finishing_stats['total_xg'].max()], [0, finishing_stats['total_xg'].max()], 'r--')
plt.xlabel("Total xG")
plt.ylabel("Total Goals")
plt.title("Goals vs xG per Player")
plt.grid(True)
plt.tight_layout()
plt.show()

# Same plot using Plotly for easier inspection of good players

# Create the scatter plot
fig = px.scatter(
    finishing_stats,
    x="total_xg",
    y="total_goals",
    hover_data=["player_id", "total_shots"],
    labels={"total_xg": "Total xG", "total_goals": "Total Goals"},
    title="Goals vs xG per Player (Min 10 Shots)"
)

# Add reference line (y = x)
fig.add_shape(
    type="line",
    x0=0, y0=0,
    x1=finishing_stats["total_xg"].max(),
    y1=finishing_stats["total_xg"].max(),
    line=dict(color="red", dash="dash")
)

# plot as html
fig.write_html("goals_vs_xg_plot.html")

## 1b. Midfield Suitability ##
df = pd.read_parquet("event_data.parquet")  # make sure pyarrow is installed

# Focus on actions that matter for central midfield profiles
actions = ['pass', 'dribble','shot', 'reception', 'interception', 'loose_ball_regain']
df_mid = df[df['action_type'].isin(actions)].copy()

# Total minutes per player
minutes_df = (
    df.groupby(['player_id', 'match_id'])['game_time_in_sec'].max()
    .groupby('player_id').sum().reset_index()
)
minutes_df.columns = ['player_id', 'total_seconds']
minutes_df['minutes_played'] = minutes_df['total_seconds'] / 60

# Count number of each action
action_counts = (
    df_mid.groupby(['player_id', 'action_type'])
    .size().unstack(fill_value=0).reset_index()
)

# Total bypassed opponents
bypassed_df = df_mid.groupby('player_id')['bypassed_opponents'].sum().reset_index()
bypassed_df.columns = ['player_id', 'bypassed_total']

# Merge base features
df_stats = (
    action_counts.merge(minutes_df[['player_id', 'minutes_played']], on='player_id')
    .merge(bypassed_df, on='player_id', how='left')
)
df_stats = df_stats[df_stats['minutes_played'] >= 30]

# Per90 features for count-based actions
for col in ['reception', 'interception', 'loose_ball_regain']:
    if col in df_stats.columns:
        df_stats[col + '_per90'] = df_stats[col] / df_stats['minutes_played'] * 90

df_stats['bypassed_per90'] = df_stats['bypassed_total'] / df_stats['minutes_played'] * 90

# Composure: success under pressure × high-pressure actions per 90
df_mid['is_success'] = df_mid['result'].str.lower() == 'success'
high_pressure = df_mid[df_mid['pressure'] >= 70]

# 1) Success rate under pressure
success_under_pressure = (
    high_pressure
    .groupby('player_id')['is_success']
    .mean()
    .reset_index(name='success_under_pressure')
)

# 2) Count high-pressure actions per player
hp_counts = (
    high_pressure
    .groupby('player_id')
    .size()
    .reset_index(name='hp_count')
)

# 3) Merge minutes played and compute high-pressure actions per 90 minutes
hp_counts = hp_counts.merge(
    minutes_df[['player_id', 'minutes_played']],
    on='player_id',
    how='left'
).fillna({'hp_count': 0, 'minutes_played': 0})
hp_counts['pressure_per90'] = hp_counts['hp_count'] / hp_counts['minutes_played'] * 90

# 4) Combine into composure metric
composure_df = success_under_pressure.merge(
    hp_counts[['player_id', 'pressure_per90']],
    on='player_id',
    how='outer'
).fillna(0)
composure_df['composure_per90'] = (
    composure_df['success_under_pressure'] * composure_df['pressure_per90']
)

# Merge composure into stats
df_stats = df_stats.merge(
    composure_df[['player_id', 'composure_per90']],
    on='player_id',
    how='left'
).fillna({'composure_per90': 0})

# Pass, dribble and shot success rates
success_subset = df_mid[df_mid['action_type'].isin(['pass', 'dribble'])].copy()
success_rates = (
    success_subset.groupby(['player_id', 'action_type'])['result']
    .value_counts().unstack(fill_value=0).reset_index()
)
 # Only sum the 'success' and 'fail' counts for total attempts
numeric_cols = ['success', 'fail']
success_rates['total'] = success_rates[numeric_cols].sum(axis=1)
success_rates['success_rate'] = success_rates['success'] / success_rates['total']

pivot_success = success_rates.pivot(index='player_id', columns='action_type', values='success_rate').reset_index()
pivot_success = pivot_success.rename(columns={'pass': 'pass_success', 'dribble': 'dribble_success'})

# Merge final metrics
df_final = df_stats.merge(pivot_success, on='player_id', how='left').fillna(0)

# Normalize features before computing the final score
from sklearn.preprocessing import MinMaxScaler

# Normalize features before computing the final score
scaler = MinMaxScaler()
scale_features = [
    'bypassed_per90',
    'reception_per90',
    'interception_per90',
    'loose_ball_regain_per90',
    'composure_per90'
]

# Fill missing values before scaling
df_final[scale_features] = df_final[scale_features].fillna(0)
# Fit and transform
df_final[scale_features] = scaler.fit_transform(df_final[scale_features])

# Final score based on weighted mix of quality + volume
df_final['midfield_score'] = (
    0.1 * df_final['pass_success'] +
    0.1 * df_final['dribble_success'] +
    0.2 * df_final['bypassed_per90'] +
    0.1 * df_final['reception_per90'] +
    0.1 * df_final['interception_per90'] +
    0.1 * df_final['loose_ball_regain_per90'] +
    0.2 * df_final['composure_per90']
)

print(df_final[scale_features].agg(['min','max','mean']).T)

# Rank players and show top 15
df_final['rank'] = df_final['midfield_score'].rank(ascending=False)
top_mids = df_final.sort_values('midfield_score', ascending=False).reset_index(drop=True)

print(top_mids[['player_id', 'midfield_score', 'rank',
                'pass_success', 'dribble_success',
                'reception_per90', 'interception_per90', 'loose_ball_regain_per90',
                'composure_per90', 'bypassed_per90']].head(15))


# Visualize the results
# Pick top 5 midfielders
top = df_final.sort_values("midfield_score", ascending=False).head(5)

# Features to plot (including shot_success)
features = [
    'pass_success', 'dribble_success',
    'reception_per90', 'interception_per90',
    'loose_ball_regain_per90', 'bypassed_per90',
    'composure_per90'
]

# Build normalization maxima: success rates in [0,1]; per90 metrics scaled to top performer
max_norm = {}
for f in features:
    if f in ['pass_success', 'dribble_success']:
        max_norm[f] = 1.0
    else:
        max_norm[f] = top[f].max()

# Normalize top-5 profiles
normalized = top.copy()
for f in features:
    normalized[f] = normalized[f] / max_norm[f]

# Compute average profile: raw mean for success rates; per90 mean normalized by top
avg_profile = {}
for f in features:
    if f in ['pass_success', 'dribble_success']:
        avg_profile[f] = df_final[f].mean()
    else:
        avg_profile[f] = df_final[f].mean() / max_norm[f]
avg_values = list(avg_profile.values()) + list(avg_profile.values())[:1]

# Prepare angles for radar chart
labels = features
num_vars = len(labels)
angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
angles += angles[:1]

plt.figure(figsize=(8, 8))

# Plot each of the top 5 player profiles
for _, row in normalized.iterrows():
    vals = row[features].tolist()
    vals += vals[:1]
    plt.polar(angles, vals, label=f"Player {int(row['player_id'])}")

# Plot the average profile
plt.polar(angles, avg_values, label="Average Player", linestyle='--', linewidth=2)

plt.xticks(angles[:-1], labels, color='grey', size=8)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.title("Top 5 vs Average Midfielder Profiles (Normalized)", size=12)
plt.show()

sns.scatterplot(data=df_final, x='pass_success', y='dribble_success', hue='midfield_score', size='bypassed_per90')
plt.title("Pass vs Dribble Success (Color = Score, Size = Bypassed)")

ax = sns.heatmap(df_final[features + ['midfield_score']].corr(), annot=True, cmap='coolwarm')
if ax.get_legend() is not None:
    ax.get_legend().remove()
plt.show()

### 2. Data Science ###

# Load shot events
df = pd.read_parquet("event_data.parquet")
shots = df[df['action_type'] == 'shot'].copy()

# Create binary goal variable
shots['is_goal'] = (shots['goal'] == 1).astype(int)

# Feature: distance to goal (goal center at 105,34)
shots['dist_to_goal'] = np.sqrt((105 - shots['start_x'])**2 + (34 - shots['start_y'])**2)

# Feature: shot angle (simplified formula)
shots['angle'] = np.arctan2(7.32 / 2, (105 - shots['start_x']))

# Feature: interaction term
shots['dist_angle_interaction'] = shots['dist_to_goal'] * shots['angle']

# Feature set
features = ['dist_to_goal', 'angle', 'dist_angle_interaction', 'pressure', 'opponents']

# Train/test split
X = shots[features]
y = shots['is_goal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Build pipeline (scaling + logistic regression)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(max_iter=200)),
])
pipeline.fit(X_train, y_train)

# Evaluate performance
y_pred = pipeline.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred)
print(f"Test ROC-AUC: {auc:.3f}")

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"LogReg (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for xG Model")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Assign predicted xG values
shots['predicted_xg'] = pipeline.predict_proba(shots[features])[:, 1]

# Compare with original xG
print(shots[['start_x', 'start_y', 'shot_xg', 'predicted_xg']].head(10))


### 3. Data Engineering ###
# Load JSON file into DataFrame
df_json = pd.read_json("League_mapping_20250313.json", encoding="utf-8-sig")

# Flatten 'items' list
df_items = pd.json_normalize(df_json['items'])

# Select only needed columns
df_small = df_items[[
    'league.id', 'league.name', 'league.gender',
    'league.nation.id', 'league.nation.name',
    'league.nation.alpha3Code', 'league.nation.isActive',
    'league.nation.eu', 'source.sourceTypeCode', 'source.sourceValue'
]]

# Pivot to one row per league with source columns
df_flat = df_small.pivot_table(
    index=[
        'league.id', 'league.name', 'league.gender',
        'league.nation.id', 'league.nation.name',
        'league.nation.alpha3Code', 'league.nation.isActive',
        'league.nation.eu'
    ],
    columns='source.sourceTypeCode',
    values='source.sourceValue',
    aggfunc='first'
).reset_index()

# Ensure all expected source columns exist
for col in ['milkeyway', 'skillbot', 'statstrue', 'system_tech']:
    if col not in df_flat.columns:
        df_flat[col] = ''

# Reorder columns to match end_result.csv
df_flat = df_flat[
    ['league.id', 'league.name', 'league.gender',
     'league.nation.id', 'league.nation.name',
     'league.nation.alpha3Code', 'league.nation.isActive',
     'league.nation.eu', 'milkeyway', 'skillbot',
     'statstrue', 'system_tech']
]

# Save to CSV
df_flat.to_csv("end_result.csv", index=False)

print("Flattened JSON saved to end_result.csv")