import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# Load data
df = pd.read_csv("Price.csv", sep=';')

# Filter and preprocess the data
filtered_df = df[df['PriceArea'] == 'DK2'][['HourDK', 'PriceArea', 'PriceEUR']].copy()
filtered_df['PriceEUR'] = filtered_df['PriceEUR'].str.replace(',', '.').astype(float)

# Parse HourDK column as datetime
filtered_df['HourDK'] = pd.to_datetime(filtered_df['HourDK'], format='%d/%m/%Y', dayfirst=True)

# Reset index
filtered_df.reset_index(drop=True, inplace=True)

# Clustering prices into 3 clusters
prices = filtered_df['PriceEUR'].values.reshape(-1, 1)
kmeans = KMeans(n_clusters=3, random_state=42)
filtered_df['PriceCluster'] = kmeans.fit_predict(prices)

# Sort clusters by price levels
cluster_centers = kmeans.cluster_centers_.flatten()
sorted_clusters = sorted(range(len(cluster_centers)), key=lambda i: cluster_centers[i])
cluster_mapping = {old: new for new, old in enumerate(sorted_clusters)}
filtered_df['PriceCluster'] = filtered_df['PriceCluster'].map(cluster_mapping)

# Calculate transition probabilities
price_cluster_sequence = filtered_df['PriceCluster'].values
n_clusters = 3
transition_counts = np.zeros((n_clusters, n_clusters), dtype=int)

for i in range(len(price_cluster_sequence) - 1):
    current_cluster = price_cluster_sequence[i]
    next_cluster = price_cluster_sequence[i + 1]
    transition_counts[current_cluster, next_cluster] += 1

# Normalize to get probabilities
transition_probabilities = transition_counts / transition_counts.sum(axis=1, keepdims=True)
transition_probabilities = np.nan_to_num(transition_probabilities)  # Handle NaNs

# Convert transition probabilities to a DataFrame with rounded values
transition_probabilities_df = pd.DataFrame(
    transition_probabilities.round(4), 
    columns=['Cluster_0', 'Cluster_1', 'Cluster_2'], 
    index=['Cluster_0', 'Cluster_1', 'Cluster_2']
)

# Save the rounded matrix for further use
transition_probabilities_decimal = transition_probabilities.round(4)

# Display the rounded DataFrame
print("Transition Probabilities (Rounded):")
print(transition_probabilities_df)

# Define SOC levels and Price Clusters
SOC_levels = [0, 100, 200, 300, 400, 500]  # 6 SOC levels
PriceClusters = [0, 1, 2]  # 3 Price Clusters
actions = {0: 'charge', 1: 'discharge', 2: 'idle'}  # Actions mapping

# Define all possible states (SOC, PriceCluster)
states = [(soc, cluster) for soc in SOC_levels for cluster in PriceClusters]

# Define valid actions for each state
valid_actions = {}
for soc in SOC_levels:
    for cluster in PriceClusters:
        state = (soc, cluster)
        if soc == 0:
            valid_actions[state] = [0, 2]  # Charge, Idle
        elif soc == 500:
            valid_actions[state] = [1, 2]  # Discharge, Idle
        else:
            valid_actions[state] = [0, 1, 2]  # Charge, Discharge, Idle

# Dynamic reward function based on market price
def calculate_reward_dynamic(state, action, market_price):
    """
    Calculate reward dynamically based on the market price at a given time.
    
    Parameters:
    - state: Tuple (SOC, PriceCluster)
    - action: int (0=Charge, 1=Discharge, 2=Idle)
    - market_price: float, the actual price at the current time
    
    Returns:
    - reward: float, the calculated reward
    """
    soc, _ = state
    if action == 0:  # Charge
        return -market_price * 100
    elif action == 1:  # Discharge
        return market_price * 100
    elif action == 2:  # Idle
        return 0
    else:
        raise ValueError("Invalid action")


# Example transition probabilities for Price Clusters (replace with actual computed values)
transition_probabilities = transition_probabilities_decimal

import numpy as np

# Initialize the state-action transition matrix (54 rows, 18 columns)
n_states = len(states)
n_actions = len(actions)
P_sa = np.zeros((n_states * n_actions, n_states))

# Populate the transition probabilities
for state_idx, current_state in enumerate(states):
    soc, current_cluster = current_state
    for action_idx, action in enumerate(actions.keys()):  # 0=Charge, 1=Discharge, 2=Idle
        # Check if the action is valid for the current state
        if action == 0 and soc == 500:  # Cannot charge when SOC=500
            continue
        if action == 1 and soc == 0:  # Cannot discharge when SOC=0
            continue

        # Compute new SOC based on the action
        if action == 0:  # Charge
            new_soc = min(500, soc + 100)
        elif action == 1:  # Discharge
            new_soc = max(0, soc - 100)
        elif action == 2:  # Idle
            new_soc = soc
        else:
            raise ValueError("Invalid action")

        # Compute row index for this state-action pair
        row_index = state_idx * n_actions + action_idx

        # Populate the row for this state-action pair
        for next_state_idx, next_state in enumerate(states):
            next_soc, next_cluster = next_state
            if next_soc == new_soc:  # Only assign probabilities to states with updated SOC
                # Assign probability based on price cluster transitions
                prob = transition_probabilities[current_cluster, next_cluster]
                P_sa[row_index, next_state_idx] = prob

# Normalize rows where valid actions exist
row_sums = P_sa.sum(axis=1)
for i, row_sum in enumerate(row_sums):
    if row_sum > 0:  # Avoid dividing rows with no transitions
        P_sa[i] /= row_sum

# Transition probability matrix
P_sa = np.round(P_sa, 5).tolist()

#print(P_sa[0])
#print(P_sa[1])
#print(P_sa[4])
#print(P_sa[7])





