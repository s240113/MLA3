import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv(r"D:/MLA 3/MLA3/Price.csv", sep=';')

# Filter and preprocess the data
filtered_df = df[df['PriceArea'] == 'DK2'][['HourDK', 'PriceArea', 'PriceEUR']].copy()
filtered_df['PriceEUR'] = filtered_df['PriceEUR'].str.replace(',', '.').astype(float)
# plt.figure(figsize=(10, 6))
# plt.plot(filtered_df['PriceEUR'], marker='o', linestyle='-', color='b', label='PriceEUR')
# plt.title('Price EUR Values')
# plt.xlabel('Index')
# plt.ylabel('Price EUR')
# plt.legend()
# plt.grid(True)
# plt.show()
# Parse HourDK column as datetime
filtered_df['HourDK'] = pd.to_datetime(filtered_df['HourDK'], format="%Y-%m-%d", dayfirst=True)

# Reset index
filtered_df.reset_index(drop=True, inplace=True)

# set the 0.25,0.5,0.75 as the initial centers for the KMeans clustering
quantiles = filtered_df['PriceEUR'].quantile([0.25, 0.5, 0.75]).values
print("Quantile Centers:", quantiles)
initial_centers = quantiles.reshape(-1, 1)
# Clustering prices into 3 clusters
prices = filtered_df['PriceEUR'].values.reshape(-1, 1)
kmeans = KMeans(n_clusters=3, init=initial_centers, n_init=1, random_state=42)
filtered_df['PriceCluster'] = kmeans.fit_predict(prices)
print("Final Cluster Centers:", kmeans.cluster_centers_)

# 5. Visualize the clustering results
plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green']

for cluster_id in range(3):
    cluster_points = filtered_df[filtered_df['PriceCluster'] == cluster_id]
    plt.scatter(
        cluster_points.index,
        cluster_points['PriceEUR'],
        color=colors[cluster_id],
        label=f'Cluster {cluster_id}',
        alpha=0.7
    )
# plot cluster centers
plt.scatter(
    [filtered_df.index.to_numpy().mean()] * 3,
    kmeans.cluster_centers_.flatten(),
    color='black',
    label='Cluster Centers',
    marker='X',
    s=200
)

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
price_transition_probabilities = transition_probabilities.round(4)

# Display the rounded DataFrame
print("Transition Probabilities (Rounded):")
print(transition_probabilities_df)

battery_SOC = [0, 100, 200, 300, 400, 500]
n_price_clusters = price_transition_probabilities.shape[0]
n_battery_SOC = len(battery_SOC)
n_combined_states = n_price_clusters * n_battery_SOC

# initial new transition matrix
combined_transition_probabilities = np.zeros((n_combined_states, n_combined_states))

# transion matrix with SOC of battery
for price_from in range(n_price_clusters):
    for price_to in range(n_price_clusters):
        prob_price_transition = price_transition_probabilities[price_from, price_to]
        for battery_from in range(n_battery_SOC):
            for battery_to in range(n_battery_SOC):
                # no-matter the SOC of battery
                from_state = price_from * n_battery_SOC + battery_from
                to_state = price_to * n_battery_SOC + battery_to
                combined_transition_probabilities[from_state, to_state] = prob_price_transition
price_clusters = ['Low', 'Medium', 'High']
state_labels = [f"{price}_{state}" for price in price_clusters for state in battery_SOC]

transition_df = pd.DataFrame(
    combined_transition_probabilities, \
    columns=state_labels,
    index=state_labels
)
# print(transition_df)
actions = ["charge", "discharge", "nothing"]
n_actions = len(actions)
n_action_states = n_combined_states * n_actions
final_transition_probabilities = np.zeros((n_action_states, n_combined_states))

for price_from in range(n_price_clusters):
    for price_to in range(n_price_clusters):
        prob_price_transition = price_transition_probabilities[price_from, price_to]
        for battery_from_idx, battery_from in enumerate(battery_SOC):
            for action_idx, action in enumerate(actions):
                #  make sure the battery state after the action
                if action == "charge":
                    battery_to = battery_from + 100
                elif action == "discharge":
                    battery_to = battery_from - 100
                elif action == "nothing":
                    battery_to = battery_from
                else:
                    continue

                # make sure the battery state is within the limits
                if battery_to in battery_SOC:
                    battery_to_idx = battery_SOC.index(battery_to)

                    # calculate the from_state and to_state indices
                    from_state = (price_from * n_battery_SOC + battery_from_idx) * n_actions + action_idx
                    to_state = price_to * n_battery_SOC + battery_to_idx

                    # fill the final transition matrix
                    final_transition_probabilities[from_state, to_state] = prob_price_transition
final_state_labels = [f"{price}_{state}_{action}" for price in price_clusters for state in battery_SOC for action in
                      actions]
final_transition_df = pd.DataFrame(
    final_transition_probabilities, \
    columns=state_labels,
    index=final_state_labels
)
# print(final_transition_df)

# initialize the valid actions array


Battery_valid_actions = []

for soc in battery_SOC:
    if soc == 0:
        Battery_valid_actions.append([0, 2])  # SOC = 0
    elif soc == 500:
        Battery_valid_actions.append([1, 2])  # SOC = 500
    else:
        Battery_valid_actions.append([0, 1, 2])  # other SOC

Battery_valid_actions = Battery_valid_actions * n_price_clusters

min_cluster_center = cluster_centers[0]  # Low price
medium_cluster_center = cluster_centers[1]  # Medium price
max_cluster_center = cluster_centers[2]  # High price

# Define all states (combinations of price cluster and battery SOC)
state_labels = [(price, soc) for price in price_clusters for soc in battery_SOC]
n_states = len(state_labels)

# Initialize the reward array: rows = states, columns = actions (charge, discharge, nothing)
reward_array = np.zeros((n_states, 3))

# Populate the reward array
for i, (price_cluster, soc) in enumerate(state_labels):
    # Determine the price based on the price cluster
    if price_cluster == "Low":
        eprice = min_cluster_center
    elif price_cluster == "Medium":
        eprice = medium_cluster_center
    elif price_cluster == "High":
        eprice = max_cluster_center

    # Assign rewards based on SOC and actions
    if soc == 0:  # SOC = 0: only charging or nothing is allowed
        reward_array[i, 0] = -100 * eprice  # Charging
        reward_array[i, 1] = 0  # assume is 0,because of valid actions, this situation won't happen
        reward_array[i, 2] = 0  # Doing nothing
    elif soc == 500:  # SOC = 500: only discharging or nothing is allowed
        reward_array[i, 0] = 0  # assume is 0,because of valid actions, this situation won't happen
        reward_array[i, 1] = 100 * eprice  # Discharging
        reward_array[i, 2] = 0  # Doing nothing
    else:  # Intermediate SOC values: all actions are allowed
        reward_array[i, 0] = -100 * eprice  # Charging
        reward_array[i, 1] = 100 * eprice  # Discharging
        reward_array[i, 2] = 0  # Doing nothing

reward_df = pd.DataFrame(
    reward_array, \
    columns=actions,
    index=state_labels
)


# # Print the reward array
# print("Reward Array:")
# print(reward_df)

class BatteryProblem:
    def __init__(self, gamma=1, P_sa=None, REWARDS=None, VALID_ACTIONS=None):
        self.gamma = gamma
        self.P_sa = P_sa
        self.REWARDS = REWARDS
        self.VALID_ACTIONS = VALID_ACTIONS

    # Define the actions and states
    ACTIONS = np.array(actions)
    STATES = np.array(state_labels)

    # Value Iteration

    def value_iteration(self, epsilon=1e-6):
        # Initialize the value function with V(s) = 0
        V = np.zeros(len(self.STATES))
        while True:
            # Initialize the new value function
            V_new = np.zeros(len(self.STATES))
            for s in range(len(self.STATES)):
                # Get the valid actions for the current state
                # 每个state都有一个对应的valid_actions
                valid_actions = self.VALID_ACTIONS[s]
                # Extract the rewards and transitions for the current state
                # 每个state都有一个对应的rewards和transitions
                rewards = self.REWARDS[s]
                transitions = self.P_sa[s * len(self.ACTIONS):(s + 1) * len(self.ACTIONS)]
                # Update the value function
                V_new[s] = np.max([rewards[a] + self.gamma * np.sum(transitions[a] * V) for a in valid_actions])
            # Check for convergence
            if np.max(np.abs(V - V_new)) < epsilon:
                break
            V = V_new
        return V

    def policy_from_value(self, V):
        # Initialize the policy with a random action for each state
        policy = np.zeros(len(self.STATES), dtype=int)
        for s in range(len(self.STATES)):
            # Get the valid actions for the current state
            valid_actions = self.VALID_ACTIONS[s]
            # Initialize the action values with -inf
            action_values = np.ones(len(self.ACTIONS)) * (-np.inf)
            # Extract the rewards and transitions for the current state
            rewards = self.REWARDS[s]
            transitions = self.P_sa[s * len(self.ACTIONS):(s + 1) * len(self.ACTIONS)]
            # Compute the value of each action
            for a in valid_actions:
                action_values[a] = rewards[a] + self.gamma * np.sum(transitions[a] * V)
            # Select the action with the highest value
            policy[s] = np.argmax(action_values)
        return policy

    # Policy Iteration

    def _evaluate_policy(self, policy):
        # Initialize rewards and transitions for each state
        transitions = np.zeros((len(self.STATES), len(self.STATES)))
        rewards = np.zeros(len(self.STATES))
        for s in range(len(self.STATES)):
            # Extract the rewards and transitions for the current state
            rewards[s] = self.REWARDS[s, policy[s]]
            transitions[s] = self.P_sa[s * len(self.ACTIONS) + policy[s]]
        # Update the value function
        V = np.linalg.inv(np.eye(len(self.STATES)) - self.gamma * transitions).dot(rewards)
        return V

    def policy_iteration(self):
        # Initialize the policy with a random action for each state
        policy = np.random.randint(0, len(self.ACTIONS), len(self.STATES))
        while True:
            # Evaluate the current policy
            V = self._evaluate_policy(policy)
            new_policy = np.zeros(len(self.STATES), dtype=int)
            for s in range(len(self.STATES)):
                valid_actions = self.VALID_ACTIONS[s]
                # Initialize the action values with -inf
                action_values = np.ones(len(self.ACTIONS)) * (-np.inf)
                # Extract the rewards and transitions for the current state
                rewards = self.REWARDS[s]
                transitions = self.P_sa[s * len(self.ACTIONS):(s + 1) * len(self.ACTIONS)]
                # Compute the value of each action
                for a in valid_actions:
                    action_values[a] = rewards[a] + self.gamma * np.sum(transitions[a] * V)
                # Select the action with the highest value and update the policy
                new_policy[s] = np.argmax(action_values)
                # Check for convergence
            if np.array_equal(policy, new_policy):
                break
            policy = new_policy
        return policy, V


def main():
    # Define the rewards, where we just use 0 for impossible transitions, as they will be ignored anyway
    rewards = np.array(reward_array)  # Broken

    # Define valid actions for each state (e.g. maintain in the broken state is not valid)
    valid_actions = Battery_valid_actions

    # Define the transition probabilities
    P_sa = np.array(final_transition_probabilities)

    # Create the maintenance problem
    Battery_strategy = BatteryProblem(0.98, P_sa, rewards, valid_actions)

    # Value Iteration
    print("Value Iteration:")
    optimal_values = Battery_strategy.value_iteration()
    optimal_policy = Battery_strategy.policy_from_value(optimal_values)
    for i, state in enumerate(Battery_strategy.STATES):
        print(f"{state} {optimal_values[i]} {Battery_strategy.ACTIONS[optimal_policy[i]]}")

    # Policy Iteration
    print("\nPolicy Iteration:")
    optimal_policy, optimal_values = Battery_strategy.policy_iteration()
    for i, state in enumerate(Battery_strategy.STATES):
        print(f"{state} {optimal_values[i]} {Battery_strategy.ACTIONS[optimal_policy[i]]}")


if __name__ == "__main__":
    main()


# 获取 optimal_policy 和 STATES

# Define the rewards, where we just use 0 for impossible transitions, as they will be ignored anyway
rewards = np.array(reward_array)  # Broken
# Define valid actions for each state (e.g. maintain in the broken state is not valid)
valid_actions = Battery_valid_actions
# Define the transition probabilities
P_sa = np.array(final_transition_probabilities)

Battery_strategy = BatteryProblem(0.98, P_sa, rewards, valid_actions)
optimal_values = Battery_strategy.value_iteration()
optimal_policy = Battery_strategy.policy_from_value(optimal_values)
states = Battery_strategy.STATES  # (price_cluster, SOC)
actions = Battery_strategy.ACTIONS  # action list

optimal_policy_dict = {
    tuple(state): actions[action_idx] for state, action_idx in zip(states, optimal_policy)
}

# 打印结果
print("Optimal Policy Dictionary:")
for state, action in optimal_policy_dict.items():
    print(f"State: {state}, Optimal Action: {action}")

# Initial SoC = 0
initial_soc = 0
max_SOC = max(battery_SOC)
min_SOC = min(battery_SOC)

# find the price cluster
filtered_df['PriceCluster'] = filtered_df['PriceEUR'].apply(
    lambda price: sorted_clusters[min(len(sorted_clusters) - 1, max(0, np.searchsorted(cluster_centers, price)))]
)

# print(filtered_df[['PriceEUR', 'PriceCluster']])
# calculate SOC
soc_sequence = [initial_soc]
current_soc = initial_soc

optimal_actions_list = []

for _, row in filtered_df.iterrows():
    # Current Price
    price_cluster = row['PriceCluster']

    # current SoC
    current_state = (price_clusters[price_cluster], str(current_soc))

    # Find the optimal action
    optimal_action = optimal_policy_dict[current_state]
    optimal_actions_list.append(optimal_action)  # 将 optimal_action 添加到列表中

    # refresh the SoC
    if optimal_action == "charge":
        current_soc += 100
    elif optimal_action == "discharge":
        current_soc -= 100
    current_soc = max(min_SOC, min(max_SOC, current_soc))

    # save the SoC of this moment
    soc_sequence.append(current_soc)

# # 输出所有的 optimal_action
# print(optimal_actions_list)
action_mapping = {"charge": 1, "discharge": -1, "nothing": 0}

numeric_actions = [action_mapping[action] for action in optimal_actions_list]

# print(numeric_actions)

revenues = [-action * 100 * price for action, price in zip(numeric_actions, prices)]

# Calculate the total revenue
total_revenue = sum(revenues)
print(total_revenue)

# df_numeric_actions = pd.DataFrame({
#     "Numeric Action": numeric_actions
# })

# # Save the DataFrame to a CSV file
# action_output_path = "C:/Users/10063/Desktop/dtu/开学/选课/2nd semester/ML for energy system/battery_action.csv"
# df_numeric_actions.to_csv(action_output_path, index=False)


# set point with a step of 100
sample_interval = 100
sampled_soc_sequence = soc_sequence[::sample_interval]

# plot SOC
plt.figure(figsize=(12, 6))
plt.plot(sampled_soc_sequence, marker='o', linestyle='-', label='SOC', color='b')
plt.title('Battery SOC over Time (Sampled)')
plt.xlabel('Time Step')
plt.ylabel('State of Charge (SOC)')
plt.grid(True)
plt.legend()
plt.show()



