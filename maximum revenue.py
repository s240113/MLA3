import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
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
price = filtered_df['PriceEUR'].values.reshape(-1, 1)
# print(price[2,0])
hours = price.shape[0]
print(hours)
inital_SOC = 0
Battery_capacity = 500

class Expando(object):
    '''
        A small class which can have attributes set
    '''
    pass

class OptimizationModel:
    '''
        A class to represent the optimization model
    '''
    def __init__(self):
        self.model = gp.Model("Optimization Model")
        self.variables = Expando()
        self.constraints = Expando()
        self.results = Expando()
        # self._build_model()

    def _build_variables(self):

        self.variables.Batt_ch = self.model.addVars(hours, lb=0, ub=GRB.INFINITY, name='Battery charge power')
        self.variables.Batt_dis = self.model.addVars(hours, lb=0, ub=GRB.INFINITY, name='Battery discharge power')
        self.variables.Batt_Energy = self.model.addVars(hours, lb=0, ub=GRB.INFINITY, name='Battery state of charge')
        self.variables.charge_flag = self.model.addVars(hours, vtype=GRB.BINARY, name='charge_flag')


    def _build_constraints(self):
        self.model.addConstrs(
            (self.variables.Batt_ch[t] <= 100 * self.variables.charge_flag[t] for t in range(hours)), name='charge flag'
        )
        self.model.addConstrs(
            (self.variables.Batt_dis[t] <= 100 * (1 - self.variables.charge_flag[t]) for t in range(hours)), name='discharge flag'
        )
        self.constraints.Energy_T_0 = self.model.addConstr(
            self.variables.Batt_Energy[0] == inital_SOC * Battery_capacity + self.variables.Batt_ch[0] - self.variables.Batt_dis[0], name='Initial Energy'
        )
        self.constraints.Energy_T = self.model.addConstrs(
            (self.variables.Batt_Energy[t] == self.variables.Batt_Energy[t-1] + self.variables.Batt_ch[t] - self.variables.Batt_dis[t]
            for t in range(1, hours)), name='Energy in T'
        )
        self.constraints.Energy_capacity = self.model.addConstrs(
            (self.variables.Batt_Energy[t] <= Battery_capacity for t in range(hours)), name='Energy boundary'
        )
    def _build_objective(self):
        Revenue = gp.quicksum(
            -self.variables.Batt_ch[t] * price[t, 0] + self.variables.Batt_dis[t] * price[t, 0] for t in range(hours)
        )
        self.model.setObjective(Revenue, GRB.MAXIMIZE)

    def _save_results(self):
        self.results.objective_value = self.model.ObjVal
        self.results.Batt_ch = {t: self.variables.Batt_ch[t].x for t in range(hours)}
        self.results.Batt_dis = {t: self.variables.Batt_dis[t].x for t in range(hours)}
        self.results.Batt_Energy = {t: self.variables.Batt_Energy[t].x for t in range(hours)}
     
    def optimize(self):

        print("Building model...")
        self._build_variables()
        self._build_constraints()
        self._build_objective()
        
        print("Optimizing model...")
        self.model.optimize()
        
        if self.model.status == GRB.OPTIMAL:
            self._save_results()
            print("Optimization successful!")
        else:
            print("Optimization was not successful.")

if __name__ == '__main__':
    model = OptimizationModel()
    model.optimize()
    batt_energy_values_first_30 = list(model.results.Batt_Energy.values())[:30]
    plt.figure(figsize=(10, 6))
    plt.plot(batt_energy_values_first_30, marker='o', linestyle='-', color='b', label='Battery Energy (First 30 Hours)')
    plt.title('Battery Energy over First 30 Hours')
    plt.xlabel('Hour')
    plt.ylabel('Battery Energy (kWh)')
    plt.grid(True)
    plt.legend()
    plt.show()

    print("Time Step | Charge - Discharge (kW)")
    for t in range(hours):
        net_value = model.results.Batt_ch[t] - model.results.Batt_dis[t]
        # print(f"{t:9} | {net_value:21.2f}")

perfect_actions = []

for t in range(hours):
    net_value = model.results.Batt_ch[t] - model.results.Batt_dis[t]
    if net_value == 100:
        action = 1  # charge
    elif net_value == -100:
        action = -1  # discharge
    else:
        action = 0  # nothing
    
    # add action to the list
    perfect_actions.append(action)
    
    # # print answer
    # print(f"{t:9} | {net_value:21.2f} | Action: {action}")

# # print(perfect_actions)
# print(perfect_actions)


df_ave_action = pd.read_csv(r"C:/Users/10063/Desktop/dtu/开学/选课/2nd semester/ML for energy system/battery_action.csv")

excel_actions = df_ave_action['Action'].tolist()
if len(perfect_actions) != len(excel_actions):
    raise ValueError("The two action lists must have the same length.")
correct_predictions = sum(1 for a, b in zip(perfect_actions, excel_actions) if a == b)
accuracy = correct_predictions / len(perfect_actions) * 100

# 输出结果
print(f"Total Actions Compared: {len(perfect_actions)}")
print(f"Correct Predictions: {correct_predictions}")
print(f"Accuracy: {accuracy:.2f}%")

time_steps = np.arange(len(perfect_actions))

interval = 100
x_ticks = time_steps[::1000]
sampled_time_steps = time_steps[::interval]
sampled_perfect_actions = perfect_actions[::interval]
sampled_excel_actions = excel_actions[::interval]

# 绘制图表
plt.figure(figsize=(14, 7))

# 绘制 sampled_perfect_actions
plt.plot(sampled_time_steps, sampled_perfect_actions, marker='o', linestyle='-', linewidth=2, color='blue', label='Perfect Actions')

# 绘制 sampled_excel_actions
plt.plot(sampled_time_steps, sampled_excel_actions, marker='x', linestyle='--', linewidth=2, color='red', label='Excel Actions')

# 添加标签和标题
plt.xlabel('Time Step', fontsize=12)
plt.ylabel('Action (-1: Discharge, 0: Nothing, 1: Charge)', fontsize=12)
plt.title('Comparison of Actions Over Time (Sampled Every 100 Steps)', fontsize=14)

# 设置 X 轴和 Y 轴刻度
plt.xticks(x_ticks , fontsize=10)  # 仅显示采样的时间步
plt.yticks([-1, 0, 1], ['Discharge (-1)', 'Nothing (0)', 'Charge (1)'], fontsize=10)

# 添加网格和图例
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

# 显示图表
plt.tight_layout()
plt.show()
