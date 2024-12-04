import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import matplotlib.pyplot as plt 

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