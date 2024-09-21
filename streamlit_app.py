import streamlit as st
import pandas as pd
import pulp 
import matplotlib.pyplot as plt

st.title("🎈 Procurement Scenario Planning")
st.write(
 #   "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)

# Function to load supplier and demand data from an uploaded CSV file
def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file, sep=';')
    return data

# Function to optimize sourcing based on emissions cap and CarbonCost
def optimize_sourcing(demand, capacities, costs, emissions, emissions_cap, CarbonCost):
    prob = pulp.LpProblem("Supplier_Sourcing_Problem", pulp.LpMinimize)

    # Decision variables for each supplier
    suppliers = len(capacities)
    x = [pulp.LpVariable(f'x{i+1}', lowBound=0, upBound=capacities[i], cat='Integer') for i in range(suppliers)]

    # Objective function: Minimize cost + emissions penalty
    total_cost = pulp.lpSum([costs[i] * x[i] for i in range(suppliers)]) + CarbonCost * pulp.lpSum([emissions[i] * x[i] for i in range(suppliers)])
    prob += total_cost, "Total_Cost_and_Emissions_Penalty"

    # Constraints
    prob += pulp.lpSum(x) == demand, "Demand_Constraint"
    prob += pulp.lpSum([emissions[i] * x[i] for i in range(suppliers)]) <= emissions_cap, "Emissions_Constraint"

    # Solve the problem
    prob.solve()

    if pulp.LpStatus[prob.status] == 'Optimal':
        x_vals = [pulp.value(var) for var in x]
        total_emissions = sum(emissions[i] * x_vals[i] for i in range(suppliers))
        total_cost = sum(costs[i] * x_vals[i] for i in range(suppliers)) + CarbonCost * total_emissions
    else:
        x_vals, total_emissions, total_cost = [0]*suppliers, 0, 0

    return x_vals, total_emissions, total_cost

# Function to plot the results
def plot_sourcing_and_emissions(data, emissions_cap, CarbonCost, month):
    # Extract data for the selected month
    demand = data[f'Demand_Month_{month}'].values[0]
    capacities = data['Capacity'].values
    costs = data['Cost'].values
    emissions = data['Emissions'].values

    # Get optimization results
    x_vals, total_emissions, total_cost = optimize_sourcing(demand, capacities, costs, emissions, emissions_cap, CarbonCost)

    # Create plot
    labels = data['Supplier'].values
    units = x_vals
    emissions_vals = [emissions[i] * units[i] for i in range(len(units))]
    costs_vals = [(costs[i]+CarbonCost) * units[i] for i in range(len(units))]

    fig, ax1 = plt.subplots(figsize=(9, 6))
    ax1.bar(labels, units, color='#1f77b4', alpha=0.7, label='Units Sourced')
    ax1.set_xlabel('Suppliers', fontsize=14)
    ax1.set_ylabel('Units Sourced', color='#1f77b4', fontsize=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    ax2 = ax1.twinx()
    ax2.bar(labels, emissions_vals, color='#ff7f0e', alpha=0.7, width=0.4, align='edge', label='Emissions')
    ax2.set_ylabel('Emissions (units)', color='#ff7f0e', fontsize=12)

    ax2.plot(labels, costs_vals, color='#2ca02c', marker='o', markersize=8, label='Costs')
    ax2.set_ylabel('Costs ($)', color='#2ca02c', fontsize=12)

    plt.title(f"Sourcing Optimization for Month {month}: Emissions Cap = {emissions_cap}, Carbon Cost = {CarbonCost}", fontsize=16)
    fig.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)

    # Display a table with results
    table_data = [[labels[i], f"{units[i]:.0f}", f"{emissions_vals[i]:.2f}", f"${costs_vals[i]:.2f}"] for i in range(len(units))]
    table_data.append(['Total', f"{sum(units):.0f}", f"{total_emissions:.2f}", f"${total_cost:.2f}"])
    column_labels = ['Supplier', 'Units Sourced', 'Emissions', 'Cost']

    st.table(pd.DataFrame(table_data, columns=column_labels))

# Main Streamlit interface
st.title('Supply Chain Sourcing Optimization')

# Upload the CSV file
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    data = load_data(uploaded_file)

    # Sliders for emissions cap and CarbonCost
    emissions_cap = st.slider('Emissions Cap', 500, 2000, 1000)
    CarbonCost = st.slider('Carbon Cost', 0.0, 10.0, 1.0)
    month = st.slider('Month', 1, 12, 1)

    # Generate plot and table
    plot_sourcing_and_emissions(data, emissions_cap, CarbonCost, month)

