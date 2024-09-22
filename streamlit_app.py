import streamlit as st 
import pandas as pd
import pulp 
import altair as alt

st.title("Procurement Scenario Planning")
st.write(
    "The app is designed to solve a supply chain optimization problem by calculating an optimal sourcing strategy for multiple suppliers. "
    "It leverages data on supplier capacity, cost, emissions, and monthly demand, which users upload in CSV format. The app computes the optimal solution "
    "that minimizes total costs while meeting demand and adhering to emissions constraints, such as an emissions cap and a carbon cost penalty for emissions."
    "Interactive sliders allow sensitivity analysis by adjusting the emissions cap and carbon cost in real-time, providing valuable insights into the trade-offs between costs and emissions."
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

# Function to plot sourcing, emissions, and costs
def plot_sourcing_and_emissions(data, emissions_cap, CarbonCost, month, unit_color, emission_color, cost_color):
    demand = data[f'Demand_Month_{month}'].values[0]
    capacities = data['Capacity'].values
    costs = data['Cost'].values
    emissions = data['Emissions'].values

    # Get optimization results
    x_vals, total_emissions, total_cost = optimize_sourcing(demand, capacities, costs, emissions, emissions_cap, CarbonCost)

    # Prepare data for Altair plot
    labels = data['Supplier'].values
    emissions_vals = [emissions[i] * x_vals[i] for i in range(len(x_vals))]
    costs_vals = [(costs[i] + CarbonCost) * x_vals[i] for i in range(len(x_vals))]

    chart_data = pd.DataFrame({
        'Supplier': labels,
        'Units Sourced': x_vals,
        'Emissions': emissions_vals,
        'Costs': costs_vals
    })

    # Display table before chart
    table_data = [[labels[i], f"{x_vals[i]:.0f}", f"{emissions_vals[i]:.2f}", f"${costs_vals[i]:.2f}"] for i in range(len(x_vals))]
    table_data.append(['Total', f"{sum(x_vals):.0f}", f"{sum(emissions_vals):.2f}", f"${sum(costs_vals):.2f}"])
    column_labels = ['Supplier', 'Units Sourced', 'Emissions', 'Cost']

    st.table(pd.DataFrame(table_data, columns=column_labels))

    # Create an Altair chart based on user's selected colors
    base = alt.Chart(chart_data).encode(
        x='Supplier:N',
        tooltip=['Supplier', 'Units Sourced', 'Emissions', 'Costs']
    )

    units_bar = base.mark_bar(color=unit_color).encode(
        y=alt.Y('Units Sourced:Q', axis=alt.Axis(title='Units Sourced', titleColor=unit_color))
    )

    emissions_bar = base.mark_bar(color=emission_color).encode(
        y=alt.Y('Emissions:Q', axis=alt.Axis(title='Emissions (units)', titleColor=emission_color, offset=60))
    ).properties(width=600)

    costs_line = base.mark_line(color=cost_color).encode(
        y=alt.Y('Costs:Q', axis=alt.Axis(title='Costs ($)', titleColor=cost_color, offset=120))
    )

    # Combine charts and adjust spacing between the y-axes
    combined_chart = alt.layer(units_bar, emissions_bar, costs_line).resolve_scale(
        y='independent'
    ).properties(
        width=800,
        height=600,
        title=f"Sourcing Optimization for Month {month}: Emissions Cap = {emissions_cap}, Carbon Cost = {CarbonCost}"
    )

    st.altair_chart(combined_chart)

# Function to plot pie chart for emissions percentage using Altair
def plot_emissions_pie_chart(suppliers, emissions_vals, color_scheme):
    demand = data[f'Demand_Month_{month}'].values[0]
    capacities = data['Capacity'].values
    costs = data['Cost'].values
    emissions = data['Emissions'].values

    # Get optimization results
    x_vals, total_emissions, total_cost = optimize_sourcing(demand, capacities, costs, emissions, emissions_cap, CarbonCost)

    emissions_vals = [emissions[i] * x_vals[i] for i in range(len(x_vals))]
    percentages = [round(val / total_emissions * 100, 2) if total_emissions > 0 else 0 for val in emissions_vals]  # Calculate percentages

    pie_data = pd.DataFrame({
        'Supplier': suppliers,
        'Emissions': emissions_vals,
        'Percentage': percentages
    })

    pie_chart = alt.Chart(pie_data).mark_arc().encode(
        theta=alt.Theta(field='Emissions', type='quantitative'),
        color=alt.Color(field='Supplier', type='nominal', scale=alt.Scale(scheme=color_scheme)),
        tooltip=[
            'Supplier',
            'Emissions',
            alt.Tooltip('Percentage:Q', format='.2f')  # Format as number with two decimals
        ]
    ).properties(
        title='Emissions by Supplier'
    )

    st.altair_chart(pie_chart, use_container_width=True)

# Main interface
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    data = load_data(uploaded_file)

    # Sliders for emissions cap and CarbonCost
    emissions_cap = st.slider('Emissions Cap', 500, 2000, 1000)
    CarbonCost = st.slider('Carbon Cost', 0.0, 10.0, 1.0)
    month = st.slider('Month', 1, 12, 1)

    # Color pickers for bars and lines
    col1, col2, col3 = st.columns(3)
    with col1:
        unit_color = st.color_picker('Pick a color for Units Sourced bar', '#4c78a8')
    with col2:
        emission_color = st.color_picker('Pick a color for Emissions bar', '#72b7b2')
    with col3:
        cost_color = st.color_picker('Pick a color for Costs line', '#9ecae9')

    # Generate plot and table
    plot_sourcing_and_emissions(data, emissions_cap, CarbonCost, month, unit_color, emission_color, cost_color)

    # Prepare data for pie chart based on the latest emissions values
    labels = data['Supplier'].values
    emissions_vals = [data['Emissions'][i] * data[f'Demand_Month_{month}'][0] for i in range(len(data['Supplier']))]

    # Let user pick a color scheme for the pie chart
    pie_color_scheme = st.selectbox('Choose a color scheme for the pie chart', ['blues', 'greens', 'reds'])

    # Generate pie chart for emissions percentage by supplier
    plot_emissions_pie_chart(labels, emissions_vals, pie_color_scheme)
