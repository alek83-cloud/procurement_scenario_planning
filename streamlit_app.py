import streamlit as st 
import pandas as pd
import pulp 
import altair as alt
import plotly.express as px


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

def baseline_sourcing(demand, capacities, costs):
    prob = pulp.LpProblem("Supplier_Sourcing_Problem", pulp.LpMinimize)

    # Decision variables for each supplier
    suppliers = len(capacities)
    x = [pulp.LpVariable(f'x{i+1}', lowBound=0, upBound=capacities[i], cat='Integer') for i in range(suppliers)]

    # Objective function: Minimize cost
    total_cost = pulp.lpSum([costs[i] * x[i] for i in range(suppliers)])
    prob += total_cost, "Total_Cost"

    # Constraints
    prob += pulp.lpSum(x) == demand, "Demand_Constraint"

    # Solve the problem
    prob.solve()

    if pulp.LpStatus[prob.status] == 'Optimal':
        x_vals_bl = [pulp.value(var) for var in x]
        total_cost = sum(costs[i] * x_vals_bl[i] for i in range(suppliers))
    else:
        x_vals_bl, total_cost = [0]*suppliers, 0, 0

    return x_vals_bl, total_cost

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

# Main interface
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    data = load_data(uploaded_file)

    capacities = data['Capacity'].values
    costs = data['Cost'].values
    emissions = data['Emissions'].values
    suppliers = data['Supplier'].values

    ec_max = round(sum([emissions[i] * capacities[i] for i in range(len(suppliers))]),-2)

    # Sidebar for input sliders
    st.sidebar.header("Configuration Settings")

    # Description for the emissions cap slider
    st.sidebar.write("Set the emissions cap to limit the total emissions allowed during the sourcing process.")
    emissions_cap = st.sidebar.slider('Emissions Cap', 0, ec_max, ec_max)

    # Description for the carbon cost slider
    st.sidebar.write("Define the carbon cost per unit of emissions. This will influence the sourcing strategy.")
    CarbonCost = st.sidebar.slider('Carbon Cost', 0.0, 3.0, 0.0)

    # Description for the month slider
    st.sidebar.write("Select the month or period for which you want to analyze the sourcing data.")
    month = st.sidebar.slider('Month', 1, 12, 1)

    demand = data[f'Demand_Month_{month}'].values[0]

    # Calculate baseline values
    x_vals_bl, total_cost_bl = baseline_sourcing(demand, capacities, costs)
    emissions_vals_bl = [emissions[i] * x_vals_bl[i] for i in range(len(x_vals_bl))]
    costs_vals_bl = [(costs[i] + CarbonCost) * x_vals_bl[i] for i in range(len(x_vals_bl))]

    # Calculate current scenario values
    x_vals_opt, total_emissions_opt, total_cost_opt = optimize_sourcing(demand, capacities, costs, emissions, emissions_cap, CarbonCost)
    emissions_vals_opt = [emissions[i] * x_vals_opt[i] for i in range(len(x_vals_opt))]
    costs_vals_opt = [(costs[i] + CarbonCost) * x_vals_opt[i] for i in range(len(x_vals_opt))]

    table_data = [[suppliers[i], f"{x_vals_bl[i]:.0f}", f"{x_vals_opt[i]:.0f}", f"{emissions_vals_bl[i]:.2f}", f"{emissions_vals_opt[i]:.2f}", f"${costs_vals_bl[i]:.2f}", f"${costs_vals_opt[i]:.2f}"] for i in range(len(x_vals_opt))]
    table_data.append(['Total', f"{sum(x_vals_bl):.0f}", f"{sum(x_vals_opt):.0f}", f"{sum(emissions_vals_bl):.2f}", f"{sum(emissions_vals_opt):.2f}", f"${sum(costs_vals_bl):.2f}", f"${sum(costs_vals_opt):.2f}"])
    column_labels = ['Supplier', 'Baseline Units Sourced', 'Current Scenario Units Sourced', 'Baseline Emissions','Current Scenario Emissions','Baseline Cost','Current Scenario Cost']
    
    # Create DataFrame without displaying the index
    st.dataframe(pd.DataFrame(table_data, columns=column_labels), hide_index=True)

    c1, c2= st.columns(2)
    c3, c4= st.columns(2)
    c5, c6= st.columns(2)

    with st.container():
        c1.write("Emissions, Current Scenario vs Baseline")
        c2.write("Emissions perc. by supplier")

    with c1:
            # Create DataFrame for Altair chart
        chart_data = pd.DataFrame({
            'Supplier': data['Supplier'],
            'Current Scenario Emissions': emissions_vals_opt,
            'Baseline Emissions': emissions_vals_bl
        })

        # Create a "Total" row with summed values
        total_row = pd.DataFrame({
            'Supplier': ['Total'],
            'Current Scenario Emissions': [sum(emissions_vals_opt)],
            'Baseline Emissions': [sum(emissions_vals_bl)]
        })

        # Use pd.concat() to append the "Total" row to the chart_data DataFrame
        chart_data = pd.concat([chart_data, total_row], ignore_index=True)

        # aqua color scheme
        custom_colors = ['#7cf9d6', '#6feeb7', '#7fccb6', '#009f8b','#007667']
        
        # Plot using Altair for Emissions with dynamic color based on values
        st.altair_chart(
            # Layer 1: Bar chart for Current Scenario emissions
            alt.Chart(chart_data)
            .mark_bar()
            .encode(
                x=alt.X('Current Scenario Emissions:Q', axis=alt.Axis(title='Emissions CO2e kg')),
                y=alt.Y('Supplier:N', sort='-x', axis=alt.Axis(title='Supplier')),
                color=alt.Color('Current Scenario Emissions:Q', 
                                scale=alt.Scale(domain=[min(emissions_vals_opt), sum(emissions_vals_opt)], 
                                                range=custom_colors),  # Custom color scheme
                                legend=None)  # No legend needed for colors
            )
            # Layer 2: Diamond markers for baseline emissions
            + alt.Chart(chart_data)
            .mark_point(
                shape="diamond",
                filled=True,
                color="orange",
                size=120,
            )
            .encode(
                x=alt.X('Baseline Emissions:Q'),
                y='Supplier:N'
            ),
            use_container_width=True,
        )


    with st.container():
        c3.write("Emissions perc. by supplier")
        c4.write("Costs perc. by supplier")

    with c2:
        # Create DataFrame for Altair chart
        cost_chart_data = pd.DataFrame({
            'Supplier': data['Supplier'],
            'Scenario Costs': [costs[i] * x_vals_opt[i] for i in range(len(x_vals_opt))],
            'Baseline Costs': [costs[i] * x_vals_bl[i] for i in range(len(x_vals_bl))]
        })

        # Create a "Total" row with summed values
        total_cost_row = pd.DataFrame({
            'Supplier': ['Total'],
            'Scenario Costs': [sum(costs[i] * x_vals_opt[i] for i in range(len(x_vals_opt)))],
            'Baseline Costs': [sum(costs[i] * x_vals_bl[i] for i in range(len(x_vals_bl)))]
        })

        # Use pd.concat() to append the "Total" row to the cost_chart_data DataFrame
        cost_chart_data = pd.concat([cost_chart_data, total_cost_row], ignore_index=True)

        # Plot using Altair for Costs with dynamic color based on values
        st.altair_chart(
            alt.Chart(cost_chart_data)
            .mark_bar()
            .encode(
                x=alt.X('Scenario Costs:Q', axis=alt.Axis(title='Costs $')),
                y=alt.Y('Supplier:N', sort='-x', axis=alt.Axis(title='Supplier')),
                color=alt.Color('Scenario Costs:Q', 
                                scale=alt.Scale(domain=[min(costs_vals_opt), sum(costs_vals_opt)], 
                                                range=custom_colors),
                                legend=None)
            )
            + alt.Chart(cost_chart_data)
            .mark_point(
                shape="diamond",
                filled=True,
                color="orange",
                size=120,
            )
            .encode(
                x=alt.X('Baseline Costs:Q', axis=alt.Axis(title='')),
                y='Supplier:N'
            ),
            use_container_width=True,
        )

    with c3:

        total_emissions_opt = sum(emissions_vals_opt)

        # Calculate percentages for the pie chart
        percentages = [round(val / total_emissions_opt * 100, 2) if total_emissions_opt > 0 else 0 for val in emissions_vals_opt]

        # Create a DataFrame with the data for the pie chart
        pie_data = pd.DataFrame({
            'Supplier': suppliers,
            'Emissions': emissions_vals_opt,
            'Percentage': percentages
        })

        f_pie_data = pie_data[pie_data['Emissions'] > 0]

        # Create the pie chart using Plotly
        fig = px.pie(f_pie_data, names='Supplier', values='Emissions', color_discrete_sequence=custom_colors)

        # Set hover template to display the rounded percentage
        fig.update_traces(hovertemplate='%{label}: %{value:.2f} emissions (%{percent:.2%})')

        fig.update_layout(
            width=300,  # Adjust the width
            height=300,  # Adjust the height
            legend=dict(
                x=-0.4,  # Move the legend to the left
                y=0.5,   # Center it vertically
        ))

        # Display the pie chart in Streamlit
        st.plotly_chart(fig)

    with c4:

        total_costs_opt = sum(costs_vals_opt)

        # Calculate percentages for the pie chart
        percentages = [round(val / total_costs_opt * 100, 2) if total_cost_opt > 0 else 0 for val in costs_vals_opt]

        # Create a DataFrame with the data for the pie chart
        pie_data = pd.DataFrame({
            'Supplier': suppliers,
            'Costs': costs_vals_opt,
            'Percentage': percentages
        })

        f_pie_data = pie_data[pie_data['Costs'] > 0]

        # Create the pie chart using Plotly
        fig = px.pie(f_pie_data, names='Supplier', values='Costs', color_discrete_sequence=custom_colors)

        # Set hover template to display the rounded percentage
        fig.update_traces(hovertemplate='%{label}: %{value:.2f} costs (%{percent:.2%})')

        fig.update_layout(
            width=300,  # Adjust the width
            height=300,  # Adjust the height
            legend=dict(
                x=-0.4,  # Move the legend to the left
                y=0.5,   # Center it vertically
        ))

        # Display the pie chart in Streamlit
        st.plotly_chart(fig)
            
    with st.container():
    #    c5.write("Emission vs Cost")
    #    c6.write("Units sourced perc. by supplier")
  
