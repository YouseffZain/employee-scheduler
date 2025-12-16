import streamlit as st
import pandas as pd
import io
from genetic_algorithm import GeneticScheduler # Importing your new class

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Genetic Scheduler: Modular Edition", layout="wide")
st.title("🧬 AI Employee Scheduler")

# ==========================================
# 2. SIDEBAR - PARAMETERS
# ==========================================
st.sidebar.header("⚙️ Settings")
uploaded_file = st.sidebar.file_uploader("Upload Schedule Excel", type=["xlsx"])

# Params
params = {
    'generations': st.sidebar.slider("Generations", 50, 1000, 300),
    'pop_size': st.sidebar.slider("Population Size", 50, 500, 150),
    'mutation_rate': st.sidebar.slider("Mutation Rate", 0.0, 1.0, 0.3),
    'elitism': st.sidebar.number_input("Elitism Count", 1, 10, 3)
}

st.sidebar.markdown("---")
st.sidebar.subheader("⚖️ Weights")
weights = {
    'unavailable': st.sidebar.number_input("Unavailable", value=2500),
    'double': st.sidebar.number_input("Double Shift", value=1800),
    'overtime': st.sidebar.number_input("Overtime", value=80),
    'under_min': st.sidebar.number_input("Under Min", value=30),
    'fairness': st.sidebar.number_input("Fairness", value=50)
}

# ==========================================
# 3. MAIN APP
# ==========================================
if uploaded_file:
    # Load raw data to show preview
    xls = pd.ExcelFile(uploaded_file)
    avail_df = pd.read_excel(xls, sheet_name="Availibility")
    demand_df = pd.read_excel(xls, sheet_name="Demand")
    
    c1, c2 = st.columns(2)
    with c1: st.dataframe(avail_df, height=300)
    with c2: st.dataframe(demand_df, height=300)

    if st.button("🚀 Run Optimizer"):
        # Initialize Scheduler with config
        scheduler = GeneticScheduler(weights, params)
        
        # Load Data
        scheduler.load_and_process_data(avail_df, demand_df)
        
        # Define callback for progress bar
        prog_bar = st.progress(0)
        status_text = st.empty()
        
        def update_ui(gen, total_gen, fit, std_dev):
            prog_bar.progress(gen / total_gen)
            if gen % 10 == 0:
                status_text.text(f"Gen {gen}/{total_gen} | Fitness: {fit:.0f} | Fairness Dev: {std_dev:.2f}")

        # Run Algorithm
        best_genome, best_fit, best_det = scheduler.optimize(progress_callback=update_ui)
        
        # Get Results
        long_df = scheduler.get_results_dataframe(best_genome)
        explain_df = scheduler.explain_penalties(best_genome)
        
        # Build Pivot
        DAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        pivot_df = long_df.groupby(["day","shift","group"])["employee_name"].apply(lambda x: ", ".join(x.tolist())).reset_index()
        pivot_df["col"] = pivot_df["group"].astype(str) + "_" + pivot_df["shift"].astype(str)
        pivot_df = pivot_df.pivot(index="day", columns="col", values="employee_name").reindex(DAYS).fillna("")
        
        # Display
        st.success("Optimization Complete!")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📅 Weekly", "📋 List", "📈 Stats", "🔍 Explanations"])
        
        with tab1:
            st.dataframe(pivot_df, use_container_width=True)
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                pivot_df.to_excel(writer, sheet_name='Schedule')
            st.download_button("Download Excel", buffer, "schedule.xlsx")
            
        with tab2: st.dataframe(long_df, use_container_width=True)
        
        with tab3:
            counts = long_df['employee_name'].value_counts().reset_index()
            counts.columns = ['Employee', 'Shifts']
            counts['Hours'] = counts['Shifts'] * 8
            st.bar_chart(counts, x='Employee', y='Hours')
            
        with tab4:
            st.dataframe(explain_df, use_container_width=True)
