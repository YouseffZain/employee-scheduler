import streamlit as st
import pandas as pd
import io
from genetic_algorithm import GeneticScheduler

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Genetic Scheduler: Modular & Audited", layout="wide")
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
    
    # Initialize Scheduler early so we can audit
    scheduler = GeneticScheduler(weights, params)
    scheduler.load_and_process_data(avail_df, demand_df)

    tab_audit, tab_run = st.tabs(["🛡️ Data Audit", "🚀 Optimizer"])

    with tab_audit:
        st.subheader("Pre-Optimization Sanity Check")
        if st.button("Run Data Audit"):
            report_df = scheduler.audit_schedule()
            
            if not report_df.empty:
                st.error(f"🚨 FOUND {len(report_df)} IMPOSSIBLE SHIFTS (Supply < Demand)")
                st.write("The AI cannot solve these without penalties because you don't have enough staff.")
                st.dataframe(report_df, use_container_width=True)
            else:
                st.success("✅ Data looks feasible! No obvious day-by-day shortages.")

    with tab_run:
        if st.button("🚀 Run Optimizer"):
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
            
            r1, r2, r3, r4 = st.tabs(["📅 Weekly", "📋 List", "📈 Stats", "🔍 Explanations"])
            
            with r1:
                st.dataframe(pivot_df, use_container_width=True)
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    pivot_df.to_excel(writer, sheet_name='Schedule')
                st.download_button("Download Excel", buffer, "schedule.xlsx")
                
            with r2: st.dataframe(long_df, use_container_width=True)
            
            with r3:
                counts = long_df['employee_name'].value_counts().reset_index()
                counts.columns = ['Employee', 'Shifts']
                counts['Hours'] = counts['Shifts'] * 8
                st.bar_chart(counts, x='Employee', y='Hours')
                
            with r4:
                st.dataframe(explain_df, use_container_width=True)
                if not explain_df.empty:
                    st.warning(f"Total Penalty Points Explained: {explain_df['Penalty Cost'].sum()}")
