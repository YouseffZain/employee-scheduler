import streamlit as st
import pandas as pd
import random
import numpy as np
from collections import Counter
import io

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Genetic Scheduler: Detective Edition", layout="wide")

st.title("🧬 AI Employee Scheduler (with Penalty Detective)")
st.markdown("I added a **'Penalty Explanation'** tab to the results. It lists every specific rule violation so you know exactly why the score is high.")

# ==========================================
# 2. SIDEBAR - PARAMETERS
# ==========================================
st.sidebar.header("⚙️ Algorithm Settings")

uploaded_file = st.sidebar.file_uploader("Upload Schedule Excel", type=["xlsx"])

# Dynamic Parameters
GENERATIONS = st.sidebar.slider("Generations", 50, 1000, 300)
POP_SIZE = st.sidebar.slider("Population Size", 50, 500, 150)
MUTATION_RATE = st.sidebar.slider("Mutation Rate", 0.0, 1.0, 0.3)
ELITISM = st.sidebar.number_input("Elitism Count", 1, 10, 3)

st.sidebar.markdown("---")
st.sidebar.subheader("⚖️ Penalty Weights")
W_UNAVAILABLE = st.sidebar.number_input("Unavailable Penalty", value=2500)
W_DOUBLE_DAY = st.sidebar.number_input("Double Shift Penalty", value=1800)
W_OVERTIME = st.sidebar.number_input("Overtime Penalty", value=80)
W_UNDER_MIN = st.sidebar.number_input("Under Min Hours Penalty", value=30)
W_FAIRNESS = st.sidebar.number_input("Fairness (Variance) Penalty", value=50)

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

@st.cache_data 
def load_data(file):
    xls = pd.ExcelFile(file)
    avail_df = pd.read_excel(xls, sheet_name="Availibility")
    demand_df = pd.read_excel(xls, sheet_name="Demand")
    avail_df.columns = [c.strip() for c in avail_df.columns]
    demand_df.columns = [c.strip() for c in demand_df.columns]
    return avail_df, demand_df

def run_pre_audit(avail_df, demand_df):
    """Checks for impossible shortages BEFORE running the AI."""
    report = []
    DAYS = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]
    
    # Parse Employees & Groups
    employees = []
    for i, row in avail_df.iterrows():
        employees.append({
            "name": str(row["Name"]).strip(),
            "group": str(row.get("Group code", "all")).strip()
        })
    df_emp = pd.DataFrame(employees)
    
    avail_map = {(g, d): 0 for g in df_emp['group'].unique() for d in DAYS}
    for i, row in avail_df.iterrows():
        g = str(row.get("Group code", "all")).strip()
        for d in DAYS:
            val = str(row.get(d, "")).strip().upper()
            if val == "A":
                avail_map[(g, d)] = avail_map.get((g, d), 0) + 1

    if demand_df.iloc[0].astype(str).str.contains("Weekday", case=False).any():
        demand_df.columns = demand_df.iloc[0].astype(str).tolist()
        demand_df = demand_df.iloc[1:].reset_index(drop=True)
    
    weekday_col = next((c for c in demand_df.columns if str(c).strip().lower() == "weekday"), demand_df.columns[0])
    group_cols = [c for c in demand_df.columns if str(c).strip().lower() not in ["weekday", "date"]]
    
    for _, row in demand_df.iterrows():
        day = str(row[weekday_col]).strip()
        if day not in DAYS: continue
        for g in group_cols:
            group_name = str(g).strip()
            req = int(row.get(g, 0) if pd.notna(row.get(g, 0)) else 0)
            available_count = avail_map.get((group_name, day), 0)
            if req > available_count:
                report.append({
                    "Day": day, "Group": group_name, "Need": req, 
                    "Have": available_count, "Missing": req - available_count
                })
    return pd.DataFrame(report)

def run_optimization(avail_df, demand_df, progress_bar, w_unavailable, w_double_day, w_overtime, w_under_min, w_fairness):
    random.seed(7)
    SHIFT_HOURS = 8
    DAYS = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]

    # --- PRE-PROCESSING ---
    employees = []
    name_to_id = {}
    for i, row in avail_df.iterrows():
        name = str(row["Name"]).strip()
        if name not in name_to_id:
            emp_id = len(employees)
            name_to_id[name] = emp_id
            employees.append({
                "id": emp_id,
                "name": name,
                "group": str(row.get("Group code", "all")).strip(),
                "min_hours": int(row.get("Min_Hours", 0)) if pd.notna(row.get("Min_Hours", 0)) else 0,
                "max_hours": int(row.get("Max_Hours", 40)) if pd.notna(row.get("Max_Hours", 40)) else 40,
            })
            
    emp_group = {e["id"]: e["group"] for e in employees}
    emp_min = {e["id"]: e["min_hours"] for e in employees}
    emp_max = {e["id"]: e["max_hours"] for e in employees}
    N_EMP = len(employees)
    
    if "Start time" in avail_df.columns:
        shift_times = sorted(avail_df["Start time"].dropna().astype(str).unique().tolist())
    else:
        shift_times = ["S"]
    SHIFTS = shift_times

    availability = {eid: {d: {s: False for s in SHIFTS} for d in DAYS} for eid in range(N_EMP)}
    for _, row in avail_df.iterrows():
        name = str(row["Name"]).strip()
        eid = name_to_id[name]
        sh = str(row["Start time"]) if "Start time" in avail_df.columns else "S"
        sh = sh.strip()
        for d in DAYS:
            v = row.get(d, None)
            is_av = (str(v).strip().upper() == "A")
            availability[eid][d][sh] = availability[eid][d][sh] or is_av

    if demand_df.iloc[0].astype(str).str.contains("Weekday", case=False).any():
        demand_df.columns = demand_df.iloc[0].astype(str).tolist()
        demand_df = demand_df.iloc[1:].reset_index(drop=True)

    weekday_col = next((c for c in demand_df.columns if str(c).strip().lower() == "weekday"), demand_df.columns[0])
    group_cols = [c for c in demand_df.columns if str(c).strip().lower() not in ["weekday", "date"]]
    GROUPS = [str(g).strip() for g in group_cols]

    avail_count = {(d, sh, g): 0 for d in DAYS for sh in SHIFTS for g in GROUPS}
    for eid in range(N_EMP):
        g = emp_group[eid]
        for d in DAYS:
            for sh in SHIFTS:
                if availability[eid][d][sh]:
                    avail_count[(d, sh, g)] += 1

    demand = {}
    for _, row in demand_df.iterrows():
        day = str(row[weekday_col]).strip()
        if day not in DAYS: continue
        for g in GROUPS:
            req_total = int(row.get(g, 0) if pd.notna(row.get(g, 0)) else 0)
            counts = [avail_count[(day, sh, g)] for sh in SHIFTS]
            total = sum(counts)
            if req_total == 0:
                for sh in SHIFTS: demand[(day, sh, g)] = 0
                continue
            if total == 0:
                for sh in SHIFTS: demand[(day, sh, g)] = 0
                demand[(day, SHIFTS[0], g)] = req_total
                continue
            raw = [req_total * (c / total) for c in counts]
            alloc = [int(x) for x in raw]
            remainder = req_total - sum(alloc)
            frac_order = sorted(range(len(SHIFTS)), key=lambda i: (raw[i] - alloc[i]), reverse=True)
            for i in frac_order:
                if remainder <= 0: break
                if counts[i] > 0:
                    alloc[i] += 1
                    remainder -= 1
            for i, sh in enumerate(SHIFTS):
                demand[(day, sh, g)] = alloc[i]

    slots = []
    day_ranges = {}
    start = 0
    for d in DAYS:
        day_slots = []
        for sh in SHIFTS:
            for g in GROUPS:
                req = demand.get((d, sh, g), 0)
                for _ in range(req):
                    day_slots.append((d, sh, g))
        slots.extend(day_slots)
        end = start + len(day_slots)
        day_ranges[d] = (start, end)
        start = end
    
    NUM_SLOTS = len(slots)

    # --- GA FUNCTIONS ---
    def is_available(eid, day, sh): return availability[eid][day][sh]

    def candidates_for(day, sh, group, used_set):
        c1 = [eid for eid in range(N_EMP) if emp_group[eid] == group and is_available(eid, day, sh) and eid not in used_set]
        if c1: return c1
        c2 = [eid for eid in range(N_EMP) if emp_group[eid] == group and eid not in used_set]
        if c2: return c2
        c3 = [eid for eid in range(N_EMP) if emp_group[eid] == group]
        if c3: return c3
        return list(range(N_EMP))

    def make_initial():
        genome = [0]*NUM_SLOTS
        used_by_day = {d:set() for d in DAYS}
        hours = [0]*N_EMP 
        for i,(day,sh,g) in enumerate(slots):
            cand = candidates_for(day, sh, g, used_by_day[day])
            if cand:
                min_h = min(hours[c] for c in cand)
                best_candidates = [c for c in cand if hours[c] <= min_h + SHIFT_HOURS] 
                pick = random.choice(best_candidates)
            else:
                pick = random.choice(range(N_EMP))
            genome[i] = pick
            used_by_day[day].add(pick)
            hours[pick] += SHIFT_HOURS
        return genome

    def get_hours_array(genome):
        h = [0] * N_EMP
        for eid in genome:
            h[eid] += SHIFT_HOURS
        return h

    def repair(genome):
        current_hours = get_hours_array(genome)
        for d in DAYS:
            used = set()
            a,b = day_ranges[d]
            for i in range(a,b):
                day,sh,g = slots[i]
                eid = genome[i]
                bad = False
                if eid in used: bad = True
                if emp_group[eid] != g: bad = True
                if not is_available(eid, day, sh): bad = True
                if not bad:
                    used.add(eid)
                    continue
                cand = candidates_for(day, sh, g, used)
                if cand:
                    cand_sorted = sorted(cand, key=lambda x: current_hours[x])
                    new_eid = random.choice(cand_sorted[:3])
                else:
                    new_eid = random.choice(list(range(N_EMP)))
                current_hours[new_eid] += SHIFT_HOURS
                if genome[i] < N_EMP:
                     current_hours[genome[i]] = max(0, current_hours[genome[i]] - SHIFT_HOURS)
                genome[i] = new_eid
                used.add(new_eid)
        return genome

    def evaluate(genome):
        used_by_day = {d: [] for d in DAYS}
        hours = [0]*N_EMP
        unavailable = 0
        wrong_group = 0
        for i,eid in enumerate(genome):
            day,sh,g = slots[i]
            used_by_day[day].append(eid)
            hours[eid] += SHIFT_HOURS
            if emp_group[eid] != g: wrong_group += 1
            if not is_available(eid, day, sh): unavailable += 1
        double_day = 0
        for d,arr in used_by_day.items():
            c = Counter(arr)
            for _,cnt in c.items():
                if cnt > 1: double_day += (cnt-1)
        overtime_hours = 0
        under_min_hours = 0
        for eid,h in enumerate(hours):
            if h > emp_max[eid]: overtime_hours += (h - emp_max[eid])
            if h < emp_min[eid]: under_min_hours += (emp_min[eid] - h)
        active_hours = [h for h in hours if h > 0]
        if len(active_hours) > 1: std_dev = np.std(active_hours)
        else: std_dev = 0
        penalty = 0
        penalty += 6000 * wrong_group
        penalty += w_unavailable * unavailable
        penalty += w_double_day * double_day
        penalty += w_overtime * overtime_hours
        penalty += w_under_min * under_min_hours
        penalty += w_fairness * std_dev
        return -penalty, {
            "wrong_group": wrong_group, "unavailable": unavailable, 
            "double_day": double_day, "overtime": overtime_hours, 
            "under_min": under_min_hours, "std_dev": round(std_dev, 2), "penalty": penalty
        }
    
    # --- DETECTIVE FUNCTION ---
    # This runs ONLY on the final best schedule to explain errors
    def explain_penalties(genome):
        violations = []
        used_by_day = {d: [] for d in DAYS}
        hours = [0]*N_EMP
        
        # 1. Slot Violations (Unavailable / Wrong Group)
        for i, eid in enumerate(genome):
            day, sh, g = slots[i]
            emp_name = employees[eid]["name"]
            used_by_day[day].append(emp_name)
            hours[eid] += SHIFT_HOURS
            
            # Check Unavailable
            if not is_available(eid, day, sh):
                violations.append({
                    "Type": "Unavailable",
                    "Employee": emp_name,
                    "Detail": f"Assigned {day} {sh} but is unavailable.",
                    "Penalty Cost": w_unavailable
                })
            
            # Check Group
            if emp_group[eid] != g:
                violations.append({
                    "Type": "Wrong Group",
                    "Employee": emp_name,
                    "Detail": f"Assigned Group '{g}' but is Group '{emp_group[eid]}'.",
                    "Penalty Cost": 6000
                })

        # 2. Daily Violations (Double Shift)
        for d, names in used_by_day.items():
            counts = Counter(names)
            for name, count in counts.items():
                if count > 1:
                    violations.append({
                        "Type": "Double Shift",
                        "Employee": name,
                        "Detail": f"Working {count} shifts on {d}.",
                        "Penalty Cost": (count-1) * w_double_day
                    })

        # 3. Aggregate Violations (Overtime / Under Min)
        for eid, h in enumerate(hours):
            emp_name = employees[eid]["name"]
            
            if h > emp_max[eid]:
                diff = h - emp_max[eid]
                violations.append({
                    "Type": "Overtime",
                    "Employee": emp_name,
                    "Detail": f"Worked {h}h (Max allowed: {emp_max[eid]}h).",
                    "Penalty Cost": diff * w_overtime
                })
            
            if h < emp_min[eid]:
                diff = emp_min[eid] - h
                violations.append({
                    "Type": "Under Min Hours",
                    "Employee": emp_name,
                    "Detail": f"Worked {h}h (Min required: {emp_min[eid]}h).",
                    "Penalty Cost": diff * w_under_min
                })
                
        return pd.DataFrame(violations)

    def tournament_select(pop, fits, k=4):
        best_i = None; best_f = None
        for _ in range(k):
            i = random.randrange(len(pop))
            if best_i is None or fits[i] > best_f: best_i = i; best_f = fits[i]
        return pop[best_i][:]

    def day_block_crossover(p1, p2):
        if random.random() > 0.9: return p1[:], p2[:]
        cut_day_idx = random.randint(1, len(DAYS)-1)
        cut_pos = day_ranges[DAYS[cut_day_idx]][0]
        c1, c2 = p1[:], p2[:]
        c1[cut_pos:], c2[cut_pos:] = c2[cut_pos:], c1[cut_pos:]
        return c1, c2

    def mutate_robin_hood(genome):
        if random.random() > MUTATION_RATE: return genome
        hours = get_hours_array(genome)
        emp_indices = list(range(N_EMP))
        random.shuffle(emp_indices)
        sorted_emps = sorted(emp_indices, key=lambda x: hours[x])
        poor_emp = sorted_emps[0]; rich_emp = sorted_emps[-1]
        if hours[rich_emp] <= hours[poor_emp]: return genome
        rich_indices = [i for i, x in enumerate(genome) if x == rich_emp]
        if not rich_indices: return genome
        slot_idx = random.choice(rich_indices)
        day, sh, g = slots[slot_idx]
        d_range = day_ranges[day]
        emps_this_day = set(genome[d_range[0]:d_range[1]])
        can_take = True
        if emp_group[poor_emp] != g: can_take = False
        if not is_available(poor_emp, day, sh): can_take = False
        if poor_emp in emps_this_day: can_take = False
        if can_take: genome[slot_idx] = poor_emp
        return genome

    # --- RUN GA ---
    pop = [repair(make_initial()) for _ in range(POP_SIZE)]
    best_g, best_f, best_det = None, float("-inf"), None
    status_text = st.empty()
    
    for gen in range(1, GENERATIONS+1):
        fits, dets = [], []
        for g in pop:
            f, d = evaluate(g)
            fits.append(f)
            dets.append(d)
        bi = max(range(len(pop)), key=lambda i: fits[i])
        if fits[bi] > best_f:
            best_f = fits[bi]; best_g = pop[bi][:]; best_det = dets[bi]
        progress_bar.progress(gen / GENERATIONS)
        if gen % 10 == 0: status_text.text(f"Gen {gen} | Fit: {best_f:.0f} | StdDev: {best_det.get('std_dev', 0)}")
        elite_idx = sorted(range(len(pop)), key=lambda i: fits[i], reverse=True)[:ELITISM]
        new_pop = [pop[i][:] for i in elite_idx]
        while len(new_pop) < POP_SIZE:
            p1 = tournament_select(pop, fits)
            p2 = tournament_select(pop, fits)
            c1, c2 = day_block_crossover(p1, p2)
            c1 = repair(mutate_robin_hood(c1))
            c2 = repair(mutate_robin_hood(c2))
            new_pop.append(c1)
            if len(new_pop) < POP_SIZE: new_pop.append(c2)
        pop = new_pop
    status_text.empty()
    
    # Generate Output
    rows = []
    for i,eid in enumerate(best_g):
        day, sh, g = slots[i]
        rows.append({
            "day": day, "shift": sh, "group": g,
            "employee_id": eid, "employee_name": employees[eid]["name"]
        })
    df_long = pd.DataFrame(rows)
    df_pivot = df_long.groupby(["day","shift","group"])["employee_name"].apply(lambda x: ", ".join(x.tolist())).reset_index()
    df_pivot["col"] = df_pivot["group"].astype(str) + "_" + df_pivot["shift"].astype(str)
    df_pivot = df_pivot.pivot(index="day", columns="col", values="employee_name").reindex(DAYS).fillna("")
    
    # Run Explanation on Best Genome
    explanation_df = explain_penalties(best_g)
    
    return df_long, df_pivot, best_det, explanation_df

# ==========================================
# 4. MAIN APP LOGIC
# ==========================================

if uploaded_file is not None:
    avail_df, demand_df = load_data(uploaded_file)
    
    st.write("### 📂 Data Preview")
    c1, c2 = st.columns(2)
    with c1: 
        st.subheader("Availability")
        st.dataframe(avail_df, height=300)
    with c2: 
        st.subheader("Demand")
        st.dataframe(demand_df, height=300)

    # TABS FOR AUDIT VS RUN
    tab_audit, tab_run = st.tabs(["🛡️ Pre-Check Audit", "🚀 Optimizer"])
    
    with tab_audit:
        st.subheader("Data Sanity Check")
        if st.button("Run Pre-Check Audit"):
            report_df = run_pre_audit(avail_df, demand_df)
            if not report_df.empty:
                st.error(f"🚨 FOUND {len(report_df)} IMPOSSIBLE SHIFTS (Supply < Demand)")
                st.dataframe(report_df, use_container_width=True)
            else:
                st.success("✅ Data looks feasible! No obvious shortages.")

    with tab_run:
        if st.button("🚀 Start Optimization"):
            progress = st.progress(0)
            try:
                long_df, pivot_df, details, explanation_df = run_optimization(
                    avail_df, demand_df, progress, 
                    W_UNAVAILABLE, W_DOUBLE_DAY, W_OVERTIME, W_UNDER_MIN, W_FAIRNESS
                )
                st.session_state['result_long'] = long_df
                st.session_state['result_pivot'] = pivot_df
                st.session_state['details'] = details
                st.session_state['explanation'] = explanation_df
                st.success("Optimization Complete!")
            except Exception as e:
                st.error(f"An error occurred: {e}")

# ==========================================
# 5. RESULTS DISPLAY
# ==========================================

if 'result_long' in st.session_state:
    det = st.session_state['details']
    explain_df = st.session_state['explanation']
    
    st.markdown("---")
    st.header("📊 Results Dashboard")
    
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Penalty Score", f"{det['penalty']:.0f}")
    m2.metric("Unavailable", det['unavailable'])
    m3.metric("Double Shifts", det['double_day'])
    m4.metric("Overtime Hrs", det['overtime'])
    m5.metric("Hr Variance", f"{det.get('std_dev',0):.2f}")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📅 Weekly View", "📋 List View", "📈 Employee Stats", "🔍 Penalty Explanation"])
    
    with tab1:
        st.dataframe(st.session_state['result_pivot'], use_container_width=True)
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            st.session_state['result_pivot'].to_excel(writer, sheet_name='Schedule')
        st.download_button(label="Download Schedule", data=buffer, file_name="optimized_schedule.xlsx", mime="application/vnd.ms-excel")

    with tab2:
        st.dataframe(st.session_state['result_long'], use_container_width=True)

    with tab3:
        df = st.session_state['result_long']
        counts = df['employee_name'].value_counts().reset_index()
        counts.columns = ['Employee', 'Shifts']
        counts['Hours'] = counts['Shifts'] * 8
        st.bar_chart(counts, x='Employee', y='Hours')

    with tab4:
        st.subheader("Why did I get this penalty score?")
        if not explain_df.empty:
            st.dataframe(explain_df, use_container_width=True)
            total_explained = explain_df["Penalty Cost"].sum()
            st.info(f"These violations account for **{total_explained:.0f}** points of your total penalty.")
        else:
            st.success("🎉 No specific violations found! (Penalty might be 0 or purely from Variance)")
