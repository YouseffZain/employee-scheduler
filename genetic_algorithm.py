import pandas as pd
import random
import numpy as np
from collections import Counter

class GeneticScheduler:
    def __init__(self, weights, params):
        """
        Initialize the scheduler with penalty weights and GA parameters.
        weights: dict with keys 'unavailable', 'double', 'overtime', 'under_min', 'fairness'
        params: dict with keys 'pop_size', 'generations', 'mutation_rate', 'elitism'
        """
        self.weights = weights
        self.params = params
        self.SHIFT_HOURS = 8
        self.DAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        
        # Internal State
        self.employees = []
        self.name_to_id = {}
        self.availability = {}  # Map: eid -> day -> shift -> bool
        self.demand = {}        # Map: (day, shift, group) -> count
        self.slots = []         # List of (day, shift, group) for every required slot
        self.day_ranges = {}    # Map: day -> (start_idx, end_idx) in slots list
        self.emp_group = {}
        self.emp_min = {}
        self.emp_max = {}
        self.N_EMP = 0

    def load_and_process_data(self, avail_df, demand_df):
        """Parses the raw DataFrames into internal structures."""
        # 1. Parse Employees
        self.employees = []
        self.name_to_id = {}
        
        # Clean headers just in case
        avail_df.columns = [c.strip() for c in avail_df.columns]
        demand_df.columns = [c.strip() for c in demand_df.columns]
        
        for i, row in avail_df.iterrows():
            name = str(row["Name"]).strip()
            if name not in self.name_to_id:
                emp_id = len(self.employees)
                self.name_to_id[name] = emp_id
                self.employees.append({
                    "id": emp_id,
                    "name": name,
                    "group": str(row.get("Group code", "all")).strip(),
                    "min_hours": int(row.get("Min_Hours", 0)) if pd.notna(row.get("Min_Hours", 0)) else 0,
                    "max_hours": int(row.get("Max_Hours", 40)) if pd.notna(row.get("Max_Hours", 40)) else 40,
                })

        self.N_EMP = len(self.employees)
        self.emp_group = {e["id"]: e["group"] for e in self.employees}
        self.emp_min = {e["id"]: e["min_hours"] for e in self.employees}
        self.emp_max = {e["id"]: e["max_hours"] for e in self.employees}

        # 2. Parse Shifts & Availability
        if "Start time" in avail_df.columns:
            self.SHIFTS = sorted(avail_df["Start time"].dropna().astype(str).unique().tolist())
        else:
            self.SHIFTS = ["S"]

        self.availability = {eid: {d: {s: False for s in self.SHIFTS} for d in self.DAYS} for eid in range(self.N_EMP)}
        for _, row in avail_df.iterrows():
            name = str(row["Name"]).strip()
            eid = self.name_to_id[name]
            sh = str(row["Start time"]) if "Start time" in avail_df.columns else "S"
            sh = sh.strip()
            for d in self.DAYS:
                v = row.get(d, None)
                is_av = (str(v).strip().upper() == "A")
                self.availability[eid][d][sh] = self.availability[eid][d][sh] or is_av

        # 3. Parse Demand
        if demand_df.iloc[0].astype(str).str.contains("Weekday", case=False).any():
            demand_df.columns = demand_df.iloc[0].astype(str).tolist()
            demand_df = demand_df.iloc[1:].reset_index(drop=True)

        weekday_col = next((c for c in demand_df.columns if str(c).strip().lower() == "weekday"), demand_df.columns[0])
        group_cols = [c for c in demand_df.columns if str(c).strip().lower() not in ["weekday", "date"]]
        GROUPS = [str(g).strip() for g in group_cols]

        # Precompute availability counts for proportional demand allocation
        avail_count = {(d, sh, g): 0 for d in self.DAYS for sh in self.SHIFTS for g in GROUPS}
        for eid in range(self.N_EMP):
            g = self.emp_group[eid]
            for d in self.DAYS:
                for sh in self.SHIFTS:
                    if self.availability[eid][d][sh]:
                        avail_count[(d, sh, g)] += 1

        self.demand = {}
        for _, row in demand_df.iterrows():
            day = str(row[weekday_col]).strip()
            if day not in self.DAYS: continue
            
            for g in GROUPS:
                req_total = int(row.get(g, 0) if pd.notna(row.get(g, 0)) else 0)
                counts = [avail_count[(day, sh, g)] for sh in self.SHIFTS]
                total = sum(counts)
                
                if req_total == 0:
                    for sh in self.SHIFTS: self.demand[(day, sh, g)] = 0
                    continue
                if total == 0:
                    for sh in self.SHIFTS: self.demand[(day, sh, g)] = 0
                    self.demand[(day, self.SHIFTS[0], g)] = req_total # Force assignment if no one avail
                    continue

                # Distribute demand across shifts based on availability
                raw = [req_total * (c / total) for c in counts]
                alloc = [int(x) for x in raw]
                remainder = req_total - sum(alloc)
                frac_order = sorted(range(len(self.SHIFTS)), key=lambda i: (raw[i] - alloc[i]), reverse=True)
                for i in frac_order:
                    if remainder <= 0: break
                    if counts[i] > 0:
                        alloc[i] += 1
                        remainder -= 1
                for i, sh in enumerate(self.SHIFTS):
                    self.demand[(day, sh, g)] = alloc[i]

        # 4. Build Slots (The "Genome" Structure)
        self.slots = []
        self.day_ranges = {}
        start = 0
        for d in self.DAYS:
            day_slots = []
            for sh in self.SHIFTS:
                for g in GROUPS:
                    req = self.demand.get((d, sh, g), 0)
                    for _ in range(req):
                        day_slots.append((d, sh, g))
            self.slots.extend(day_slots)
            end = start + len(day_slots)
            self.day_ranges[d] = (start, end)
            start = end
        
        self.NUM_SLOTS = len(self.slots)

    # --- Core Helpers ---
    def is_available(self, eid, day, sh):
        return self.availability[eid][day][sh]

    def candidates_for(self, day, sh, group, used_set):
        c1 = [eid for eid in range(self.N_EMP) if self.emp_group[eid] == group and self.is_available(eid, day, sh) and eid not in used_set]
        if c1: return c1
        c2 = [eid for eid in range(self.N_EMP) if self.emp_group[eid] == group and eid not in used_set]
        if c2: return c2
        c3 = [eid for eid in range(self.N_EMP) if self.emp_group[eid] == group]
        if c3: return c3
        return list(range(self.N_EMP))

    def get_hours_array(self, genome):
        h = [0] * self.N_EMP
        for eid in genome:
            h[eid] += self.SHIFT_HOURS
        return h

    # --- Genetic Operators ---
    def make_initial(self):
        genome = [0] * self.NUM_SLOTS
        used_by_day = {d: set() for d in self.DAYS}
        hours = [0] * self.N_EMP 
        for i, (day, sh, g) in enumerate(self.slots):
            cand = self.candidates_for(day, sh, g, used_by_day[day])
            if cand:
                min_h = min(hours[c] for c in cand)
                best_candidates = [c for c in cand if hours[c] <= min_h + self.SHIFT_HOURS] 
                pick = random.choice(best_candidates)
            else:
                pick = random.choice(range(self.N_EMP))
            genome[i] = pick
            used_by_day[day].add(pick)
            hours[pick] += self.SHIFT_HOURS
        return genome

    def repair(self, genome):
        current_hours = self.get_hours_array(genome)
        for d in self.DAYS:
            used = set()
            a, b = self.day_ranges[d]
            for i in range(a, b):
                day, sh, g = self.slots[i]
                eid = genome[i]
                bad = False
                if eid in used: bad = True
                if self.emp_group[eid] != g: bad = True
                if not self.is_available(eid, day, sh): bad = True
                
                if not bad:
                    used.add(eid)
                    continue

                cand = self.candidates_for(day, sh, g, used)
                if cand:
                    cand_sorted = sorted(cand, key=lambda x: current_hours[x])
                    new_eid = random.choice(cand_sorted[:3]) # pick from least worked
                else:
                    new_eid = random.choice(list(range(self.N_EMP)))
                
                current_hours[new_eid] += self.SHIFT_HOURS
                if genome[i] < self.N_EMP:
                     current_hours[genome[i]] = max(0, current_hours[genome[i]] - self.SHIFT_HOURS)
                genome[i] = new_eid
                used.add(new_eid)
        return genome

    def evaluate(self, genome):
        used_by_day = {d: [] for d in self.DAYS}
        hours = [0] * self.N_EMP
        unavailable = 0
        wrong_group = 0
        
        for i, eid in enumerate(genome):
            day, sh, g = self.slots[i]
            used_by_day[day].append(eid)
            hours[eid] += self.SHIFT_HOURS
            if self.emp_group[eid] != g: wrong_group += 1
            if not self.is_available(eid, day, sh): unavailable += 1

        double_day = 0
        for d, arr in used_by_day.items():
            c = Counter(arr)
            for _, cnt in c.items():
                if cnt > 1: double_day += (cnt-1)

        overtime_hours = 0
        under_min_hours = 0
        for eid, h in enumerate(hours):
            if h > self.emp_max[eid]: overtime_hours += (h - self.emp_max[eid])
            if h < self.emp_min[eid]: under_min_hours += (self.emp_min[eid] - h)

        active_hours = [h for h in hours if h > 0]
        if len(active_hours) > 1:
            std_dev = np.std(active_hours)
        else:
            std_dev = 0

        penalty = 0
        penalty += 6000 * wrong_group
        penalty += self.weights['unavailable'] * unavailable
        penalty += self.weights['double'] * double_day
        penalty += self.weights['overtime'] * overtime_hours
        penalty += self.weights['under_min'] * under_min_hours
        penalty += self.weights['fairness'] * std_dev
        
        return -penalty, {
            "wrong_group": wrong_group, 
            "unavailable": unavailable, 
            "double_day": double_day, 
            "overtime": overtime_hours, 
            "under_min": under_min_hours, 
            "std_dev": round(std_dev, 2),
            "penalty": penalty
        }

    def mutate_robin_hood(self, genome):
        if random.random() > self.params['mutation_rate']: return genome
        hours = self.get_hours_array(genome)
        emp_indices = list(range(self.N_EMP))
        random.shuffle(emp_indices)
        sorted_emps = sorted(emp_indices, key=lambda x: hours[x])
        
        poor_emp = sorted_emps[0]
        rich_emp = sorted_emps[-1]
        
        if hours[rich_emp] <= hours[poor_emp]: return genome
            
        rich_indices = [i for i, x in enumerate(genome) if x == rich_emp]
        if not rich_indices: return genome
        
        slot_idx = random.choice(rich_indices)
        day, sh, g = self.slots[slot_idx]
        
        d_range = self.day_ranges[day]
        emps_this_day = set(genome[d_range[0]:d_range[1]])
        
        can_take = True
        if self.emp_group[poor_emp] != g: can_take = False
        if not self.is_available(poor_emp, day, sh): can_take = False
        if poor_emp in emps_this_day: can_take = False
        
        if can_take:
            genome[slot_idx] = poor_emp
            
        return genome

    def day_block_crossover(self, p1, p2):
        if random.random() > 0.9: return p1[:], p2[:]
        cut_day_idx = random.randint(1, len(self.DAYS)-1)
        cut_pos = self.day_ranges[self.DAYS[cut_day_idx]][0]
        c1, c2 = p1[:], p2[:]
        c1[cut_pos:], c2[cut_pos:] = c2[cut_pos:], c1[cut_pos:]
        return c1, c2

    def tournament_select(self, pop, fits, k=4):
        best_i = None; best_f = None
        for _ in range(k):
            i = random.randrange(len(pop))
            if best_i is None or fits[i] > best_f: best_i = i; best_f = fits[i]
        return pop[best_i][:]

    # --- Main Optimization Loop ---
    def optimize(self, progress_callback=None):
        random.seed(7)
        pop = [self.repair(self.make_initial()) for _ in range(self.params['pop_size'])]
        best_g, best_f, best_det = None, float("-inf"), None

        for gen in range(1, self.params['generations'] + 1):
            fits, dets = [], []
            for g in pop:
                f, d = self.evaluate(g)
                fits.append(f)
                dets.append(d)

            bi = max(range(len(pop)), key=lambda i: fits[i])
            if fits[bi] > best_f:
                best_f = fits[bi]
                best_g = pop[bi][:]
                best_det = dets[bi]

            if progress_callback:
                progress_callback(gen, self.params['generations'], best_f, best_det.get('std_dev', 0))

            elite_idx = sorted(range(len(pop)), key=lambda i: fits[i], reverse=True)[:self.params['elitism']]
            new_pop = [pop[i][:] for i in elite_idx]

            while len(new_pop) < self.params['pop_size']:
                p1 = self.tournament_select(pop, fits)
                p2 = self.tournament_select(pop, fits)
                c1, c2 = self.day_block_crossover(p1, p2)
                c1 = self.repair(self.mutate_robin_hood(c1))
                c2 = self.repair(self.mutate_robin_hood(c2))
                new_pop.append(c1)
                if len(new_pop) < self.params['pop_size']:
                    new_pop.append(c2)
            pop = new_pop

        return best_g, best_f, best_det

    # --- Result Formatting Helpers ---
    def get_results_dataframe(self, genome):
        rows = []
        for i, eid in enumerate(genome):
            day, sh, g = self.slots[i]
            rows.append({
                "day": day, "shift": sh, "group": g,
                "employee_id": eid, "employee_name": self.employees[eid]["name"]
            })
        return pd.DataFrame(rows)

    def explain_penalties(self, genome):
        violations = []
        used_by_day = {d: [] for d in self.DAYS}
        hours = [0] * self.N_EMP
        
        for i, eid in enumerate(genome):
            day, sh, g = self.slots[i]
            emp_name = self.employees[eid]["name"]
            used_by_day[day].append(emp_name)
            hours[eid] += self.SHIFT_HOURS
            
            if not self.is_available(eid, day, sh):
                violations.append({"Type": "Unavailable", "Employee": emp_name, 
                                   "Detail": f"Assigned {day} {sh}", "Penalty Cost": self.weights['unavailable']})
            if self.emp_group[eid] != g:
                violations.append({"Type": "Wrong Group", "Employee": emp_name, 
                                   "Detail": f"Assigned {g} but is {self.emp_group[eid]}", "Penalty Cost": 6000})

        for d, names in used_by_day.items():
            counts = Counter(names)
            for name, count in counts.items():
                if count > 1:
                    violations.append({"Type": "Double Shift", "Employee": name, 
                                       "Detail": f"{count} shifts on {d}", "Penalty Cost": (count-1) * self.weights['double']})

        for eid, h in enumerate(hours):
            emp_name = self.employees[eid]["name"]
            if h > self.emp_max[eid]:
                violations.append({"Type": "Overtime", "Employee": emp_name, 
                                   "Detail": f"{h}h > {self.emp_max[eid]}h", "Penalty Cost": (h - self.emp_max[eid]) * self.weights['overtime']})
            if h < self.emp_min[eid]:
                violations.append({"Type": "Under Min", "Employee": emp_name, 
                                   "Detail": f"{h}h < {self.emp_min[eid]}h", "Penalty Cost": (self.emp_min[eid] - h) * self.weights['under_min']})
                
        return pd.DataFrame(violations)
