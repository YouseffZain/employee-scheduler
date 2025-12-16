🧬 AI Employee Scheduler
A powerful, intelligent scheduling tool that uses a Genetic Algorithm (GA) to solve the Nurse Scheduling Problem. It automatically generates optimal rosters while balancing business constraints, employee availability, and workload fairness.
🔗 Live Demo: Click here to try the App
🎯 Project Overview
Scheduling employees is an NP-Hard optimization problem. This project allows managers to upload an Excel file containing employee availability and daily demand, and uses an evolutionary algorithm to "evolve" the perfect schedule.
Unlike basic schedulers, this AI optimizes for Fairness. It actively prevents "shift hoarding" by using a variance penalty and a custom "Robin Hood" mutation operator to redistribute hours from overworked to underworked staff.
✨ Key Features
Modular Architecture: Business logic (genetic_algorithm.py) is completely separated from the presentation layer (app.py).
Fairness Engine: Uses Standard Deviation calculations to ensure workloads are distributed evenly.
🛡️ Data Audit: Built-in sanity checker detects "Impossible Shifts" (where Demand > Supply) before the AI even starts.
🔍 Penalty Detective: A post-processing engine that explains exactly why a schedule received a certain score (e.g., "Dexter worked 8 hours overtime").
Dynamic Configuration: Adjust population size, mutation rates, and penalty weights on the fly.
🏗️ Project Structure
This project follows the Separation of Concerns principle:
genetic_algorithm.py ( The Logic )
Contains the GeneticScheduler class.
Handles data parsing, population initialization, crossover, mutation, and fitness evaluation.
Contains the custom "Robin Hood Mutation" logic.
app.py ( The Interface )
A clean Streamlit dashboard.
Handles file uploads, sidebar configuration, and data visualization.
Calls the scheduler class to run computations.
⚙️ How It Works (The Algorithm)
The solver uses a standard Genetic Algorithm lifecycle with custom operators:
Initialization: Creates a population of random schedules (respecting basic availability).
Evaluation (Fitness Function): Calculates a "Penalty Score" based on:
Hard Constraints: Unavailable staff (Penalty: 2500), Double Shifts (Penalty: 1800).
Soft Constraints: Overtime (Penalty: 80/hr), Under-utilization (Penalty: 30/hr).
Fairness: Variance in hours worked across the team.
Selection: Uses Tournament Selection to pick the best schedules.
Crossover: Uses "Day Block Crossover" to swap whole days between schedules.
Mutation:
Random Mutation: Randomly reassigns a slot.
Robin Hood Mutation: Specifically targets the most overworked employee and moves their shift to the most underworked employee.
Elitism: Preserves the top N solutions to guarantee improvement.
🚀 Local Installation
Clone the repository
git clone [https://github.com/your-username/employee-scheduler.git](https://github.com/your-username/employee-scheduler.git)
cd employee-scheduler


Install dependencies
pip install pandas numpy streamlit xlsxwriter openpyxl


Run the App
streamlit run app.py


📊 Input Data Format
The app expects an Excel file (.xlsx) with two specific sheets:
Availibility: Columns for Name, Group code, Min_Hours, Max_Hours, and days of the week (Monday, Tuesday...) containing "A" (Available) or "NW" (No Work).
Demand: Columns for Weekday and group names (e.g., a, b, c) specifying how many people are needed per group per day.
🛠️ Configuration Guide
Parameter
Description
Generations
Rounds of evolution. Higher = better results, slower speed.
Mutation Rate
Probability of random changes. Keeps the AI from getting "stuck."
Fairness Penalty
The "Anti-Burnout" slider. High values force even workload distribution.
Overtime Penalty
Cost per hour of exceeding Max_Hours.

License
This project is open-source and available under the MIT License.
