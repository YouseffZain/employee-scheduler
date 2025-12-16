🧬 AI Employee Scheduler (Genetic Algorithm)

A robust, intelligent scheduling system that solves the Nurse Scheduling Problem (NSP) using an Evolutionary Genetic Algorithm. It optimizes for hard constraints (availability) and soft constraints (overtime, fairness) to generate the mathematically optimal roster.

🚀 Live Demo

Click below to test the application in your browser:


🎯 Project Overview

Scheduling employees while respecting individual availability, group requirements, and labor laws is an NP-Hard optimization problem. This project automates that process.

Unlike standard schedulers, this engine prioritizes Workload Fairness. It uses a custom "Robin Hood" mutation operator to actively redistribute hours from overworked employees to underutilized ones, ensuring no single employee (e.g., "Dexter") is unfairly burdened.

Key Features

🧠 Genetic Optimization: Evolve schedules over hundreds of generations to minimize penalties.

⚖️ Fairness Engine: A variance-based penalty system that forces even distribution of shifts.

🛡️ Data Audit: Pre-computation checks to detect "Impossible Shifts" (Demand > Supply) before execution.

🔍 Penalty Detective: A post-processing analysis tool that explains exactly why a schedule received a specific score (e.g., "3 Double Shifts detected").

modular Architecture: Strict separation between the UI (app.py) and the algorithmic logic (genetic_algorithm.py).

📂 Repository Structure

This project follows the Separation of Concerns software design principle:

File

Description

genetic_algorithm.py

The Core Engine. Contains the GeneticScheduler class. Handles population initialization, crossover, mutation logic, and fitness evaluation.

app.py

The Interface. A Streamlit dashboard that handles file I/O, visualization, and user configuration. It imports the engine as a module.

requirements.txt

Dependencies required to run the project.

Employee Schedule.xlsx

Sample dataset for testing.

⚙️ How the Algorithm Works

The solver operates on a population of potential schedules ("Chromosomes") and evolves them using the following pipeline:

Initialization:

Generates N random schedules.

Smart Init: Prefers assigning shifts to employees with the lowest current hours to start with a balanced base.

Evaluation (Fitness Function):

Calculates a "Penalty Score" based on weighted constraints:

Hard Constraints: Unavailable staff (Penalty: 2500), Double Shifts (Penalty: 1800).

Soft Constraints: Overtime (Penalty: 80/hr), Under-utilization (Penalty: 30/hr).

Fairness: Standard Deviation of hours worked across the team.

Selection:

Uses Tournament Selection to pick the fittest parents for the next generation.

Crossover:

Day-Block Crossover: Swaps entire daily schedules between two parents to preserve valid daily structures.

Mutation (The "Robin Hood" Operator):

Standard mutation randomly swaps a shift.

Robin Hood Mutation: Specifically identifies the employee with the most hours and attempts to move one of their shifts to the employee with the fewest hours. This aggressively converges towards fairness.

Elitism:

The top k schedules are copied unchanged to the next generation to guarantee monotonically non-decreasing performance.

🛠️ Local Installation

To run this project on your local machine:

Clone the repository

git clone [https://github.com/your-username/employee-scheduler.git](https://github.com/your-username/employee-scheduler.git)
cd employee-scheduler


Install dependencies

pip install pandas numpy streamlit xlsxwriter openpyxl


Run the App

streamlit run app.py


📊 Input Data Format

The application expects an Excel file (.xlsx) with two sheets:

Availibility

Columns: Name, Group code, Min_Hours, Max_Hours.

Days: Columns for Monday, Tuesday, etc., containing "A" (Available) or "NW" (No Work).

Demand

Columns: Weekday and Group Columns (e.g., a, b, c).

Values: Integers representing how many staff members are needed for that group on that day.

📸 Screenshots

(Optional: Add screenshots of your "Data Audit" tab or "Results Dashboard" here to make the repo look even better)

📝 License

This project is open-source and available under the MIT License.
