# Student-Pass-Prediction-System

A CLI-based Machine Learning application that utilizes a Decision Tree Classifier to predict student academic success based on behavioral metrics. This project demonstrates data synthesis, feature engineering, and model evaluation within a professional Python environment.

🚀 Overview
This system identifies "at-risk" students by analyzing:

Study Habits: Hours studied per week.

Engagement: Assignment completion and class participation.

Attendance: Historical presence in the classroom.

The model provides not just a prediction, but a Visual Logic Tree (results_tree.png) that explains the "why" behind every "Pass" or "Fail" decision.

🛠️ Installation & Setup
1. Prerequisites
Python 3.8+ installed on your system.

Git installed for repository cloning.

2. Clone the Repository
Open your terminal and run:

Bash
git clone https://github.com/{your-github-username}/{your-repo-name}
cd {your-repo-name}
3. Environment Configuration (Recommended)
It is best practice to use a virtual environment to avoid dependency conflicts:

Bash
# Create the environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
4. Install Dependencies
Install the required libraries (pandas, scikit-learn, matplotlib, numpy):

Bash
pip install -r requirements.txt
💻 Execution (CLI)
This project is fully executable via the command line. You can specify the number of students to simulate using the --samples argument.

Run the default analysis (200 students):

Bash
python main.py
Run with a custom sample size (e.g., 500 students):

Bash
python main.py --samples 500
📊 Project Structure
Plaintext
├── main.py              # Core application logic & ML pipeline
├── requirements.txt     # Project dependencies
├── REPORT.md            # In-depth technical analysis & syllabus coverage
├── results_tree.png     # Generated visualization of the Decision Tree
└── README.md            # Project documentation (this file)
📈 Expected Output
Upon successful execution, the script will:

Generate a synthetic dataset of student profiles.

Perform Feature Engineering to calculate an effort-based score.

Train a Decision Tree Classifier.

Print Accuracy, Precision, Recall, and a Confusion Matrix to the terminal.

Save a high-resolution visualization of the model's logic as results_tree.png.

📝 Syllabus Coverage
This project serves as a practical application of the following course concepts:

Supervised Learning: Binary classification using Decision Trees.

Regularization: Preventing overfitting via tree depth constraints.

Data Preprocessing: Handling train-test splits and feature synthesis.

Evaluation Metrics: Interpreting Confusion Matrices and F1-Scores.
