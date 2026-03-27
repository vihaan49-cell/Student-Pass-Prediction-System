import pandas as pd
import numpy as np
import matplotlib
# Force matplotlib to not use any X-windows backend (Crucial for CLI/Server execution)
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def run_project():
    # 1. Create Dataset (Synthetic Example)
    np.random.seed(42)
    num_students = 200
    data = {
        "hours_studied": np.random.randint(1, 20, num_students),
        "attendance": np.random.randint(50, 101, num_students),
        "assignments_completed": np.random.randint(0, 10, num_students),
        "class_participation": np.random.randint(1, 10, num_students)
    }

    df = pd.DataFrame(data)
    # Define passing logic
    df['pass'] = ((df['hours_studied'] + df['assignments_completed'] + df['class_participation'] > 22) & 
                  (df['attendance'] > 65)).astype(int)

    print("--- Data Sample ---")
    print(df.head())

    # 2. Prepare Data
    X = df.drop('pass', axis=1)
    y = df['pass']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Train Model
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    # 4. Evaluation
    y_pred = model.predict(X_test)
    print("\n--- Model Results ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 5. Save Visualization (Instead of plt.show)
    plt.figure(figsize=(10, 6))
    plot_tree(model, feature_names=X.columns, class_names=['Fail', 'Pass'], filled=True)
    plt.title("Decision Tree Logic")
    
    output_file = "model_visualization.png"
    plt.savefig(output_file)
    print(f"\nSuccess! Visualization saved as: {output_file}")

if __name__ == "__main__":
    run_project()