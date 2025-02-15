import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sklearn.utils as utils
import sklearn.preprocessing as preprocessing
import sklearn.tree as tree
import sklearn.ensemble as ensemble
import sklearn.linear_model as linear_model
import sklearn.neural_network as neural_network
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

def evaluate_model(y_true, y_pred):
    """Evaluate a single model's performance."""
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted'),
        'Recall': recall_score(y_true, y_pred, average='weighted'),
        'F1 Score': f1_score(y_true, y_pred, average='weighted')
    }

def compare_models(y_true, predictions):
    """Compare multiple models' performance and calculate total accuracy."""
    results = []
    total_accuracy = 0
    
    for model_name, y_pred in predictions.items():
        metrics = evaluate_model(y_true, y_pred)
        total_accuracy += metrics['Accuracy']
        results.append({
            'Model': model_name,
            **metrics
        })
    
    avg_accuracy = (total_accuracy / len(predictions)) * 100
    results_df = pd.DataFrame(results)
    
    return results_df, avg_accuracy

def load_and_preprocess_data(file_path="AI-Data.csv"):
    """Load and preprocess the student data."""
    try:
        data = pd.read_csv(file_path)
        
        # Drop columns that won't be used for prediction
        columns_to_drop = [
            "gender", "StageID", "GradeID", "NationalITy", "PlaceofBirth", 
            "SectionID", "Topic", "Semester", "Relation", 
            "ParentschoolSatisfaction", "ParentAnsweringSurvey", 
            "AnnouncementsView"
        ]
        data = data.drop(columns_to_drop, axis=1)
        
        # Encode categorical variables
        for column in data.columns:
            if data[column].dtype == type(object):
                le = preprocessing.LabelEncoder()
                data[column] = le.fit_transform(data[column])
        
        return data
    except FileNotFoundError:
        print(f"Error: Could not find {file_path}. Please ensure the file exists in the current directory.")
        exit(1)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        exit(1)

def split_data(data, train_size=0.7):
    """Split the data into training and testing sets."""
    data_shuffled = utils.shuffle(data)
    split_idx = int(len(data_shuffled) * train_size)
    
    features = data_shuffled.values[:, 0:4]
    labels = data_shuffled.values[:, 4]
    
    return (
        features[0:split_idx],
        features[split_idx:],
        labels[0:split_idx],
        labels[split_idx:]
    )

def train_models(X_train, y_train):
    """Train multiple classification models."""
    models = {
        'Decision Tree': tree.DecisionTreeClassifier(),
        'Random Forest': ensemble.RandomForestClassifier(),
        'Perceptron': linear_model.Perceptron(),
        'Logistic Regression': linear_model.LogisticRegression(),
        'Neural Network': neural_network.MLPClassifier(activation="logistic")
    }
    
    try:
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
        return models
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        exit(1)

def plot_class_distributions(data):
    """Plot various distribution graphs for the data."""
    plot_options = {
        1: ('Class', None, 'Marks Class Count'),
        2: ('StudentAbsenceDays', 'Class', 'Marks Class Absent Days-wise'),
        3: ('VisITedResources', 'Class', 'Resources Visited by Class'),
        4: ('Discussion', 'Class', 'Discussion Participation by Class')
    }
    
    while True:
        print("\nVisualization Options:")
        for key, (x, hue, title) in plot_options.items():
            print(f"{key}. {title}")
        print("5. Exit Visualization")
        
        try:
            choice = int(input("\nEnter choice (1-5): "))
            if choice == 5:
                break
            elif choice in plot_options:
                x, hue, title = plot_options[choice]
                plt.figure(figsize=(10, 6))
                if hue:
                    sns.countplot(data=data, x=x, hue=hue, hue_order=['L', 'M', 'H'])
                else:
                    sns.countplot(data=data, x=x, order=['L', 'M', 'H'])
                plt.title(title)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()
            else:
                print("Invalid choice. Please enter a number between 1 and 5.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def predict_new_student():
    """Predict performance for a new student."""
    try:
        print("\nEnter student information:")
        features = {}
        
        features['raised_hands'] = int(input("Enter number of raised hands: "))
        features['visited_resources'] = int(input("Enter number of visited resources: "))
        features['discussions'] = int(input("Enter number of discussions: "))
        
        while True:
            absences = input("Enter absences (Under-7 or Above-7): ")
            if absences in ['Under-7', 'Above-7']:
                features['absences'] = 1 if absences == "Under-7" else 0
                break
            print("Please enter either 'Under-7' or 'Above-7'")
        
        return np.array([
            features['raised_hands'],
            features['visited_resources'],
            features['discussions'],
            features['absences']
        ]).reshape(1, -1)
    except ValueError:
        print("Invalid input. Please enter numeric values for the first three inputs.")
        return None

def main():
    # Load and prepare data
    print("Loading and preprocessing data...")
    data = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = split_data(data)
    
    # Training phase
    print("\nTraining models...")
    models = train_models(X_train, y_train)
    
    # Evaluation phase
    print("\nEvaluating models...")
    predictions = {name: model.predict(X_test) for name, model in models.items()}
    results_df, total_accuracy = compare_models(y_test, predictions)
    print("\nModel Comparison:")
    print(results_df.to_string(index=False))
    print(f"\nOverall Model Accuracy: {total_accuracy:.2f}%")
    
    while True:
        print("\nOptions:")
        print("1. View data distributions")
        print("2. Predict for new student")
        print("3. Exit")
        
        choice = input("\nEnter choice (1-3): ")
        
        if choice == '1':
            plot_class_distributions(data)
        elif choice == '2':
            new_student = predict_new_student()
            if new_student is not None:
                print("\nPredictions for the new student:")
                for name, model in models.items():
                    pred = model.predict(new_student)[0]
                    class_map = {0: 'H', 1: 'M', 2: 'L'}
                    print(f"{name}: {class_map[pred]}")
        elif choice == '3':
            print("Thank you for using the Student Progress Predictor!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")
