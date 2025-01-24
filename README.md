# Student Enrollment Prediction Model

This repository contains a machine learning project that predicts student enrollment status based on given features. It includes data preprocessing, exploratory data analysis, model training, evaluation, and deployment. The best-performing model is saved and ready for real-world predictions.

---

## 📌 Features

- **Data Preprocessing**:
  - Handles missing values using imputation.
  - Standardizes numerical features and one-hot encodes categorical features.
- **Exploratory Data Analysis (EDA)**:
  - Visualizes the distribution of enrollment status.
  - Displays a correlation matrix of numerical features.
- **Model Training**:
  - Implements Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting models.
  - Evaluates models using metrics like accuracy, AUC-ROC, and classification reports.
- **Hyperparameter Tuning**:
  - Fine-tunes Random Forest using `GridSearchCV`.
- **Model Deployment**:
  - Saves the best-performing model using `joblib`.
  - Includes feature importance visualization.

---

## 🚀 Getting Started

### Prerequisites

Make sure you have Python installed along with the following libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `joblib`

Install the required libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
Running the Code
Clone this repository:
bash
git clone https://github.com/your-username/student-enrollment-model.git
Navigate to the project directory:
bash

cd student-enrollment-model
Ensure the dataset (combined_student_enrollment_data.csv) is in the project directory.
Run the script:
bash

load.py
📊 Project Workflow
1. Load the Dataset
The dataset is loaded using Pandas:

Victor = pd.read_csv('combined_student_enrollment_data.csv')
2. Preprocessing Pipelines
Handles numerical and categorical features using pipelines:

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
3. Exploratory Data Analysis (EDA)
Includes basic statistics and visualizations:

sns.countplot(x='enrollment_status', data=Victor)
plt.show()
4. Model Training
Four models are trained and evaluated:

models = [
    LogisticRegression(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GradientBoostingClassifier()
]
5. Hyperparameter Tuning
Fine-tunes the Random Forest model using GridSearchCV:

param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
6. Save and Load the Model
Saves the best model for deployment:

joblib.dump(best_model, 'student_enrollment_model.pkl')

## 📈 Example Output

## After running the script, you’ll see:

Dataset statistics and visualizations.
Accuracy and classification reports for each model.
The best model saved as student_enrollment_model.pkl.
Feature importance visualization for the Random Forest model.
