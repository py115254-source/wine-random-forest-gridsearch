
# Wine Classification using Random Forest and GridSearchCV

## Project Overview
This project optimizes a Random Forest Classifier using GridSearchCV on the Scikit-learn Wine dataset.

## Objectives
- Load the Wine dataset.
- Train a Random Forest Classifier.
- Optimize model parameters using GridSearchCV.
- Apply 5-fold cross-validation.
- Evaluate the best model.

## Technologies Used
- Python
- Scikit-learn
- Pandas
- Random Forest
- GridSearchCV

## Methodology
1. Load the Wine dataset.
2. Split data into training and testing sets.
3. Define the Random Forest parameter grid.
4. Run GridSearchCV with 5-fold cross-validation.
5. Select the best estimator.
6. Evaluate test accuracy and classification report.

## Results
- Best Cross-Validation Score: 0.9862
- Test Accuracy: 1.00 (100%)
- Cross-Validation: 5-fold

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python main.py
```
