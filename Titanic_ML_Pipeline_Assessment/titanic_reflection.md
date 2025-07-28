# Part 5: Reflection & Documentation

## My Approach and Rationale

In this Assignment, I aimed to build a complete machine learning pipeline using the Titanic dataset. I approached this systematically in five key steps: preprocessing, model building, evaluation, pipeline integration, and reflection.

First, I explored and cleaned the dataset. Handling missing values was crucial—‘Age’ and ‘Embarked’ had missing entries, so I used the median and mode respectively. I dropped ‘Cabin’ due to excessive missing data. I encoded categorical variables using one-hot encoding for ‘Sex’ and ‘Embarked’. Then I scaled numerical features like ‘Age’ and ‘Fare’ to ensure that models like Logistic Regression and SVM performed optimally.

Next, I built model using supervised learning algorithm: Random Forest. I used cross-validation to assess their performance more robustly and applied GridSearchCV for hyperparameter tuning.Random Forest performed the best in terms of both accuracy and generalization.

## Challenges Faced and Solutions

One major challenge was handling missing data, especially the ‘Age’ column. Imputing it with the mean skewed the distribution, so I used the median instead. Another challenge was encoding and scaling in the correct order to avoid data leakage, which I solved by using `ColumnTransformer` inside a `Pipeline`.

While tuning hyperparameters, the GridSearch process was slow. I restricted the parameter grid size and number of cross-validation folds to speed it up without losing performance significantly. Visualizations like the confusion matrix and ROC curve also helped me confirm the model’s strengths and limitations.

## Suggestions for Improvement

If I had more time, I would include feature engineering techniques like extracting titles from names or creating a ‘FamilySize’ feature, which are known to boost performance on the Titanic dataset. Ensemble methods such as Gradient Boosting (e.g., XGBoost) could also be explored for better accuracy.

For deployment, saving the trained pipeline with `joblib` or `pickle`, and integrating it with a Flask API or a web dashboard would make it production-ready. I also suggest adding logging and performance monitoring if deployed in a real-time system.

Overall, this project helped me understand the end-to-end flow of a machine learning solution—from data cleaning to deployment strategy—while also highlighting the importance of reproducibility using pipelines.
