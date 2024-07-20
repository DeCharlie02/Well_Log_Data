# Well Log Data Prediction Using AI Algorithms

## Introduction

### Background
Well log data is essential in geological and petroleum engineering, providing detailed records of geological formations penetrated by a borehole. It captures various physical properties at different depths, including Caliper (CALI), Gamma Ray (GR), Neutron Porosity (NPHI), Bulk Density (RHOB), Resistivity (RT), and Compressional Sonic Travel Time (DTCO). Despite its importance, missing intervals in well log data due to equipment failure or adverse conditions can impact subsurface characterization and model accuracy. This project focuses on predicting these missing values using advanced AI algorithms to improve data integrity and decision-making in exploration and production.

### Problem Statement
Missing log data intervals pose challenges in well log interpretation. Traditional methods like interpolation or mean imputation may fail to capture complex, non-linear relationships among log measurements. This project explores AI algorithms—specifically Random Forest and Artificial Neural Networks (ANNs)—to better model these relationships and predict missing well log values.

### Objectives
1. **Predict Missing Well Log Data:** Utilize Random Forest and ANN models to fill gaps in well log data.
2. **Compare Model Performance:** Evaluate and compare the accuracy, reliability, and efficiency of Random Forest and ANN models.
3. **Implement Data Visualization:** Develop visualizations to analyze and interpret prediction results.
4. **Provide Recommendations:** Offer insights for improving data completeness and optimizing AI algorithms in future projects.

## Challenges and Basic Concepts

### Well Log Data and Its Challenges
Well log data is crucial but often incomplete due to various factors. Traditional methods like linear interpolation or mean imputation may not effectively capture the non-linear relationships in the data.

### Machine Learning in Well Log Prediction
Machine learning models, particularly Random Forest and ANNs, can learn from existing data to accurately estimate missing values. 

### Random Forest
An ensemble learning method combining multiple decision trees. It is effective in handling large datasets, missing values, and provides insights into feature importance.

### Artificial Neural Networks
Computational models inspired by the human brain, capable of capturing complex relationships in data. ANNs use advanced techniques to enhance performance and generalization.

### Comparative Studies
Studies indicate Random Forest is effective for interpretability and handling small datasets, while ANNs excel in capturing complex patterns. This project leverages both models to predict missing well log values.

## Methodology

### Data Collection and Preparation
1. **Loading Data:** Imported from an Excel file using pandas.
2. **Handling Missing Values:** Replaced placeholder values with NaN.
3. **Feature Selection:** Included CALI, GR, NPHI, RHOB, and RT for predicting DTCO.
4. **Data Imputation:** Used mean value strategy, with alternative methods like LOCF, Interpolation, and KNN considered.
5. **Data Standardization:** Standardized features using StandardScaler.

### Model Development

#### Random Forest Model
1. **Initialization:** Configured using scikit-learn with key hyperparameters.
2. **Training:** Fit to the training data.
3. **Prediction:** Predicted missing DTCO values and evaluated performance.
4. **Feature Importance:** Analyzed and visualized using feature_importances_.

#### Artificial Neural Network (ANN)
1. **Initialization:** Designed with TensorFlow and Keras, using ReLU and linear activation functions.
2. **Compilation:** Used MSE as the loss function and Adam optimizer.
3. **Training:** Implemented with early stopping, batch size of 32, and up to 100 epochs.
4. **Prediction:** Compared predicted DTCO values with actual values.
5. **Evaluation Metrics:** Calculated RMSE for accuracy assessment.
6. **Visualization:** Plotted loss curves and predicted vs. actual values.

### Data Visualization
1. **Correlation Matrix:** Visualized relationships between features.
2. **Feature Importance:** Compared Random Forest and ANN feature importances.
3. **Density Plots:** Displayed distributions of features and target variable.
4. **Prediction Results:** Scatter plots of predicted vs. actual DTCO values.

## Results

Original Log Data

![image](https://github.com/user-attachments/assets/ab68ff6b-bc52-4029-bef9-110c8e8835ab)

Predicted DTCO Log Values using Random Forest.

![image](https://github.com/user-attachments/assets/b70bdbe0-b9a1-41fd-8f03-5fbd3c0ad1ce)

Correlation Matrix and Feature Importance Chart for Random Forest.

![image](https://github.com/user-attachments/assets/ab0b4eac-22ed-4adc-837a-f2eb592ef0f7)

Original vs Predicted DTCO Values Graph for Random Forest Algorithm.

![image](https://github.com/user-attachments/assets/542aa03b-3ee5-45a6-88eb-8b68a5246e9f)

Predicted DTCO Log Values using Artificial Neural Network(ANN).

![image](https://github.com/user-attachments/assets/48c92e73-c4ee-4b03-aa8d-2b11e68a1399)

Values Loss by each step in Artificial Neural Network(ANN)

![image](https://github.com/user-attachments/assets/b2d057a9-f87c-41c6-b23f-13c1e220e243)

Correlation Matrix and Features Importance Chart using Artificial Neural Network(ANN)

![image](https://github.com/user-attachments/assets/b0aa8624-78e4-41b4-9a5a-30ce3c65b4b0)

Original vs Predicted DTCO Values Graph for Artificial Neural Network.

![image](https://github.com/user-attachments/assets/0a63bc8b-859e-4f88-9959-be7dad44ff62)


### Random Forest Results

![image](https://github.com/user-attachments/assets/e968962a-8a68-4786-9efc-921d9f98ce4a)

## Conclusion

### Summary of Key Findings
- **Random Forest Performance:** Demonstrated robust performance with lower RMSE values compared to the ANN model, indicating its effectiveness in capturing linear relationships within the well log data.
- **ANN Performance:** Showed promise but with higher RMSE values than the Random Forest model. This suggests that non-linear relationships might not be as significant, or further tuning and optimization are needed.
- **Data Visualization:** Techniques such as feature importance graphs, correlation matrices, and density plots provided valuable insights into the data structure and the predictive power of different features.

### Strengths and Limitations

#### Strengths
- **Random Forest (RF):**
  - **Feature Importance:** Excels in identifying and ranking feature importance, offering clear insights into which features drive predictions, such as RHOB.
  - **Robustness to Overfitting:** The ensemble learning approach reduces overfitting tendencies, maintaining stable performance across various datasets.
  - **Linear Relationships:** Effective in capturing complex linear relationships, contributing to superior performance where such relationships are predominant.

- **Artificial Neural Network (ANN):**
  - **Nonlinear Mapping:** Models intricate nonlinear relationships, allowing for nuanced predictions where relationships between features and measurements are complex.
  - **Adaptability to Data Complexity:** Demonstrates adaptability to varying degrees of data complexity, showing potential in handling intricate patterns.
  - **Potential for Optimization:** Shows promise for further optimization with appropriate hyperparameter tuning and architecture design, indicating scalability for larger datasets.

#### Limitations
- **Random Forest (RF):**
  - **Model Interpretability:** While providing feature importance, interpreting individual decision trees can be challenging, limiting detailed insights into predictions.
  - **Dependency on Hyperparameters:** Performance is sensitive to hyperparameter settings, requiring careful tuning that can be time-consuming and computationally intensive.

- **Artificial Neural Network (ANN):**
  - **Black-Box Nature:** Opaque decision-making process limits understanding of how predictions are derived, affecting interpretability.
  - **Data Dependency:** Heavily reliant on the quantity and quality of training data, with issues like data sparsity or outliers adversely impacting accuracy.
  - **Computational Intensity:** Training, especially with large datasets or complex architectures, demands substantial computational resources, making it less practical in resource-constrained environments.

---
