import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for seaborn
sns.set(style="whitegrid")

# Global variable to store the data
data = None

# Function to load the data
def load_data():
    uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")
    if uploaded_file is not None:
        try:
            global data
            data = pd.read_excel(uploaded_file)
            st.success("Data loaded successfully")
            st.write(f"File: {uploaded_file.name}")
            st.write(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")
        except Exception as e:
            st.error(f"Failed to load data: {e}")
    return data

# Function to perform prediction and visualization
def predict_and_visualize():
    if data is None:
        st.error("No data loaded")
        return

    # Get model parameters from user input
    n_estimators = st.sidebar.slider("Number of Estimators", min_value=10, max_value=1000, value=100, step=10)
    max_depth = st.sidebar.slider("Max Depth", min_value=1, max_value=100, value=10, step=1)

    # Replace all -999.25 values with NaN
    data.replace(-999.25, np.nan, inplace=True)

    # Define features and check if they exist
    features = ['CALI', 'GR', 'NPHI', 'RHOB', 'RT']
    for feature in features:
        if feature not in data.columns:
            st.error(f"Missing column in the data: {feature}")
            return

    # Define x_trainwell and y_trainwell
    try:
        x_trainwell = data[features].values
        y_trainwell = data['DTCO'].values
    except KeyError as e:
        st.error(f"Missing column in the data: {e}")
        return

    # Impute missing values with the mean
    imputer = SimpleImputer(strategy='mean')
    x_trainwell = imputer.fit_transform(x_trainwell)

    # Standardize the matrix for training data
    scaler = StandardScaler()
    x_trainwell = scaler.fit_transform(x_trainwell)

    # Filter the dataset for prediction
    prediction_mask = data['DTCO'].isna() & (data['DEPTH'] > 1251)
    x_testwell = data.loc[prediction_mask, features].values

    # Debug statements
    st.write("Shape of x_testwell before imputation:", x_testwell.shape)
    st.write("Number of rows meeting the prediction mask condition:", prediction_mask.sum())
    st.write("Content of x_testwell before imputation:", x_testwell)

    if x_testwell.shape[0] == 0:
        st.error("No rows meet the condition for prediction. Please check your data and conditions.")
        return

    x_testwell = imputer.transform(x_testwell)  # Impute missing values in test data
    x_testwell = scaler.transform(x_testwell)  # Standardize test data

    # Filter the training data to exclude the rows where DTCO is NaN
    train_mask = ~data['DTCO'].isna()
    x_trainwell_filtered = x_trainwell[train_mask]
    y_trainwell_filtered = y_trainwell[train_mask]

    # Random Forest Regressor with Randomized Search CV
    RF = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100, 300, 500],
        'max_features': [1.0, 'sqrt', 'log2'],
        'max_depth': [4, 8, 12]
    }
    random_search = RandomizedSearchCV(estimator=RF, param_distributions=param_grid, n_iter=10, cv=3, random_state=42, n_jobs=-1)

    with st.spinner('Finding best parameters...'):
        random_search.fit(x_trainwell_filtered, y_trainwell_filtered)

    # Random forest model with best parameters
    best_params = random_search.best_params_
    RF = RandomForestRegressor(n_estimators=n_estimators, max_features=best_params['max_features'], max_depth=max_depth, min_samples_leaf=1, random_state=42)

    with st.spinner('Fitting the model...'):
        RF.fit(x_trainwell_filtered, y_trainwell_filtered)

    # Predict DTCO values for the entire dataset
    x_all = imputer.transform(data[features].values)  # Impute missing values in entire data
    x_all = scaler.transform(x_all)  # Standardize the entire data
    y_all_predict = RF.predict(x_all)

    # Calculate RMSE for the training data
    y_train_pred = RF.predict(x_trainwell_filtered)
    rmse_train = np.sqrt(mean_squared_error(y_trainwell_filtered, y_train_pred))
    st.write(f'RMSE (Training Data): {rmse_train:.4f}')

    # Calculate RMSE for the overlapping original and predicted values in the entire dataset
    overlap_mask = ~data['DTCO'].isna()
    rmse_entire = np.sqrt(mean_squared_error(data['DTCO'][overlap_mask], y_all_predict[overlap_mask]))
    st.write(f'RMSE (Entire Data Overlap): {rmse_entire:.4f}')

    # Create a DataFrame for the predicted data
    predicted_data = data[['DEPTH']].copy()
    predicted_data['Predicted_DTCO'] = y_all_predict

    # Combine the original and predicted DTCO values
    combined_data = data.copy()
    combined_data['Predicted_DTCO'] = y_all_predict

    # Set predicted DTCO to NaN where original DTCO is present
    predicted_data_with_nan = combined_data.copy()
    predicted_data_with_nan['Predicted_DTCO'] = np.nan
    predicted_data_with_nan.loc[prediction_mask, 'Predicted_DTCO'] = y_all_predict[prediction_mask]

    # Plot the original and predicted DTCO values along with other features
    fig, axs = plt.subplots(1, 6, figsize=(30, 18), sharey=True)

    # Define x-axis limits for specific features
    x_limits = {
        'CALI': (5, 25),
        'GR': (0, 200),
        'NPHI': (0, 1),
        'RHOB': (1, 3),
        'RT': (0.1, 1000)  # Log scale typically uses a wide range
    }
    x_units = {
        'CALI': ' (inches)',
        'GR': ' (API)',
        'NPHI': ' (v/v)',
        'RHOB': ' (g/cm³)',
        'RT': ' (ohm·m)',
        'DTCO': ' (µs/ft)'
    }

    # Plot each feature
    for i, feature in enumerate(features):
        ax = axs[i]
        if feature == 'RT':
            ax.semilogx(data[feature], data['DEPTH'], label=feature, color='black', linewidth=1)
        else:
            ax.plot(data[feature], data['DEPTH'], label=feature, color='black', linewidth=1)
        ax.set_xlabel(f"{feature}{x_units.get(feature, '')}", fontsize=24)  # Set font size and units for x-axis label
        if i == 0:
            ax.set_ylabel('Depth', fontsize=24)  # Set font size for y-axis label only for the leftmost plot
        ax.tick_params(axis='both', which='major', labelsize=20)  # Set font size for tick labels
        if feature in x_limits:
            ax.set_xlim(x_limits[feature])
        ax.legend(fontsize=18)

    # Plot the original and predicted DTCO values in the last subplot
    ax = axs[5]
    ax.plot(combined_data['DTCO'], combined_data['DEPTH'], label='Original DTCO', color='black', linewidth=1)
    ax.plot(predicted_data_with_nan['Predicted_DTCO'], predicted_data_with_nan['DEPTH'], label='Predicted DTCO', color='magenta', linewidth=1)
    ax.plot(predicted_data['Predicted_DTCO'], combined_data['DEPTH'], label='Predicted DTCO (All)', color='red', linestyle='dotted', linewidth=0.5)
    ax.set_xlabel(f'DTCO{x_units["DTCO"]}', fontsize=24)  # Set font size and unit for x-axis label
    ax.tick_params(axis='both', which='major', labelsize=20)  # Set font size for tick labels
    ax.set_xlim(0, 200)  # Set the x-axis limit from 0 to 200
    ax.invert_yaxis()  # Invert the y-axis
    ax.legend(fontsize=18)

    st.pyplot(fig)

    # Scatter plot of original vs. predicted DTCO values for the entire dataset where both exist
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.scatter(data['DTCO'][overlap_mask], y_all_predict[overlap_mask], alpha=0.5, color='blue', label='Original vs. Predicted')
    ax2.set_xlabel('Original DTCO', fontsize=14)
    ax2.set_ylabel('Predicted DTCO', fontsize=14)
    ax2.set_title('Original vs. Predicted DTCO Scatter Plot', fontsize=16)
    ax2.legend()
    st.pyplot(fig2)

    # Feature Importance
    feature_importances = RF.feature_importances_
    feature_names = features
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.sort_values(by='Importance', ascending=False), ax=ax3)
    ax3.set_title('Feature Importances', fontsize=16)
    st.pyplot(fig3)

# Main function to run the app
def main():
    st.title("Well Log Data Prediction and Visualization")
    st.sidebar.title("Model Parameters")

    data = load_data()
    if data is not None:
        st.write(data.head())

    if st.sidebar.button('Run Prediction'):
        predict_and_visualize()

if __name__ == '__main__':
    main()
