import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers

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

# Function to build and train the neural network model
def build_train_nn_model(X_train, y_train, X_val, y_val):
    # Define neural network architecture
    model = Sequential()
    model.add(Dense(128, input_shape=(X_train.shape[1],), activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(1))

    # Compile the model
    lr = 0.001
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=lr))

    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    # Train the model
    history = model.fit(X_train, y_train, batch_size=32, epochs=500, validation_data=(X_val, y_val), callbacks=[early_stopping])
    return model

# Function to perform prediction and visualization using the trained neural network model
def predict_and_visualize_nn(model):
    if data is None:
        st.error("No data loaded")
        return

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
    x_testwell = imputer.transform(x_testwell)  # Impute missing values in test data
    x_testwell = scaler.transform(x_testwell)  # Standardize test data

    # Filter the training data to exclude the rows where DTCO is NaN
    train_mask = ~data['DTCO'].isna()
    x_trainwell_filtered = x_trainwell[train_mask]
    y_trainwell_filtered = y_trainwell[train_mask]

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(x_trainwell_filtered, y_trainwell_filtered, test_size=0.2, random_state=42)

    # Normalize the target variable for better training
    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val = scaler_y.transform(y_val.reshape(-1, 1)).flatten()

    # Build and train the neural network model
    model = build_train_nn_model(X_train, y_train, X_val, y_val)

    # Predict DTCO values for the entire dataset
    x_all = scaler.transform(imputer.transform(data[features].values))
    y_all_predict = scaler_y.inverse_transform(model.predict(x_all).flatten().reshape(-1, 1)).flatten()

    # Calculate RMSE for the training data
    y_train_pred = scaler_y.inverse_transform(model.predict(X_train).flatten().reshape(-1, 1)).flatten()
    rmse_train = np.sqrt(mean_squared_error(scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten(), y_train_pred))
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
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.plot([0, 200], [0, 200], color='cyan', linestyle='--')
    ax2.set_xlim(0, 200)
    ax2.set_ylim(0, 200)
    ax2.legend(fontsize=12)
    ax2.grid(True)
    st.pyplot(fig2)

    # Correlation matrix heatmap and feature importance side by side
    fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(20, 10))

    # Correlation matrix heatmap
    corr_matrix = data[features + ['DTCO']].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax3)
    ax3.set_title('Correlation Matrix', fontsize=16)

    # Feature importance calculation
    if model is not None:
        # Retrieve the weights of the first layer (assuming it corresponds to feature importances)
        feature_importances = model.layers[0].get_weights()[0]
        feature_importances = np.mean(np.abs(feature_importances), axis=1)  # Take the average across all neurons
        # Sort features by importance
        features_sorted_idx = np.argsort(feature_importances)
        ax4.barh(range(len(features)), feature_importances[features_sorted_idx], align='center')
        ax4.set_yticks(range(len(features)))
        ax4.set_yticklabels(np.array(features)[features_sorted_idx], fontsize=14)
        ax4.set_title('Feature Importances', fontsize=16)

    plt.tight_layout()
    st.pyplot(fig3)

    # Residuals plot
    fig4, ax5 = plt.subplots(figsize=(8, 6))
    residuals = scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten() - y_train_pred
    ax5.scatter(scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten(), residuals, alpha=0.5, color='blue')
    ax5.axhline(0, color='cyan', linestyle='--')
    ax5.set_xlabel('Original DTCO', fontsize=14)
    ax5.set_ylabel('Residuals', fontsize=14)
    ax5.tick_params(axis='both', which='major', labelsize=12)
    ax5.set_title('Residuals Plot', fontsize=16)
    ax5.grid(True)
    st.pyplot(fig4)

    # Density plot with histograms
    fig5, ax6 = plt.subplots(figsize=(8, 6))
    sns.histplot(y_trainwell_filtered, bins=30, kde=False, color='blue', alpha=0.5, label='Actual DTCO', ax=ax6)
    sns.histplot(y_train_pred, bins=30, kde=False, color='orange', alpha=0.5, label='Predicted DTCO', ax=ax6)
    ax6.set_xlabel('DTCO', fontsize=14)
    ax6.set_ylabel('Density', fontsize=14)
    ax6.tick_params(axis='both', which='major', labelsize=12)
    ax6.set_title('Density Plot', fontsize=16)
    ax6.legend(fontsize=12)
    st.pyplot(fig5)

    # Display the dataset with predicted values in a table
    st.subheader('Combined Data with Predicted DTCO')
    st.dataframe(combined_data)

# Streamlit UI layout
def main():
    st.title('Well Log Prediction with Neural Network')
    st.markdown('Upload your Excel file (.xlsx) containing well log data.')

    # Load data
    global data
    data = load_data()

    if data is not None:
        # Sidebar with prediction options
        st.sidebar.subheader('Prediction Options')
        predict_button = st.sidebar.button('Predict and Visualize')
        if predict_button:
            # Perform prediction and visualization using neural network model
            predict_and_visualize_nn(model=None)  # Pass your trained neural network model here

if __name__ == '__main__':
    main()
