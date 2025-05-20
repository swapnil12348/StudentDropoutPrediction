import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Page configuration
st.set_page_config(page_title="Student Dropout Prediction", page_icon="🎓", layout="wide")

# Create a mapping dictionary for target labels to ensure consistency
TARGET_MAPPING = {
    'Graduate': 1,
    'Dropout': 0,
    'Enrolled': 2  # In case this label exists in some datasets
}


def display_dataframe_info(df):
    """Display comprehensive dataframe information in Streamlit"""
    st.header("Data Overview")

    # Dataset basic information
    st.subheader("Dataset Dimensions")
    st.write(f"Number of Rows: {df.shape[0]}")
    st.write(f"Number of Columns: {df.shape[1]}")

    # Display first few rows
    st.subheader("First Few Rows")
    st.dataframe(df.head())

    # Column information
    st.subheader("Column Information")
    column_info = pd.DataFrame({
        'Column Name': df.columns,
        'Data Type': df.dtypes.astype(str),  # Convert to string to avoid Arrow conversion issues
        'Non-Null Count': df.count(),
        'Null Count': df.isnull().sum(),
        'Null Percentage': (df.isnull().sum() / len(df) * 100).round(2)
    })
    st.dataframe(column_info)

    # Basic statistics for numeric columns
    st.subheader("Numeric Columns Statistics")
    st.dataframe(df.describe())


def load_data(uploaded_file=None):
    """Load data from file or sample dataset"""
    # List of potential file paths to try
    potential_paths = [
        'C:/Users/swapn/StudentDropoutPrediction/.venv/data/student_dropout_data.csv',
        'C:/Users/swapn/StudentDropoutPrediction/data/student_dropout_data.csv',
        './data/student_dropout_data.csv',
        '../data/student_dropout_data.csv'
    ]

    # If a file is uploaded through Streamlit, use that first
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("Successfully loaded uploaded file!")
            return df
        except Exception as e:
            st.error(f"Error loading uploaded file: {e}")

    # Try loading from potential local paths
    for path in potential_paths:
        try:
            df = pd.read_csv(path)
            st.success(f"Successfully loaded data from {path}")
            return df
        except FileNotFoundError:
            continue
        except Exception as e:
            st.error(f"Error loading file from {path}: {e}")

    # If no file is found, create a minimal sample dataset based on the provided structure
    st.warning("No dataset found. Creating a minimal sample dataset.")
    df = pd.DataFrame({
        'id': [0, 1, 2],
        'Marital status': [1, 1, 1],
        'Application mode': [1, 17, 17],
        'Application order': [1, 1, 2],
        'Course': [9238, 9238, 9254],
        'Daytime/evening attendance': [1, 1, 1],
        'Previous qualification': [1, 1, 1],
        'Previous qualification (grade)': [126.0, 125.0, 137.0],
        'Nacionality': [1, 1, 1],
        'Mother\'s qualification': [1, 19, 3],
        'Father\'s qualification': [19, 19, 19],
        'Mother\'s occupation': [5, 9, 2],
        'Father\'s occupation': [5, 9, 3],
        'Admission grade': [122.6, 119.8, 144.7],
        'Displaced': [0, 1, 0],
        'Educational special needs': [0, 0, 0],
        'Debtor': [0, 0, 0],
        'Tuition fees up to date': [1, 1, 1],
        'Gender': [0, 0, 1],
        'Scholarship holder': [1, 0, 0],
        'Age at enrollment': [18, 18, 18],
        'International': [0, 0, 0],
        'Curricular units 1st sem (credited)': [0, 0, 0],
        'Curricular units 1st sem (enrolled)': [6, 6, 6],
        'Curricular units 1st sem (evaluations)': [6, 8, 0],
        'Curricular units 1st sem (approved)': [6, 4, 0],
        'Curricular units 1st sem (grade)': [14.5, 11.6, 0.0],
        'Curricular units 1st sem (without evaluations)': [0, 0, 0],
        'Curricular units 2nd sem (credited)': [0, 0, 0],
        'Curricular units 2nd sem (enrolled)': [6, 6, 6],
        'Curricular units 2nd sem (evaluations)': [7, 9, 0],
        'Curricular units 2nd sem (approved)': [6, 0, 0],
        'Curricular units 2nd sem (grade)': [12.428571428571429, 0.0, 0.0],
        'Curricular units 2nd sem (without evaluations)': [0, 0, 0],
        'Unemployment rate': [11.1, 11.1, 16.2],
        'Inflation rate': [0.6, 0.6, 0.3],
        'GDP': [2.02, 2.02, -0.92],
        'Target': ['Graduate', 'Dropout', 'Dropout']
    })
    return df


def preprocess_data(df):
    """Preprocess the data for model training with consistent target encoding"""
    # Make a copy to avoid overwriting original data
    processed_df = df.copy()

    # Drop student ID if present
    if 'id' in processed_df.columns:
        processed_df = processed_df.drop('id', axis=1)

    # Log information about target values
    st.write("Target values found:", processed_df['Target'].unique().tolist())

    # Handle missing values
    processed_df = processed_df.fillna(processed_df.median(numeric_only=True))

    # Encode categorical variables
    le = LabelEncoder()
    categorical_cols = processed_df.select_dtypes(include=['object']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col != 'Target']  # Exclude Target

    for col in categorical_cols:
        processed_df[col] = le.fit_transform(processed_df[col])

    # Special handling for Target column to ensure consistent encoding
    if 'Target' in processed_df.columns:
        # Create a mapping table to show the encoding
        target_mapping_df = pd.DataFrame({
            'Original': list(TARGET_MAPPING.keys()),
            'Encoded': list(TARGET_MAPPING.values())
        })

        # Display the mapping
        with st.expander("Target Label Encoding"):
            st.dataframe(target_mapping_df)

        # Apply the fixed encoding
        processed_df['Target'] = processed_df['Target'].map(TARGET_MAPPING)
        processed_df['Target'] = processed_df['Target'].fillna(0)  # Default to 'Dropout' for any unknown values

    # Ensure all columns are numeric
    for col in processed_df.columns:
        processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')

    # Drop any rows with NaN values after conversion
    processed_df.dropna(inplace=True)

    # Prepare features and target
    X = processed_df.drop('Target', axis=1)
    y = processed_df['Target']

    return X, y, processed_df


def train_model(X_train, y_train):
    """Train a Random Forest Classifier with robust parameters"""
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    status_text.text("Starting model training...")

    # Use more trees and handle class imbalance
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1  # Use all available cores for faster training
    )

    status_text.text("Fitting the model...")
    progress_bar.progress(25)

    model.fit(X_train, y_train)

    progress_bar.progress(100)
    status_text.text("Model training complete!")

    return model


def individual_dropout_prediction(model, X):
    """
    Create interactive widgets for individual dropout prediction
    Args:
        model: Trained Random Forest Classifier
        X: Feature DataFrame used for training
    """
    st.subheader("Student Dropout Probability Predictor")

    # Optional debug mode
    show_debug = st.checkbox("Show debug information")

    # Show model information if debug mode is on
    if show_debug:
        st.subheader("Model Information")
        st.write("Model Classes:", model.classes_)

        # Map numeric classes back to labels
        reverse_mapping = {v: k for k, v in TARGET_MAPPING.items()}
        st.write("Class mapping:", {f"Class {k}": reverse_mapping.get(k, f"Unknown-{k}") for k in model.classes_})

    # Create columns for better layout
    col1, col2, col3 = st.columns(3)

    # Demographics
    with col1:
        st.markdown("### Demographics")
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.slider("Age at enrollment", min_value=17, max_value=70, value=20)
        marital_status = st.selectbox("Marital status", [1, 2, 3, 4, 5, 6])
        international = st.selectbox("International", [0, 1])
        displaced = st.selectbox("Displaced", [0, 1])
        educational_needs = st.selectbox("Educational special needs", [0, 1])
        scholarship = st.selectbox("Scholarship holder", [0, 1])

    # Academic Background
    with col2:
        st.markdown("### Academic Background")
        application_mode = st.selectbox("Application mode", list(range(1, 18)))
        application_order = st.slider("Application order", min_value=1, max_value=9, value=1)
        course = st.number_input("Course Code", min_value=9000, max_value=9999, value=9238)
        attendance = st.selectbox("Daytime/evening attendance", [0, 1])
        prev_qualification = st.selectbox("Previous qualification", list(range(1, 18)))
        prev_grade = st.slider("Previous qualification grade", min_value=0.0, max_value=200.0, value=120.0, step=5.0)
        admission_grade = st.slider("Admission grade", min_value=0.0, max_value=200.0, value=120.0, step=5.0)

    # Academic Performance - Most important features
    with col3:
        st.markdown("### Academic Performance")
        units_1st_approved = st.slider("Units approved (1st sem)", min_value=0, max_value=20, value=5)
        units_1st_grade = st.slider("Average grade (1st sem)", min_value=0.0, max_value=20.0, value=12.0, step=1.0)
        units_2nd_approved = st.slider("Units approved (2nd sem)", min_value=0, max_value=20, value=5)
        units_2nd_grade = st.slider("Average grade (2nd sem)", min_value=0.0, max_value=20.0, value=12.0, step=1.0)

    # Economic Indicators
    st.markdown("### Economic Indicators")
    eco_col1, eco_col2, eco_col3 = st.columns(3)
    with eco_col1:
        unemployment = st.slider("Unemployment rate", min_value=0.0, max_value=20.0, value=11.0, step=1.0)
    with eco_col2:
        inflation = st.slider("Inflation rate", min_value=-2.0, max_value=5.0, value=0.6, step=0.5)
    with eco_col3:
        gdp = st.slider("GDP", min_value=-5.0, max_value=5.0, value=2.0, step=0.5)

    # Create default values for remaining features
    default_values = {
        'Mother\'s qualification': 10,
        'Father\'s qualification': 10,
        'Mother\'s occupation': 5,
        'Father\'s occupation': 5,
        'Debtor': 0,
        'Tuition fees up to date': 1,
        'Nacionality': 1,
        'Curricular units 1st sem (credited)': 0,
        'Curricular units 1st sem (enrolled)': 6,
        'Curricular units 1st sem (evaluations)': 6,
        'Curricular units 1st sem (without evaluations)': 0,
        'Curricular units 2nd sem (credited)': 0,
        'Curricular units 2nd sem (enrolled)': 6,
        'Curricular units 2nd sem (evaluations)': 6,
        'Curricular units 2nd sem (without evaluations)': 0,
    }

    # Show advanced options toggle
    show_advanced = st.checkbox("Show advanced options")
    if show_advanced:
        st.markdown("### Advanced Features")
        col_adv1, col_adv2 = st.columns(2)

        with col_adv1:
            for key in list(default_values.keys())[:len(default_values) // 2]:
                if key in ['Nacionality', 'Debtor', 'Tuition fees up to date']:
                    default_values[key] = st.selectbox(key, [0, 1], index=default_values[key])
                else:
                    default_values[key] = st.number_input(key, value=default_values[key])

        with col_adv2:
            for key in list(default_values.keys())[len(default_values) // 2:]:
                default_values[key] = st.number_input(key, value=default_values[key])

    # Prediction button
    if st.button("Predict Dropout Probability"):
        # Create a spinner to show progress
        with st.spinner('Calculating prediction...'):
            # Create input data dictionary
            input_dict = {
                'Marital status': marital_status,
                'Application mode': application_mode,
                'Application order': application_order,
                'Course': course,
                'Daytime/evening attendance': attendance,
                'Previous qualification': prev_qualification,
                'Previous qualification (grade)': prev_grade,
                'Admission grade': admission_grade,
                'Displaced': displaced,
                'Educational special needs': educational_needs,
                'Gender': 1 if gender == "Male" else 0,
                'Scholarship holder': scholarship,
                'Age at enrollment': age,
                'International': international,
                'Curricular units 1st sem (approved)': units_1st_approved,
                'Curricular units 1st sem (grade)': units_1st_grade,
                'Curricular units 2nd sem (approved)': units_2nd_approved,
                'Curricular units 2nd sem (grade)': units_2nd_grade,
                'Unemployment rate': unemployment,
                'Inflation rate': inflation,
                'GDP': gdp
            }

            # Add the default values
            for key, value in default_values.items():
                input_dict[key] = value

            # Create DataFrame from dictionary
            input_data = pd.DataFrame([input_dict])

            # Show debug information
            if show_debug:
                st.subheader("Debug Information")
                st.write("Raw Input Data:")
                st.write(input_data)

            try:
                # Ensure input data has the same columns and types as training data
                for col in X.columns:
                    if col not in input_data.columns:
                        input_data[col] = X[col].median()

                # Keep only columns from training data and ensure same order
                input_data = input_data[X.columns]

                # Get prediction probabilities
                prediction_probs = model.predict_proba(input_data)[0]

                # Map class indices to human-readable labels
                reverse_mapping = {v: k for k, v in TARGET_MAPPING.items()}
                class_names = [reverse_mapping.get(i, f"Class {i}") for i in model.classes_]

                # Create results dataframe
                results = pd.DataFrame({
                    'Outcome': class_names,
                    'Probability': [f"{p * 100:.2f}%" for p in prediction_probs],
                    'Raw Value': prediction_probs
                })

                # Display results
                st.subheader("Prediction Results")
                st.dataframe(results)

                # Show probability bars
                for outcome, prob in zip(class_names, prediction_probs):
                    st.write(f"{outcome}: {prob:.2%}")
                    st.progress(float(prob))

                # Highlight most likely outcome
                max_class_idx = np.argmax(prediction_probs)
                max_class_name = class_names[max_class_idx]
                max_prob = prediction_probs[max_class_idx]

                st.subheader("Prediction Summary")

                if max_class_name == "Dropout":
                    st.warning(f"⚠️ Risk of student dropout detected! Probability: {max_prob:.2%}")
                    st.write("Recommended actions:")
                    st.write("- Schedule academic counseling")
                    st.write("- Review financial aid options")
                    st.write("- Consider tutoring or additional support")
                else:  # Graduate or Enrolled
                    st.success(f"✅ Student likely to succeed! Probability of {max_class_name}: {max_prob:.2%}")
                    st.write("Recommended actions:")
                    st.write("- Continue regular monitoring")
                    st.write("- Provide ongoing encouragement and support")

            except Exception as e:
                st.error(f"Prediction error: {str(e)}")

                # Show detailed error information in debug mode
                if show_debug:
                    import traceback
                    st.code(traceback.format_exc())


def visualize_model_results(model, X_test, y_test):
    """Visualize model evaluation results"""
    # Get predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Get accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Display accuracy with a metric
    st.metric("Model Accuracy", f"{accuracy:.2%}")

    # Create tabs for different visualizations
    tabs = st.tabs(["Classification Report", "Confusion Matrix", "Feature Importance"])

    with tabs[0]:
        # Classification Report
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)

    with tabs[1]:
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)

        # Map numeric classes to labels
        reverse_mapping = {v: k for k, v in TARGET_MAPPING.items()}
        class_labels = [reverse_mapping.get(c, f"Class {c}") for c in sorted(np.unique(y_test))]

        ax.set_title('Confusion Matrix')
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        ax.set_xticklabels(class_labels)
        ax.set_yticklabels(class_labels)
        st.pyplot(fig)

    with tabs[2]:
        # Feature Importance
        st.subheader("Feature Importance")

        feature_importance = pd.DataFrame({
            'Feature': X_test.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)

        fig, ax = plt.subplots(figsize=(10, 8))
        top_features = feature_importance.head(15)  # Show top 15 features
        sns.barplot(x='Importance', y='Feature', data=top_features, ax=ax)
        ax.set_title('Top 15 Feature Importances')
        plt.tight_layout()
        st.pyplot(fig)


def main():
    st.title("🎓 Student Dropout Prediction Dashboard")
    st.markdown("""
    This dashboard helps predict and analyze factors that lead to student dropout.
    Upload your data or use our sample dataset to explore this critical educational issue.
    """)

    # Initialize session state variables if they don't exist
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False

    if 'show_prediction' not in st.session_state:
        st.session_state.show_prediction = False

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    menu = ["Data Overview", "Exploratory Data Analysis", "Model Training & Evaluation", "Dropout Prediction"]
    choice = st.sidebar.radio("Select Module", menu)

    # File upload
    st.sidebar.header("Data Input")
    uploaded_file = st.sidebar.file_uploader("Upload your Student Dropout CSV", type=['csv'])

    # Load data
    df = load_data(uploaded_file)

    # Preprocess data
    X, y, processed_df = preprocess_data(df)

    # Split data once for consistency
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Handle different menu choices
    if choice == "Data Overview":
        display_dataframe_info(df)

    elif choice == "Exploratory Data Analysis":
        st.header("Exploratory Data Analysis")

        # Create tabs for different visualizations
        tabs = st.tabs(["Distribution Analysis", "Correlation Analysis", "Outcome Analysis"])

        with tabs[0]:
            st.subheader("Feature Distributions")

            # Select columns to visualize
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            selected_columns = st.multiselect(
                "Select features to visualize",
                options=numeric_cols,
                default=['Age at enrollment', 'Admission grade', 'Curricular units 1st sem (grade)']
            )

            if selected_columns:
                for column in selected_columns:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(df[column], kde=True, ax=ax)
                    ax.set_title(f'Distribution of {column}')
                    st.pyplot(fig)
            else:
                st.info("Please select at least one column to visualize")

        with tabs[1]:
            st.subheader("Correlation Analysis")

            # Select columns for correlation
            correlation_features = st.multiselect(
                "Select features for correlation analysis",
                options=numeric_cols,
                default=['Age at enrollment', 'Admission grade', 'Curricular units 1st sem (grade)',
                         'Curricular units 2nd sem (grade)', 'Unemployment rate']
            )

            if len(correlation_features) > 1:
                correlation_matrix = df[correlation_features].corr()
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
                st.pyplot(fig)
            else:
                st.info("Please select at least two features for correlation analysis")

        with tabs[2]:
            st.subheader("Student Outcome Analysis")

            # Simple count plot of target variable
            fig, ax = plt.subplots(figsize=(8, 6))
            target_counts = df['Target'].value_counts().reset_index()
            target_counts.columns = ['Outcome', 'Count']
            sns.barplot(x='Outcome', y='Count', data=target_counts, ax=ax)
            ax.set_title('Distribution of Student Outcomes')
            st.pyplot(fig)

            # Target distribution by key factors
            factor_cols = st.multiselect(
                "Select factors to analyze against outcomes",
                options=['Gender', 'Scholarship holder', 'Displaced', 'International'],
                default=['Gender', 'Scholarship holder']
            )

            if factor_cols:
                for col in factor_cols:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    # Convert to numeric for consistency in crosstab
                    cross_tab = pd.crosstab(df[col], df['Target'])
                    cross_tab.plot(kind='bar', stacked=True, ax=ax)
                    ax.set_title(f'Outcome Distribution by {col}')
                    ax.set_ylabel('Count')
                    ax.legend(title='Outcome')
                    st.pyplot(fig)

    elif choice == "Model Training & Evaluation":
        st.header("Model Training & Evaluation")

        # Train model button
        if st.button("Train Model") or st.session_state.model_trained:
            # Train the model
            model = train_model(X_train, y_train)

            # Store model in session state
            st.session_state.model = model
            st.session_state.model_trained = True

            # Visualize results
            visualize_model_results(model, X_test, y_test)

            # Option to proceed to prediction
            if st.button("Proceed to Student Dropout Prediction"):
                st.session_state.show_prediction = True
                st.experimental_rerun()
        else:
            st.info("Click 'Train Model' to start the training process")

    elif choice == "Dropout Prediction" or st.session_state.show_prediction:
        st.header("Student Dropout Prediction")

        # Check if model exists in session state
        if 'model' in st.session_state:
            individual_dropout_prediction(st.session_state.model, X)
        else:
            st.warning("Please train the model first!")
            if st.button("Go to Model Training"):
                st.session_state.show_prediction = False
                st.experimental_rerun()


if __name__ == "__main__":
    main()
