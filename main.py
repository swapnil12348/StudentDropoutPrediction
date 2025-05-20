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
st.set_page_config(page_title="Student Dropout Prediction", page_icon=":school:", layout="wide")


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
        'Data Type': df.dtypes,
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
    """Preprocess the data for model training"""
    # Drop student ID
    if 'id' in df.columns:
        df = df.drop('id', axis=1)

    # Handle missing values
    df = df.fillna(df.median(numeric_only=True))

    # Encode categorical variables including Target
    le = LabelEncoder()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    # Ensure all columns are numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop any rows with NaN values after conversion
    df.dropna(inplace=True)

    # Prepare features and target
    X = df.drop('Target', axis=1)
    y = df['Target']

    return X, y, df


def train_model(X_train, y_train):
    """Train a Random Forest Classifier"""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model


def individual_dropout_prediction(model, X):
    """
    Create interactive widgets for individual dropout prediction
    Args:
        model: Trained Random Forest Classifier
        X: Feature DataFrame used for training
    """
    st.subheader("Student Dropout Probability Predictor")

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
        prev_grade = st.slider("Previous qualification grade", min_value=0.0, max_value=200.0, value=120.0, step=0.1)
        admission_grade = st.slider("Admission grade", min_value=0.0, max_value=200.0, value=120.0, step=0.1)

    # Family & Financial
    with col3:
        st.markdown("### Family & Financial")
        mother_qual = st.selectbox("Mother's qualification", list(range(1, 35)))
        father_qual = st.selectbox("Father's qualification", list(range(1, 35)))
        mother_occ = st.selectbox("Mother's occupation", list(range(1, 46)))
        father_occ = st.selectbox("Father's occupation", list(range(1, 46)))
        debtor = st.selectbox("Debtor", [0, 1])
        tuition_up_to_date = st.selectbox("Tuition fees up to date", [0, 1])

    # Academic Performance
    st.markdown("### Academic Performance")
    col4, col5 = st.columns(2)

    with col4:
        st.markdown("#### 1st Semester")
        units_1st_credited = st.slider("Units credited (1st sem)", min_value=0, max_value=20, value=0)
        units_1st_enrolled = st.slider("Units enrolled (1st sem)", min_value=0, max_value=20, value=6)
        units_1st_evaluations = st.slider("Units evaluations (1st sem)", min_value=0, max_value=20, value=6)
        units_1st_approved = st.slider("Units approved (1st sem)", min_value=0, max_value=20, value=5)
        units_1st_grade = st.slider("Average grade (1st sem)", min_value=0.0, max_value=20.0, value=12.0, step=0.1)
        units_1st_without_eval = st.slider("Units without evaluations (1st sem)", min_value=0, max_value=20, value=0)

    with col5:
        st.markdown("#### 2nd Semester")
        units_2nd_credited = st.slider("Units credited (2nd sem)", min_value=0, max_value=20, value=0)
        units_2nd_enrolled = st.slider("Units enrolled (2nd sem)", min_value=0, max_value=20, value=6)
        units_2nd_evaluations = st.slider("Units evaluations (2nd sem)", min_value=0, max_value=20, value=6)
        units_2nd_approved = st.slider("Units approved (2nd sem)", min_value=0, max_value=20, value=5)
        units_2nd_grade = st.slider("Average grade (2nd sem)", min_value=0.0, max_value=20.0, value=12.0, step=0.1)
        units_2nd_without_eval = st.slider("Units without evaluations (2nd sem)", min_value=0, max_value=20, value=0)

    # Economic Indicators
    st.markdown("### Economic Indicators")
    eco_col1, eco_col2, eco_col3 = st.columns(3)

    with eco_col1:
        unemployment = st.slider("Unemployment rate", min_value=0.0, max_value=20.0, value=11.0, step=0.1)

    with eco_col2:
        inflation = st.slider("Inflation rate", min_value=-2.0, max_value=5.0, value=0.6, step=0.1)

    with eco_col3:
        gdp = st.slider("GDP", min_value=-5.0, max_value=5.0, value=2.0, step=0.01)

    # Add debug information option
    show_debug = st.checkbox("Show debug information")

    # Prediction button
    if st.button("Predict Dropout Probability"):
        # Create input data dictionary using raw values (no text conversion needed)
        input_dict = {
            'Marital status': marital_status,
            'Application mode': application_mode,
            'Application order': application_order,
            'Course': course,
            'Daytime/evening attendance': attendance,
            'Previous qualification': prev_qualification,
            'Previous qualification (grade)': prev_grade,
            'Nacionality': 1,  # Default value
            'Mother\'s qualification': mother_qual,
            'Father\'s qualification': father_qual,
            'Mother\'s occupation': mother_occ,
            'Father\'s occupation': father_occ,
            'Admission grade': admission_grade,
            'Displaced': displaced,
            'Educational special needs': educational_needs,
            'Debtor': debtor,
            'Tuition fees up to date': tuition_up_to_date,
            'Gender': 1 if gender == "Male" else 0,
            'Scholarship holder': scholarship,
            'Age at enrollment': age,
            'International': international,
            'Curricular units 1st sem (credited)': units_1st_credited,
            'Curricular units 1st sem (enrolled)': units_1st_enrolled,
            'Curricular units 1st sem (evaluations)': units_1st_evaluations,
            'Curricular units 1st sem (approved)': units_1st_approved,
            'Curricular units 1st sem (grade)': units_1st_grade,
            'Curricular units 1st sem (without evaluations)': units_1st_without_eval,
            'Curricular units 2nd sem (credited)': units_2nd_credited,
            'Curricular units 2nd sem (enrolled)': units_2nd_enrolled,
            'Curricular units 2nd sem (evaluations)': units_2nd_evaluations,
            'Curricular units 2nd sem (approved)': units_2nd_approved,
            'Curricular units 2nd sem (grade)': units_2nd_grade,
            'Curricular units 2nd sem (without evaluations)': units_2nd_without_eval,
            'Unemployment rate': unemployment,
            'Inflation rate': inflation,
            'GDP': gdp
        }

        # Create DataFrame from dictionary
        input_data = pd.DataFrame([input_dict])

        # Show debug information if requested
        if show_debug:
            st.subheader("Debug Information")
            st.write("Input Data:")
            st.write(input_data)
            st.write("Model Classes:", model.classes_)
            st.write("Training Data Sample:")
            st.write(X.head())
            st.write("Input Data Shape:", input_data.shape)
            st.write("Training Data Shape:", X.shape)
            st.write("Input Data Columns:", input_data.columns.tolist())
            st.write("Training Data Columns:", X.columns.tolist())

        # Ensure matching columns by using the training feature names
        try:
            # Add any missing columns from training data
            for col in X.columns:
                if col not in input_data.columns:
                    input_data[col] = 0

            # Keep only columns that exist in training data
            input_data = input_data[X.columns]

            # Make prediction
            dropout_prob = model.predict_proba(input_data)

            # Get class names - assuming they might be encoded
            class_names = []
            for i in range(len(model.classes_)):
                if i == 0:
                    class_names.append("Dropout")
                elif i == 1:
                    if len(model.classes_) == 2:
                        class_names.append("Graduate")  # Binary case: 0=Dropout, 1=Graduate
                    else:
                        class_names.append("Enrolled")  # Multi-class case
                elif i == 2:
                    class_names.append("Graduate")
                else:
                    class_names.append(f"Class {i}")

            # Create results dataframe for display
            results = pd.DataFrame({
                'Outcome': class_names,
                'Probability': [f"{p * 100:.2f}%" for p in dropout_prob[0]],
                'Raw Probability': dropout_prob[0]
            })

            # Display results
            st.subheader("Prediction Results")
            st.dataframe(results[['Outcome', 'Probability']])

            # Show progress bars
            for i, (outcome, prob) in enumerate(zip(class_names, dropout_prob[0])):
                st.write(f"{outcome}:")
                st.progress(float(prob))

            # Highlight prediction and provide guidance
            max_class_idx = np.argmax(dropout_prob[0])
            max_class_name = class_names[max_class_idx]
            max_prob = dropout_prob[0][max_class_idx]

            st.subheader("Prediction Summary")
            if max_class_name == "Dropout":
                st.warning(f"⚠️ High risk of student dropout! Probability: {max_prob * 100:.2f}%")
                st.write("Recommended actions:")
                st.write("- Schedule academic counseling")
                st.write("- Review financial aid options")
                st.write("- Consider tutoring or additional support")
            elif max_class_name == "Enrolled":
                st.info(f"ℹ️ Student likely to remain enrolled. Probability: {max_prob * 100:.2f}%")
                st.write("Recommended actions:")
                st.write("- Continue regular academic monitoring")
                st.write("- Check in periodically on academic progress")
            else:  # Graduate
                st.success(f"✅ Student shows strong graduation potential! Probability: {max_prob * 100:.2f}%")
                st.write("Recommended actions:")
                st.write("- Encourage continued excellence")
                st.write("- Consider mentorship opportunities")

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.info("Try adjusting the input values or check the debug information for clues.")

def main():
    st.title("🎓 Student Dropout Prediction Dashboard")

    # Sidebar for navigation
    menu = ["Data Overview", "Exploratory Data Analysis", "Model Training", "Dropout Prediction"]
    choice = st.sidebar.selectbox("Select Module", menu)

    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload your Student Dropout CSV", type=['csv'])

    # Load data
    df = load_data(uploaded_file)

    # Preprocess data
    X, y, full_df = preprocess_data(df)

    # Store the model as a session state variable to persist across reruns
    if 'model' not in st.session_state:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Train model
        st.session_state.model = train_model(X_train, y_train)

    if choice == "Data Overview":
        display_dataframe_info(df)

    elif choice == "Exploratory Data Analysis":
        st.header("Exploratory Data Analysis")

        # Numerical Features Distribution
        st.subheader("Numerical Features Distribution")
        numerical_features = ['Age at enrollment', 'Admission grade', 'Curricular units 1st sem (grade)']
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i, feature in enumerate(numerical_features):
            try:
                sns.histplot(full_df[feature], kde=True, ax=axes[i])
                axes[i].set_title(f'Distribution of {feature}')
            except:
                axes[i].text(0.5, 0.5, f"Error plotting {feature}", ha='center')
        plt.tight_layout()
        st.pyplot(fig)

        # Correlation Heatmap
        st.subheader("Correlation Heatmap")
        try:
            correlation_features = ['Age at enrollment', 'Admission grade', 'Curricular units 1st sem (grade)',
                                    'Curricular units 2nd sem (grade)', 'Unemployment rate', 'Inflation rate', 'GDP']
            correlation_matrix = full_df[correlation_features].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        except:
            st.error("Could not generate correlation heatmap with the current data.")

        # Dropout Rate
        st.subheader("Student Outcome Distribution")
        try:
            outcome_count = full_df['Target'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(x=outcome_count.index, y=outcome_count.values, ax=ax)
            ax.set_title('Student Outcome Distribution')
            ax.set_ylabel('Count')
            ax.set_xlabel('Target')
            st.pyplot(fig)
        except:
            st.error("Could not generate outcome distribution with the current data.")

    elif choice == "Model Training":
        st.header("Model Training")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = train_model(X_train, y_train)

        # Model evaluation
        y_pred = model.predict(X_test)

        st.subheader("Model Performance")
        st.write("Accuracy:", accuracy_score(y_test, y_pred))

        # Classification Report
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        st.pyplot(fig)

        # Feature Importance
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)

        fig, ax = plt.subplots(figsize=(10, 8))
        top_features = feature_importance.head(15)  # Show top 15 features
        sns.barplot(x='Importance', y='Feature', data=top_features, ax=ax)
        ax.set_title('Top 15 Feature Importances')
        plt.tight_layout()
        st.pyplot(fig)

    elif choice == "Dropout Prediction":
        # Call the individual prediction function
        individual_dropout_prediction(st.session_state.model, X)


if __name__ == "__main__":
    main()