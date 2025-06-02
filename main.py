import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(page_title="Student Dropout Prediction", page_icon="🎓", layout="wide")

# Create a mapping dictionary for target labels to ensure consistency
TARGET_MAPPING = {
    'Graduate': 1,
    'Dropout': 0,
    'Enrolled': 2  # In case this label exists in some datasets
}


# Fix for Data Quality Assessment alignment (around line 45-80)


def display_enhanced_dataframe_info(df):
    """Display enhanced and insightful dataframe information with improved layout"""
    st.header("📊 Data Overview & Insights")

    # Key metrics at the top with consistent styling
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Students", f"{df.shape[0]:,}")
    with col2:
        dropout_rate = (df['Target'] == 'Dropout').sum() / len(df) * 100
        st.metric("Dropout Rate", f"{dropout_rate:.1f}%")
    with col3:
        graduate_rate = (df['Target'] == 'Graduate').sum() / len(df) * 100
        st.metric("Graduate Rate", f"{graduate_rate:.1f}%")
    with col4:
        missing_data_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
        st.metric("Missing Data", f"{missing_data_pct:.1f}%")

    # Data Quality Overview - Improved alignment and consistent sizing
    st.subheader("🔍 Data Quality Assessment")
    col1, col2 = st.columns([1, 1])  # Equal width columns

    with col1:
        st.write("**Missing Data Analysis**")
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)

        if len(missing_data) > 0:
            fig, ax = plt.subplots(figsize=(6, 4))  # Standardized size
            missing_data.plot(kind='bar', ax=ax, color='coral')
            ax.set_title('Missing Data by Column', fontsize=12, fontweight='bold')
            ax.set_ylabel('Number of Missing Values', fontsize=10)
            ax.set_xlabel('Columns', fontsize=10)
            ax.tick_params(axis='both', labelsize=8)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.success("✅ No missing data found!")

    with col2:
        st.write("**Data Types Distribution**")
        dtype_counts = df.dtypes.value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))  # Standardized size
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
        dtype_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%', colors=colors[:len(dtype_counts)],
                         textprops={'fontsize': 8})
        ax.set_title('Distribution of Data Types', fontsize=12, fontweight='bold')
        ax.set_ylabel('')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Student Demographics Insights - Improved alignment and consistent sizing
    st.subheader("👥 Student Demographics Insights")
    col1, col2, col3 = st.columns([1, 1, 1])  # Equal width columns

    with col1:
        st.write("**Age Distribution**")
        if 'Age at enrollment' in df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))  # Standardized size
            df['Age at enrollment'].hist(bins=20, ax=ax, color='lightblue', alpha=0.7, edgecolor='black')
            ax.axvline(df['Age at enrollment'].mean(), color='red', linestyle='--',
                       label=f'Mean: {df["Age at enrollment"].mean():.1f}')
            ax.set_title('Age at Enrollment', fontsize=12, fontweight='bold')
            ax.set_xlabel('Age', fontsize=10)
            ax.set_ylabel('Count', fontsize=10)
            ax.tick_params(axis='both', labelsize=8)
            ax.legend(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with col2:
        st.write("**Gender Distribution**")
        if 'Gender' in df.columns:
            gender_counts = df['Gender'].value_counts()
            gender_labels = ['Female' if x == 0 else 'Male' for x in gender_counts.index]
            fig, ax = plt.subplots(figsize=(6, 4))  # Standardized size
            ax.pie(gender_counts.values, labels=gender_labels, autopct='%1.1f%%',
                   colors=['pink', 'lightblue'], textprops={'fontsize': 8})
            ax.set_title('Gender Distribution', fontsize=12, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with col3:
        st.write("**Scholarship Distribution**")
        if 'Scholarship holder' in df.columns:
            scholarship_counts = df['Scholarship holder'].value_counts()
            scholarship_labels = ['No Scholarship' if x == 0 else 'Scholarship' for x in scholarship_counts.index]
            fig, ax = plt.subplots(figsize=(6, 4))  # Standardized size
            ax.pie(scholarship_counts.values, labels=scholarship_labels, autopct='%1.1f%%',
                   colors=['lightcoral', 'lightgreen'], textprops={'fontsize': 8})
            ax.set_title('Scholarship Status', fontsize=12, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # Academic Performance Overview
    st.subheader("📚 Academic Performance Overview")
    st.info("💡 **What this shows:** How well students performed academically and how it relates to their final outcome (Graduate/Dropout)")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.write("** Admission Grades vs Final Outcome**")
        st.caption("Shows how entrance exam scores relate to student success")
        if 'Admission grade' in df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))  # Standardized size
            colors = {'Graduate': 'green', 'Dropout': 'red', 'Enrolled': 'orange'}
            for outcome in df['Target'].unique():
                subset = df[df['Target'] == outcome]['Admission grade']
                ax.hist(subset, alpha=0.6, label=f'{outcome} ({len(subset)} students)',
                        bins=15, color=colors.get(outcome, 'gray'))
            ax.set_title('Admission Grade by Outcome', fontsize=12, fontweight='bold')
            ax.set_xlabel('Admission Grade', fontsize=10)
            ax.set_ylabel('Number of Students', fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', labelsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.caption("📊 **Key Insight:** Students with higher admission grades are more likely to graduate")

    with col2:
        st.write("**First Semester Performance vs Final Outcome**")
        st.caption("Shows how first semester grades predict student success")
        if 'Curricular units 1st sem (grade)' in df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))  # Standardized size
            colors = {'Graduate': 'green', 'Dropout': 'red', 'Enrolled': 'orange'}
            for outcome in df['Target'].unique():
                subset = df[df['Target'] == outcome]['Curricular units 1st sem (grade)']
                ax.hist(subset, alpha=0.6, label=f'{outcome} ({len(subset)} students)',
                        bins=15, color=colors.get(outcome, 'gray'))
            ax.set_title('1st Semester Grade by Outcome', fontsize=12, fontweight='bold')
            ax.set_xlabel('Average Grade (0-20)', fontsize=10)
            ax.set_ylabel('Number of Students', fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', labelsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.caption("📊 **Key Insight:** Strong first semester performance is crucial for graduation")

    # Performance Summary
    if 'Admission grade' in df.columns and 'Curricular units 1st sem (grade)' in df.columns:
        st.subheader("📈 Performance Summary")
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        with summary_col1:
            st.metric(
                label="Average Admission Grade (Graduates)",
                value=f"{df[df['Target'] == 'Graduate']['Admission grade'].mean():.1f}",
                delta=f"{df[df['Target'] == 'Graduate']['Admission grade'].mean() - df[df['Target'] == 'Dropout']['Admission grade'].mean():.1f} vs Dropouts"
            )
        with summary_col2:
            st.metric(
                label="Average 1st Sem Grade (Graduates)",
                value=f"{df[df['Target'] == 'Graduate']['Curricular units 1st sem (grade)'].mean():.1f}",
                delta=f"{df[df['Target'] == 'Graduate']['Curricular units 1st sem (grade)'].mean() - df[df['Target'] == 'Dropout']['Curricular units 1st sem (grade)'].mean():.1f} vs Dropouts"
            )
        with summary_col3:
            high_performers = df[df['Curricular units 1st sem (grade)'] > 15]
            success_rate = (high_performers['Target'] == 'Graduate').mean() * 100 if len(high_performers) > 0 else 0
            st.metric(
                label="Success Rate (Grade >15)",
                value=f"{success_rate:.1f}%"
            )

    # Key Risk Factors Analysis
    st.subheader("⚠️ Key Risk Factors Analysis")
    risk_factors = []
    if 'Curricular units 1st sem (approved)' in df.columns:
        low_performance = df['Curricular units 1st sem (approved)'] <= 2
        risk_factors.append(('Low 1st Sem Performance (≤2 units)', low_performance.sum()))
    if 'Age at enrollment' in df.columns:
        mature_students = df['Age at enrollment'] > 25
        risk_factors.append(('Mature Students (>25 years)', mature_students.sum()))
    if 'Scholarship holder' in df.columns:
        no_scholarship = df['Scholarship holder'] == 0
        risk_factors.append(('No Financial Aid', no_scholarship.sum()))
    if 'Displaced' in df.columns:
        displaced = df['Displaced'] == 1
        risk_factors.append(('Displaced Students', displaced.sum()))

    if risk_factors:
        risk_df = pd.DataFrame(risk_factors, columns=['Risk Factor', 'Number of Students'])
        fig, ax = plt.subplots(figsize=(6, 4))  # Standardized size
        bars = ax.bar(risk_df['Risk Factor'], risk_df['Number of Students'],
                      color=['red', 'orange', 'yellow', 'coral'][:len(risk_factors)])
        ax.set_title('Students by Risk Factor', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Students', fontsize=10)
        ax.tick_params(axis='both', labelsize=8)
        plt.xticks(rotation=45, ha='right')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Sample Data Preview with Context
    st.subheader("📋 Data Sample & Structure")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("**Sample of Student Records:**")
        sample_df = df.head(10)
        def highlight_outcome(val):
            if val == 'Dropout':
                return 'background-color: #ffcccc'
            elif val == 'Graduate':
                return 'background-color: #ccffcc'
            else:
                return 'background-color: #ffffcc'
        styled_df = sample_df.style.map(highlight_outcome, subset=['Target'])
        st.dataframe(styled_df)
    with col2:
        st.write("**Dataset Composition:**")
        st.write(f"• **Rows:** {df.shape[0]:,} students")
        st.write(f"• **Columns:** {df.shape[1]} features")
        demographic_cols = ['Gender', 'Age at enrollment', 'Marital status', 'Nacionality']
        academic_cols = [col for col in df.columns if 'grade' in col.lower() or 'units' in col.lower()]
        economic_cols = ['Unemployment rate', 'Inflation rate', 'GDP']
        st.write("**Feature Categories:**")
        st.write(f"• Demographics: {len([c for c in demographic_cols if c in df.columns])}")
        st.write(f"• Academic: {len([c for c in academic_cols if c in df.columns])}")
        st.write(f"• Economic: {len([c for c in economic_cols if c in df.columns])}")
        st.write(
            f"• Other: {df.shape[1] - len([c for c in demographic_cols + academic_cols + economic_cols if c in df.columns])}")

    # Interactive Feature Explorer
    st.subheader("🔧 Interactive Feature Explorer")
    col_to_explore = st.selectbox(
        "Select a feature to explore in detail:",
        options=[col for col in df.columns if col != 'Target' and col != 'id']
    )
    if col_to_explore:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.write(f"**Statistics for {col_to_explore}:**")
            if df[col_to_explore].dtype in ['int64', 'float64']:
                stats = df[col_to_explore].describe()
                for stat, value in stats.items():
                    st.write(f"• **{stat}:** {value:.2f}")
            else:
                unique_vals = df[col_to_explore].nunique()
                st.write(f"• **Unique values:** {unique_vals}")
                st.write(f"• **Most common:** {df[col_to_explore].mode().iloc[0]}")
        with col2:
            st.write(f"**{col_to_explore} vs Dropout Rate:**")
            if df[col_to_explore].dtype in ['int64', 'float64']:
                correlation = df[col_to_explore].corr(df['Target'].map({'Graduate': 1, 'Dropout': 0, 'Enrolled': 2}))
                st.write(f"• **Correlation with dropout:** {correlation:.3f}")
                df_temp = df.copy()
                try:
                    df_temp[f'{col_to_explore}_binned'] = pd.cut(df_temp[col_to_explore], bins=5)
                    dropout_by_bin = df_temp.groupby(f'{col_to_explore}_binned', observed=True)['Target'].apply(
                        lambda x: (x == 'Dropout').mean() * 100 if len(x) > 0 else 0
                    ).round(1)
                    for bin_range, dropout_rate in dropout_by_bin.items():
                        st.write(f"• **{bin_range}:** {dropout_rate:.1f}% dropout rate")
                except Exception as e:
                    st.write(f"• Unable to create bins for this feature: {str(e)}")
            else:
                dropout_by_cat = df.groupby(col_to_explore, observed=True)['Target'].apply(
                    lambda x: (x == 'Dropout').mean() * 100 if len(x) > 0 else 0
                ).round(1)
                for category, dropout_rate in dropout_by_cat.items():
                    st.write(f"• **{category}:** {dropout_rate:.1f}% dropout rate")

    # Expandable Detailed Column Information
    with st.expander("📋 Detailed Column Information"):
        column_info = pd.DataFrame({
            'Column Name': df.columns,
            'Data Type': df.dtypes.astype(str),
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Null Percentage': (df.isnull().sum() / len(df) * 100).round(2),
            'Unique Values': df.nunique()
        })
        def interpret_column(row):
            if row['Null Percentage'] > 10:
                return "⚠️ High missing data"
            elif row['Unique Values'] == 1:
                return "❌ No variation"
            elif row['Unique Values'] == len(df):
                return "🔑 Unique identifier"
            elif row['Unique Values'] < 10:
                return "📊 Categorical"
            else:
                return "📈 Continuous"
        column_info['Interpretation'] = column_info.apply(interpret_column, axis=1)
        st.dataframe(column_info)

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
    """Visualize model evaluation results with improved layout and smaller graphs"""
    st.subheader("Model Evaluation Results")

    # Get predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Display accuracy with a metric
    accuracy = accuracy_score(y_test, y_pred)
    st.metric("Model Accuracy", f"{accuracy:.2%}")

    # Option to choose display mode: Tabs or Panels
    display_mode = st.radio("Select display mode:", ["Tabs", "Panels"], index=1)

    if display_mode == "Tabs":
        # Original tab-based layout
        tabs = st.tabs(["Classification Report", "Confusion Matrix", "Feature Importance"])

        with tabs[0]:
            st.write("**Classification Report**")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)

        with tabs[1]:
            st.write("**Confusion Matrix**")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(6, 4))  # Smaller size
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        annot_kws={"size": 8}, cbar_kws={"shrink": 0.8})

            # Map numeric classes to labels
            reverse_mapping = {0: 'Dropout', 1: 'Graduate', 2: 'Enrolled'}
            class_labels = [reverse_mapping.get(c, f"Class {c}") for c in sorted(np.unique(y_test))]

            ax.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
            ax.set_ylabel('Actual', fontsize=10)
            ax.set_xlabel('Predicted', fontsize=10)
            ax.set_xticklabels(class_labels, fontsize=8, rotation=45, ha='right')
            ax.set_yticklabels(class_labels, fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with tabs[2]:
            st.write("**Feature Importance**")
            feature_importance = pd.DataFrame({
                'Feature': X_test.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)

            fig, ax = plt.subplots(figsize=(6, 4))  # Smaller size
            top_features = feature_importance.head(15)  # Show top 15 features
            sns.barplot(x='Importance', y='Feature', data=top_features, ax=ax, color='skyblue')
            ax.set_title('Top 15 Feature Importances', fontsize=12, fontweight='bold')
            ax.set_xlabel('Importance', fontsize=10)
            ax.set_ylabel('Feature', fontsize=10)
            ax.tick_params(axis='both', labelsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    else:
        # Panel-based layout
        st.write("**Model Evaluation Panels**")

        # Classification Report as a table
        st.write("**Classification Report**")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)

        # Create two columns for Confusion Matrix and Feature Importance
        col1, col2 = st.columns([1, 1])

        with col1:
            st.write("**Confusion Matrix**")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(6, 4))  # Smaller size
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        annot_kws={"size": 8}, cbar_kws={"shrink": 0.8})

            # Map numeric classes to labels
            reverse_mapping = {0: 'Dropout', 1: 'Graduate', 2: 'Enrolled'}
            class_labels = [reverse_mapping.get(c, f"Class {c}") for c in sorted(np.unique(y_test))]

            ax.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
            ax.set_ylabel('Actual', fontsize=10)
            ax.set_xlabel('Predicted', fontsize=10)
            ax.set_xticklabels(class_labels, fontsize=8, rotation=45, ha='right')
            ax.set_yticklabels(class_labels, fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.write("**Feature Importance**")
            feature_importance = pd.DataFrame({
                'Feature': X_test.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)

            fig, ax = plt.subplots(figsize=(6, 4))  # Smaller size
            top_features = feature_importance.head(15)  # Show top 15 features
            sns.barplot(x='Importance', y='Feature', data=top_features, ax=ax, color='skyblue')
            ax.set_title('Top 15 Feature Importances', fontsize=12, fontweight='bold')
            ax.set_xlabel('Importance', fontsize=10)
            ax.set_ylabel('Feature', fontsize=10)
            ax.tick_params(axis='both', labelsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()


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
        display_enhanced_dataframe_info(df)

    elif choice == "Exploratory Data Analysis":
        st.header("📊 Exploratory Data Analysis")

        # Create a copy of dataframe with better labels for visualization
        df_viz = df.copy()

        # Create label mappings for better visualization
        label_mappings = {
            'Gender': {0: 'Female', 1: 'Male'},
            'Scholarship holder': {0: 'No Scholarship', 1: 'Has Scholarship'},
            'Displaced': {0: 'Not Displaced', 1: 'Displaced'},
            'International': {0: 'Domestic', 1: 'International'},
            'Debtor': {0: 'No Debt', 1: 'Has Debt'},
            'Tuition fees up to date': {0: 'Fees Not Updated', 1: 'Fees Updated'},
            'Educational special needs': {0: 'No Special Needs', 1: 'Has Special Needs'},
            'Daytime/evening attendance': {0: 'Evening', 1: 'Daytime'}
        }

        # Apply label mappings to visualization dataframe
        for col, mapping in label_mappings.items():
            if col in df_viz.columns:
                df_viz[col] = df_viz[col].map(mapping).fillna(df_viz[col])

        # Create tabs for different visualizations
        tabs = st.tabs(
            ["📈 Distribution Analysis", "🔗 Correlation Analysis", "🎯 Outcome Analysis", "📊 Academic Performance"])

        with tabs[0]:
            st.subheader("Feature Distributions")

            # Select columns to visualize
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            # Remove target and id columns from selection
            numeric_cols = [col for col in numeric_cols if col not in ['Target', 'id']]

            selected_columns = st.multiselect(
                "Select features to visualize (up to 4 for better layout)",
                options=numeric_cols,
                default=['Age at enrollment', 'Admission grade', 'Curricular units 1st sem (grade)'][:3],
                max_selections=4
            )

            if selected_columns:
                # Create grid layout based on number of selected columns
                if len(selected_columns) == 1:
                    cols = st.columns(1)
                elif len(selected_columns) == 2:
                    cols = st.columns(2)
                elif len(selected_columns) == 3:
                    cols = st.columns(3)
                else:  # 4 columns
                    cols = st.columns(2)  # 2x2 grid

                for i, column in enumerate(selected_columns):
                    with cols[i % len(cols)]:
                        fig, ax = plt.subplots(figsize=(6, 4))
                        sns.histplot(df[column], kde=True, ax=ax, color='skyblue', alpha=0.7)
                        ax.set_title(f'{column}', fontsize=10, fontweight='bold')
                        ax.set_xlabel('')
                        ax.tick_params(axis='x', labelsize=8)
                        ax.tick_params(axis='y', labelsize=8)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
            else:
                st.info("Please select at least one column to visualize")

        with tabs[1]:
            st.subheader("Correlation Analysis")

            # Select columns for correlation
            correlation_features = st.multiselect(
                "Select features for correlation analysis",
                options=numeric_cols,
                default=['Age at enrollment', 'Admission grade', 'Curricular units 1st sem (grade)',
                         'Curricular units 2nd sem (grade)', 'Unemployment rate'][:4]
            )

            if len(correlation_features) > 1:
                col1, col2 = st.columns([2, 1])

                with col1:
                    correlation_matrix = df[correlation_features].corr()
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax,
                                center=0, square=True, fmt='.2f', cbar_kws={"shrink": .8})
                    ax.set_title('Feature Correlation Matrix', fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                with col2:
                    st.write("**Correlation Insights:**")
                    # Find strongest correlations
                    corr_pairs = []
                    for i in range(len(correlation_matrix.columns)):
                        for j in range(i + 1, len(correlation_matrix.columns)):
                            corr_value = correlation_matrix.iloc[i, j]
                            if abs(corr_value) > 0.5:  # Only show strong correlations
                                corr_pairs.append((
                                    correlation_matrix.columns[i],
                                    correlation_matrix.columns[j],
                                    corr_value
                                ))

                    if corr_pairs:
                        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                        for feat1, feat2, corr in corr_pairs[:5]:  # Show top 5
                            direction = "positive" if corr > 0 else "negative"
                            st.write(f"• **{feat1}** & **{feat2}**: {direction} ({corr:.2f})")
                    else:
                        st.write("No strong correlations (>0.5) found between selected features.")
            else:
                st.info("Please select at least two features for correlation analysis")

        with tabs[2]:
            st.subheader("Student Outcome Analysis")

            col1, col2 = st.columns([1, 1])

            with col1:
                # Target distribution
                fig, ax = plt.subplots(figsize=(6, 4))
                target_counts = df['Target'].value_counts()
                colors = ['#ff9999', '#66b3ff', '#99ff99']
                wedges, texts, autotexts = ax.pie(target_counts.values, labels=target_counts.index,
                                                  autopct='%1.1f%%', colors=colors, startangle=90)
                ax.set_title('Distribution of Student Outcomes', fontweight='bold')
                st.pyplot(fig)
                plt.close()

            with col2:
                # Outcome statistics
                st.write("**Outcome Statistics:**")
                total_students = len(df)
                for outcome in df['Target'].unique():
                    count = (df['Target'] == outcome).sum()
                    percentage = count / total_students * 100
                    st.write(f"• **{outcome}**: {count:,} students ({percentage:.1f}%)")

                st.write(f"\n**Total Students**: {total_students:,}")

            # Target distribution by key factors
            st.subheader("Outcome Analysis by Demographics")

            factor_cols = st.multiselect(
                "Select factors to analyze against outcomes",
                options=['Gender', 'Scholarship holder', 'Displaced', 'International'],
                default=['Gender', 'Scholarship holder']
            )

            if factor_cols:
                # Create a grid layout for factor analysis
                if len(factor_cols) <= 2:
                    cols = st.columns(len(factor_cols))
                else:
                    cols = st.columns(2)  # 2x2 grid for more factors

                for i, col in enumerate(factor_cols):
                    with cols[i % len(cols)]:
                        fig, ax = plt.subplots(figsize=(6, 4))

                        # Use the visualization dataframe with proper labels
                        cross_tab = pd.crosstab(df_viz[col], df_viz['Target'])
                        cross_tab_pct = cross_tab.div(cross_tab.sum(axis=1), axis=0) * 100

                        cross_tab_pct.plot(kind='bar', stacked=False, ax=ax,
                                           color=['#ff9999', '#66b3ff', '#99ff99'])
                        ax.set_title(f'Outcome Distribution by {col}', fontweight='bold', fontsize=10)
                        ax.set_ylabel('Percentage (%)', fontsize=8)
                        ax.set_xlabel('')
                        ax.legend(title='Outcome', fontsize=8, title_fontsize=8)
                        ax.tick_params(axis='x', labelsize=8, rotation=45)
                        ax.tick_params(axis='y', labelsize=8)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()

        with tabs[3]:
            st.subheader("Academic Performance Analysis")

            # Add user-friendly explanation
            st.info(
                "💡 **Understanding Academic Performance:** This section shows how different academic metrics relate to student success. Higher grades and more approved units typically indicate better chances of graduation.")

            # Academic performance metrics
            academic_cols = [col for col in df.columns if 'grade' in col.lower() or 'approved' in col.lower()]
            academic_cols = [col for col in academic_cols if col in numeric_cols]

            if academic_cols:
                # Simplified selection with better defaults
                default_metrics = []
                if 'Curricular units 1st sem (grade)' in academic_cols:
                    default_metrics.append('Curricular units 1st sem (grade)')
                if 'Curricular units 1st sem (approved)' in academic_cols:
                    default_metrics.append('Curricular units 1st sem (approved)')

                selected_academic = st.multiselect(
                    "Select academic performance metrics to analyze",
                    options=academic_cols,
                    default=default_metrics[:2],
                    help="Choose metrics to see how they relate to student outcomes"
                )

                if selected_academic:
                    for col in selected_academic:
                        # Create user-friendly column names
                        friendly_name = col.replace('Curricular units 1st sem (grade)', 'First Semester Grades') \
                            .replace('Curricular units 1st sem (approved)', 'First Semester Units Passed') \
                            .replace('Curricular units 2nd sem (grade)', 'Second Semester Grades') \
                            .replace('Curricular units 2nd sem (approved)', 'Second Semester Units Passed')

                        st.write(f"**{friendly_name} Analysis**")

                        col1, col2 = st.columns([2, 1])

                        with col1:
                            fig, ax = plt.subplots(figsize=(10, 5))

                            # Create a cleaner box plot with better styling
                            colors = ['lightcoral', 'lightgreen', 'lightskyblue']
                            box_plot = ax.boxplot(
                                [df[df['Target'] == outcome][col].dropna() for outcome in df['Target'].unique()],
                                labels=df['Target'].unique(),
                                patch_artist=True)

                            # Color the boxes
                            for patch, color in zip(box_plot['boxes'], colors):
                                patch.set_facecolor(color)
                                patch.set_alpha(0.7)

                            ax.set_title(f'{friendly_name} by Student Outcome', fontweight='bold', fontsize=12)
                            ax.set_xlabel('Student Outcome', fontweight='bold')
                            ax.set_ylabel(friendly_name, fontweight='bold')
                            ax.grid(True, alpha=0.3)
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()

                        with col2:
                            # Simplified summary statistics with interpretation
                            st.write("**Quick Summary:**")

                            summary_stats = df.groupby('Target')[col].agg(['mean', 'count']).round(2)

                            for outcome in summary_stats.index:
                                mean_val = summary_stats.loc[outcome, 'mean']
                                count_val = int(summary_stats.loc[outcome, 'count'])

                                # Add emoji indicators
                                if outcome == 'Graduate':
                                    emoji = "🎓"
                                elif outcome == 'Dropout':
                                    emoji = "⚠️"
                                else:
                                    emoji = "📚"

                                st.write(f"{emoji} **{outcome}:**")
                                st.write(f"   Average: {mean_val}")
                                st.write(f"   Students: {count_val}")
                                st.write("")

                            # Add insight
                            graduate_mean = summary_stats.loc[
                                'Graduate', 'mean'] if 'Graduate' in summary_stats.index else 0
                            dropout_mean = summary_stats.loc[
                                'Dropout', 'mean'] if 'Dropout' in summary_stats.index else 0

                            if graduate_mean > dropout_mean:
                                st.success("✅ Graduates perform better on this metric")
                            else:
                                st.warning("⚠️ Dropouts show similar/better performance")
            else:
                st.info("No academic performance columns found in the dataset.")

            # Simplified Performance comparison with better explanation
            st.subheader("📊 Semester-by-Semester Comparison")
            st.info(
                "💡 **What this shows:** How student performance changes from first to second semester, and how this relates to their final outcome.")

            if 'Curricular units 1st sem (grade)' in df.columns and 'Curricular units 2nd sem (grade)' in df.columns:
                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Grade Comparison: 1st vs 2nd Semester**")
                    st.caption("Each dot represents one student")

                    # Simplified scatter plot with better colors and less clutter
                    fig, ax = plt.subplots(figsize=(8, 6))

                    colors = {'Graduate': '#2E8B57', 'Dropout': '#DC143C', 'Enrolled': '#FF8C00'}
                    markers = {'Graduate': 'o', 'Dropout': 's', 'Enrolled': '^'}

                    for outcome in df['Target'].unique():
                        subset = df[df['Target'] == outcome]
                        ax.scatter(subset['Curricular units 1st sem (grade)'],
                                   subset['Curricular units 2nd sem (grade)'],
                                   c=colors.get(outcome, 'gray'),
                                   label=f'{outcome} ({len(subset)} students)',
                                   alpha=0.6, s=40, marker=markers.get(outcome, 'o'))

                    # Add diagonal line for reference
                    ax.plot([0, 20], [0, 20], 'k--', alpha=0.3, label='Same performance line')

                    ax.set_xlabel('1st Semester Grade', fontweight='bold')
                    ax.set_ylabel('2nd Semester Grade', fontweight='bold')
                    ax.set_title('Academic Performance: 1st vs 2nd Semester', fontweight='bold')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    ax.set_xlim(0, 20)
                    ax.set_ylim(0, 20)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                with col2:
                    st.write("**Performance Change Analysis**")
                    st.caption("Shows if students improved or declined from 1st to 2nd semester")

                    # Performance improvement analysis with clearer visualization
                    df_temp = df.copy()
                    df_temp['Performance_Change'] = (df_temp['Curricular units 2nd sem (grade)'] -
                                                     df_temp['Curricular units 1st sem (grade)'])

                    fig, ax = plt.subplots(figsize=(8, 6))

                    # Create violin plot for better distribution visualization
                    colors = ['lightcoral', 'lightgreen', 'lightskyblue']
                    outcomes = df_temp['Target'].unique()

                    violin_parts = ax.violinplot([df_temp[df_temp['Target'] == outcome]['Performance_Change'].dropna()
                                                  for outcome in outcomes], positions=range(len(outcomes)),
                                                 showmeans=True)

                    # Color the violin plots
                    for i, pc in enumerate(violin_parts['bodies']):
                        pc.set_facecolor(colors[i % len(colors)])
                        pc.set_alpha(0.7)

                    ax.set_xticks(range(len(outcomes)))
                    ax.set_xticklabels(outcomes)
                    ax.set_title('Grade Change from 1st to 2nd Semester', fontweight='bold')
                    ax.set_xlabel('Student Outcome', fontweight='bold')
                    ax.set_ylabel('Grade Change (2nd - 1st)', fontweight='bold')
                    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='No change')
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                # Add summary insights
                st.subheader("🔍 Key Insights")
                col1, col2, col3 = st.columns(3)

                with col1:
                    # Calculate improvement rate for graduates
                    graduates = df_temp[df_temp['Target'] == 'Graduate']
                    improved_graduates = (graduates['Performance_Change'] > 0).sum()
                    total_graduates = len(graduates)
                    improvement_rate = (improved_graduates / total_graduates * 100) if total_graduates > 0 else 0

                    st.metric(
                        label="Graduates Who Improved",
                        value=f"{improvement_rate:.1f}%",
                        help="Percentage of graduates who had better grades in 2nd semester"
                    )

                with col2:
                    # Calculate decline rate for dropouts
                    dropouts = df_temp[df_temp['Target'] == 'Dropout']
                    declined_dropouts = (dropouts['Performance_Change'] < 0).sum()
                    total_dropouts = len(dropouts)
                    decline_rate = (declined_dropouts / total_dropouts * 100) if total_dropouts > 0 else 0

                    st.metric(
                        label="Dropouts Who Declined",
                        value=f"{decline_rate:.1f}%",
                        help="Percentage of dropouts who had worse grades in 2nd semester"
                    )

                with col3:
                    # Average performance change
                    avg_change = df_temp['Performance_Change'].mean()
                    st.metric(
                        label="Average Grade Change",
                        value=f"{avg_change:+.2f}",
                        help="Overall average change from 1st to 2nd semester"
                    )

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
