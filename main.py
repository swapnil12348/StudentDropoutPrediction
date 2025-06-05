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

# Create a mapping dictionary for target labels
TARGET_MAPPING = {
    'Graduate': 1,
    'Dropout': 0,
    'Enrolled': 2
}

def display_data_quality(df):
    """Display data quality assessment with labeled charts and insights"""
    st.markdown("<h2 style='font-size: 20px; font-weight: bold;'>🔍 Data Quality Assessment</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("<h3 style='font-size: 16px;'>Missing Data by Column</h3>", unsafe_allow_html=True)
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        if len(missing_data) > 0:
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = missing_data.plot(kind='bar', ax=ax, color='coral')
            ax.set_title('Missing Data Across Columns', fontsize=14, fontweight='bold', pad=15)
            ax.set_ylabel('Number of Missing Values', fontsize=12)
            ax.set_xlabel('Dataset Columns', fontsize=12)
            ax.tick_params(axis='both', labelsize=10)
            plt.xticks(rotation=45, ha='right')
            for bar in bars.patches:
                height = bar.get_height()
                ax.annotate(f'{int(height)}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords='offset points',
                           ha='center', va='bottom', fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown("**Insight**: Columns with high missing values may require data imputation or removal to ensure reliable analysis.", unsafe_allow_html=True)
            plt.close()
        else:
            st.success("✅ No missing data found in the dataset!")
            st.markdown("**Insight**: The dataset is complete, ensuring robust analysis without the need for imputation.", unsafe_allow_html=True)

    with col2:
        st.markdown("<h3 style='font-size: 16px;'>Distribution of Data Types</h3>", unsafe_allow_html=True)
        dtype_counts = df.dtypes.value_counts()
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
        labels = [str(dtype) for dtype in dtype_counts.index]
        ax.pie(dtype_counts.values, labels=labels, autopct='%1.1f%%',
               colors=colors[:len(dtype_counts)], textprops={'fontsize': 10})
        ax.set_title('Proportion of Data Types in Dataset', fontsize=14, fontweight='bold', pad=15)
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown("**Insight**: The mix of data types (e.g., numeric, categorical) influences preprocessing needs for modeling.", unsafe_allow_html=True)
        plt.close()

def display_demographics(df):
    """Display student demographics insights with labeled charts and insights"""
    st.markdown("<h2 style='font-size: 20px; font-weight: bold;'>👥 Student Demographics Insights</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown("<h3 style='font-size: 16px;'>Age Distribution at Enrollment</h3>", unsafe_allow_html=True)
        if 'Age at enrollment' in df.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            df['Age at enrollment'].hist(bins=20, ax=ax, color='lightblue', alpha=0.7, edgecolor='black')
            mean_age = df['Age at enrollment'].mean()
            ax.axvline(mean_age, color='red', linestyle='--',
                      label=f'Mean Age: {mean_age:.1f}')
            ax.set_title('Distribution of Student Age at Enrollment', fontsize=14, fontweight='bold', pad=15)
            ax.set_xlabel('Age (Years)', fontsize=12)
            ax.set_ylabel('Number of Students', fontsize=12)
            ax.tick_params(axis='both', labelsize=10)
            ax.legend(fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown(f"**Insight**: Most students enroll around age {int(mean_age)}, with older students potentially facing higher dropout risks.", unsafe_allow_html=True)
            plt.close()

    with col2:
        st.markdown("<h3 style='font-size: 16px;'>Gender Distribution</h3>", unsafe_allow_html=True)
        if 'Gender' in df.columns:
            gender_counts = df['Gender'].value_counts()
            gender_labels = ['Female' if x == 0 else 'Male' for x in gender_counts.index]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.pie(gender_counts.values, labels=gender_labels, autopct='%1.1f%%',
                  colors=['pink', 'lightblue'], textprops={'fontsize': 10})
            ax.set_title('Gender Distribution of Students', fontsize=14, fontweight='bold', pad=15)
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown("**Insight**: Gender balance affects dropout rates, with potential differences in academic support needs.", unsafe_allow_html=True)
            plt.close()

    with col3:
        st.markdown("<h3 style='font-size: 16px;'>Scholarship Status</h3>", unsafe_allow_html=True)
        if 'Scholarship holder' in df.columns:
            scholarship_counts = df['Scholarship holder'].value_counts()
            scholarship_labels = ['No Scholarship' if x == 0 else 'Scholarship' for x in scholarship_counts.index]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.pie(scholarship_counts.values, labels=scholarship_labels, autopct='%1.1f%%',
                  colors=['lightcoral', 'lightgreen'], textprops={'fontsize': 10})
            ax.set_title('Proportion of Scholarship Holders', fontsize=14, fontweight='bold', pad=15)
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown("**Insight**: Students without scholarships may face financial stress, increasing dropout likelihood.", unsafe_allow_html=True)
            plt.close()

def display_academic_performance(df):
    """Display academic performance overview with labeled charts and insights"""
    st.markdown("<h2 style='font-size: 20px; font-weight: bold;'>📚 Academic Performance Overview</h2>", unsafe_allow_html=True)
    st.info("💡 **What this shows:** How well students performed academically and how it relates to their final outcome")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("<h3 style='font-size: 16px;'>Admission Grades by Outcome</h3>", unsafe_allow_html=True)
        if 'Admission grade' in df.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = {'Graduate': 'green', 'Dropout': 'red', 'Enrolled': 'orange'}
            for outcome in df['Target'].unique():
                subset = df[df['Target'] == outcome]['Admission grade']
                ax.hist(subset, alpha=0.6, label=f'{outcome} ({len(subset)})',
                       bins=15, color=colors.get(outcome, 'gray'))
                mean_val = subset.mean()
                ax.axvline(mean_val, color=colors.get(outcome, 'gray'), linestyle='--',
                          label=f'{outcome} Mean: {mean_val:.1f}')
            ax.set_title('Admission Grades by Student Outcome', fontsize=14, fontweight='bold', pad=15)
            ax.set_xlabel('Admission Grade (0-200)', fontsize=12)
            ax.set_ylabel('Number of Students', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', labelsize=10)
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown("**Insight**: Higher admission grades are strongly associated with graduation, while lower grades correlate with dropouts.", unsafe_allow_html=True)
            plt.close()

    with col2:
        st.markdown("<h3 style='font-size: 16px;'>First Semester Grades by Outcome</h3>", unsafe_allow_html=True)
        if 'Curricular units 1st sem (grade)' in df.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = {'Graduate': 'green', 'Dropout': 'red', 'Enrolled': 'orange'}
            for outcome in df['Target'].unique():
                subset = df[df['Target'] == outcome]['Curricular units 1st sem (grade)']
                ax.hist(subset, alpha=0.6, label=f'{outcome} ({len(subset)})',
                       bins=15, color=colors.get(outcome, 'gray'))
                mean_val = subset.mean()
                ax.axvline(mean_val, color=colors.get(outcome, 'gray'), linestyle='--',
                          label=f'{outcome} Mean: {mean_val:.1f}')
            ax.set_title('First Semester Grades by Student Outcome', fontsize=14, fontweight='bold', pad=15)
            ax.set_xlabel('Average Grade (0-20)', fontsize=12)
            ax.set_ylabel('Number of Students', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', labelsize=10)
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown("**Insight**: Strong first-semester performance is a key predictor of graduation success.", unsafe_allow_html=True)
            plt.close()

def display_performance_summary(df):
    """Display performance summary metrics"""
    st.markdown("<h2 style='font-size: 20px; font-weight: bold;'>📈 Performance Summary</h2>", unsafe_allow_html=True)
    if 'Admission grade' in df.columns and 'Curricular units 1st sem (grade)' in df.columns:
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
        st.markdown("**Insight**: Graduates typically have higher admission and first-semester grades compared to dropouts, indicating academic performance as a key success factor.", unsafe_allow_html=True)

def display_risk_factors(df):
    """Display key risk factors analysis with labeled charts and insights"""
    st.markdown("<h2 style='font-size: 20px; font-weight: bold;'>⚠️ Key Risk Factors Analysis</h2>", unsafe_allow_html=True)
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
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(risk_df['Risk Factor'], risk_df['Number of Students'],
                     color=['red', 'orange', 'yellow', 'coral'][:len(risk_factors)])
        ax.set_title('Number of Students by Risk Factor', fontsize=14, fontweight='bold', pad=15)
        ax.set_ylabel('Number of Students', fontsize=12)
        ax.set_xlabel('Risk Factors', fontsize=12)
        ax.tick_params(axis='both', labelsize=10)
        plt.xticks(rotation=45, ha='right')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown("**Insight**: Risk factors like low academic performance and lack of financial aid significantly increase dropout probability.", unsafe_allow_html=True)
        plt.close()

def display_data_structure(df):
    """Display data sample and structure"""
    st.markdown("<h2 style='font-size: 20px; font-weight: bold;'>📋 Data Sample & Structure</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("<h3 style='font-size: 16px;'>Sample of Student Records</h3>", unsafe_allow_html=True)
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
        st.markdown("**Insight**: This sample shows key student attributes, with outcomes highlighted for quick reference.", unsafe_allow_html=True)
    with col2:
        st.markdown("<h3 style='font-size: 16px;'>Dataset Composition</h3>", unsafe_allow_html=True)
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
        st.markdown("**Insight**: The dataset includes a mix of demographic, academic, and economic features critical for predicting dropout risk.", unsafe_allow_html=True)

def display_feature_explorer(df):
    """Interactive feature explorer with labeled insights"""
    st.markdown("<h2 style='font-size: 20px; font-weight: bold;'>🔧 Interactive Feature Explorer</h2>", unsafe_allow_html=True)
    col_to_explore = st.selectbox(
        "Select a feature to explore in detail:",
        options=[col for col in df.columns if col != 'Target' and col != 'id']
    )
    if col_to_explore:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(f"<h3 style='font-size: 16px;'>Statistics for {col_to_explore}</h3>", unsafe_allow_html=True)
            if df[col_to_explore].dtype in ['int64', 'float64']:
                stats = df[col_to_explore].describe()
                for stat, value in stats.items():
                    st.write(f"• **{stat}:** {value:.2f}")
                st.markdown(f"**Insight**: The distribution of {col_to_explore} shows its variability and central tendency.", unsafe_allow_html=True)
            else:
                unique_vals = df[col_to_explore].nunique()
                st.write(f"• **Unique values:** {unique_vals}")
                st.write(f"• **Most common:** {df[col_to_explore].mode().iloc[0]}")
                st.markdown(f"**Insight**: Categorical feature with {unique_vals} unique values, indicating diversity in {col_to_explore}.", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<h3 style='font-size: 16px;'>{col_to_explore} vs Dropout Rate</h3>", unsafe_allow_html=True)
            if df[col_to_explore].dtype in ['int64', 'float64']:
                correlation = df[col_to_explore].corr(df['Target'].map({'Graduate': 1, 'Dropout': 0, 'Enrolled': 2}))
                st.write(f"• **Correlation with dropout:** {correlation:.3f}")
                df_temp = df.copy()
                try:
                    df_temp[f'{col_to_explore}_binned'] = pd.cut(df_temp[col_to_explore], bins=5)
                    dropout_by_bin = df_temp.groupby(f'{col_to_explore}_binned', observed=True)['Target'].apply(
                        lambda x: (x == 'Dropout').mean() * 100 if len(x) > 0 else 0
                    ).round(1)
                    fig, ax = plt.subplots(figsize=(8, 5))
                    dropout_by_bin.plot(kind='bar', ax=ax, color='coral')
                    ax.set_title(f'Dropout Rate by {col_to_explore} Range', fontsize=14, fontweight='bold', pad=15)
                    ax.set_xlabel(f'{col_to_explore} Bins', fontsize=12)
                    ax.set_ylabel('Dropout Rate (%)', fontsize=12)
                    ax.tick_params(axis='both', labelsize=10)
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig)
                    st.markdown(f"**Insight**: Variations in {col_to_explore} show differing dropout risks across its range.", unsafe_allow_html=True)
                    plt.close()
                except Exception as e:
                    st.write(f"• Unable to create bins for this feature: {str(e)}")
            else:
                dropout_by_cat = df.groupby(col_to_explore, observed=True)['Target'].apply(
                    lambda x: (x == 'Dropout').mean() * 100 if len(x) > 0 else 0
                ).round(1)
                fig, ax = plt.subplots(figsize=(8, 5))
                dropout_by_cat.plot(kind='bar', ax=ax, color='coral')
                ax.set_title(f'Dropout Rate by {col_to_explore}', fontsize=14, fontweight='bold', pad=15)
                ax.set_xlabel(col_to_explore, fontsize=12)
                ax.set_ylabel('Dropout Rate (%)', fontsize=12)
                ax.tick_params(axis='both', labelsize=10)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
                st.markdown(f"**Insight**: Different categories of {col_to_explore} have varying dropout rates, highlighting its impact.", unsafe_allow_html=True)
                plt.close()

def display_column_info(df):
    """Display detailed column information"""
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
        st.markdown("**Insight**: Understanding column types and missing data helps prioritize features for modeling.", unsafe_allow_html=True)

def load_data(uploaded_file=None):
    """Load data from file or sample dataset"""
    potential_paths = [
        'C:/Users/swapn/StudentDropoutPrediction/.venv/data/student_dropout_data.csv',
        'C:/Users/swapn/StudentDropoutPrediction/data/student_dropout_data.csv',
        './data/student_dropout_data.csv',
        '../data/student_dropout_data.csv'
    ]

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("Successfully loaded uploaded file!")
            return df
        except Exception as e:
            st.error(f"Error loading uploaded file: {e}")

    for path in potential_paths:
        try:
            df = pd.read_csv(path)
            st.success(f"Successfully loaded data from {path}")
            return df
        except FileNotFoundError:
            continue
        except Exception as e:
            st.error(f"Error loading file from {path}: {e}")

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
    processed_df = df.copy()
    if 'id' in processed_df.columns:
        processed_df = processed_df.drop('id', axis=1)
    processed_df = processed_df.fillna(processed_df.median(numeric_only=True))
    le = LabelEncoder()
    categorical_cols = processed_df.select_dtypes(include=['object']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col != 'Target']
    for col in categorical_cols:
        processed_df[col] = le.fit_transform(processed_df[col])
    if 'Target' in processed_df.columns:
        processed_df['Target'] = processed_df['Target'].map(TARGET_MAPPING)
        processed_df['Target'] = processed_df['Target'].fillna(0)
    for col in processed_df.columns:
        processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
    processed_df.dropna(inplace=True)
    X = processed_df.drop('Target', axis=1)
    y = processed_df['Target']
    return X, y, processed_df

def train_model(X_train, y_train):
    """Train a Random Forest Classifier"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Starting model training...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    status_text.text("Fitting the model...")
    progress_bar.progress(25)
    model.fit(X_train, y_train)
    progress_bar.progress(100)
    status_text.text("Model training complete!")
    return model

def individual_dropout_prediction(model, X):
    """Create interactive widgets for individual dropout prediction"""
    st.markdown("<h2 style='font-size: 20px; font-weight: bold;'>Student Dropout Probability Predictor</h2>", unsafe_allow_html=True)
    show_debug = st.checkbox("Show debug information")
    if show_debug:
        st.markdown("<h3 style='font-size: 16px;'>Model Information</h3>", unsafe_allow_html=True)
        st.write("Model Classes:", model.classes_)
        reverse_mapping = {v: k for k, v in TARGET_MAPPING.items()}
        st.write("Class mapping:", {f"Class {k}": reverse_mapping.get(k, f"Unknown-{k}") for k in model.classes_})
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<h3 style='font-size: 16px;'>Demographics</h3>", unsafe_allow_html=True)
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.slider("Age at enrollment", min_value=17, max_value=70, value=20)
        marital_status = st.selectbox("Marital status", [1, 2, 3, 4, 5, 6])
        international = st.selectbox("International", [0, 1])
        displaced = st.selectbox("Displaced", [0, 1])
        educational_needs = st.selectbox("Educational special needs", [0, 1])
        scholarship = st.selectbox("Scholarship holder", [0, 1])
    with col2:
        st.markdown("<h3 style='font-size: 16px;'>Academic Background</h3>", unsafe_allow_html=True)
        application_mode = st.selectbox("Application mode", list(range(1, 18)))
        application_order = st.slider("Application order", min_value=1, max_value=9, value=1)
        course = st.number_input("Course Code", min_value=9000, max_value=9999, value=9238)
        attendance = st.selectbox("Daytime/evening attendance", [0, 1])
        prev_qualification = st.selectbox("Previous qualification", list(range(1, 18)))
        prev_grade = st.slider("Previous qualification grade", min_value=0.0, max_value=200.0, value=120.0, step=5.0)
        admission_grade = st.slider("Admission grade", min_value=0.0, max_value=200.0, value=120.0, step=5.0)
    with col3:
        st.markdown("<h3 style='font-size: 16px;'>Academic Performance</h3>", unsafe_allow_html=True)
        units_1st_approved = st.slider("Units approved (1st sem)", min_value=0, max_value=20, value=5)
        units_1st_grade = st.slider("Average grade (1st sem)", min_value=0.0, max_value=20.0, value=12.0, step=1.0)
        units_2nd_approved = st.slider("Units approved (2nd sem)", min_value=0, max_value=20, value=5)
        units_2nd_grade = st.slider("Average grade (2nd sem)", min_value=0.0, max_value=20.0, value=12.0, step=1.0)
    st.markdown("<h3 style='font-size: 16px;'>Economic Indicators</h3>", unsafe_allow_html=True)
    eco_col1, eco_col2, eco_col3 = st.columns(3)
    with eco_col1:
        unemployment = st.slider("Unemployment rate", min_value=0.0, max_value=20.0, value=11.0, step=1.0)
    with eco_col2:
        inflation = st.slider("Inflation rate", min_value=-2.0, max_value=5.0, value=0.6, step=0.5)
    with eco_col3:
        gdp = st.slider("GDP", min_value=-5.0, max_value=5.0, value=2.0, step=0.5)
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
    show_advanced = st.checkbox("Show advanced options")
    if show_advanced:
        st.markdown("<h3 style='font-size: 16px;'>Advanced Features</h3>", unsafe_allow_html=True)
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
    if st.button("Predict Dropout Probability"):
        with st.spinner('Calculating prediction...'):
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
            for key, value in default_values.items():
                input_dict[key] = value
            input_data = pd.DataFrame([input_dict])
            if show_debug:
                st.markdown("<h3 style='font-size: 16px;'>Debug Information</h3>", unsafe_allow_html=True)
                st.write("Raw Input Data:")
                st.write(input_data)
            try:
                for col in X.columns:
                    if col not in input_data.columns:
                        input_data[col] = X[col].median()
                input_data = input_data[X.columns]
                prediction_probs = model.predict_proba(input_data)[0]
                reverse_mapping = {v: k for k, v in TARGET_MAPPING.items()}
                class_names = [reverse_mapping.get(i, f"Class {i}") for i in model.classes_]
                results = pd.DataFrame({
                    'Outcome': class_names,
                    'Probability': [f"{p * 100:.2f}%" for p in prediction_probs],
                    'Raw Value': prediction_probs
                })
                st.markdown("<h3 style='font-size: 16px;'>Prediction Results</h3>", unsafe_allow_html=True)
                st.dataframe(results)
                for outcome, prob in zip(class_names, prediction_probs):
                    st.write(f"{outcome}: {prob:.2%}")
                    st.progress(float(prob))
                max_class_idx = np.argmax(prediction_probs)
                max_class_name = class_names[max_class_idx]
                max_prob = prediction_probs[max_class_idx]
                st.markdown("<h3 style='font-size: 16px;'>Prediction Summary</h3>", unsafe_allow_html=True)
                if max_class_name == "Dropout":
                    st.warning(f"⚠️ Risk of student dropout detected! Probability: {max_prob:.2%}")
                    st.write("Recommended actions:")
                    st.write("- Schedule academic counseling")
                    st.write("- Review financial aid options")
                    st.write("- Consider tutoring or additional support")
                else:
                    st.success(f"✅ Student likely to succeed! Probability of {max_class_name}: {max_prob:.2%}")
                    st.write("Recommended actions:")
                    st.write("- Continue regular monitoring")
                    st.write("- Provide ongoing encouragement and support")
                st.markdown(f"**Insight**: The model predicts a {max_prob:.2%} chance of {max_class_name}, guiding targeted interventions.", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                if show_debug:
                    import traceback
                    st.code(traceback.format_exc())

def visualize_model_results(model, X_test, y_test):
    """Visualize model evaluation results with labeled charts and insights"""
    st.markdown("<h2 style='font-size: 20px; font-weight: bold;'>Model Evaluation Results</h2>", unsafe_allow_html=True)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.metric("Model Accuracy", f"{accuracy:.2%}")
    st.markdown(f"**Insight**: The model achieves {accuracy:.2%} accuracy in predicting student outcomes.", unsafe_allow_html=True)
    display_mode = st.radio("Select display mode:", ["Tabs", "Panels"], index=1)
    if display_mode == "Tabs":
        tabs = st.tabs(["Classification Report", "Confusion Matrix", "Feature Importance"])
        with tabs[0]:
            st.markdown("<h3 style='font-size: 16px;'>Classification Report</h3>", unsafe_allow_html=True)
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
            st.markdown("**Insight**: Precision, recall, and F1-scores show the model's performance across Graduate, Dropout, and Enrolled classes.", unsafe_allow_html=True)
        with tabs[1]:
            st.markdown("<h3 style='font-size: 16px;'>Confusion Matrix</h3>", unsafe_allow_html=True)
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       annot_kws={"size": 10}, cbar_kws={"shrink": 0.8})
            reverse_mapping = {0: 'Dropout', 1: 'Graduate', 2: 'Enrolled'}
            class_labels = [reverse_mapping.get(c, f"Class {c}") for c in sorted(np.unique(y_test))]
            ax.set_title('Confusion Matrix of Model Predictions', fontsize=14, fontweight='bold', pad=15)
            ax.set_ylabel('Actual Outcome', fontsize=12)
            ax.set_xlabel('Predicted Outcome', fontsize=12)
            ax.set_xticklabels(class_labels, fontsize=10, rotation=45, ha='right')
            ax.set_yticklabels(class_labels, fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown("**Insight**: The matrix shows correct and incorrect predictions, with higher diagonal values indicating better performance.", unsafe_allow_html=True)
            plt.close()
        with tabs[2]:
            st.markdown("<h3 style='font-size: 16px;'>Feature Importance</h3>", unsafe_allow_html=True)
            feature_importance = pd.DataFrame({
                'Feature': X_test.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            fig, ax = plt.subplots(figsize=(8, 5))
            top_features = feature_importance.head(15)
            sns.barplot(x='Importance', y='Feature', data=top_features, ax=ax, color='skyblue')
            ax.set_title('Top 15 Features Influencing Dropout Prediction', fontsize=14, fontweight='bold', pad=15)
            ax.set_xlabel('Feature Importance Score', fontsize=12)
            ax.set_ylabel('Features', fontsize=12)
            ax.tick_params(axis='both', labelsize=10)
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown("**Insight**: Features like first-semester grades and units approved are critical drivers of the model's predictions.", unsafe_allow_html=True)
            plt.close()
    else:
        st.markdown("<h3 style='font-size: 16px;'>Classification Report</h3>", unsafe_allow_html=True)
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)
        st.markdown("**Insight**: Precision, recall, and F1-scores show the model's performance across Graduate, Dropout, and Enrolled classes.", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("<h3 style='font-size: 16px;'>Confusion Matrix</h3>", unsafe_allow_html=True)
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       annot_kws={"size": 10}, cbar_kws={"shrink": 0.8})
            reverse_mapping = {0: 'Dropout', 1: 'Graduate', 2: 'Enrolled'}
            class_labels = [reverse_mapping.get(c, f"Class {c}") for c in sorted(np.unique(y_test))]
            ax.set_title('Confusion Matrix of Model Predictions', fontsize=14, fontweight='bold', pad=15)
            ax.set_ylabel('Actual Outcome', fontsize=12)
            ax.set_xlabel('Predicted Outcome', fontsize=12)
            ax.set_xticklabels(class_labels, fontsize=10, rotation=45, ha='right')
            ax.set_yticklabels(class_labels, fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown("**Insight**: The matrix shows correct and incorrect predictions, with higher diagonal values indicating better performance.", unsafe_allow_html=True)
            plt.close()
        with col2:
            st.markdown("<h3 style='font-size: 16px;'>Feature Importance</h3>", unsafe_allow_html=True)
            feature_importance = pd.DataFrame({
                'Feature': X_test.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            fig, ax = plt.subplots(figsize=(8, 5))
            top_features = feature_importance.head(15)
            sns.barplot(x='Importance', y='Feature', data=top_features, ax=ax, color='skyblue')
            ax.set_title('Top 15 Features Influencing Dropout Prediction', fontsize=14, fontweight='bold', pad=15)
            ax.set_xlabel('Feature Importance Score', fontsize=12)
            ax.set_ylabel('Features', fontsize=12)
            ax.tick_params(axis='both', labelsize=10)
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown("**Insight**: Features like first-semester grades and units approved are critical drivers of the model's predictions.", unsafe_allow_html=True)
            plt.close()

def main():
    st.markdown("<div style='font-size: 24px; font-weight: bold;'>🎓 Student Dropout Prediction Dashboard</div>", unsafe_allow_html=True)
    st.markdown("""
    This interactive dashboard helps predict and analyze factors contributing to student dropout. 
    Navigate through the sections below to explore data, analyze trends, train models, and predict outcomes.
    """)
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'show_prediction' not in st.session_state:
        st.session_state.show_prediction = False
    st.sidebar.markdown("<div style='font-size: 20px; font-weight: bold;'>Navigation</div>", unsafe_allow_html=True)
    menu = ["Data Overview", "Exploratory Data Analysis", "Model Training & Evaluation", "Dropout Prediction"]
    choice = st.sidebar.radio("Select Module", menu, label_visibility="collapsed")
    st.sidebar.markdown("<div style='font-size: 16px;'>Data Input</div>", unsafe_allow_html=True)
    uploaded_file = st.sidebar.file_uploader("Upload your Student Dropout CSV", type=['csv'])
    df = load_data(uploaded_file)
    X, y, processed_df = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    st.sidebar.markdown("---")
    if st.sidebar.button("Back to Top"):
        st.markdown("<script>window.scrollTo(0, 0);</script>", unsafe_allow_html=True)
    if choice == "Data Overview":
        st.markdown("<div style='font-size: 20px; font-weight: bold;'>📊 Data Overview</div>", unsafe_allow_html=True)
        st.markdown("Explore the dataset structure and key statistics.")
        sub_menu = st.selectbox("Select Analysis", [
            "Data Quality Assessment",
            "Demographics Insights",
            "Academic Performance",
            "Performance Summary",
            "Risk Factors Analysis",
            "Data Structure",
            "Feature Explorer",
            "Column Information"
        ], key="data_overview_submenu")
        with st.expander(f"{sub_menu}", expanded=True):
            if sub_menu == "Data Quality Assessment":
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
                display_data_quality(df)
            elif sub_menu == "Demographics Insights":
                display_demographics(df)
            elif sub_menu == "Academic Performance":
                display_academic_performance(df)
            elif sub_menu == "Performance Summary":
                display_performance_summary(df)
            elif sub_menu == "Risk Factors Analysis":
                display_risk_factors(df)
            elif sub_menu == "Data Structure":
                display_data_structure(df)
            elif sub_menu == "Feature Explorer":
                display_feature_explorer(df)
            elif sub_menu == "Column Information":
                display_column_info(df)
    elif choice == "Exploratory Data Analysis":
        st.markdown("<div style='font-size: 20px; font-weight: bold;'>📈 Exploratory Data Analysis</div>", unsafe_allow_html=True)
        st.markdown("Analyze data distributions, correlations, and relationships with student outcomes.")
        sub_menu = st.selectbox("Select Analysis", [
            "Distribution Analysis",
            "Correlation Analysis",
            "Outcome Analysis",
            "Academic Performance"
        ], key="eda_submenu")
        df_viz = df.copy()
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
        for col, mapping in label_mappings.items():
            if col in df_viz.columns:
                df_viz[col] = df_viz[col].map(mapping).fillna(df_viz[col])
        with st.expander(f"{sub_menu}", expanded=True):
            if sub_menu == "Distribution Analysis":
                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
                numeric_cols = [col for col in numeric_cols if col not in ['Target', 'id']]
                selected_columns = st.multiselect(
                    "Select features to visualize (up to 4)",
                    options=numeric_cols,
                    default=['Age at enrollment', 'Admission grade', 'Curricular units 1st sem (grade)'][:3],
                    max_selections=4
                )
                if selected_columns:
                    cols = st.columns(min(len(selected_columns), 2))
                    for i, column in enumerate(selected_columns):
                        with cols[i % len(cols)]:
                            fig, ax = plt.subplots(figsize=(8, 5))
                            sns.histplot(df[column], kde=True, ax=ax, color='skyblue', alpha=0.7)
                            ax.set_title(f'Distribution of {column}', fontsize=14, fontweight='bold', pad=15)
                            ax.set_xlabel(column, fontsize=12)
                            ax.set_ylabel('Number of Students', fontsize=12)
                            ax.tick_params(axis='both', labelsize=10)
                            plt.tight_layout()
                            st.pyplot(fig)
                            st.markdown(f"**Insight**: The distribution of {column} shows its range and central tendency, impacting dropout risk.", unsafe_allow_html=True)
                            plt.close()
                else:
                    st.info("Select at least one column to visualize.")
            elif sub_menu == "Correlation Analysis":
                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
                correlation_features = st.multiselect(
                    "Select features for correlation analysis",
                    options=numeric_cols,
                    default=['Age at enrollment', 'Admission grade', 'Curricular units 1st sem (grade)',
                            'Curricular units 2nd sem (grade)'][:4]
                )
                if len(correlation_features) > 1:
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        correlation_matrix = df[correlation_features].corr()
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax, center=0, fmt='.2f',
                                   annot_kws={'size': 10})
                        ax.set_title('Correlation Matrix of Selected Features', fontsize=14, fontweight='bold', pad=15)
                        ax.tick_params(axis='both', labelsize=10)
                        plt.tight_layout()
                        st.pyplot(fig)
                        st.markdown("**Insight**: Strong correlations indicate features that move together, influencing dropout prediction.", unsafe_allow_html=True)
                        plt.close()
                    with col2:
                        st.markdown("<h3 style='font-size: 16px;'>Correlation Insights</h3>", unsafe_allow_html=True)
                        corr_pairs = []
                        for i in range(len(correlation_matrix.columns)):
                            for j in range(i + 1, len(correlation_matrix.columns)):
                                corr_value = correlation_matrix.iloc[i, j]
                                if abs(corr_value) > 0.5:
                                    corr_pairs.append(
                                        (correlation_matrix.columns[i], correlation_matrix.columns[j], corr_value))
                        if corr_pairs:
                            corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                            for feat1, feat2, corr in corr_pairs[:5]:
                                direction = "positive" if corr > 0 else "negative"
                                st.write(f"• **{feat1}** & **{feat2}**: {direction} ({corr:.2f})")
                            st.markdown("**Insight**: High correlations highlight key feature relationships affecting student outcomes.", unsafe_allow_html=True)
                        else:
                            st.write("No strong correlations (>0.5) found.")
                            st.markdown("**Insight**: Weak correlations suggest features are independent, requiring diverse predictors.", unsafe_allow_html=True)
                else:
                    st.info("Select at least two features for correlation analysis.")
            elif sub_menu == "Outcome Analysis":
                col1, col2 = st.columns([1, 1])
                with col1:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    target_counts = df['Target'].value_counts()
                    ax.pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%',
                          colors=['#ff9999', '#66b3ff', '#99ff99'], textprops={'fontsize': 10})
                    ax.set_title('Distribution of Student Outcomes', fontsize=14, fontweight='bold', pad=15)
                    plt.tight_layout()
                    st.pyplot(fig)
                    st.markdown("**Insight**: The proportion of Graduates, Dropouts, and Enrolled students shows the dataset's outcome balance.", unsafe_allow_html=True)
                    plt.close()
                with col2:
                    st.markdown("<h3 style='font-size: 16px;'>Outcome Statistics</h3>", unsafe_allow_html=True)
                    total_students = len(df)
                    for outcome in df['Target'].unique():
                        count = (df['Target'] == outcome).sum()
                        percentage = count / total_students * 100
                        st.write(f"• **{outcome}**: {count:,} students ({percentage:.1f}%)")
                    st.markdown("**Insight**: Dropout rates indicate the scale of the challenge in improving student retention.", unsafe_allow_html=True)
                factor_cols = st.multiselect(
                    "Select factors to analyze against outcomes",
                    options=['Gender', 'Scholarship holder', 'Displaced', 'International'],
                    default=['Gender', 'Scholarship holder']
                )
                if factor_cols:
                    cols = st.columns(min(len(factor_cols), 2))
                    for i, col in enumerate(factor_cols):
                        with cols[i % len(cols)]:
                            fig, ax = plt.subplots(figsize=(8, 5))
                            cross_tab = pd.crosstab(df_viz[col], df_viz['Target'])
                            cross_tab_pct = cross_tab.div(cross_tab.sum(axis=1), axis=0) * 100
                            cross_tab_pct.plot(kind='bar', ax=ax, color=['#ff9999', '#66b3ff', '#99ff99'])
                            ax.set_title(f'Outcome Distribution by {col}', fontsize=14, fontweight='bold', pad=15)
                            ax.set_xlabel(col, fontsize=12)
                            ax.set_ylabel('Percentage of Students', fontsize=12)
                            ax.tick_params(axis='both', labelsize=10)
                            ax.legend(title='Outcome', fontsize=10)
                            plt.tight_layout()
                            st.pyplot(fig)
                            st.markdown(f"**Insight**: {col} influences student outcomes, with certain categories linked to higher dropout rates.", unsafe_allow_html=True)
                            plt.close()
            elif sub_menu == "Academic Performance":
                academic_cols = [col for col in df.columns if 'grade' in col.lower() or 'approved' in col.lower()]
                selected_academic = st.multiselect(
                    "Select academic performance metrics",
                    options=academic_cols,
                    default=['Curricular units 1st sem (grade)', 'Curricular units 1st sem (approved)'][:2]
                )
                if selected_academic:
                    for col in selected_academic:
                        friendly_name = col.replace('Curricular units 1st sem (grade)', 'First Semester Grades') \
                            .replace('Curricular units 1st sem (approved)', 'First Semester Units Passed') \
                            .replace('Curricular units 2nd sem (grade)', 'Second Semester Grades') \
                            .replace('Curricular units 2nd sem (approved)', 'Second Semester Units Passed')
                        st.markdown(f"<h3 style='font-size: 16px;'>{friendly_name} Analysis</h3>", unsafe_allow_html=True)
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            fig, ax = plt.subplots(figsize=(10, 5))
                            colors = ['lightcoral', 'lightgreen', 'lightskyblue']
                            box_plot = ax.boxplot(
                                [df[df['Target'] == outcome][col].dropna() for outcome in df['Target'].unique()],
                                labels=df['Target'].unique(),
                                patch_artist=True)
                            for patch, color in zip(box_plot['boxes'], colors):
                                patch.set_facecolor(color)
                                patch.set_alpha(0.7)
                            ax.set_title(f'{friendly_name} by Student Outcome', fontsize=14, fontweight='bold', pad=15)
                            ax.set_xlabel('Student Outcome', fontsize=12)
                            ax.set_ylabel(friendly_name, fontsize=12)
                            ax.tick_params(axis='both', labelsize=10)
                            plt.tight_layout()
                            st.pyplot(fig)
                            st.markdown(f"**Insight**: Higher {friendly_name.lower()} are associated with graduates, while lower values correlate with dropouts.", unsafe_allow_html=True)
                            plt.close()
                        with col2:
                            st.markdown("<h3 style='font-size: 16px;'>Quick Summary</h3>", unsafe_allow_html=True)
                            summary_stats = df.groupby('Target')[col].agg(['mean', 'count']).round(2)
                            for outcome in summary_stats.index:
                                mean_val = summary_stats.loc[outcome, 'mean']
                                count_val = int(summary_stats.loc[outcome, 'count'])
                                emoji = "🎓" if outcome == 'Graduate' else "⚠️" if outcome == 'Dropout' else "📚"
                                st.write(f"{emoji} **{outcome}:** Average: {mean_val}, Students: {count_val}")
                            st.markdown(f"**Insight**: {friendly_name} significantly differentiates outcomes, guiding intervention strategies.", unsafe_allow_html=True)
    elif choice == "Model Training & Evaluation":
        st.markdown("<div style='font-size: 20px; font-weight: bold;'>🤖 Model Training & Evaluation</div>", unsafe_allow_html=True)
        st.markdown("Train a machine learning model and evaluate its performance.")
        sub_menu = st.selectbox("Select Action", [
            "Train Model",
            "View Results"
        ], key="model_submenu")
        if sub_menu == "Train Model":
            with st.expander("Model Training", expanded=True):
                target_mapping_df = pd.DataFrame({
                    'Original': list(TARGET_MAPPING.keys()),
                    'Encoded': list(TARGET_MAPPING.values())
                })
                st.markdown("<h3 style='font-size: 16px;'>Target Label Encoding</h3>", unsafe_allow_html=True)
                st.dataframe(target_mapping_df)
                st.write("Target values found:", processed_df['Target'].unique().tolist())
                st.markdown("**Insight**: Encoded labels ensure the model can process categorical outcomes effectively.", unsafe_allow_html=True)
                if st.button("Start Training"):
                    with st.spinner("Training model..."):
                        model = train_model(X_train, y_train)
                        st.session_state.model = model
                        st.session_state.model_trained = True
                        st.success("Model trained successfully!")
                        st.button("View Results", key="go_to_results")
        else:
            if st.session_state.model_trained:
                with st.expander("Model Results", expanded=True):
                    visualize_model_results(st.session_state.model, X_test, y_test)
                    if st.button("Proceed to Dropout Prediction"):
                        st.session_state.show_prediction = True
                        st.experimental_rerun()
            else:
                st.warning("Please train the model first!")
    elif choice == "Dropout Prediction" or st.session_state.show_prediction:
        st.markdown("<div style='font-size: 20px; font-weight: bold;'>🔮 Dropout Prediction</div>", unsafe_allow_html=True)
        st.markdown("Predict the likelihood of dropout for an individual student.")
        if 'model' in st.session_state:
            with st.expander("Prediction Tool", expanded=True):
                individual_dropout_prediction(st.session_state.model, X)
        else:
            st.warning("Please train the model first!")
            if st.button("Go to Model Training"):
                st.session_state.show_prediction = False
                st.experimental_rerun()

if __name__ == "__main__":
    main()
