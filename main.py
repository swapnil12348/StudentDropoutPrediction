import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lime
import lime.lime_tabular
from sklearn.inspection import permutation_importance
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    """Display detailed column information using tabs instead of nested expanders"""

    st.subheader("📋 Detailed Column Information")

    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Statistics", "❌ Missing Data", "🔍 Data Types"])

    with tab1:
        st.write("**Dataset Overview**")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            st.metric("Memory Usage", f"{df.memory_usage().sum() / 1024 ** 2:.2f} MB")

        st.write("**Column Names**")
        st.write(", ".join(df.columns.tolist()))

    with tab2:
        st.write("**Statistical Summary**")

        # Numeric columns statistics
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            st.write("*Numeric Columns:*")
            st.dataframe(df[numeric_cols].describe())

        # Categorical columns statistics
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            st.write("*Categorical Columns:*")
            cat_stats = []
            for col in categorical_cols:
                cat_stats.append({
                    'Column': col,
                    'Unique Values': df[col].nunique(),
                    'Most Frequent': df[col].mode().iloc[0] if not df[col].mode().empty else 'N/A',
                    'Frequency': df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 0
                })
            st.dataframe(cat_stats)

    with tab3:
        st.write("**Missing Data Analysis**")

        missing_data = df.isnull().sum()
        missing_percentage = (missing_data / len(df)) * 100

        missing_df = pd.DataFrame({
            'Column': missing_data.index.tolist(),
            'Missing Count': missing_data.values.tolist(),
            'Missing Percentage': missing_percentage.values.tolist()
        })

        # Only show columns with missing data
        missing_df = missing_df[missing_df['Missing Count'] > 0]

        if len(missing_df) > 0:
            st.dataframe(missing_df)

            # Visual representation
            if len(missing_df) <= 10:  # Only show chart if not too many columns
                import plotly.express as px
                fig = px.bar(missing_df, x='Column', y='Missing Percentage',
                             title='Missing Data Percentage by Column')
                fig.update_xaxis(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing data found in the dataset!")

    with tab4:
        st.write("**Data Types Information**")

        dtype_info = pd.DataFrame({
            'Column': df.columns.tolist(),
            'Data Type': [str(dtype) for dtype in df.dtypes],
            'Non-Null Count': df.count().tolist(),
            'Null Count': df.isnull().sum().tolist()
        })

        st.dataframe(dtype_info)

        # Data type distribution
        dtype_counts = df.dtypes.value_counts()
        col1, col2 = st.columns([1, 1])

        with col1:
            st.write("**Data Type Distribution**")
            for dtype, count in dtype_counts.items():
                st.write(f"• {dtype}: {count} columns")

        with col2:
            if len(dtype_counts) > 1:
                import plotly.express as px
                # Convert to standard Python types to avoid JSON serialization error
                fig = px.pie(values=dtype_counts.values.tolist(),
                             names=[str(name) for name in dtype_counts.index],
                             title='Distribution of Data Types')
                st.plotly_chart(fig, use_container_width=True)

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


def display_global_feature_importance(model, feature_names):
    """Display global feature importance using SHAP"""
    st.markdown("#### 🌍 Global Feature Importance")
    st.markdown("Understanding which features are most important across all predictions.")

    try:
        with st.spinner("Calculating global feature importance..."):
            # Create SHAP explainer
            explainer = shap.TreeExplainer(model)

            # Get SHAP values for a sample of the training data
            # Use a smaller sample for performance
            sample_size = min(100, len(st.session_state.X_train))
            X_sample = st.session_state.X_train.sample(n=sample_size, random_state=42)

            shap_values = explainer.shap_values(X_sample)

            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                # Multi-class case: shap_values is a list of arrays
                st.markdown("**Multi-class model detected**")

                # Calculate mean absolute SHAP values for each class
                class_importance = {}
                for class_idx, class_shap_values in enumerate(shap_values):
                    mean_abs_shap = np.mean(np.abs(class_shap_values), axis=0)
                    class_importance[f'Class_{class_idx}'] = mean_abs_shap

                # Overall importance (mean across all classes)
                overall_importance = np.mean([np.abs(class_shap) for class_shap in shap_values], axis=(0, 1))

                # Create DataFrame for overall importance
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Mean_Abs_SHAP': overall_importance
                }).sort_values('Mean_Abs_SHAP', ascending=False)

                # Display overall importance
                st.markdown("**🏆 Overall Feature Importance (All Classes)**")
                col1, col2 = st.columns([2, 1])

                with col1:
                    fig = px.bar(
                        importance_df.head(15),
                        x='Mean_Abs_SHAP',
                        y='Feature',
                        orientation='h',
                        title='Top 15 Most Important Features (Overall)',
                        color='Mean_Abs_SHAP',
                        color_continuous_scale='viridis'
                    )
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.markdown("**📊 Top 10 Features:**")
                    for i, (idx, row) in enumerate(importance_df.head(10).iterrows()):
                        st.write(f"{i + 1}. **{row['Feature']}**: {row['Mean_Abs_SHAP']:.4f}")

                # Class-specific importance
                st.markdown("**🎯 Class-Specific Feature Importance**")

                class_tabs = st.tabs([f"Class {i}" for i in range(len(shap_values))])

                for class_idx, tab in enumerate(class_tabs):
                    with tab:
                        class_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Mean_Abs_SHAP': class_importance[f'Class_{class_idx}']
                        }).sort_values('Mean_Abs_SHAP', ascending=False)

                        # Readable class names
                        class_name = "Dropout" if class_idx == 0 else "Graduate" if class_idx == 1 else "Enrolled"
                        st.markdown(f"**Most Important Features for {class_name} Prediction:**")

                        fig = px.bar(
                            class_df.head(10),
                            x='Mean_Abs_SHAP',
                            y='Feature',
                            orientation='h',
                            title=f'Top 10 Features for {class_name}',
                            color='Mean_Abs_SHAP',
                            color_continuous_scale='plasma'
                        )
                        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)

            else:
                # Binary classification case
                st.markdown("**Binary classification model detected**")

                # Calculate mean absolute SHAP values
                mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

                # Create DataFrame
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Mean_Abs_SHAP': mean_abs_shap
                }).sort_values('Mean_Abs_SHAP', ascending=False)

                # Display results
                col1, col2 = st.columns([2, 1])

                with col1:
                    fig = px.bar(
                        importance_df.head(15),
                        x='Mean_Abs_SHAP',
                        y='Feature',
                        orientation='h',
                        title='Top 15 Most Important Features',
                        color='Mean_Abs_SHAP',
                        color_continuous_scale='viridis'
                    )
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.markdown("**📊 Top 10 Features:**")
                    for i, (idx, row) in enumerate(importance_df.head(10).iterrows()):
                        st.write(f"{i + 1}. **{row['Feature']}**: {row['Mean_Abs_SHAP']:.4f}")

            # SHAP Summary Plot
            st.markdown("**📈 SHAP Summary Plot**")

            if isinstance(shap_values, list):
                # For multi-class, show summary for the first class (usually most interesting)
                fig_summary = plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values[0], X_sample, feature_names=feature_names, show=False)
                st.pyplot(fig_summary)
                plt.close()
                st.caption(
                    "Summary plot for Class 0 (Dropout). Each dot represents a student, color indicates feature value.")
            else:
                fig_summary = plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
                st.pyplot(fig_summary)
                plt.close()
                st.caption("Each dot represents a student, color indicates feature value.")

            # Insights
            st.markdown("### 💡 Key Insights")
            top_3_features = importance_df.head(3)['Feature'].tolist()
            st.write(f"**Top 3 most important features:** {', '.join(top_3_features)}")
            st.write("• Higher SHAP values indicate greater impact on model predictions")
            st.write("• Focus intervention strategies on the most important features")
            st.write("• Consider collecting more data for less important features that might be noise")

    except Exception as e:
        st.error(f"Error calculating feature importance: {str(e)}")
        st.info("This might be due to model compatibility issues. Try retraining the model or check your data format.")

        # Fallback: Show model feature importance if available
        try:
            if hasattr(model, 'feature_importances_'):
                st.markdown("**🔄 Fallback: Model Built-in Feature Importance**")
                fallback_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)

                fig = px.bar(
                    fallback_df.head(15),
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Top 15 Features (Model Built-in Importance)'
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
        except:
            st.write("Could not display alternative feature importance.")

def display_local_explanation(model, X_train, X_test, feature_names):
    """Display local explanations for individual predictions"""
    st.markdown("#### 🔍 Local Prediction Explanation")
    st.markdown("Understand why the model made a specific prediction for an individual student.")

    # Student selection
    st.markdown("**Select a student to explain:**")
    student_idx = st.selectbox(
        "Choose student index from test set",
        range(len(X_test)),
        format_func=lambda x: f"Student {x + 1}"
    )

    # Get the selected student data
    student_data = X_test.iloc[student_idx:student_idx + 1]
    prediction = model.predict(student_data)[0]
    prediction_proba = model.predict_proba(student_data)[0]

    # Display prediction
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Predicted Outcome", f"Class {prediction}")
    with col2:
        max_proba = np.max(prediction_proba)
        st.metric("Confidence", f"{max_proba:.2%}")
    with col3:
        # Map prediction to readable format if you have TARGET_MAPPING
        readable_prediction = "Dropout" if prediction == 0 else "Graduate" if prediction == 1 else "Enrolled"
        st.metric("Prediction", readable_prediction)

    # Show all class probabilities
    st.markdown("**📊 Prediction Probabilities:**")
    prob_df = pd.DataFrame({
        'Class': [f'Class {i}' for i in range(len(prediction_proba))],
        'Probability': prediction_proba
    })
    fig = px.bar(prob_df, x='Class', y='Probability',
                 title="Prediction Probabilities for Selected Student")
    st.plotly_chart(fig, use_container_width=True)

    # Explanation method selection
    explanation_method = st.selectbox(
        "Select Explanation Method",
        ["SHAP Local Explanation", "LIME Explanation"]
    )

    try:
        if explanation_method == "SHAP Local Explanation":
            with st.spinner("Generating SHAP explanation..."):
                # Create SHAP explainer
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(student_data)

                # Handle multi-class output
                if isinstance(shap_values, list):
                    shap_values_for_prediction = shap_values[prediction]  # Use predicted class
                else:
                    shap_values_for_prediction = shap_values

                # SHAP waterfall plot - FIXED VERSION
                st.markdown("**🌊 SHAP Waterfall Plot**")

                # Create the explanation object properly
                if isinstance(explainer.expected_value, np.ndarray):
                    base_value = explainer.expected_value[prediction]
                else:
                    base_value = explainer.expected_value

                shap_explanation = shap.Explanation(
                    values=shap_values_for_prediction[0],  # Single instance SHAP values
                    base_values=base_value,
                    data=student_data.iloc[0].values,
                    feature_names=feature_names
                )

                # Use the newer shap.plots.waterfall function
                fig_waterfall = plt.figure(figsize=(10, 8))
                shap.plots.waterfall(shap_explanation, show=False)
                st.pyplot(fig_waterfall)
                plt.close()

                # Top contributing features
                shap_importance = pd.DataFrame({
                    'Feature': feature_names,
                    'SHAP_Value': shap_values_for_prediction[0],
                    'Feature_Value': student_data.iloc[0].values
                })
                shap_importance['Abs_SHAP'] = np.abs(shap_importance['SHAP_Value'])
                shap_importance = shap_importance.sort_values('Abs_SHAP', ascending=False)

                st.markdown("**🎯 Top Contributing Features:**")
                for i, (idx, row) in enumerate(shap_importance.head(5).iterrows()):
                    impact = "increases" if row['SHAP_Value'] > 0 else "decreases"
                    st.write(
                        f"{i + 1}. **{row['Feature']}** (value: {row['Feature_Value']:.2f}) {impact} prediction by {abs(row['SHAP_Value']):.4f}")

        elif explanation_method == "LIME Explanation":
            with st.spinner("Generating LIME explanation..."):
                # Create LIME explainer
                explainer = lime.lime_tabular.LimeTabularExplainer(
                    X_train.values,
                    feature_names=feature_names,
                    class_names=[f'Class_{i}' for i in range(len(prediction_proba))],
                    mode='classification'
                )

                # Generate explanation
                explanation = explainer.explain_instance(
                    student_data.iloc[0].values,
                    model.predict_proba,
                    num_features=10
                )

                # Display LIME plot
                st.markdown("**🍃 LIME Explanation**")
                fig_lime = explanation.as_pyplot_figure()
                st.pyplot(fig_lime)
                plt.close()

                # Extract and display feature contributions
                lime_values = explanation.as_list()
                st.markdown("**📋 LIME Feature Contributions:**")
                for i, (feature_desc, contribution) in enumerate(lime_values[:5]):
                    impact = "supports" if contribution > 0 else "opposes"
                    st.write(f"{i + 1}. {feature_desc} {impact} the prediction (weight: {contribution:.4f})")

    except Exception as e:
        st.error(f"Error generating explanation: {str(e)}")
        st.info("Try selecting a different student or explanation method.")

def display_feature_impact_analysis(model, X_train, X_test, feature_names, df):
    """Display how different feature values impact predictions"""
    st.markdown("#### 📈 Feature Impact Analysis")
    st.markdown("Explore how changing feature values affects prediction probabilities.")

    # Feature selection for analysis
    selected_feature = st.selectbox(
        "Select feature to analyze",
        feature_names
    )

    if selected_feature:
        # Get feature statistics
        feature_stats = df[selected_feature].describe()

        col1, col2 = st.columns([2, 1])

        with col1:
            # Create range of values for the selected feature
            feature_min = feature_stats['min']
            feature_max = feature_stats['max']
            feature_mean = feature_stats['mean']

            # Generate range of values
            if feature_max - feature_min > 0:
                feature_range = np.linspace(feature_min, feature_max, 50)
            else:
                feature_range = [feature_mean]

            # Use a representative sample from test set
            sample_student = X_test.iloc[0:1].copy()

            # Calculate predictions for different feature values
            predictions_over_range = []
            for value in feature_range:
                temp_student = sample_student.copy()
                temp_student[selected_feature] = value
                pred_proba = model.predict_proba(temp_student)[0]
                predictions_over_range.append(pred_proba)

            # Convert to DataFrame for plotting
            pred_df = pd.DataFrame(predictions_over_range)
            pred_df['Feature_Value'] = feature_range

            # Melt for plotting
            pred_melted = pred_df.melt(id_vars=['Feature_Value'],
                                       value_vars=list(range(len(pred_df.columns) - 1)),
                                       var_name='Class', value_name='Probability')
            pred_melted['Class'] = pred_melted['Class'].apply(lambda x: f'Class_{x}')

            # Plot feature impact
            fig = px.line(pred_melted, x='Feature_Value', y='Probability',
                          color='Class', title=f'Impact of {selected_feature} on Predictions')
            fig.add_vline(x=feature_mean, line_dash="dash",
                          annotation_text=f"Mean: {feature_mean:.2f}")
            fig.update_layout(
                xaxis_title=selected_feature,
                yaxis_title='Prediction Probability'
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(
                f"**Insight**: This shows how changing {selected_feature} from {feature_min:.2f} to {feature_max:.2f} affects the model's predictions.")

        with col2:
            st.markdown("**📊 Feature Statistics**")
            st.write(f"**Min**: {feature_stats['min']:.2f}")
            st.write(f"**Max**: {feature_stats['max']:.2f}")
            st.write(f"**Mean**: {feature_stats['mean']:.2f}")
            st.write(f"**Std**: {feature_stats['std']:.2f}")

            # Feature value distribution
            st.markdown("**📈 Value Distribution**")
            fig_hist = px.histogram(df, x=selected_feature, nbins=30)
            st.plotly_chart(fig_hist, use_container_width=True)

            # Risk zones
            st.markdown("**⚠️ Risk Analysis**")
            if len(feature_range) > 1:
                high_risk_threshold = np.percentile(feature_range, 75)
                low_risk_threshold = np.percentile(feature_range, 25)
                st.write(f"**High Risk Zone**: > {high_risk_threshold:.2f}")
                st.write(f"**Medium Risk Zone**: {low_risk_threshold:.2f} - {high_risk_threshold:.2f}")
                st.write(f"**Low Risk Zone**: < {low_risk_threshold:.2f}")


def individual_dropout_prediction_with_explanation(model, X, X_train, feature_names):
    """Enhanced individual prediction with explanations"""
    st.markdown("#### 🎯 Individual Student Prediction with Explanation")

    # Create input form for student data
    st.markdown("**Enter Student Information:**")

    # Create columns for better layout
    col1, col2 = st.columns(2)

    input_data = {}
    feature_list = feature_names.tolist()

    # Split features into two columns
    mid_point = len(feature_list) // 2

    with col1:
        for feature in feature_list[:mid_point]:
            if feature in X.columns:
                min_val = float(X[feature].min())
                max_val = float(X[feature].max())
                mean_val = float(X[feature].mean())

                input_data[feature] = st.number_input(
                    f"{feature}",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    key=f"input_{feature}"
                )

    with col2:
        for feature in feature_list[mid_point:]:
            if feature in X.columns:
                min_val = float(X[feature].min())
                max_val = float(X[feature].max())
                mean_val = float(X[feature].mean())

                input_data[feature] = st.number_input(
                    f"{feature}",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    key=f"input_{feature}"
                )

    if st.button("🔮 Predict with Explanation", type="primary"):
        # Create DataFrame from input
        input_df = pd.DataFrame([input_data])

        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]

        # Display results
        st.markdown("---")
        st.markdown("### 📊 Prediction Results")

        col1, col2, col3 = st.columns(3)
        with col1:
            readable_prediction = "Dropout" if prediction == 0 else "Graduate" if prediction == 1 else "Enrolled"
            st.metric("🎯 Prediction", readable_prediction)

        with col2:
            confidence = np.max(prediction_proba)
            st.metric("🎯 Confidence", f"{confidence:.1%}")

        with col3:
            risk_level = "High" if prediction == 0 else "Low"
            st.metric("⚠️ Risk Level", risk_level)

        # Probability breakdown
        st.markdown("**📊 Detailed Probabilities:**")
        prob_data = {
            'Outcome': ['Dropout', 'Graduate', 'Enrolled'][:len(prediction_proba)],
            'Probability': prediction_proba
        }
        prob_df = pd.DataFrame(prob_data)

        fig = px.bar(prob_df, x='Outcome', y='Probability',
                     color='Probability', color_continuous_scale='RdYlGn_r',
                     title="Prediction Probabilities")
        st.plotly_chart(fig, use_container_width=True)

        # Generate explanation
        try:
            st.markdown("### 🔍 Why This Prediction?")

            with st.spinner("Generating explanation..."):
                # SHAP explanation
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(input_df)

                if isinstance(shap_values, list):
                    shap_values_single = shap_values[prediction]
                else:
                    shap_values_single = shap_values

                # Feature contributions
                feature_contrib = pd.DataFrame({
                    'Feature': feature_names,
                    'Value': input_df.iloc[0].values,
                    'SHAP_Value': shap_values_single[0]
                })
                feature_contrib['Abs_SHAP'] = np.abs(feature_contrib['SHAP_Value'])
                feature_contrib = feature_contrib.sort_values('Abs_SHAP', ascending=False)

                # Display top contributing features
                st.markdown("**🎯 Top Factors Influencing This Prediction:**")

                for i, (idx, row) in enumerate(feature_contrib.head(5).iterrows()):
                    impact = "increases" if row['SHAP_Value'] > 0 else "decreases"
                    color = "🔴" if row['SHAP_Value'] > 0 else "🟢"

                    st.write(f"{i + 1}. {color} **{row['Feature']}** (value: {row['Value']:.2f}) "
                             f"{impact} dropout risk by {abs(row['SHAP_Value']):.4f}")

                # Visualization of feature contributions
                top_features = feature_contrib.head(10)
                fig = px.bar(top_features, x='SHAP_Value', y='Feature',
                             orientation='h', color='SHAP_Value',
                             color_continuous_scale='RdBu_r',
                             title="Feature Contributions to Prediction")
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

                # Recommendations
                st.markdown("### 💡 Recommendations")
                if prediction == 0:  # Dropout prediction
                    st.warning("**High Dropout Risk Detected!**")
                    st.markdown("**Recommended Interventions:**")

                    # Find features that increase dropout risk
                    risk_factors = feature_contrib[feature_contrib['SHAP_Value'] > 0].head(3)
                    for idx, row in risk_factors.iterrows():
                        st.write(f"• Address **{row['Feature']}** - current value: {row['Value']:.2f}")

                    st.markdown("**Support Strategies:**")
                    st.write("• Academic counseling and tutoring")
                    st.write("• Financial aid consultation")
                    st.write("• Student engagement programs")
                    st.write("• Regular progress monitoring")

                else:  # Graduate/Enrolled prediction
                    st.success("**Low Dropout Risk - Student on Track!**")
                    st.markdown("**Maintain Success Factors:**")

                    protective_factors = feature_contrib[feature_contrib['SHAP_Value'] < 0].head(3)
                    for idx, row in protective_factors.iterrows():
                        st.write(f"• Continue supporting **{row['Feature']}** - current value: {row['Value']:.2f}")

        except Exception as e:
            st.error(f"Could not generate explanation: {str(e)}")
            st.info("Prediction completed, but explanation feature is unavailable.")


def main():
    st.markdown("<div style='font-size: 24px; font-weight: bold;'>🎓 Student Dropout Prediction Dashboard</div>",
                unsafe_allow_html=True)
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
        st.markdown("<div style='font-size: 20px; font-weight: bold;'>📈 Exploratory Data Analysis</div>",
                    unsafe_allow_html=True)
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
                            st.markdown(
                                f"**Insight**: The distribution of {column} shows its range and central tendency, impacting dropout risk.",
                                unsafe_allow_html=True)
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
                        st.markdown(
                            "**Insight**: Strong correlations indicate features that move together, influencing dropout prediction.",
                            unsafe_allow_html=True)
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
                            st.markdown(
                                "**Insight**: High correlations highlight key feature relationships affecting student outcomes.",
                                unsafe_allow_html=True)
                        else:
                            st.write("No strong correlations (>0.5) found.")
                            st.markdown(
                                "**Insight**: Weak correlations suggest features are independent, requiring diverse predictors.",
                                unsafe_allow_html=True)
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
                    st.markdown(
                        "**Insight**: The proportion of Graduates, Dropouts, and Enrolled students shows the dataset's outcome balance.",
                        unsafe_allow_html=True)
                    plt.close()

                with col2:
                    st.markdown("<h3 style='font-size: 16px;'>Outcome Statistics</h3>", unsafe_allow_html=True)
                    total_students = len(df)
                    for outcome in df['Target'].unique():
                        count = (df['Target'] == outcome).sum()
                        percentage = count / total_students * 100
                        st.write(f"• **{outcome}**: {count:,} students ({percentage:.1f}%)")
                    st.markdown(
                        "**Insight**: Dropout rates indicate the scale of the challenge in improving student retention.",
                        unsafe_allow_html=True)

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
                            st.markdown(
                                f"**Insight**: {col} influences student outcomes, with certain categories linked to higher dropout rates.",
                                unsafe_allow_html=True)
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

                        st.markdown(f"<h3 style='font-size: 16px;'>{friendly_name} Analysis</h3>",
                                    unsafe_allow_html=True)
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
                            st.markdown(
                                f"**Insight**: Higher {friendly_name.lower()} are associated with graduates, while lower values correlate with dropouts.",
                                unsafe_allow_html=True)
                            plt.close()

                        with col2:
                            st.markdown("<h3 style='font-size: 16px;'>Quick Summary</h3>", unsafe_allow_html=True)
                            summary_stats = df.groupby('Target')[col].agg(['mean', 'count']).round(2)
                            for outcome in summary_stats.index:
                                mean_val = summary_stats.loc[outcome, 'mean']
                                count_val = int(summary_stats.loc[outcome, 'count'])
                                emoji = "🎓" if outcome == 'Graduate' else "⚠️" if outcome == 'Dropout' else "📚"
                                st.write(f"{emoji} **{outcome}:** Average: {mean_val}, Students: {count_val}")
                            st.markdown(
                                f"**Insight**: {friendly_name} significantly differentiates outcomes, guiding intervention strategies.",
                                unsafe_allow_html=True)

    elif choice == "Model Training & Evaluation":
        st.markdown("<div style='font-size: 20px; font-weight: bold;'>🤖 Model Training & Evaluation</div>",
                    unsafe_allow_html=True)
        st.markdown("Train a machine learning model and evaluate its performance.")

        sub_menu = st.selectbox("Select Action", [
            "Train Model",
            "View Results",
            "Model Explainability"  # NEW: Added explainability option
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
                st.markdown(
                    "**Insight**: Encoded labels ensure the model can process categorical outcomes effectively.",
                    unsafe_allow_html=True)

                if st.button("Start Training"):
                    with st.spinner("Training model..."):
                        model = train_model(X_train, y_train)
                        st.session_state.model = model
                        st.session_state.model_trained = True
                        # Store training data for explainability
                        st.session_state.X_train = X_train
                        st.session_state.X_test = X_test
                        st.session_state.y_test = y_test
                        st.session_state.feature_names = X.columns.tolist()
                        st.success("Model trained successfully!")
                        st.button("View Results", key="go_to_results")

        elif sub_menu == "View Results":
            if st.session_state.model_trained:
                with st.expander("Model Results", expanded=True):
                    visualize_model_results(st.session_state.model, X_test, y_test)
                    if st.button("Proceed to Dropout Prediction"):
                        st.session_state.show_prediction = True
                        st.experimental_rerun()
            else:
                st.warning("Please train the model first!")

        # NEW: Model Explainability Section
        elif sub_menu == "Model Explainability":
            if st.session_state.model_trained and 'model' in st.session_state:
                with st.expander("Model Explainability Dashboard", expanded=True):
                    st.markdown("### 🔍 Understanding Model Decisions")
                    st.markdown("Explore how the model makes predictions and which features are most important.")

                    explainability_type = st.selectbox(
                        "Select Explainability Analysis",
                        ["Global Feature Importance", "Local Prediction Explanation", "Feature Impact Analysis"]
                    )

                    if explainability_type == "Global Feature Importance":
                        display_global_feature_importance(
                            st.session_state.model,
                            st.session_state.feature_names
                        )

                    elif explainability_type == "Local Prediction Explanation":
                        display_local_explanation(
                            st.session_state.model,
                            st.session_state.X_train,
                            st.session_state.X_test,
                            st.session_state.feature_names
                        )

                    elif explainability_type == "Feature Impact Analysis":
                        display_feature_impact_analysis(
                            st.session_state.model,
                            st.session_state.X_train,
                            st.session_state.X_test,
                            st.session_state.feature_names,
                            df
                        )
            else:
                st.warning("Please train the model first to access explainability features!")
                if st.button("Go to Model Training"):
                    st.experimental_rerun()

    elif choice == "Dropout Prediction" or st.session_state.show_prediction:
        st.markdown("<div style='font-size: 20px; font-weight: bold;'>🔮 Dropout Prediction</div>",
                    unsafe_allow_html=True)
        st.markdown("Predict the likelihood of dropout for an individual student.")

        if 'model' in st.session_state:
            with st.expander("Prediction Tool", expanded=True):
                # Modified to include explainability in individual predictions
                individual_dropout_prediction_with_explanation(
                    st.session_state.model,
                    X,
                    st.session_state.X_train,
                    st.session_state.feature_names
                )
        else:
            st.warning("Please train the model first!")
            if st.button("Go to Model Training"):
                st.session_state.show_prediction = False
                st.experimental_rerun()


if __name__ == "__main__":
    main()
