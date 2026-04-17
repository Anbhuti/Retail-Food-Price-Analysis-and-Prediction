import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, f1_score

def set_plot_theme(theme='dark'):
    if theme == 'light':
        text_color = '#1e1b4b'
        label_color = '#4b5563'
        palette = "magma"
    else:
        text_color = '#f8fafc'
        label_color = '#94a3b8'
        palette = "viridis"

    plt.rcParams.update({
        'figure.facecolor': (0, 0, 0, 0),
        'axes.facecolor': (0, 0, 0, 0),
        'savefig.facecolor': (0, 0, 0, 0),
        'text.color': text_color,
        'axes.labelcolor': label_color,
        'xtick.color': label_color,
        'ytick.color': label_color,
        'font.family': 'sans-serif'
    })
    return palette

def parse_file(file_content, filename):
    if filename.endswith('.csv'):
        df = pd.read_csv(io.BytesIO(file_content))
    elif filename.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(io.BytesIO(file_content))
    elif filename.endswith('.json'):
        df = pd.read_json(io.BytesIO(file_content))
    else:
        raise ValueError("Unsupported file format")
    return df

def detect_columns(df):
    """Categorizes columns into Numeric, Categorical, DateTime, and ID-like."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number, 'datetime']).columns.tolist()
    datetime_cols = []
    
    # Attempt to detect datetime columns
    for col in categorical_cols[:]:
        try:
            if df[col].nunique() > 1: # Avoid constants
                series = pd.to_datetime(df[col], errors='coerce')
                if series.notnull().mean() > 0.8: # If 80% are dates
                    datetime_cols.append(col)
                    categorical_cols.remove(col)
        except:
            pass

    # Filter out ID-like columns (high cardinality categories or sequential numbers)
    potential_ids = []
    for col in categorical_cols[:]:
        if df[col].nunique() == len(df) and len(df) > 10:
            potential_ids.append(col)
            # categorical_cols.remove(col) # Keep for now but flag

    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "datetime": datetime_cols,
        "ids": potential_ids
    }

def get_basic_stats(df):
    cols_info = detect_columns(df)
    missing_values = df.isnull().sum().to_dict()
    dtypes = df.dtypes.apply(lambda x: str(x)).to_dict()
    
    # Outlier detection (IQR)
    outliers = {}
    for col in cols_info['numeric']:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        outlier_count = ((df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))).sum()
        outliers[col] = int(outlier_count)

    summary = {
        "Total Rows": len(df),
        "Total Columns": len(df.columns),
        "Numeric Columns": len(cols_info['numeric']),
        "Categorical Columns": len(cols_info['categorical']),
        "DateTime Columns": len(cols_info['datetime']),
        "Missing Cells": int(df.isnull().sum().sum()),
        "Memory Usage": f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
    }
    
    return {
        "columns": df.columns.tolist(),
        "cols_info": cols_info,
        "missing_values": missing_values,
        "dtypes": dtypes,
        "outliers": outliers,
        "summary": summary
    }

def generate_base64_plot():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_str

def get_visualizations(df, theme='dark'):
    palette = set_plot_theme(theme)
    visuals = {}
    
    # Numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 1. Histograms for top 4 numeric columns
    if numeric_cols:
        plt.figure(figsize=(12, 8))
        cols_to_plot = numeric_cols[:4]
        for i, col in enumerate(cols_to_plot):
            plt.subplot(2, 2, i+1)
            sns.histplot(df[col], kde=True, color='#8b5cf6' if theme == 'dark' else '#d946ef')
            plt.title(f'Distribution of {col}', color=plt.rcParams['text.color'])
        plt.tight_layout()
        visuals['histograms'] = generate_base64_plot()

    # 2. Correlation Heatmap
    if len(numeric_cols) > 1:
        plt.figure(figsize=(10, 8))
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, cmap=palette, fmt='.2f')
        plt.title('Correlation Heatmap', color=plt.rcParams['text.color'])
        visuals['heatmap'] = generate_base64_plot()

    # 3. Box Plots
    if numeric_cols:
        plt.figure(figsize=(12, 6))
        cols_to_plot = numeric_cols[:4]
        sns.boxplot(data=df[cols_to_plot], palette="Set2")
        plt.title('Box Plot of Numeric Columns', color=plt.rcParams['text.color'])
        visuals['boxplot'] = generate_base64_plot()

    return visuals

def generate_scatter_plot(df, x_col, y_col, theme='dark'):
    set_plot_theme(theme)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col, color='#8b5cf6' if theme == 'dark' else '#d946ef', alpha=0.6)
    plt.title(f'Relationship: {x_col} vs {y_col}', color=plt.rcParams['text.color'])
    return generate_base64_plot()

def get_custom_analysis(df, index_col, column_col, val_col=None, agg_func='count'):
    if agg_func == 'count':
        pivot = pd.crosstab(df[index_col], df[column_col])
    else:
        if val_col and df[val_col].dtype in [np.number, 'int64', 'float64']:
            pivot = df.pivot_table(index=index_col, columns=column_col, values=val_col, aggfunc=agg_func)
        else:
            return None # Cannot aggregate non-numeric
    return pivot

def perform_prediction(df, target_col, theme='dark', feature_cols=None):
    set_plot_theme(theme)
    # Drop rows with missing target
    df = df.dropna(subset=[target_col])
    
    if feature_cols:
        cols_to_keep = list(set(feature_cols + [target_col]))
        df = df[cols_to_keep]
    
    df = df.dropna(axis=1, how='all')
    
    # Process Categorical Features (One-Hot for Predictability)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Task Detection: 
    # Classify if not numeric (strings, categories, booleans) or if integer with low cardinality.
    is_classification = False
    if not pd.api.types.is_numeric_dtype(y):
        is_classification = True
    elif pd.api.types.is_integer_dtype(y) and y.nunique() < 10:
        is_classification = True
    else:
        is_classification = False
    
    # Simple Imputation & Encoding
    X_processed = X.copy()
    metadata = {
        "cat_mappings": {},
        "date_cols": [],
        "num_cols": []
    }

    for col in X_processed.columns:
        if pd.api.types.is_numeric_dtype(X_processed[col]):
            X_processed[col] = X_processed[col].fillna(X_processed[col].median())
            metadata["num_cols"].append(col)
        elif pd.api.types.is_datetime64_any_dtype(X_processed[col]):
            metadata["date_cols"].append(col)
            # Convert to numeric for model input
            X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce').fillna(0)
        else:
            # Categorical: Clean then map
            X_processed[col] = X_processed[col].fillna(X_processed[col].mode()[0] if not X_processed[col].mode().empty else "Missing")
            # Convert to string and sort unique labels for the UI
            unique_labels = sorted([str(x) for x in X_processed[col].unique().tolist()])
            metadata["cat_mappings"][col] = unique_labels
            
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col].astype(str))

    if is_classification:
        # Clean and encode target
        y_clean = y.astype(str).fillna("Missing")
        y_le = LabelEncoder()
        y = y_le.fit_transform(y_clean)
        metadata["target_mapping"] = [str(c) for c in y_le.classes_.tolist()]

    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    
    results = {
        'is_classification': bool(is_classification),
        'metrics': {},
        'plots': {},
        'models': {},
        'metadata': metadata,
        'X_sample': X.head(1).to_dict('records')[0] # Original values for predictor defaults
    }

    if is_classification:
        m1 = LogisticRegression(max_iter=1000)
        m2 = RandomForestClassifier(n_estimators=100, random_state=42)
        m1.fit(X_train, y_train)
        m2.fit(X_train, y_train)
        p1 = m1.predict(X_test)
        p2 = m2.predict(X_test)
        
        results['metrics']['Logistic Regression'] = {'Accuracy': round(accuracy_score(y_test, p1), 4)}
        results['metrics']['Random Forest'] = {'Accuracy': round(accuracy_score(y_test, p2), 4)}
        results['models'] = {'Logistic Regression': m1, 'Random Forest': m2}
    else:
        m1 = LinearRegression()
        m2 = RandomForestRegressor(n_estimators=100, random_state=42)
        m1.fit(X_train, y_train)
        m2.fit(X_train, y_train)
        p1 = m1.predict(X_test)
        p2 = m2.predict(X_test)
        
        results['metrics']['Linear Regression'] = {
            'R2 Score': round(r2_score(y_test, p1), 4),
            'RMSE': round(np.sqrt(mean_squared_error(y_test, p1)), 4)
        }
        results['metrics']['Random Forest'] = {
            'R2 Score': round(r2_score(y_test, p2), 4),
            'RMSE': round(np.sqrt(mean_squared_error(y_test, p2)), 4)
        }
        results['models'] = {'Linear/Logistic Regression': m1, 'Random Forest': m2}

    # Generate Feature Importance Plots
    for name, model in results['models'].items():
        try:
            plt.figure(figsize=(10, 5))
            if hasattr(model, 'feature_importances_'):
                imp = model.feature_importances_
            else:
                imp = np.abs(model.coef_[0] if is_classification else model.coef_)
            
            pd.Series(imp, index=X_processed.columns).nlargest(10).plot(kind='barh', color='#8b5cf6')
            plt.title(f'Feature Importance: {name}')
            results['plots'][name] = generate_base64_plot()
        except: pass

    return results

def generate_html_report(df, stats, visuals, ml_results=None):
    """Compiles a standalone HTML report with embedded styles and images."""
    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f8fafc; color: #1e293b; padding: 40px; }}
            .container {{ max-width: 1000px; margin: auto; background: white; padding: 40px; border-radius: 20px; box-shadow: 0 10px 25px rgba(0,0,0,0.05); }}
            h1 {{ color: #7c3aed; text-align: center; font-size: 32px; border-bottom: 2px solid #f1f5f9; padding-bottom: 20px; }}
            .section {{ margin-top: 40px; }}
            .card-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-top: 20px; }}
            .card {{ background: #f1f5f9; padding: 20px; border-radius: 12px; text-align: center; }}
            .card .label {{ font-size: 12px; text-transform: uppercase; color: #64748b; font-weight: bold; }}
            .card .value {{ font-size: 24px; font-weight: bold; color: #1e1b4b; margin-top: 10px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #f1f5f9; }}
            th {{ background: #f8fafc; color: #64748b; font-size: 12px; text-transform: uppercase; }}
            .chart {{ width: 100%; border-radius: 12px; margin-top: 20px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); }}
            .footer {{ text-align: center; margin-top: 60px; color: #94a3b8; font-size: 12px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Anubhuti AI | Data Intelligence Report</h1>
            
            <div class="section">
                <h2>Overview: {stats.get('filename', 'Unknown Dataset')}</h2>
                <div class="card-grid">
                    <div class="card"><div class="label">Rows</div><div class="value">{stats['summary']['Total Rows']}</div></div>
                    <div class="card"><div class="label">Columns</div><div class="value">{stats['summary']['Total Columns']}</div></div>
                    <div class="card"><div class="label">Numeric</div><div class="value">{stats['summary']['Numeric Columns']}</div></div>
                    <div class="card"><div class="label">Outliers Detect</div><div class="value">{sum(stats['outliers'].values())}</div></div>
                </div>
            </div>

            <div class="section">
                <h2>Visual Insights</h2>
    """
    if 'histograms' in visuals:
        html += f'<img src="data:image/png;base64,{visuals["histograms"]}" class="chart" />'
    if 'heatmap' in visuals:
        html += f'<img src="data:image/png;base64,{visuals["heatmap"]}" class="chart" />'
    
    if ml_results:
        html += """
            <div class="section">
                <h2>Machine Learning Benchmark</h2>
                <table>
                    <tr><th>Model</th><th>Metric</th><th>Value</th></tr>
        """
        for m_name, m_metrics in ml_results['metrics'].items():
            for metric, val in m_metrics.items():
                html += f"<tr><td>{m_name}</td><td>{metric}</td><td>{val}</td></tr>"
        html += "</table></div>"

    html += """
            <div class="footer">Generated by Anubhuti AI v5.0 | AI-Powered Data Analysis Platform</div>
        </div>
    </body>
    </html>
    """
    return html

def get_sample_data():
    """Generates a synthetic Sales & Marketing dataset for demonstration."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range(start='2024-01-01', periods=n, freq='D')
    campaigns = ['Social Media', 'Email', 'TV Ad', 'Search Engine']
    regions = ['North', 'South', 'East', 'West']
    
    data = {
        'Date': dates,
        'Campaign': np.random.choice(campaigns, n),
        'Region': np.random.choice(regions, n),
        'Budget ($)': np.random.randint(1000, 50000, n),
        'Clicks': np.random.randint(50, 2000, n),
        'Conversions': np.random.randint(1, 100, n),
        'Satisfaction_Score': np.random.uniform(1, 5, n),
        'Is_Active': np.random.choice([True, False], n)
    }
    df = pd.DataFrame(data)
    # Add some outliers
    df.loc[0, 'Budget ($)'] = 500000 
    df.loc[1, 'Conversions'] = 1000
    return df
