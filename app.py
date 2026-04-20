import streamlit as st
import pandas as pd
import numpy as np
import ml_utils
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="Model Training | SaaS Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Session State Initialization ---
if 'theme' not in st.session_state: st.session_state.theme = 'dark'
if 'master_df' not in st.session_state: st.session_state.master_df = None
if 'df' not in st.session_state: st.session_state.df = None
if 'ml_results' not in st.session_state: st.session_state.ml_results = None

# --- UI Definitions ---
themes = {
    'dark': {
        'bg_gradient': 'radial-gradient(circle at 20% 20%, #0f172a 0%, #020617 100%)',
        'card_bg': 'rgba(30, 41, 59, 0.5)',
        'text': '#f8fafc',
        'muted': '#94a3b8',
        'accent': '#7c3aed',
        'accent_lite': '#a78bfa',
        'border': 'rgba(255, 255, 255, 0.08)'
    },
    'light': {
        'bg_gradient': 'linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%)',
        'card_bg': 'rgba(255, 255, 255, 0.7)',
        'text': '#0f172a',
        'muted': '#64748b',
        'accent': '#6366f1',
        'accent_lite': '#818cf8',
        'border': 'rgba(0, 0, 0, 0.05)'
    }
}
t = themes[st.session_state.theme]

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
    
    .stApp {{ background: {t['bg_gradient']}; color: {t['text']} !important; }}
    
    /* Header Styling */
    .premium-header {{
        padding: 1.5rem 0; text-align: left; border-bottom: 1px solid {t['border']}; margin-bottom: 2rem;
        display: flex; align-items: center; gap: 15px;
    }}
    .logo-text {{ font-size: 2rem; font-weight: 800; background: linear-gradient(to right, {t['accent']}, {t['accent_lite']}); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
    
    /* Card Component */
    .glass-card {{
        background: {t['card_bg']}; backdrop-filter: blur(12px); border-radius: 20px;
        border: 1px solid {t['border']}; padding: 1.5rem; margin-bottom: 1.5rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }}
    .glass-card:hover {{ transform: translateY(-3px); box-shadow: 0 12px 24px rgba(0,0,0,0.1); }}
    
    /* Metric Cards */
    .kpi-card {{ text-align: center; }}
    .kpi-label {{ color: {t['muted']}; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }}
    .kpi-value {{ font-size: 1.8rem; font-weight: 700; color: {t['text']}; margin: 0.5rem 0; }}
    
    /* Sidebar */
    section[data-testid="stSidebar"] {{ background-color: {t['card_bg']} !important; border-right: 1px solid {t['border']}; backdrop-filter: blur(20px); }}
    
    /* Tabs Customization */
    .stTabs [data-baseweb="tab-list"] {{ gap: 24px; }}
    .stTabs [data-baseweb="tab"] {{ height: 50px; border-radius: 8px; color: {t['muted']}; }}
    .stTabs [aria-selected="true"] {{ color: {t['accent']} !important; font-weight: 600; }}

    /* Fix White Backgrounds in Dataframes */
    [data-testid="stDataFrame"] {{ background: transparent !important; border-radius: 12px; overflow: hidden; }}
    
    /* Button Customization */
    .stButton>button {{
        background: {t['accent']} !important; border-radius: 10px !important; color: white !important;
        border: none !important; font-weight: 600 !important; transition: 0.3s !important;
    }}
    .stButton>button:hover {{ opacity: 0.9 !important; transform: scale(1.02) !important; }}
    
    /* Fix text colors */
    h1, h2, h3, p, span, label, div.stMarkdown, div.stMarkdown p {{ color: {t['text']} !important; }}
</style>
""", unsafe_allow_html=True)

# --- Sidebar Content ---
with st.sidebar:
    st.markdown(f"<div class='logo-text'>Insight Flow</div>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: {t['muted']}; font-size: 0.85rem;'>Universal Insights Platform</p>", unsafe_allow_html=True)
    st.markdown("[Tableau Dashboard](https://public.tableau.com/views/RetailFoodPriceAnalysisPredictionDashboard/Dashboard1?:language=en-GB&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)", unsafe_allow_html=False)
    
    st.write("---")
    
    # Theme Toggle
    theme_icon = "☀️" if st.session_state.theme == 'dark' else "🌙"
    if st.button(f"{theme_icon} Switch Theme", use_container_width=True):
        st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'
        st.rerun()

    st.write("---")
    st.subheader("📁 Data Management")
    uploaded_file = st.file_uploader("Upload CSV/Excel/JSON", type=['csv', 'xlsx', 'json'])
    
    if st.button("✨ Load Sample Dataset", use_container_width=True):
        st.session_state.master_df = ml_utils.get_sample_data()
        st.session_state.filename = "Sample_Marketing_Data.csv"
        st.success("Sample data loaded!")
        st.rerun()

    if uploaded_file:
        try:
            content = uploaded_file.read()
            st.session_state.master_df = ml_utils.parse_file(content, uploaded_file.name)
            st.session_state.filename = uploaded_file.name
            st.success(f"File '{uploaded_file.name}' uploaded successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"Error uploading file: {str(e)}")
            st.info("Please check the file format and try again. Supported formats: CSV, XLSX")
        st.session_state.filename = uploaded_file.name

    # Filter Section (Only if DF exists)
    if st.session_state.master_df is not None:
        st.write("---")
        st.subheader("🔍 Local Filters")
        all_cols = st.session_state.master_df.columns.tolist()
        
        filter_col = st.selectbox("Select Filter Column", all_cols)
        unique_vals = st.session_state.master_df[filter_col].unique()
        
        if len(unique_vals) < 25:
            selected_vals = st.multiselect("Select Values", sorted(unique_vals), default=None)
            if selected_vals:
                st.session_state.df = st.session_state.master_df[st.session_state.master_df[filter_col].isin(selected_vals)]
            else:
                st.session_state.df = st.session_state.master_df
        else:
            if pd.api.types.is_numeric_dtype(st.session_state.master_df[filter_col]):
                min_v, max_v = float(st.session_state.master_df[filter_col].min()), float(st.session_state.master_df[filter_col].max())
                range_v = st.slider("Select Range", min_v, max_v, (min_v, max_v))
                st.session_state.df = st.session_state.master_df[(st.session_state.master_df[filter_col] >= range_v[0]) & (st.session_state.master_df[filter_col] <= range_v[1])]
            else:
                st.session_state.df = st.session_state.master_df

# --- Main App Logic ---
if st.session_state.master_df is not None:
    df = st.session_state.df if st.session_state.df is not None else st.session_state.master_df
    stats = ml_utils.get_basic_stats(df)
    stats['filename'] = getattr(st.session_state, 'filename', 'Custom_Upload.csv')
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Overview", "📊 Advanced EDA", "📈 Dashboard", "🤖 Model Training", "🔮 Predictions"
    ])

    # --- Tab 1: Overview ---
    with tab1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        col1.markdown(f"<div class='kpi-card'><div class='kpi-label'>Total Rows</div><div class='kpi-value'>{stats['summary']['Total Rows']}</div></div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='kpi-card'><div class='kpi-label'>Total Columns</div><div class='kpi-value'>{stats['summary']['Total Columns']}</div></div>", unsafe_allow_html=True)
        col3.markdown(f"<div class='kpi-card'><div class='kpi-label'>Missing Cells</div><div class='kpi-value'>{stats['summary']['Missing Cells']}</div></div>", unsafe_allow_html=True)
        col4.markdown(f"<div class='kpi-card'><div class='kpi-label'>Memory</div><div class='kpi-value'>{stats['summary']['Memory Usage']}</div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Dataset Snapshot")
        st.dataframe(df.head(20), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Tab 2: Advanced EDA ---
    with tab2:
       st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
       st.subheader("Automated Visual Intelligence")

    # 👉 User se selection
    plot_option = st.selectbox(
        "Select Visualization",
        ["Histogram", "Correlation Heatmap", "Boxplot"]
    )

    with st.spinner("Generating visualization..."):

        if plot_option == "Histogram":
            column = st.selectbox("Select numeric column", df.select_dtypes(include='number').columns)

            if column:
                import matplotlib.pyplot as plt
                import seaborn as sns
                import numpy as np

                data = df[column].dropna()
                mean_val = np.mean(data)
                median_val = np.median(data)

                fig, ax = plt.subplots()
                sns.histplot(data, kde=True, ax=ax)

                ax.axvline(mean_val, linestyle='--', label=f"Mean: {mean_val:.2f}")
                ax.axvline(median_val, linestyle='-', label=f"Median: {median_val:.2f}")

                ax.legend()
                ax.set_title(f"Histogram of {column}")

                st.pyplot(fig)

        elif plot_option == "Correlation Heatmap":
            import matplotlib.pyplot as plt
            import seaborn as sns

            corr = df.select_dtypes(include='number').corr()

            fig, ax = plt.subplots()
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)

            ax.set_title("Correlation Heatmap")
            st.pyplot(fig)

        elif plot_option == "Boxplot":
            column = st.selectbox("Select column for boxplot", df.select_dtypes(include='number').columns)

            if column:
                import matplotlib.pyplot as plt
                import seaborn as sns

                fig, ax = plt.subplots()
                sns.boxplot(y=df[column], ax=ax)

                ax.set_title(f"Boxplot of {column}")
                st.pyplot(fig)

            st.markdown("</div>", unsafe_allow_html=True)
        
        # Trend Analysis
        if stats['cols_info']['datetime']:
            st.write("---")
            st.subheader("📅 Time-Series Trend Analysis")
            date_col = st.selectbox("Select Date Column", stats['cols_info']['datetime'])
            val_col = st.selectbox("Select Value to Trend", stats['cols_info']['numeric'])
            
            trend_df = df.sort_values(date_col).groupby(date_col)[val_col].mean().reset_index()
            st.line_chart(trend_df.set_index(date_col))

        st.markdown("</div>", unsafe_allow_html=True)

    # --- Tab 3: Dashboard ---
    with tab3:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Interactive KPI Intelligence")
        
        kpi_cols = st.columns(min(len(stats['cols_info']['numeric']), 4) if stats['cols_info']['numeric'] else 1)
        if stats['cols_info']['numeric']:
            for i, col_name in enumerate(stats['cols_info']['numeric'][:4]):
                val = df[col_name].mean()
                kpi_cols[i % 4].markdown(f"""
                    <div class='kpi-card' style='border: 1px solid {t['border']}; padding: 10px; border-radius: 12px;'>
                        <div class='kpi-label'>Avg {col_name}</div>
                        <div class='kpi-value'>{val:,.2f}</div>
                    </div>
                """, unsafe_allow_html=True)
        
        st.write("---")
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            if stats['cols_info']['categorical']:
                cat_col = st.selectbox("Categorical Breakdown", stats['cols_info']['categorical'], key='dash_cat')
                st.bar_chart(df[cat_col].value_counts())
            else:
                st.info("No categorical columns detected.")
        with d_col2:
            if stats['cols_info']['numeric']:
                num_col = st.selectbox("Numeric Density Breakdown", stats['cols_info']['numeric'], key='dash_num')
                st.area_chart(df[num_col].value_counts().sort_index())
            else:
                st.info("No numeric columns detected.")
        
        # Report Export
        st.write("---")
        visuals = "This section shows all charts"
        st.write(visuals)
        if st.button("📄 Generate Comprehensive Report", use_container_width=True):
            report_html = ml_utils.generate_html_report(df, stats, visuals, st.session_state.ml_results)
            st.download_button(
                label="📥 Download Data Report (HTML)",
                data=report_html,
                file_name=f"Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html",
                use_container_width=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Tab 4: AutoML ---
    with tab4:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Model training")
        
        bench_col1, bench_col2 = st.columns([1, 2])
        with bench_col1:
            # Smart default target: look for Price, Quantity, etc.
            target_options = stats['columns']
            default_ix = len(target_options) - 1
            for i, col in enumerate(target_options):
                if any(k in col.lower() for k in ['price', 'qunatity', 'quantity', 'amount', 'total']):
                    default_ix = i
                    break
            
            target = st.selectbox("Target Variable (Y)", target_options, index=default_ix)
            features = st.multiselect("Predictor Features (X)", [c for c in stats['columns'] if c != target], default=[c for c in stats['columns'] if c != target][:5])
        
        if st.button("🚀 Run Training"):
            with st.spinner("Training models and analyzing performance..."):
                results = ml_utils.perform_prediction(df, target, theme=st.session_state.theme, feature_cols=features)
                results['target_name'] = target # Track which target was used for training
                st.session_state.ml_results = results
                
                with bench_col2:
                    st.write("### Model Performance Comparison")
                    metrics_df = pd.DataFrame(results['metrics']).T
                    st.table(metrics_df)
                    
                    main_metric = 'Accuracy' if results['is_classification'] else 'R2 Score'
                    st.bar_chart(metrics_df[main_metric])

        if st.session_state.ml_results:
            st.write("---")
            st.subheader("Feature Influence Analysis")
            res_plots = st.session_state.ml_results['plots']
            imp_cols = st.columns(len(res_plots) if res_plots else 1)
            for i, (name, p_data) in enumerate(res_plots.items()):
                with imp_cols[i]:
                    st.image(base64.b64decode(p_data), caption=name, use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Tab 5: Predictions ---
    with tab5:
        if st.session_state.ml_results is None:
            st.warning("Please run Model Training first to initialize models.")
        elif st.session_state.ml_results.get('target_name') != target:
            st.error(f"⚠️ Target mismatch! Model was trained to predict '{st.session_state.ml_results.get('target_name')}', but you have now selected '{target}'. Please rerun the AutoML benchmark.")
        else:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("🔮 Instant AI Prediction")
            
            res = st.session_state.ml_results
            meta = res['metadata']
            model_name = st.selectbox("Select Model for Prediction", list(res['models'].keys()))
            model = res['models'][model_name]
            
            st.write("### Input Feature Values")
            user_inputs = {}
            input_cols = st.columns(3)
            
            # Use original feature order from X_sample keys
            for i, feat in enumerate(res['X_sample'].keys()):
                with input_cols[i % 3]:
                    if feat in meta['cat_mappings']:
                        user_inputs[feat] = st.selectbox(f"{feat}", meta['cat_mappings'][feat])
                    elif feat in meta['date_cols']:
                        # Default to current date if possible
                        default_date = pd.to_datetime(res['X_sample'][feat])
                        user_inputs[feat] = st.date_input(f"{feat}", value=default_date)
                    else:
                        user_inputs[feat] = st.number_input(f"{feat}", value=float(res['X_sample'][feat]), format="%.4f")
            
            if st.button("Calculate Prediction", use_container_width=True):
                # Prepare data for model: Encode selections
                processed_input = {}
                for feat, val in user_inputs.items():
                    if feat in meta['cat_mappings']:
                        # Manual Label Encoding based on alphabetical sort in ml_utils
                        # We use the list index as the encoded value
                        try:
                            processed_input[feat] = meta['cat_mappings'][feat].index(val)
                        except:
                            processed_input[feat] = 0
                    elif feat in meta['date_cols']:
                        # Convert selected date to numeric timestamp
                        processed_input[feat] = pd.to_numeric(pd.Series([pd.Timestamp(val)]))[0]
                    else:
                        processed_input[feat] = val
                
                input_df = pd.DataFrame([processed_input])
                pred = model.predict(input_df)[0]
                
                # If classification, try to map back to original label
                display_pred = pred
                if res['is_classification'] and 'target_mapping' in meta:
                    try:
                        display_pred = meta['target_mapping'][int(pred)]
                    except:
                        pass

                st.markdown(f"""
                <div style='background: {t['accent']}20; border: 2px solid {t['accent']}; border-radius: 15px; padding: 2rem; text-align: center; margin-top: 2rem;'>
                    <h2 style='margin:0; color: {t['text']} !important;'>Predicted {target}</h2>
                    <div style='font-size: 3rem; font-weight: 800; color: {t['accent']}'>{display_pred if isinstance(display_pred, str) else f"{display_pred:,.4f}"}</div>
                    <p style='color: {t['muted']};'>Based on {model_name} architecture</p>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

else:
    # Empty State
    st.markdown(f"""
    <div class='glass-card' style='text-align: center; padding: 5rem 2rem; margin-top: 5rem;'>
        <img src='https://img.icons8.com/isometric/100/data-transfer.png' style='margin-bottom: 2rem;' />
        <h1 style='margin-bottom: 1rem; color: {t['text']} !important;'>Welcome to Data Intelligence</h1>
        <p style='color: {t['muted']}; max-width: 600px; margin: 0 auto 2rem;'>
            Upload your CSV or Excel dataset in the sidebar to unlock automated insights, 
            dashboard visualizations, and machine learning predictions.
        </p>
        <div style='color: {t['accent']}; font-weight: 600;'>← Start by uploading a file or loading sample data</div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown(f"<div style='text-align: center; color: {t['muted']}; font-size: 0.8rem; margin-top: 4rem; padding-bottom: 2rem;'>Insight Flow v5.5 | Professional Data Science Suite | </div>", unsafe_allow_html=True)
