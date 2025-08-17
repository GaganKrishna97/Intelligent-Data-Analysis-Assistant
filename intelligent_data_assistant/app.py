import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from uuid import uuid4
import re
import scipy.stats as stats
from collections import Counter

from core import data_processing, report_generator, nlp_module, ai_integration
from utils import query_parser, cache_utils
from streamlit_webrtc import webrtc_streamer
import sklearn
import sklearn.metrics
import inspect





st.set_page_config(page_title="Intelligent Data Assistant", layout="wide")
st.title("📊 Intelligent Data Assistant")

# Initialize session
session = cache_utils.get_session_state()
if "workspace_id" not in session:
    session.workspace_id = str(uuid4())

# File upload UI
uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

# Define tabs
tab_names = [
    "Natural Language Query",
    "AI/NLP (Hugging Face & Ollama)",
    "Export & Reporting",
    "Machine Learning Auto Modeling",
    "Statistical Inference",
    "Text Analytics",
    "Interactive Dashboard"
]

if uploaded:
    try:
        df = data_processing.load_dataset(uploaded)
        if df is None or df.empty or len(df.columns) == 0:
            st.error("Uploaded file is empty or invalid. Please check and upload a valid file.")
            st.stop()
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()
    
    col_types = data_processing.detect_column_types(df)

    st.markdown("### Dataset Preview")
    st.dataframe(df.head())

    with st.expander("Detected Column Types"):
        for k, v in col_types.items():
            st.markdown(f"**{k.capitalize()}**: {', '.join(v) if v else 'None'}")

    st.divider()
    st.header("Auto Dashboard Visualizations")

    if col_types["numeric"]:
        for col in col_types["numeric"][:2]:
            st.markdown(f"Distribution of **{col}**")
            fig, ax = plt.subplots()
            sns.histplot(df[col].dropna(), kde=True, ax=ax)
            st.pyplot(fig)
            fig2, ax2 = plt.subplots()
            sns.boxplot(y=df[col].dropna(), ax=ax2)
            st.pyplot(fig2)
    if len(col_types["numeric"]) > 1:
        st.markdown("Correlation Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(df[col_types["numeric"]].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    if col_types["categorical"]:
        for col in col_types["categorical"][:2]:
            st.markdown(f"Value Counts for **{col}**")
            fig, ax = plt.subplots()
            df[col].value_counts().head(20).plot(kind='bar', ax=ax)
            st.pyplot(fig)

    st.divider()
    st.header("Advanced Analysis & Multi-AI")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(tab_names)

    # TAB 1: Natural Language Query with Voice
    with tab1:
        st.subheader("Natural Language Query")
        if "show_voicebox" not in st.session_state:
            st.session_state.show_voicebox = False
        if st.button("Toggle Voice Input"):
            st.session_state.show_voicebox = not st.session_state.show_voicebox
        if st.session_state.show_voicebox:
            webrtc_streamer(key="speech_to_text")
            st.info("Listening... Speak your query.")

        user_query = st.text_input("Or type your question in plain English")
        if user_query:
            intent = query_parser.detect_intent(user_query)
            st.write(f"Detected Intent: **{intent}**")
            question = user_query.lower()
            answered = False

            if "missing" in question and ("value" in question or "values" in question):
                missing_counts = df.isnull().sum()
                if missing_counts.any():
                    max_col = missing_counts.idxmax()
                    st.success(f"Column with most missing values: **{max_col}** ({missing_counts[max_col]})")
                else:
                    st.info("No missing values detected.")
                answered = True
            elif "average" in question or "mean" in question:
                found_col = next((col for col in df.columns if col.lower() in question), None)
                if found_col:
                    mean_val = df[found_col].mean()
                    st.success(f"Average of **{found_col}** is {mean_val:.2f}")
                else:
                    st.write(df.select_dtypes(include="number").mean())
                answered = True
            elif "correlation" in question:
                cols = [col for col in df.columns if col.lower() in question]
                if len(cols) == 2:
                    corr_val = df[cols[0]].corr(df[cols[1]])
                    st.success(f"Correlation between **{cols}** and **{cols[1]}**: {corr_val:.2f}")
                else:
                    st.dataframe(df.select_dtypes(include="number").corr())
                answered = True
            elif "categorical" in question:
                st.success(f"Categorical columns: {', '.join(col_types.get('categorical', []))}")
                answered = True
            elif "rows" in question and "greater" in question:
                match = re.search(r'rows?.*?(\w+).*?greater than (\d+\.?\d*)', question)
                if match:
                    col, val = match.group(1), float(match.group(2))
                    if col in df.columns:
                        filtered_df = df[df[col] > val]
                        st.dataframe(filtered_df)
                    else:
                        st.warning(f"Column {col} not found.")
                    answered = True
            elif "numeric" in question:
                st.success(f"Numeric columns: {', '.join(col_types.get('numeric', []))}")
                answered = True
            elif any(word in question for word in ["preview", "show", "head"]):
                st.dataframe(df.head())
                answered = True
            if not answered:
                st.info("Sorry, I couldn't answer that. Please try rephrasing.")

    # TAB 2: AI/NLP
    with tab2:
        st.subheader("AI/NLP with Hugging Face & Ollama")
        raw_text = st.text_area("Enter text for AI analysis")
        ai_model = st.radio("Choose AI Model", ["Hugging Face Sentiment", "Hugging Face Summarize", "Ollama Phi", "Ollama TinyLlama"])
        if raw_text:
            if ai_model == "Hugging Face Sentiment":
                res = ai_integration.hf_sentiment(raw_text)
                st.json(res)
            elif ai_model == "Hugging Face Summarize":
                res = ai_integration.hf_summarize(raw_text)
                st.write(res)
            elif ai_model == "Ollama Phi":
                res = ai_integration.ollama_infer(raw_text, "phi")
                st.write(res)
            else:
                res = ai_integration.ollama_infer(raw_text, "tinyllama")
                st.write(res)

        text_cols = col_types.get("text", [])
        if text_cols:
            colSel = st.selectbox("Select text column for analysis", text_cols)
            text_data = df[colSel].dropna().astype(str)
            if not text_data.empty:
                keywords = nlp_module.get_keywords(text_data)
                sentiment = nlp_module.get_sentiment(text_data.sample().iloc[0])
                st.write(f"Top Keywords: {keywords}")
                st.write(f"Sample Sentiment: {sentiment}")

    # TAB 3: Export & Reporting
    with tab3:
        st.subheader("Export & Reporting")
        profile = data_processing.profile_data(df)
        insights_text = "\n".join(profile.get("suggestions", []))
        if st.button("Generate PDF Report"):
            pdf_content = report_generator.generate_enhanced_pdf_report(df, col_types, insights_text)
            st.download_button("Download PDF Report", pdf_content, "report.pdf", "application/pdf")

    # TAB 4: Machine Learning Auto Modeling
    with tab4:
        st.subheader("Machine Learning Auto Modeling")
        target = st.selectbox("Select target column", df.columns)
        if st.button("Build Model"):
            try:
                model, X_train, score, task_type = ai_integration.auto_ml(df, target)
                if task_type == "classification":
                    st.success(f"Classification accuracy: {score:.2%}")
                else:
                    st.success(f"Regression RMSE: {score:.4f}")

                if hasattr(model, "feature_importances_"):
                    imp_df = pd.DataFrame({
                        "Feature": X_train.columns,
                        "Importance": model.feature_importances_
                    }).sort_values("Importance", ascending=False)
                    st.dataframe(imp_df)
                    fig, ax = plt.subplots()
                    imp_df.set_index("Feature").plot.barh(legend=False, ax=ax)
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"Error during model training: {e}")

    # TAB 5: Statistical Inference
    with tab5:
        st.subheader("Statistical Inference")
        numeric_cols = col_types.get("numeric", [])
        if len(numeric_cols) >= 2:
            col1 = st.selectbox("Variable 1", numeric_cols)
            col2 = st.selectbox("Variable 2", numeric_cols)
            test = st.radio("Test Type", ["T-test", "Correlation"])
            if st.button("Run Test"):
                x = df[col1].dropna()
                y = df[col2].dropna()
                if test == "T-test":
                    stat, pval = stats.ttest_ind(x, y)
                    st.write(f"**T-statistic:** {stat:.4f}")
                    st.write(f"**P-value:** {pval:.4f}")
                    if pval < 0.05:
                        st.success("The difference is statistically significant (p < 0.05).")
                    else:
                        st.warning("The difference is not statistically significant (p ≥ 0.05).")
                else:
                    corr, pval = stats.pearsonr(x, y)
                    if abs(corr) < 0.3:
                        strength = "very weak"
                    elif abs(corr) < 0.5:
                        strength = "weak"
                    elif abs(corr) < 0.7:
                        strength = "moderate"
                    else:
                        strength = "strong"

                    st.write(f"**Pearson correlation coefficient (r):** {corr:.4f}")
                    st.write(f"**P-value:** {pval:.4f}")
                    st.write(f"**Interpretation:** There is a {strength} linear relationship between {col1} and {col2}.")

                    if pval < 0.05:
                        st.success("The correlation is statistically significant (p < 0.05).")
                    else:
                        st.warning("The correlation is not statistically significant (p ≥ 0.05).")

                    fig, ax = plt.subplots()
                    ax.scatter(x, y)
                    ax.set_xlabel(col1)
                    ax.set_ylabel(col2)
                    ax.set_title(f"Scatter Plot of {col1} vs {col2}")
                    st.pyplot(fig)
                    st.caption("Note: Correlation does not imply causation.")
        else:
            st.info("Need at least two numeric columns.")

    # TAB 6: Text Analytics
    with tab6:
        st.subheader("Text Analytics")
        text_cols = col_types.get("text", [])
        if text_cols:
            sel_col = st.selectbox("Select text column", text_cols)
            texts = df[sel_col].dropna().astype(str)
            if st.button("Show Top Words"):
                words = " ".join(texts).lower().split()
                most_common = Counter(words).most_common(20)
                st.write(most_common)
            if st.button("Sentiment Analysis"):
                sample_text = texts.sample(1).iloc[0]
                sentiment = ai_integration.hf_sentiment(sample_text)
                st.write(f"Sample Text: {sample_text}")
                st.json(sentiment)
            if st.button("Summarize"):
                combined_text = "\n".join(texts.sample(min(len(texts), 3)))
                summary = ai_integration.hf_summarize(combined_text)
                st.write(summary)
        else:
            st.warning("No text columns available.")

    # TAB 7: Interactive Dashboard
    with tab7:
        st.subheader("Interactive Dashboard")
        numeric_cols = col_types.get("numeric", [])
        categorical_cols = col_types.get("categorical", [])
        plot_type = st.selectbox("Select plot type", ["Scatter", "Line", "Box"])
        if len(numeric_cols) >= 2:
            x_axis = st.selectbox("X Axis", numeric_cols)
            y_axis = st.selectbox("Y Axis", numeric_cols)
            color = st.selectbox("Color (optional)", ["None"] + categorical_cols)
            if st.button("Plot"):
                color_arg = color if color != "None" else None
                if plot_type == "Scatter":
                    fig = px.scatter(df, x=x_axis, y=y_axis, color=color_arg)
                elif plot_type == "Line":
                    fig = px.line(df, x=x_axis, y=y_axis, color=color_arg)
                else:
                    fig = px.box(df, x=color_arg, y=y_axis) if color_arg else px.box(df, y=y_axis)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Please upload a dataset with at least two numeric columns to plot.")

else:
    st.info("Please upload a CSV or Excel file to begin.")
