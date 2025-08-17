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

st.set_page_config(page_title="Intelligent Data Analysis Assistant", layout="wide")
st.title("📊 Intelligent Data Analysis Assistant")

session = cache_utils.get_session_state()

if "workspace_id" not in session:
    session.workspace_id = str(uuid4())

uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

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
    df = data_processing.load_dataset(uploaded)
    col_types = data_processing.detect_column_types(df)

    st.markdown("### 📋 Dataset Preview")
    st.dataframe(df.head())

    with st.expander("🧩 Detected Column Types"):
        for k, v in col_types.items():
            st.markdown(f"**{k.capitalize()}**: {', '.join(v) if v else 'None'}")

    if col_types["numeric"]:
        st.markdown("#### 🧮 Numeric Summary Statistics")
        num_sum = pd.DataFrame({
            "Mean": [df[c].mean() for c in col_types["numeric"]],
            "Std": [df[c].std() for c in col_types["numeric"]],
            "Missing": [df[c].isnull().sum() for c in col_types["numeric"]]
        }, index=col_types["numeric"])
        st.dataframe(num_sum.style.format({"Mean": "{:.2f}", "Std": "{:.2f}"}))

    st.divider()
    st.header("Auto Dashboard Visualizations")

    if col_types["numeric"]:
        for col in col_types["numeric"][:2]:
            st.markdown(f"##### Distribution of **{col}**")
            fig, ax = plt.subplots()
            sns.histplot(df[col].dropna(), kde=True, ax=ax)
            st.pyplot(fig)
            fig2, ax2 = plt.subplots()
            sns.boxplot(y=df[col].dropna(), ax=ax2)
            st.pyplot(fig2)
    if len(col_types["numeric"]) > 1:
        st.markdown("##### 🔥 Correlation Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(df[col_types["numeric"]].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    if col_types["categorical"]:
        for col in col_types["categorical"][:2]:
            st.markdown(f"##### Value Counts for **{col}**")
            fig, ax = plt.subplots()
            df[col].value_counts().head(20).plot(kind='bar', ax=ax)
            st.pyplot(fig)

    st.divider()
    st.header("Advanced Analysis & Multi-AI (HF + Ollama)")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(tab_names)

    # 1. Natural Language Query + Voice
    with tab1:
        st.subheader("🔎 Natural Language Data Query")
        if "show_voicebox" not in st.session_state:
            st.session_state.show_voicebox = False
        if st.button("Toggle Voice Input"):
            st.session_state.show_voicebox = not st.session_state.show_voicebox
        transcription = ""
        if st.session_state.show_voicebox:
            webrtc_ctx = webrtc_streamer(key="speech-to-text")
            st.info("Voice UI box active. (Transcription may be added here.)")

        user_q = st.text_input("Or type your question in plain English", value=transcription)
        if user_q:
            intent = query_parser.detect_intent(user_q)
            st.write(f"Detected intent: **{intent}**")
            question = user_q.strip().lower()
            answered = False
            if "missing" in question and ("value" in question or "values" in question):
                missing = df.isnull().sum()
                if (missing > 0).any():
                    col = missing.idxmax()
                    st.success(f"Column with the most missing values: **{col}** ({missing[col]} missing)")
                else:
                    st.info("No missing values in any column.")
                answered = True
            elif ("average" in question or "mean" in question):
                col = None
                for name in df.columns:
                    if name.lower() in question:
                        col = name
                        break
                if col:
                    mean_val = df[col].mean()
                    st.success(f"Mean (average) of {col}: {mean_val:.2f}")
                else:
                    means = df.select_dtypes(include='number').mean()
                    st.write("Mean values for numeric columns:")
                    st.dataframe(means.rename("Mean"))
                answered = True
            elif "correlation" in question:
                found_cols = [col for col in df.columns if col.lower() in question]
                if len(found_cols) == 2:
                    corr = df[found_cols[0]].corr(df[found_cols[1]])
                    st.success(f"Correlation between {found_cols} and {found_cols[1]}: {corr:.2f}")
                else:
                    st.write("Correlation matrix for numeric columns:")
                    st.dataframe(df.select_dtypes(include='number').corr())
                answered = True
            elif "categorical" in question and ("column" in question or "columns" in question):
                cats = col_types.get("categorical", [])
                st.success(f"Categorical columns: {', '.join(cats) if cats else 'None'}")
                answered = True
            elif "rows where" in question and "greater than" in question:
                match = re.search(r"rows where ([\w\s]+) is greater than (\d+(\.\d+)?)", question)
                if match:
                    col = match.group(1).strip()
                    val = float(match.group(2))
                    if col in df.columns:
                        filtered = df[df[col] > val]
                        st.write(f"Rows where {col} > {val}:")
                        st.dataframe(filtered)
                    else:
                        st.warning(f"Column '{col}' not found in dataset.")
                    answered = True
            elif "numeric" in question and ("column" in question or "columns" in question):
                nums = col_types.get("numeric", [])
                st.success(f"Numeric columns: {', '.join(nums) if nums else 'None'}")
                answered = True
            elif "preview" in question or "show data" in question or "first rows" in question or "head" in question:
                st.write("First five rows of dataset:")
                st.dataframe(df.head())
                answered = True
            if not answered:
                st.info("Sorry, I couldn't automatically answer that. Please try rephrasing or ask a different data question.")

    # 2. AI/NLP Tab
    with tab2:
        st.subheader("🤗 Hugging Face (local) + Ollama (lightweight local LLMs)")
        user_txt = st.text_area("Enter text/question for AI analysis")
        ai_type = st.radio("Choose AI:", [
            "Hugging Face Sentiment",
            "Hugging Face Summarize",
            "Ollama Phi",
            "Ollama TinyLlama"
        ])
        if user_txt:
            if ai_type == "Hugging Face Sentiment":
                st.json(ai_integration.hf_sentiment(user_txt))
            elif ai_type == "Hugging Face Summarize":
                st.write(ai_integration.hf_summarize(user_txt))
            elif ai_type == "Ollama Phi":
                st.write(ai_integration.ollama_infer(user_txt, model="phi"))
            elif ai_type == "Ollama TinyLlama":
                st.write(ai_integration.ollama_infer(user_txt, model="tinyllama"))
        text_cols = col_types.get("text", [])
        if text_cols:
            st.write("NLP on Dataset Text Columns:")
            text_col = st.selectbox("Select text column", text_cols)
            valid_texts = df[text_col].dropna().astype(str)
            if not valid_texts.empty:
                st.write("Top Keywords:", nlp_module.get_keywords(valid_texts))
                st.write("Sample Sentiment:", nlp_module.get_sentiment(valid_texts.sample(1).iloc[0]))

    # 3. Export & Reporting
    with tab3:
        st.subheader("📄 Generate PDF Report")
        profile = data_processing.profile_data(df)
        insights = "\n".join(profile.get("suggestions", []))
        if not insights.strip():
            insights = "No major issues detected. The data appears clean and ready for analysis."
        if st.button("Generate PDF Report"):
            pdf_buffer = report_generator.generate_enhanced_pdf_report(df, col_types, insights)
            st.download_button(
                label="Download PDF Report",
                data=pdf_buffer,
                file_name="intelligent_data_report.pdf",
                mime="application/pdf"
            )

    # 4. Machine Learning Auto Modeling Tab
    with tab4:
        st.subheader("🤖 Automatic Model Building and Evaluation")
        target_column = st.selectbox("Select the target column for prediction", options=df.columns)
        if st.button("Build Model"):
            model, X_train = ai_integration.auto_ml(df, target_column)
            st.success("Model training and evaluation completed.")
            # Feature Importances (tree models)
            if hasattr(model, "feature_importances_"):
                st.write("### 🔎 Feature Importances")
                importances = model.feature_importances_
                feat_imp_df = pd.DataFrame({
                    "feature": X_train.columns,
                    "importance": importances
                }).sort_values("importance", ascending=False)
                st.dataframe(feat_imp_df)
                fig, ax = plt.subplots()
                feat_imp_df.set_index("feature").plot(kind="barh", ax=ax, legend=False)
                st.pyplot(fig)
            # Actual vs Predicted (regression models)
            if hasattr(model, "predict") and not hasattr(model, "classes_"):
                y_pred = model.predict(X_train)
                y_true = df[target_column].values[:len(y_pred)]
                st.write("### 📈 Actual vs Predicted")
                fig, ax = plt.subplots()
                ax.scatter(y_true, y_pred, alpha=0.5)
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                ax.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
                st.pyplot(fig)
            if 'feat_imp_df' in locals():
                explanation = ai_integration.generate_explanation(f"The model feature importances: {feat_imp_df.to_string(index=False)}")
                st.info(explanation)

    # 5. Statistical Inference Tab
    with tab5:
        st.subheader("📊 Statistical Inference")
        numeric = col_types.get("numeric", [])
        if len(numeric) >= 2:
            col1 = st.selectbox("First variable", numeric, key="inf_col1")
            col2 = st.selectbox("Second variable", numeric, key="inf_col2")
            test_type = st.radio("Select test", ["T-test (independent)", "Correlation Test"])
            if st.button("Run Test"):
                x = df[col1].dropna()
                y = df[col2].dropna().reindex(x.index)
                if test_type == "T-test (independent)":
                    stat, p = stats.ttest_ind(x, y)
                    st.write(f"T-statistic: {stat:.4f}, p-value: {p:.4e}")
                    st.write("Significant difference!" if p < 0.05 else "No significant difference.")
                else:  # Correlation test
                    corr, p = stats.pearsonr(x, y)
                    st.write(f"Correlation: {corr:.4f}, p-value: {p:.4e}")
                    st.write("Significant correlation!" if p < 0.05 else "No significant correlation.")
        else:
            st.info("Not enough numeric columns for inference testing.")

    # 6. TEXT ANALYTICS TAB
    with tab6:
        st.subheader("📝 Text Analytics Integration")
        text_cols = col_types.get("text", [])
        if text_cols:
            text_col = st.selectbox("Select text column", text_cols, key="txt_analytics_col")
            docs = df[text_col].dropna().astype(str)
            st.write(f"Number of text samples: {len(docs)}")
            if st.button("Show Most Common Words"):
                words = " ".join(docs).lower().split()
                common = Counter(words).most_common(20)
                st.write("Most Common Words:", common)
            if st.button("Sample Sentiment (HuggingFace)"):
                sample = docs.sample(1).iloc[0]
                sentiment = ai_integration.hf_sentiment(sample)
                st.info(f"Text: {sample}")
                st.success(f"Sentiment result: {sentiment}")
            if st.button("Summarize Sample (HuggingFace)"):
                sample = "\n".join(docs.sample(min(3, len(docs))).tolist())
                summary = ai_integration.hf_summarize(sample)
                st.info(f"Summary:\n{summary}")
        else:
            st.warning("No text columns detected.")

    # 7. INTERACTIVE DASHBOARD TAB
    with tab7:
        st.subheader("📈 Multi-Panel Interactive Visualization")
        numeric = col_types.get("numeric", [])
        categorical = col_types.get("categorical", [])
        plot_type = st.selectbox("Choose plot type", ["Scatter", "Line", "Box"])
        if (numeric and len(numeric) >= 2):
            x_col = st.selectbox("X Axis", numeric, key="plot_x")
            y_col = st.selectbox("Y Axis", numeric, key="plot_y")
            color_col = st.selectbox("Color by (optional, categorical)", ["None"] + categorical, key="plot_color")
            if st.button("Show Plot", key="interactive_plot"):
                color = color_col if color_col != "None" else None
                if plot_type == "Scatter":
                    fig = px.scatter(df, x=x_col, y=y_col, color=color)
                elif plot_type == "Line":
                    fig = px.line(df, x=x_col, y=y_col, color=color)
                elif plot_type == "Box":
                    fig = px.box(df, x=color, y=y_col) if color else px.box(df, y=y_col)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least two numeric columns for interactive plots.")

else:
    st.info("Please upload a dataset to begin.")
