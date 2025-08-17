# Intelligent Data Analysis Assistant

A comprehensive and interactive data analysis platform that empowers users to explore, analyze, visualize, and model datasets efficiently using natural language queries, machine learning, and AI-powered insights.

---

## Features

- **Data Upload & Profiling**  
  Upload CSV or Excel datasets. Automatic detection of variable types (numeric, categorical, text) and generation of summary statistics, missing value reports, and correlations.

- **Natural Language Query Interface**  
  Ask questions in plain English to get immediate data insights. Includes support for voice input via speech-to-text.

- **Advanced Visualization Dashboard**  
  Interactive and customizable visualizations including scatter, line, box plots, histograms, and correlation heatmaps powered by Plotly, Matplotlib, and Seaborn.

- **AI & NLP Integration**  
  Built-in Hugging Face models for sentiment analysis and summarization. Integration with local Ollama LLMs for natural language understanding and explanations.

- **Automatic Machine Learning Modeling**  
  One-click model building with classification and regression support, feature importance visualization, prediction vs actual charts, and AI-generated model explanations.

- **Statistical Inference Tools**  
  Integrated hypothesis testing (t-test) and correlation significance testing with clear interpretations.

- **Text Analytics Module**  
  Exploration of text columns for word frequency, sentiment analysis, and summarization using NLP techniques.

- **Session Management & Collaboration (Planned)**  
  Workspace ID assignment for saving, sharing, and collaborative data analysis.

- **Export & Reporting**  
  Generate detailed PDF reports summarizing data insights, analysis results, and recommendations.

---

## Installation

1. Clone the repository:


2. Install required Python packages:


3. (Optional) Pull and run Ollama models locally for LLM features:


---

## Usage

Run the Streamlit app locally:


Access the app via the displayed local URL (usually http://localhost:8501).

---

## Project Structure

- `app.py` - Main Streamlit app with UI and feature integration.  
- `core/` - Core modules including data processing, AI integrations (Hugging Face, Ollama), NLP utilities, and report generation.  
- `utils/` - Utility scripts for query parsing, session state management, and caching.  
- `requirements.txt` - Python dependencies.  
- `README.md` - This documentation.

---

## Future Work

- Implement full multi-user collaboration with shared real-time workspaces.  
- Support real-time data sources via API and streaming integrations.  
- Extend AI capabilities for causal inference, time series forecasting, and Bayesian analysis.  
- Enhance export options including interactive HTML reports and direct model deployment.

---

## License

Specify your open source license here (e.g., MIT, Apache 2.0).

---

## Acknowledgments

Thanks to the developers of Streamlit, Hugging Face transformers, Ollama LLMs, and all open-source libraries powering this assistant.

---

## Contact

For questions or contributions, please raise issues or pull requests on the GitHub repository.

---
