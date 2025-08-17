def detect_intent(query):
    q = query.lower()
    if "mean" in q or "average" in q: return "summary_stats"
    if "trend" in q: return "trend_analysis"
    if "correlation" in q: return "correlation"
    if "model" in q or "predict" in q: return "ml_model"
    return "general_question"
