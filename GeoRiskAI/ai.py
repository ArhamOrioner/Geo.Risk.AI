# -----------------------------------------------------------------------------
# GeoRiskAI - The Final AI Storyteller
#
# CRITICAL OVERHAUL (Final Investor Mandate): This module's prompt is now
# more advanced, data-driven, and responsible. It receives aggregated risk
# statistics and SHAP values to tell a nuanced story, while acknowledging
# model uncertainty.
# -----------------------------------------------------------------------------

import logging
import json
import os
import google.generativeai as genai

def configure_genai(api_key):
    """Configures the Google Generative AI SDK."""
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        logging.error(f"Failed to configure Google Generative AI: {e}", exc_info=True)
        return False

def get_risk_narrative(roi_bounds, report_summary):
    """Generates a qualitative risk narrative using aggregated per-pixel results."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or not configure_genai(api_key):
        logging.error("Cannot generate AI narrative due to API key/configuration issue.")
        return {"narrative_title": "AI Analysis Unavailable", "risk_explanation": "Could not connect to the AI service.", "recommendations": []}

    logging.info("Generating final, data-driven AI risk narrative...")
    model = genai.GenerativeModel('gemini-1.5-flash-latest')

    # Prepare SHAP summary for the prompt
    shap_summary = ", ".join([f'{k.replace("_", " ").title()} ({v:.2f})' for k, v in list(report_summary.get('global_feature_importance', {}).items())[:3]])

    operating_threshold = report_summary.get('operating_threshold', 0.5)
    prompt = f"""
    You are an expert environmental risk analyst from a top-tier consultancy, writing the executive summary for a client report. Be professional, data-driven, and cautious.

    **Analysis Subject:** Flood and Runoff Risk for the region defined by bounding box {roi_bounds}.

    **Key Quantitative Findings from our Per-Pixel ML Model:**
    - **Average Risk Score:** {report_summary.get('avg_risk_score', 0):.3f} (90% Confidence Interval: [{report_summary.get('avg_risk_lower_90', 0):.3f} - {report_summary.get('avg_risk_upper_90', 0):.3f}])
    - **Maximum Risk Score Detected:** {report_summary.get('max_risk_score', 0):.3f}
    - **High-Risk Area:** Approximately {report_summary.get('high_risk_area_pct', 0):.1f}% of the area is classified as high risk (score ≥ {operating_threshold:.2f}).
    - **Primary Risk Drivers (from SHAP analysis):** {shap_summary}

    **Your Task:**
    Based on this data, generate a concise and insightful JSON object for the report. Do not include any text outside of the JSON object.

    {{
      "narrative_title": "<A professional, compelling title for the summary>",
      "risk_explanation": "<A multi-sentence paragraph. Start with a clear statement about the overall risk level based on the average score and its confidence interval. Discuss the significance of the spatial variance (the difference between the average and max scores, and the percentage of high-risk area). Explain how the primary risk drivers likely contribute to these findings.>",
      "recommendations": [
        "<A specific, actionable recommendation targeting the {report_summary.get('high_risk_area_pct', 0):.1f}% high-risk area (defined using a calibrated operating threshold) for immediate on-the-ground surveys...>",
        "<A second recommendation based on the top SHAP risk drivers, e.g., 'Given the high impact of [Top SHAP Factor], focus mitigation efforts on...'>",
        "<A third recommendation addressing policy, early warning systems, or the need for further validation given the model's uncertainty.>"
      ]
    }}
    """

    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip().replace("```json", "").replace("```", "")
        ai_result = json.loads(response_text)
        logging.info("Successfully generated final AI risk narrative.")
        return ai_result
    except Exception as e:
        logging.error(f"Error generating AI narrative: {e}", exc_info=True)
        return {"narrative_title": "AI Analysis Incomplete", "risk_explanation": "The AI failed to generate a complete narrative.", "recommendations": []}
