# -----------------------------------------------------------------------------
# GeoRiskAI - Production-Grade Reporting Module
#
# CRITICAL FIX (Investor Mandate): This module now generates a report that
# is suitable for real stakeholders.
# 1. HONESTY: Clearly displays both risk scores and uncertainty intervals.
# 2. EXPLAINABILITY: Includes a chart of global feature importances from SHAP.
# 3. ACTIONABLE: Provides clear, data-driven summaries.
# 4. SAFETY: Includes a prominent disclaimer about the model's limitations.
# -----------------------------------------------------------------------------

import os
import logging
import json

def get_risk_level_class(score):
    """Determines the CSS class for a given risk score."""
    if score >= 0.8: return "very-high-risk"
    if score >= 0.6: return "high-risk"
    if score >= 0.4: return "moderate-risk"
    return "low-risk"

def generate_html_report(output_dir, config, report_summary, ai_narrative, map_path):
    """Generate a comprehensive, production-grade HTML report."""
    logging.info("Generating final, production-grade HTML report...")

    avg_score = report_summary.get('avg_risk_score', 0.5)
    risk_level_class = get_risk_level_class(avg_score)
    
    narrative_title = ai_narrative.get("narrative_title", "AI Analysis Incomplete")
    risk_explanation = ai_narrative.get("risk_explanation", "The AI was unable to provide a detailed explanation.")
    recommendations = ai_narrative.get("recommendations", [])
    recommendations_html = ''.join([f'<li>{rec}</li>' for rec in recommendations])

    # Create feature importance list from SHAP summary
    feature_importance_html = ""
    global_importance = report_summary.get('global_feature_importance', {})
    if global_importance:
        # Take top 7 for display
        for feature, importance in list(global_importance.items())[:7]:
            feature_importance_html += f'<li><strong>{feature.replace("_", " ").title()}:</strong> {importance:.3f}</li>'

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{config.PROJECT_NAME} - Flood Risk Assessment Report</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 20px; background-color: #f4f7f9; color: #333; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 25px rgba(0,0,0,0.08); }}
            .header h1 {{ color: #1a237e; margin: 0; font-size: 2.8em; font-weight: 700; }}
            .header p {{ color: #555; margin: 10px 0 0 0; font-size: 1.1em; }}
            .section {{ margin: 35px 0; }}
            .section h2 {{ color: #1a237e; margin-top: 0; border-bottom: 2px solid #3f51b5; padding-bottom: 10px; font-size: 1.8em; }}
            .risk-score-card {{ text-align: center; padding: 20px; border-radius: 10px; color: white; }}
            .risk-score-card h3 {{ margin: 0 0 10px 0; font-size: 1.2em; text-transform: uppercase; letter-spacing: 1px; }}
            .risk-score-card .score {{ font-size: 3.5em; font-weight: 800; }}
            .risk-score-card .uncertainty {{ font-size: 1em; opacity: 0.8; }}
            .high-risk {{ background: linear-gradient(135deg, #d32f2f, #b71c1c); }}
            .moderate-risk {{ background: linear-gradient(135deg, #f57c00, #e65100); }}
            .low-risk {{ background: linear-gradient(135deg, #388e3c, #1b5e20); }}
            .very-high-risk {{ background: linear-gradient(135deg, #6a1b9a, #4a148c); }}
            .grid-container {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 25px; margin-top: 20px; }}
            .card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 5px solid #3f51b5; }}
            .card h3 {{ margin: 0 0 15px 0; color: #1a237e; }}
            .card p, .card li {{ line-height: 1.6; }}
            .card ul {{ list-style: none; padding: 0; }}
            .card ul li {{ background: #e8eaf6; margin: 8px 0; padding: 12px; border-radius: 5px; }}
            .map-container {{ margin-top: 20px; height: 550px; border-radius: 10px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
            .map-container iframe {{ width: 100%; height: 100%; border: none; }}
            .disclaimer {{ background-color: #fff3e0; border: 1px solid #ffe0b2; border-left: 5px solid #ff9800; padding: 15px; margin-top: 20px; border-radius: 8px; }}
            .footer {{ text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #e0e0e0; color: #777; }}
        </style>
    </head>
    <body>
        <div class="container">
            <header class="header">
                <h1>{config.PROJECT_NAME}</h1>
                <p>Scientific Flood Risk Assessment Report (Provisional)</p>
            </header>

            <div class="disclaimer">
                <h3>IMPORTANT DISCLAIMER</h3>
                <p>This report is generated by a machine learning model and is intended for strategic planning and risk identification purposes only. <strong>Current NRT data is partial: only IMERG precipitation is fully integrated; GloFAS and VIIRS NRT are in progress. Training data is based on the Global Flood Database (GFD) up to 2018 and does not include recent (2024–2025) extreme events. Model calibration against new events is critical before operational use.</strong> The model is trained on simulated and historical data and has inherent limitations. Predictions must be validated with on-the-ground surveys and professional engineering assessments before any critical infrastructure or policy decisions are made. <strong>DO NOT use this report as a sole source for decisions impacting life or property.</strong></p>
            </div>

            <section class="section">
                <h2>Executive Summary: {narrative_title}</h2>
                <div class="grid-container">
                    <div class="risk-score-card {risk_level_class}">
                        <h3>Average Risk Score</h3>
                        <div class="score">{avg_score:.3f}</div>
                        <div class="uncertainty">Avg. Interval Width (90%): {report_summary.get('avg_uncertainty', 0):.3f}</div>
                    </div>
                    <div class="card">
                        <h3>Provisional Interpretation (AI)</h3>
                        <p>{risk_explanation}</p>
                    </div>
                    <div class="card">
                        <h3>Data Coverage</h3>
                        <p>Requested pixels: {report_summary.get('requested_pixels', 'N/A')}<br/>
                           Returned pixels: {report_summary.get('returned_pixels', 'N/A')}<br/>
                           Coverage ratio: {report_summary.get('coverage_ratio', 'N/A')}</p>
                        <p><em>Note:</em> If coverage ratio is low (e.g., due to clouds), this report may be incomplete; analysis will halt below 0.8.</p>
                    </div>
                </div>
            </section>

            <section class="section">
                <h2>Key Findings & Recommendations</h2>
                <div class="grid-container">
                    <div class="card">
                        <h3>Global Feature Importance (SHAP)</h3>
                        <p>The primary drivers of risk across the entire region, as determined by the model:</p>
                        <ul>{feature_importance_html}</ul>
                    </div>
                    <div class="card">
                        <h3>Operational Priority Guide</h3>
                        <p>The system converts probabilities and uncertainty into actionable categories to guide field ops:</p>
                        <ul>
                            <li><strong>High risk (confident)</strong>: p ≥ threshold and low uncertainty — prioritize immediate mitigation.</li>
                            <li><strong>High risk (uncertain)</strong>: p ≥ threshold and higher uncertainty — prioritize rapid verification and deploy sensors/spotters.</li>
                            <li><strong>Investigate (uncertain)</strong>: p < threshold and high uncertainty — schedule site survey; improve data coverage.</li>
                            <li><strong>Low risk (confident)</strong>: monitor with routine checks.</li>
                        </ul>
                        <p><em>Threshold</em> is calibrated via precision-recall analysis. Current operating threshold: {report_summary.get('operating_threshold', 0.5):.2f}.</p>
                        <p>Uncertainty reflects <strong>conformal prediction intervals</strong> around probabilities at alpha = {report_summary.get('conformal_alpha', 0.1)} (nominal 90% coverage). Interval width is used to flag uncertainty. If empirical coverage diverges from target in backtests, <strong>treat alerts as provisional</strong> pending recalibration/NRT updates.</p>
                        <p>All inputs were resampled to a common analysis resolution of <strong>{report_summary.get('analysis_resolution_m', 'native')}</strong> meters. This is the limit of the model's spatial precision.</p>
                        <p><strong>Limitations</strong>: Training data currently covers 2000–2018 GFD events; calibration against recent extremes is required before operational use. API decay constant and coverage target are configurable and should be tuned for the region.</p>
                    </div>
                    <div class="card">
                        <h3>Actionable Recommendations</h3>
                        <p>Based on the AI's analysis, the following actions are recommended:</p>
                        <ul>{recommendations_html}</ul>
                    </div>
                </div>
            </section>

            <section class="section">
                <h2>Interactive Risk Map & Dashboard</h2>
                 <p>The map visualizes probability of high-severity flooding via a heatmap. For detailed data, view the <a href="unified_dashboard.html" target="_blank">Interactive Dashboard</a>.</p>
                <div class="map-container">
                    <iframe src="{os.path.basename(map_path)}"></iframe>
                </div>
            </section>

            <footer class="footer">
                <p><strong>{config.PROJECT_NAME}</strong> | Production-Grade Architecture</p>
                <p>Uncertainty method: Conformal Prediction Intervals. Coverage target: {(1 - report_summary.get('conformal_alpha', 0.1)):.0%}. Interval width summarizes uncertainty. Spatial precision limited to analysis resolution: {report_summary.get('analysis_resolution_m', 'native')} meters.</p>
                <p><strong>Critical Note:</strong> Training data historically covers 2000–2018 (GFD). Near real-time augmentation is <strong>partial</strong> (currently IMERG precipitation only; GloFAS/VIIRS integration pending). Calibration against 2024–2025 events is required prior to operational deployment.</p>
            </footer>
        </div>
    </body>
    </html>
    """
    
    report_path = os.path.join(output_dir, f'{config.PROJECT_NAME}_Report.html')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logging.info(f"Production-grade HTML report successfully saved to {report_path}")
    return report_path
