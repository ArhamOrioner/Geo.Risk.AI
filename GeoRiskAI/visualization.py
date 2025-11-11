# -----------------------------------------------------------------------------
# GeoRiskAI - Honest & Advanced Visualization Module
#
# CRITICAL FIX (Investor Mandate #2): This module has been completely
# overhauled to be consistent with the new unified architecture. All plots
# and maps are now derived from the single, authoritative `Final_Risk_Score`
# and its associated data. Obsolete plots have been removed, and advanced
# Plotly visualizations have been reinstated and improved.
# -----------------------------------------------------------------------------

import logging
import pandas as pd
import numpy as np
import folium
from folium import plugins
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

def create_eda_plots(df, output_dir):
    """
    Creates an interactive Plotly dashboard that is HONEST and reflects
    the current, unified architecture.
    """
    logging.info("Creating unified, interactive EDA dashboard...")

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            "<b>Risk Score Distribution</b>",
            "<b>Partial Dependence (PDP): Selected Features</b>",
            "<b>Risk Zone Distribution</b>",
            "<b>Land Cover Composition</b>",
            "<b>Uncertainty (Interval Width) Distribution</b>"
        ),
        specs=[[{"type": "histogram"}, {"type": "scatter"}, {"type": "histogram"}],
               [{"type": "bar"}, {"type": "pie"}, {"type": "table"}]]
    )

    # 1. Risk Distribution
    fig.add_trace(
        go.Histogram(x=df['Final_Risk_Score'], name='Risk Score', marker_color='#3f51b5'),
        row=1, col=1
    )

    # 2. PDP-like visualization (approximate using binning)
    def add_pdp(ax_row, ax_col, feature_name):
        if feature_name not in df.columns:
            return
        try:
            bins = np.linspace(df[feature_name].quantile(0.01), df[feature_name].quantile(0.99), 20)
            binned = pd.cut(df[feature_name], bins=bins, include_lowest=True)
            pdp = df.groupby(binned)['Final_Risk_Score'].mean().reset_index()
            centers = [c.mid for c in pdp[feature_name].cat.categories]
            fig.add_trace(go.Scatter(x=centers, y=pdp['Final_Risk_Score'], mode='lines+markers', name=f"PDP {feature_name}"), row=ax_row, col=ax_col)
        except Exception:
            pass
    for f in ['Slope', 'Max_Daily_Precipitation', 'NDVI']:
        add_pdp(1, 2, f)

    # 3. Risk Zone Distribution
    if 'Risk_Zone' in df.columns:
        zone_counts = df['Risk_Zone'].value_counts()
        fig.add_trace(
            go.Bar(x=zone_counts.index, y=zone_counts.values, name='Zone Counts', marker_color='#1a237e'),
            row=2, col=1
        )

    # 4. Land Cover Distribution
    if 'Land_Cover_Type' in df.columns:
        land_cover_counts = df['Land_Cover_Type'].value_counts()
        fig.add_trace(
            go.Pie(labels=land_cover_counts.index, values=land_cover_counts.values, name='Land Cover'),
            row=2, col=2
        )

    # 5. Uncertainty distribution (conformal interval width)
    if 'Risk_Lower_90' in df.columns and 'Risk_Upper_90' in df.columns:
        width = (df['Risk_Upper_90'] - df['Risk_Lower_90']).clip(lower=0)
        fig.add_trace(
            go.Histogram(x=width, name='Interval Width', marker_color='#00897b'),
            row=1, col=3
        )
    elif 'Uncertainty' in df.columns:
        fig.add_trace(
            go.Histogram(x=df['Uncertainty'], name='Uncertainty', marker_color='#00897b'),
            row=1, col=3
        )

    fig.update_layout(
        title_text="<b>GeoRiskAI - Unified Analysis Dashboard</b>",
        height=900,
        showlegend=True,
        legend_title_text='Factors',
        font=dict(family="Arial, sans-serif")
    )
    dashboard_path = f'{output_dir}/unified_dashboard.html'
    fig.write_html(dashboard_path)
    logging.info(f"Unified dashboard saved to {dashboard_path}")


def create_interactive_map(df, output_dir):
    """Creates an interactive map visualizing the final, authoritative risk score."""
    logging.info("Creating interactive map with authoritative risk scores...")
    
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles="CartoDB positron")
    
    folium.TileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', attr='Esri', name='Satellite Imagery').add_to(m)

    # Use a heatmap for a more intuitive visualization of risk density
    heat_data = df[['latitude', 'longitude', 'Final_Risk_Score']].values.tolist()
    plugins.HeatMap(heat_data, radius=15, name="Risk Heatmap").add_to(m)

    # Add individual risk zone points for drill-down analysis
    if 'Risk_Zone' in df.columns:
        risk_colors = {'Low Risk': 'green', 'Moderate Risk': 'orange', 'High Risk': 'red', 'Very High Risk': 'purple', 'Extreme Risk': 'black'}
        sample_size = min(1000, len(df))
        df_sample = df.sample(n=sample_size, random_state=42)
        
        zones_group = folium.FeatureGroup(name="Risk Zones (Sampled Points)")
        for _, row in df_sample.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=3,
                popup=f"<b>Risk Zone: {row['Risk_Zone']}</b><br>Score: {row['Final_Risk_Score']:.3f}",
                color=risk_colors.get(row['Risk_Zone'], 'gray'),
                fill=True,
                fillOpacity=0.7
            ).add_to(zones_group)
        zones_group.add_to(m)

    folium.LayerControl().add_to(m)
    plugins.Fullscreen().add_to(m)
    
    map_path = f'{output_dir}/authoritative_risk_map.html'
    m.save(map_path)
    
    logging.info(f"Authoritative interactive map saved to {map_path}")
    return map_path
