#!/usr/bin/env python3
"""
Simple script to use the pretrained model for predictions.
This bypasses the training pipeline entirely.
"""

import os
import sys
import logging
import pandas as pd
from datetime import datetime

# Add current directory to path
sys.path.append('.')

import ML as ml_model

def load_model():
    """Load the pretrained model."""
    if not os.path.exists("1.pkl"):
        raise FileNotFoundError("Pretrained model file '1.pkl' not found in current directory")
    
    model = ml_model.ProductionRiskModel()
    model.load_pretrained_model("1.pkl")
    return model

def predict_flood_risk(latitude, longitude, event_date, severity=2.0):
    """
    Predict flood risk for a given location and date.
    
    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate  
        event_date: Event date as string (YYYY-MM-DD)
        severity: Event severity (default 2.0)
    
    Returns:
        Dictionary with prediction results
    """
    # Create sample data
    sample_data = pd.DataFrame({
        'latitude': [latitude],
        'longitude': [longitude],
        'Began': [event_date],
        'Severity': [severity],
        'Elevation': [0.0],  # Will be filled by GEE
        'Slope': [0.0],      # Will be filled by GEE
        'NDVI': [0.0]        # Will be filled by GEE
    })
    
    # Load model
    model = load_model()
    
    # Make prediction
    results = model.predict_per_pixel(sample_data)
    
    return {
        'risk_score': results['Final_Risk_Score'].iloc[0],
        'risk_probability': results['Risk_Probability'].iloc[0],
        'uncertainty': results['Uncertainty'].iloc[0],
        'risk_lower_90': results['Risk_Lower_90'].iloc[0],
        'risk_upper_90': results['Risk_Upper_90'].iloc[0]
    }

def main():
    """Example usage of the pretrained model."""
    print("=" * 60)
    print("GeoRiskAI Pretrained Model Usage Example")
    print("=" * 60)
    
    # Set up logging
    logging.basicConfig(level=logging.WARNING)
    
    try:
        # Example predictions
        locations = [
            (40.7128, -74.0060, "2024-06-15"),  # New York
            (34.0522, -118.2437, "2024-07-20"), # Los Angeles
            (51.5074, -0.1278, "2024-08-10"),  # London
        ]
        
        print("Making predictions for sample locations...")
        print()
        
        for lat, lon, date in locations:
            try:
                result = predict_flood_risk(lat, lon, date)
                print(f"Location: {lat:.4f}, {lon:.4f} on {date}")
                print(f"  Risk Score: {result['risk_score']:.4f}")
                print(f"  Risk Probability: {result['risk_probability']:.4f}")
                print(f"  Uncertainty: {result['uncertainty']:.4f}")
                print(f"  90% Confidence Interval: [{result['risk_lower_90']:.4f}, {result['risk_upper_90']:.4f}]")
                print()
            except Exception as e:
                print(f"  ❌ Prediction failed: {e}")
                print()
        
        print("✅ Pretrained model is working correctly!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
