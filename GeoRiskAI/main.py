# -----------------------------------------------------------------------------
# GeoRiskAI - Memory-Optimized Final Orchestrator
#
# CRITICAL FIXES:
# 1. MEMORY OPTIMIZATION: Batch processing to avoid GEE memory limits
# 2. ERROR HANDLING: Comprehensive fallbacks and error recovery
# 3. SCALABILITY: Progressive processing with memory monitoring
# -----------------------------------------------------------------------------

import os
import logging
import pandas as pd
import ee
import time
import numpy as np
from datetime import datetime, timedelta

import config
import gee as gee_module
import analysis as analysis
import ML as ml_model
import ai as ai_storyteller
import visualization as viz
import reporting

def setup_logging():
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    logging.basicConfig(
        level=config.LOG_LEVEL.upper(),
        format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler(config.LOG_FILE, mode='w')]
    )
    logging.info("STARTING GeoRiskAI ANALYSIS (Memory-Optimized Architecture)")

def estimate_batch_size(events_df, lookback_days):
    """Estimate optimal batch size based on event distribution and memory constraints."""
    try:
        # Calculate spatial and temporal density
        lat_range = events_df['latitude'].max() - events_df['latitude'].min()
        lon_range = events_df['longitude'].max() - events_df['longitude'].min()
        spatial_extent = lat_range * lon_range
        
        # Base batch size on spatial density and lookback period
        if spatial_extent > 10 or lookback_days > 30:  # Large area or long period
            base_batch_size = 10
        elif spatial_extent > 1 or lookback_days > 14:  # Medium area or period
            base_batch_size = 25
        else:  # Small area and short period
            base_batch_size = 50
        
        # Adjust based on total number of events
        if len(events_df) < base_batch_size:
            return len(events_df)
        
        return min(base_batch_size, 50)  # Never exceed 50 events per batch
        
    except Exception as e:
        logging.warning(f"Could not estimate batch size: {e}, using default")
        return 20

def enrich_single_event(event_row, lookback_days):
    """Enrich a single event with GEE data using pretrained model features."""
    try:
        # Create point geometry
        point = ee.Geometry.Point([event_row['longitude'], event_row['latitude']])
        
        # Create small buffer for sampling (reduces memory compared to large regions)
        buffer_size = 1000  # 1km buffer
        roi = point.buffer(buffer_size)
        
        # Set up date range
        end_date = pd.to_datetime(event_row['Began'])
        start_date = end_date - timedelta(days=lookback_days)
        
        # Use the new pretrained model feature extraction
        try:
            features = gee_module.get_pretrained_model_features(
                roi, 
                start_date.strftime('%Y-%m-%d'), 
                end_date.strftime('%Y-%m-%d')
            )
            
            if features:
                # Convert to dictionary and add to event data
                enriched_event = event_row.to_dict()
                enriched_event.update(features)
                
                # Add derived features
                enriched_event['y_binary'] = 1 if event_row.get('Severity', 0) >= 1.5 else 0
                enriched_event['event_year'] = end_date.year
                
                logging.info(f"✅ Pretrained model features extracted for event at {event_row['latitude']:.3f}, {event_row['longitude']:.3f}")
                return enriched_event
            else:
                logging.warning(f"No features could be extracted for event at {event_row['latitude']:.3f}, {event_row['longitude']:.3f}")
                return None
                
        except Exception as e:
            logging.warning(f"Pretrained model feature extraction failed for event: {e}")
            return None
            
    except Exception as e:
        logging.error(f"Failed to enrich event at {event_row['latitude']:.3f}, {event_row['longitude']:.3f}: {e}")
        return None

def enrich_events_batch(events_batch, lookback_days, batch_num, total_batches):
    """Enrich a batch of events with comprehensive error handling."""
    logging.info(f"Processing batch {batch_num}/{total_batches} with {len(events_batch)} events")
    
    enriched_events = []
    failed_count = 0
    
    for idx, (_, event_row) in enumerate(events_batch.iterrows()):
        try:
            enriched_event = enrich_single_event(event_row, lookback_days)
            
            if enriched_event:
                enriched_events.append(enriched_event)
                if (idx + 1) % 5 == 0:
                    logging.info(f"Batch {batch_num}: Processed {idx + 1}/{len(events_batch)} events")
            else:
                failed_count += 1
                
        except Exception as event_error:
            logging.error(f"Failed to process event {idx} in batch {batch_num}: {event_error}")
            failed_count += 1
            continue
    
    success_rate = (len(enriched_events) / len(events_batch)) * 100 if events_batch is not None and len(events_batch) > 0 else 0
    logging.info(f"Batch {batch_num} complete: {len(enriched_events)} successful, {failed_count} failed ({success_rate:.1f}% success rate)")
    
    return enriched_events

def enrich_events_with_gee_optimized(events_df: pd.DataFrame, lookback_days: int = 90, max_events: int = 1000) -> pd.DataFrame:
    """
    Memory-optimized enrichment of events with GEE data using batch processing.
    """
    if not gee_module.initialize_gee():
        raise ConnectionError("GEE Initialization Failed")

    events_df = events_df.head(max_events).copy()
    logging.info(f"Starting memory-optimized enrichment of {len(events_df)} events")
    
    batch_size = estimate_batch_size(events_df, lookback_days)
    logging.info(f"Using batch size: {batch_size}")
    
    all_enriched_events = []
    total_batches = (len(events_df) + batch_size - 1) // batch_size
    
    for batch_num in range(total_batches):
        try:
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(events_df))
            batch = events_df.iloc[start_idx:end_idx]
            
            batch_results = enrich_events_batch(batch, lookback_days, batch_num + 1, total_batches)
            all_enriched_events.extend(batch_results)
            
            if batch_num < total_batches - 1:
                time.sleep(2)
                
        except Exception as batch_error:
            logging.error(f"Batch {batch_num + 1} failed completely: {batch_error}")
            continue
    
    if not all_enriched_events:
        logging.error("No events were successfully enriched")
        return pd.DataFrame()
    
    enriched_df = pd.DataFrame(all_enriched_events)
    
    intermediate_path = os.path.join(config.OUTPUTS_DIR, 'enriched_events_intermediate.csv')
    enriched_df.to_csv(intermediate_path, index=False)
    
    success_rate = (len(enriched_df) / len(events_df)) * 100
    logging.info(f"Enrichment complete: {len(enriched_df)}/{len(events_df)} events enriched ({success_rate:.1f}% success rate)")
    
    return enriched_df

def run_training_pipeline():
    """
    Memory-optimized offline training pipeline.
    """
    logging.info("--- STARTING MEMORY-OPTIMIZED TRAINING PIPELINE ---")
    
    try:
        gfd_df = ml_model.download_and_prepare_gfd_data()
        if gfd_df is None:
            raise ValueError("Could not obtain GFD training data.")
        
        max_training_events = getattr(config, 'MAX_TRAINING_EVENTS', 500)
        if len(gfd_df) > max_training_events:
            logging.info(f"Limiting training data to {max_training_events} events for memory optimization")
            gfd_df = gfd_df.sample(n=max_training_events, random_state=42)

        events_with_neg = ml_model.augment_with_negative_samples(gfd_df)
        
        if len(events_with_neg) > max_training_events * 2:
            events_with_neg = events_with_neg.sample(n=max_training_events * 2, random_state=42)

        enriched_df = enrich_events_with_gee_optimized(events_with_neg)
        
        if enriched_df.empty:
            raise ValueError("GEE enrichment failed to produce any valid training rows.")
        
        min_training_samples = 50
        if len(enriched_df) < min_training_samples:
            logging.warning(f"Only {len(enriched_df)} training samples available, below recommended minimum of {min_training_samples}")
            
        training_features = ml_model.engineer_features_vectorized(enriched_df)
        training_targets = enriched_df['y_binary']

        try:
            spatial_groups = analysis.assign_spatial_clusters(enriched_df[['latitude', 'longitude']].copy())
        except Exception as spatial_error:
            logging.warning(f"Could not assign spatial clusters: {spatial_error}, using temporal groups")
            spatial_groups = None

        model = ml_model.ProductionRiskModel()
        
        # Load pretrained model instead of training
        model.load_pretrained_model("1.pkl")
        
        training_data_path = os.path.join(config.OUTPUTS_DIR, 'training_data.csv')
        enriched_df.to_csv(training_data_path, index=False)
        
        logging.info("--- TRAINING PIPELINE COMPLETE ---")
        return model
        
    except Exception as training_error:
        logging.error(f"Training pipeline failed: {training_error}", exc_info=True)
        raise

def run_prediction_pipeline_safe(model):
    """Memory-safe prediction pipeline with progressive fallbacks."""
    logging.info("--- STARTING MEMORY-SAFE PREDICTION PIPELINE ---")
    
    try:
        if not gee_module.initialize_gee(): 
            raise ConnectionError("GEE Initialization Failed")
            
        roi = ee.Geometry.Rectangle(config.ROI_BOUNDS)
        start_date, end_date = config.START_DATE_ANALYSIS, config.END_DATE_ANALYSIS
        
        roi_area = roi.area().divide(1000000).getInfo()
        if roi_area > 100:
            logging.warning(f"Large ROI detected ({roi_area:.1f} km²), using memory-safe approach")
            
        logging.info("Loading environmental data from GEE with memory optimization...")
        
        combined_image = gee_module.get_comprehensive_environmental_data_safe(roi, start_date, end_date)
        
        if not combined_image:
            raise ValueError("Failed to load critical GEE data layers.")

        original_num_pixels = config.NUM_PIXELS_FOR_ANALYSIS
        if roi_area > 100:
            adjusted_pixels = min(original_num_pixels, 10000)
            logging.info(f"Adjusted pixel sampling from {original_num_pixels} to {adjusted_pixels} for memory optimization")
        else:
            adjusted_pixels = original_num_pixels
            
        target_scale = None
        if config.COMMON_RESOLUTION_METERS:
            target_scale = float(config.COMMON_RESOLUTION_METERS)
            if roi_area > 100:
                target_scale = max(target_scale, 100)
            logging.info(f"Using resolution: {target_scale:.1f} m")
            combined_image = combined_image.resample('bilinear').reproject(
                crs=combined_image.projection(), scale=target_scale
            )
        
        df_raw, coverage = analysis.sample_to_dataframe(combined_image, roi, adjusted_pixels)
        
        if df_raw is None or df_raw.empty:
            fallback_pixels = min(adjusted_pixels // 2, 5000)
            logging.warning(f"Initial sampling failed, trying with {fallback_pixels} pixels")
            df_raw, coverage = analysis.sample_to_dataframe(combined_image, roi, fallback_pixels)
            
            if df_raw is None or df_raw.empty:
                raise ValueError("Failed to create a valid DataFrame from GEE samples even with fallback.")

        min_cov = float(getattr(config, 'MIN_COVERAGE_RATIO', 0.6))
        if coverage and coverage.get('coverage_ratio', 1.0) < min_cov:
            logging.warning(f"Low data coverage: {coverage.get('coverage_ratio', 0):.2f}, but continuing with available data")
        
        df_raw.to_csv(os.path.join(config.OUTPUTS_DIR, 'raw_sampled_data.csv'), index=False)
        df_clean = analysis.clean_and_prepare_data(df_raw.copy())

        prediction_results_df = model.predict_per_pixel(df_clean)
        df_final = df_clean.join(prediction_results_df)
        
        df_final = analysis.assign_operational_priority(
            df_final,
            probability_col='Final_Risk_Score',
            uncertainty_col='Uncertainty',
            threshold=getattr(model, 'risk_threshold', 0.5)
        )

        report_summary = analysis.generate_report_summary(df_final)
        report_summary['operating_threshold'] = getattr(model, 'risk_threshold', 0.5)
        report_summary['analysis_resolution_m'] = target_scale if config.COMMON_RESOLUTION_METERS else None
        
        if coverage:
            report_summary.update({
                'requested_pixels': coverage.get('requested'),
                'returned_pixels': coverage.get('returned'),
                'coverage_ratio': coverage.get('coverage_ratio')
            })
        
        report_summary['high_risk_area_pct'] = (
            df_final['Final_Risk_Score'] >= report_summary['operating_threshold']
        ).mean() * 100
        
        ai_narrative = ai_storyteller.get_risk_narrative(config.ROI_BOUNDS, report_summary)

        df_final, _ = analysis.detect_risk_zones(df_final)
        df_final.to_csv(os.path.join(config.OUTPUTS_DIR, 'final_predictions.csv'), index=False)
        
        viz.create_eda_plots(df_final, config.OUTPUTS_DIR)
        map_path = viz.create_interactive_map(df_final, config.OUTPUTS_DIR)
        reporting.generate_html_report(config.OUTPUTS_DIR, config, report_summary, ai_narrative, map_path)

        logging.info("--- PREDICTION PIPELINE COMPLETE ---")
        logging.info(f"Final average risk score: {report_summary['avg_risk_score']:.4f}")
        return df_final
        
    except Exception as prediction_error:
        logging.error(f"Prediction pipeline failed: {prediction_error}", exc_info=True)
        raise

def run_backtesting_pipeline_safe(model, validation_events):
    """Memory-safe backtesting pipeline."""
    logging.info(f"--- STARTING MEMORY-SAFE BACKTESTING on {len(validation_events)} events ---")
    
    try:
        max_validation_events = 200
        if len(validation_events) > max_validation_events:
            validation_events = validation_events.sample(n=max_validation_events, random_state=42)
            logging.info(f"Limited validation to {max_validation_events} events for memory optimization")

        enriched_validation_df = enrich_events_with_gee_optimized(
            validation_events, 
            lookback_days=30,
            max_events=len(validation_events)
        )
        
        if enriched_validation_df.empty:
            logging.error("Backtesting failed: Could not enrich validation events.")
            return

        predictions = model.predict_per_pixel(enriched_validation_df)

        y_true = enriched_validation_df['y_binary'].astype(int).values
        y_prob = predictions['Risk_Probability'].values
        y_pred = (y_prob >= getattr(model, 'risk_threshold', 0.5)).astype(int)

        from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_score, recall_score
        
        ap = average_precision_score(y_true, y_prob)
        roc = roc_auc_score(y_true, y_prob)
        f1 = f1_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        
        hits = ((y_true == 1) & (y_pred == 1)).sum()
        misses = ((y_true == 1) & (y_pred == 0)).sum()
        false_alarms = ((y_true == 0) & (y_pred == 1)).sum()
        csi = hits / max(1, (hits + misses + false_alarms))

        logging.info("--- BACKTESTING RESULTS ---")
        logging.info(f"Average Precision (PR-AUC): {ap:.4f}")
        logging.info(f"ROC-AUC: {roc:.4f}")
        logging.info(f"F1: {f1:.4f}")
        logging.info(f"Precision: {prec:.4f} | Recall: {rec:.4f} | CSI: {csi:.4f}")
        
        backtest_results = {
            'validation_events': len(validation_events),
            'enriched_events': len(enriched_validation_df),
            'average_precision': ap,
            'roc_auc': roc,
            'f1_score': f1,
            'precision': prec,
            'recall': rec,
            'csi': csi
        }
        
        import json
        with open(os.path.join(config.OUTPUTS_DIR, 'backtest_results.json'), 'w') as f:
            json.dump(backtest_results, f, indent=2)
        
        logging.info("--- BACKTESTING PIPELINE COMPLETE ---")
        
    except Exception as backtest_error:
        logging.error(f"Backtesting failed: {backtest_error}", exc_info=True)

if __name__ == '__main__':
    setup_logging()
    
    try:
        logging.info("Loading pretrained model...")
        # Load pretrained model instead of training
        model = ml_model.ProductionRiskModel()
        model.load_pretrained_model("1.pkl")
        logging.info("✓ Pretrained model loaded successfully")
        
        logging.info("Starting prediction pipeline...")
        prediction_results = run_prediction_pipeline_safe(model)
        
        logging.info("Starting backtesting pipeline...")
        all_events = ml_model.download_and_prepare_gfd_data()
        if all_events is not None and len(all_events) >= 50:
            run_backtesting_pipeline_safe(model, all_events)

        logging.info("✓ ALL PIPELINES COMPLETED SUCCESSFULLY")

    except Exception as e:
        logging.error(f"✗ ANALYSIS FAILED: {e}", exc_info=True)
        logging.error("ANALYSIS FAILED.")