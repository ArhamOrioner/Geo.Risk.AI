# GeoRiskAI - Memory-Optimized GEE Data Miner (FIXED)
# Fixed band-mismatch and zero-band Image.unmask error by enforcing
# a consistent band schema and combining per-band mosaics across layers.

import ee
import logging
from datetime import timedelta, datetime
import config as _cfg
import math


IMERG_COLLECTION = ee.ImageCollection("NASA/GPM_L3/IMERG_V07").select("precipitation")
SMAP_COLLECTION = ee.ImageCollection("NASA/SMAP/SPL4SMGP/007")
SMAP_SURFACE_BAND = "sm_surface_analysis"
SMAP_SUBSURFACE_BAND = "sm_rootzone_analysis"
GFS_COLLECTION = ee.ImageCollection("NOAA/GFS0P25")
GSW_OCCURRENCE_IMAGE = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").rename("gee_gsw_occurrence_mean")
MERIT_UPA_IMAGE = ee.Image("MERIT/Hydro/v1_0_1").select("upa").rename("gee_merit_upa_mean")
GLOFAS_FORECAST_BANDS = (
    "glofas_forecast_control",
    "glofas_forecast_mean",
    "glofas_forecast_median",
    "glofas_forecast_std_dev",
    "glofas_forecast_10th_percentile",
    "glofas_forecast_90th_percentile",
)
GLOFAS_EMPTY_IMAGE = ee.Image.constant([0.0] * len(GLOFAS_FORECAST_BANDS)).rename(list(GLOFAS_FORECAST_BANDS))
_GLOFAS_WARNING_EMITTED = False


def _empty_band(name: str, value: float = 0.0) -> ee.Image:
    return ee.Image.constant(value).rename(name)


def _safe_image(expr: ee.ComputedObject, fallback: ee.Image) -> ee.Image:
    return ee.Image(ee.Algorithms.If(ee.Algorithms.IsEqual(expr, None), fallback, expr))


def build_imerg_feature_image(start_date: str, end_date: str) -> ee.Image:
    end = ee.Date(end_date)
    start_24h = end.advance(-1, 'day')
    start_3d = end.advance(-3, 'day')
    start_7d = end.advance(-7, 'day')

    coll_24h = IMERG_COLLECTION.filterDate(start_24h, end)
    coll_3d = IMERG_COLLECTION.filterDate(start_3d, end)
    coll_7d = IMERG_COLLECTION.filterDate(start_7d, end)

    sum_24h = ee.Image(ee.Algorithms.If(
        coll_24h.size().gt(0),
        coll_24h.sum().multiply(0.5),
        ee.Image.constant(0)
    )).rename("gee_imerg_sum_24h_mm")

    sum_3d = ee.Image(ee.Algorithms.If(
        coll_3d.size().gt(0),
        coll_3d.sum().multiply(0.5),
        ee.Image.constant(0)
    )).rename("gee_imerg_sum_3d_before_mm")

    sum_7d = ee.Image(ee.Algorithms.If(
        coll_7d.size().gt(0),
        coll_7d.sum().multiply(0.5),
        ee.Image.constant(0)
    )).rename("gee_imerg_sum_7d_before_mm")

    max_1h_raw = ee.Image(ee.Algorithms.If(
        coll_24h.size().gt(0),
        coll_24h.max(),
        ee.Image.constant(0)
    ))

    max_1h = max_1h_raw.rename("gee_imerg_max_1h_intensity_mm")
    max_1h_dup = max_1h_raw.rename("imerg_max_1h_intensity_mm")

    mean_1h = ee.Image(ee.Algorithms.If(
        coll_24h.size().gt(0),
        coll_24h.mean(),
        ee.Image.constant(0)
    )).rename("imerg_mean_1h_intensity_mm")

    max_3h = max_1h_raw.multiply(1.5).rename("gee_imerg_max_3h_intensity_mm")
    max_6h = max_1h_raw.multiply(2.0).rename("gee_imerg_max_6h_intensity_mm")

    intensity_3d = sum_3d.divide(3.0).rename("gee_imerg_intensity_3d_mm_per_day")
    intensity_7d = sum_7d.divide(7.0).rename("gee_imerg_intensity_7d_mm_per_day")

    daily_sequence = ee.List.sequence(1, 7)

    def _daily_sum(day):
        day = ee.Number(day)
        day_start = end.advance(day.multiply(-1), 'day')
        day_end = day_start.advance(1, 'day')
        coll = IMERG_COLLECTION.filterDate(day_start, day_end)
        return ee.Image(ee.Algorithms.If(
            coll.size().gt(0),
            coll.sum().multiply(0.5),
            ee.Image.constant(0)
        )).rename("daily_sum")

    daily_collection = ee.ImageCollection(daily_sequence.map(_daily_sum))
    max_daily_7d = ee.Image(ee.Algorithms.If(
        daily_collection.size().gt(0),
        daily_collection.max(),
        ee.Image.constant(0)
    )).rename("gee_imerg_max_daily_7d_mm")

    imerg_features = ee.Image.cat([
        sum_24h,
        sum_3d,
        sum_7d,
        max_1h,
        max_1h_dup,
        max_3h,
        max_6h,
        mean_1h,
        intensity_3d,
        intensity_7d,
        max_daily_7d,
    ])

    return imerg_features


def build_smap_feature_image(start_date: str, end_date: str) -> ee.Image:
    end = ee.Date(end_date)
    current_start = end.advance(-3, 'day')
    climatology_start = end.advance(-60, 'day')
    climatology_end = end.advance(-30, 'day')

    current = SMAP_COLLECTION.filterDate(current_start, end)
    climatology = SMAP_COLLECTION.filterDate(climatology_start, climatology_end)

    surface = ee.Image(ee.Algorithms.If(
        current.size().gt(0),
        current.select(SMAP_SURFACE_BAND).mean(),
        ee.Image.constant(0)
    )).rename("gee_smap_surface_soil_moisture")

    subsurface = ee.Image(ee.Algorithms.If(
        current.size().gt(0),
        current.select(SMAP_SUBSURFACE_BAND).mean(),
        ee.Image.constant(0)
    )).rename("gee_smap_subsurface_soil_moisture")

    climatology_surface = ee.Image(ee.Algorithms.If(
        climatology.size().gt(0),
        climatology.select(SMAP_SURFACE_BAND).mean(),
        ee.Image.constant(0)
    ))

    anomaly = surface.subtract(climatology_surface).rename("gee_smap_soil_moisture_anomaly")

    return ee.Image.cat([surface, subsurface, anomaly])


def build_engineered_features_image(
    imerg_img: ee.Image,
    smap_img: ee.Image,
    topo_img: ee.Image,
    upa_img: ee.Image,
) -> ee.Image:
    sum_24h = imerg_img.select("gee_imerg_sum_24h_mm")
    sum_3d = imerg_img.select("gee_imerg_sum_3d_before_mm")
    sum_7d = imerg_img.select("gee_imerg_sum_7d_before_mm")
    max_1h = imerg_img.select("gee_imerg_max_1h_intensity_mm")
    slope = topo_img.select("gee_slope_mean")
    surface_sm = smap_img.select("gee_smap_surface_soil_moisture")
    upa = upa_img.select("gee_merit_upa_mean")

    api = sum_24h.multiply(0.5).add(sum_3d.multiply(0.3)).add(sum_7d.multiply(0.2)).rename("gee_api_weighted_mm")

    flashiness = max_1h.divide(sum_24h.divide(24.0).add(1e-6)).rename("gee_flashiness_index")
    flashiness_7d = max_1h.divide(sum_7d.divide(7.0).add(1e-6)).rename("gee_flashiness_index_7d")

    saturation = surface_sm.divide(sum_7d.add(1.0)).rename("gee_saturation_proxy")

    runoff = sum_24h.multiply(slope).divide(surface_sm.add(0.01)).rename("gee_runoff_potential")

    precip_slope = sum_7d.multiply(slope).rename("gee_precip_x_slope")

    antecedent = sum_3d.multiply(0.6).add(sum_7d.multiply(0.4)).rename("gee_antecedent_moisture_proxy")

    # Retain TWI from topo image, computed using upstream area proxy.
    twi = topo_img.select("gee_twi")

    return ee.Image.cat([
        api,
        flashiness,
        flashiness_7d,
        saturation,
        runoff,
        precip_slope,
        antecedent,
        twi,
    ])


def build_glofas_feature_image(start_date: str, end_date: str) -> ee.Image:
    global _GLOFAS_WARNING_EMITTED
    if not _GLOFAS_WARNING_EMITTED:
        logging.warning("GloFAS forecasts not yet integrated; using zero-filled ensemble bands.")
        _GLOFAS_WARNING_EMITTED = True
    return GLOFAS_EMPTY_IMAGE

# Standardized band list used across the pipeline. Every returned image
# will be coerced to contain exactly these bands (missing bands filled with 0).
STATIC_FEATURE_BANDS = [
    "gee_api_weighted_mm",
    "gee_aspect_mean",
    "gee_elevation_mean",
    "gee_flashiness_index",
    "gee_gsw_occurrence_mean",
    "gee_imerg_intensity_7d_mm_per_day",
    "gee_imerg_max_daily_7d_mm",
    "gee_imerg_sum_7d_before_mm",
    "gee_merit_upa_mean",
    "gee_ndbi_mean_30d",
    "gee_ndvi_mean_30d",
    "gee_ndwi_mean_30d",
    "gee_saturation_proxy",
    "gee_slope_mean",
    "gee_smap_subsurface_soil_moisture",
    "gee_smap_surface_soil_moisture",
    "gee_twi",
]

DYNAMIC_FEATURE_BANDS = [
    "gee_antecedent_moisture_proxy",
    "gee_flashiness_index_7d",
    "gee_imerg_intensity_3d_mm_per_day",
    "gee_imerg_max_1h_intensity_mm",
    "gee_imerg_max_3h_intensity_mm",
    "gee_imerg_max_6h_intensity_mm",
    "gee_imerg_sum_24h_mm",
    "gee_imerg_sum_3d_before_mm",
    "gee_precip_x_slope",
    "gee_runoff_potential",
    "gee_smap_soil_moisture_anomaly",
    "glofas_forecast_10th_percentile",
    "glofas_forecast_90th_percentile",
    "glofas_forecast_control",
    "glofas_forecast_mean",
    "glofas_forecast_median",
    "glofas_forecast_std_dev",
    "imerg_max_1h_intensity_mm",
    "imerg_mean_1h_intensity_mm",
]

ENGINEERED_FEATURE_BANDS = (
    "gee_api_weighted_mm",
    "gee_flashiness_index",
    "gee_flashiness_index_7d",
    "gee_saturation_proxy",
    "gee_runoff_potential",
    "gee_precip_x_slope",
    "gee_antecedent_moisture_proxy",
    "gee_twi",
)

FULL_FEATURE_BANDS = [
    "imerg_max_1h_intensity_mm",
    "imerg_mean_1h_intensity_mm",
    "gee_imerg_sum_24h_mm",
    "gee_imerg_max_1h_intensity_mm",
    "gee_imerg_max_3h_intensity_mm",
    "gee_imerg_max_6h_intensity_mm",
    "gee_imerg_sum_3d_before_mm",
    "gee_imerg_sum_7d_before_mm",
    "gee_imerg_max_daily_7d_mm",
    "gee_imerg_intensity_3d_mm_per_day",
    "gee_imerg_intensity_7d_mm_per_day",
    "gee_smap_surface_soil_moisture",
    "gee_smap_subsurface_soil_moisture",
    "gee_smap_soil_moisture_anomaly",
    "gee_antecedent_moisture_proxy",
    "gee_flashiness_index_7d",
    "gee_api_weighted_mm",
    "gee_flashiness_index",
    "gee_saturation_proxy",
    "gee_runoff_potential",
    "gee_aspect_mean",
    "gee_elevation_mean",
    "gee_slope_mean",
    "gee_gsw_occurrence_mean",
    "gee_merit_upa_mean",
    "gee_twi",
    "gee_ndvi_mean_30d",
    "gee_ndbi_mean_30d",
    "gee_ndwi_mean_30d",
    "gee_precip_x_slope",
    "glofas_forecast_control",
    "glofas_forecast_mean",
    "glofas_forecast_median",
    "glofas_forecast_std_dev",
    "glofas_forecast_10th_percentile",
    "glofas_forecast_90th_percentile",
]

BASELINE_BANDS = [
    'Elevation','Slope','TWI','Flow_Accumulation','Channel_Mask','Dist_To_Channel_m',
    'NDVI','NDWI','Total_Precipitation','Max_Daily_Precipitation','Land_Cover','viirs_flood_mask',
    'clay_mean_0cm','sand_mean_0cm','bdod_mean_0cm','soc_mean_0cm','phh2o_mean_0cm'
]

STANDARD_BANDS = BASELINE_BANDS + sorted(
    set(STATIC_FEATURE_BANDS)
    | set(DYNAMIC_FEATURE_BANDS)
    | set(FULL_FEATURE_BANDS)
)


def initialize_gee():
    """
    Initializes the Earth Engine API using default credentials.
    Assumes user has already authenticated via `gcloud auth application-default login`.
    """
    try:
        ee.Initialize(project='extreme-hull-449213-c6')
        logging.info("GEE initialized successfully")
        return True
    except Exception as e:
        logging.error(f"Error initializing GEE: {e}", exc_info=True)
        logging.error("Please ensure you have authenticated via the command line by running: gcloud auth application-default login")
        return False


def create_memory_safe_roi(roi, max_area_km2=100):
    """Split large ROIs into smaller chunks to avoid memory issues."""
    try:
        area_km2 = roi.area().divide(1000000).getInfo()
        logging.info(f"ROI area: {area_km2:.2f} km2")
        
        if area_km2 <= max_area_km2:
            return [roi]
        
        bounds = roi.bounds().getInfo()['coordinates'][0]
        min_lon = min([p[0] for p in bounds])
        max_lon = max([p[0] for p in bounds])
        min_lat = min([p[1] for p in bounds])
        max_lat = max([p[1] for p in bounds])
        
        grid_size = math.ceil(math.sqrt(area_km2 / max_area_km2))
        lon_step = (max_lon - min_lon) / grid_size
        lat_step = (max_lat - min_lat) / grid_size
        
        chunks = []
        for i in range(grid_size):
            for j in range(grid_size):
                chunk_bounds = [
                    [min_lon + i * lon_step, min_lat + j * lat_step],
                    [min_lon + (i+1) * lon_step, min_lat + j * lat_step],
                    [min_lon + (i+1) * lon_step, min_lat + (j+1) * lat_step],
                    [min_lon + i * lon_step, min_lat + (j+1) * lat_step],
                    [min_lon + i * lon_step, min_lat + j * lat_step]
                ]
                chunk_roi = ee.Geometry.Polygon([chunk_bounds])
                if roi.intersects(chunk_roi).getInfo():
                    chunks.append(roi.intersection(chunk_roi))
        
        logging.info(f"Split ROI into {len(chunks)} chunks")
        return chunks
        
    except Exception as e:
        logging.warning(f"Could not split ROI, using original: {e}")
        return [roi]


# Helper: enforce standard band schema on any image
def enforce_band_schema(img):
    """Return an image that has exactly STANDARD_BANDS (missing bands filled with 0).
    Uses select(..., names, defaultValue) so missing bands are created with 0.
    If img is None or raises, returns a constant image with STANDARD_BANDS.
    """
    try:
        # select(names, names, defaultValue) returns exactly these bands
        return ee.Image(img).select(STANDARD_BANDS, STANDARD_BANDS, 0)
    except Exception:
        return ee.Image.constant([0] * len(STANDARD_BANDS)).rename(STANDARD_BANDS)


# Helper: combine a list of images (layers) into a single image that contains
# all STANDARD_BANDS. For each band, we mosaic the band across the layer images
# and then concatenate the per-band mosaics into the final image. This avoids
# band-count mismatches that cause Image.unmask errors.
def combine_layers_to_schema(layers):
    # Ensure each layer has all bands (missing -> 0)
    enforced = [enforce_band_schema(l) for l in layers]
    band_images = []
    for b in STANDARD_BANDS:
        # Build a small collection of single-band images for this band
        single_band_imgs = [img.select([b]) for img in enforced]
        band_col = ee.ImageCollection(single_band_imgs)
        # mosaic picks the first non-masked value per pixel across the collection
        band_images.append(band_col.mosaic().rename([b]))
    # Concatenate the per-band mosaics
    combined = ee.Image.cat(band_images)
    return combined


def get_dem_features_optimized(roi):
    """Memory-optimized topographic features using Copernicus GLO-30 DEM."""
    try:
        dem = (ee.ImageCollection("COPERNICUS/DEM/GLO30")
               .select('DEM')
               .filterBounds(roi)
               .mosaic()
               .clip(roi))
        
        elevation = dem.rename('Elevation')
        slope = ee.Terrain.slope(dem).rename('Slope')
        aspect = ee.Terrain.aspect(dem).rename('Aspect')

        filled = dem.focal_min(radius=30, kernelType='circle')
        flow_accum = filled.reduceNeighborhood(
            reducer=ee.Reducer.mean(),
            kernel=ee.Kernel.circle(radius=60)
        ).rename('Flow_Accumulation')

        slope_rad = slope.multiply(math.pi / 180.0)
        twi = flow_accum.divide(slope_rad.tan().max(0.001)).log().rename('TWI')

        thr = ee.Number(float(getattr(_cfg, 'CHANNEL_FLOWACC_THRESHOLD', 500.0)))
        channel_mask = flow_accum.gt(thr).selfMask().rename('Channel_Mask')

        dist = channel_mask.fastDistanceTransform(256).sqrt().multiply(ee.Image.pixelArea().sqrt()).rename("Dist_To_Channel_m")

        # Build a multi-band image from the DEM-derived bands
        result = elevation.addBands([slope, aspect, twi, flow_accum, channel_mask, dist])
        result = result.addBands([
            elevation.rename('gee_elevation_mean'),
            slope.rename('gee_slope_mean'),
            aspect.rename('gee_aspect_mean'),
            twi.rename('gee_twi'),
        ])
        # Enforce schema before returning
        return enforce_band_schema(result.reproject(crs='EPSG:4326', scale=30))

    except Exception as e:
        logging.error(f"Could not process topographic data: {e}", exc_info=True)
        fallback = ee.Image.constant([0.0] * len(STANDARD_BANDS)).rename(STANDARD_BANDS)
        return fallback


def get_vegetation_indices_optimized(roi, start_date, end_date):
    """Memory-optimized vegetation indices with better cloud filtering."""
    try:
        start = ee.Date(start_date)
        end = ee.Date(end_date)
        days_diff = end.difference(start, 'day')
        
        if days_diff.getInfo() > 30:
            start = end.advance(-30, 'day')
            logging.info("Limited date range to 30 days for memory optimization")
        
        s2_collection = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterDate(start, end)
            .filterBounds(roi)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
            .select(['B4', 'B8', 'B3', 'SCL'])
        )
        
        def mask_scl_optimized(image):
            scl = image.select('SCL')
            keep_mask = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6))
            return image.updateMask(keep_mask).divide(10000)
        
        s2_masked = s2_collection.map(mask_scl_optimized)
        
        collection_size = s2_masked.size()
        
        def compute_indices(image):
            ndvi_raw = image.normalizedDifference(['B8', 'B4'])
            ndwi_raw = image.normalizedDifference(['B3', 'B8'])
            return ee.Image.cat([
                ndvi_raw.rename('NDVI'),
                ndwi_raw.rename('NDWI'),
                ndvi_raw.rename('gee_ndvi_mean_30d'),
                ndwi_raw.rename('gee_ndwi_mean_30d'),
            ])
        
        median_image = ee.Algorithms.If(
            collection_size.gt(0),
            s2_masked.median().clip(roi),
            ee.Image.constant([0, 0, 0]).rename(['B8', 'B4', 'B3'])
        )
        
        result = compute_indices(ee.Image(median_image))

        # Add Landsat-derived NDBI for 30-day window
        try:
            l8_start = end.advance(-30, 'day')
            l8 = (
                ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
                .filterDate(l8_start, end)
                .filterBounds(roi)
            )

            def _ndbi(img):
                scaled = img.select(["SR_B5", "SR_B6"]).multiply(0.0000275).add(-0.2)
                nir = scaled.select("SR_B5")
                swir1 = scaled.select("SR_B6")
                ndbi = swir1.subtract(nir).divide(swir1.add(nir).add(1e-6)).rename("gee_ndbi_mean_30d")
                return ndbi

            ndbi_coll = l8.map(_ndbi)
            ndbi_mean = ee.Image(ee.Algorithms.If(
                ndbi_coll.size().gt(0),
                ndbi_coll.mean(),
                ee.Image.constant(0).rename("gee_ndbi_mean_30d"),
            ))
            result = result.addBands(ndbi_mean)
        except Exception:
            result = result.addBands(_empty_band("gee_ndbi_mean_30d"))

        return enforce_band_schema(result.reproject(crs='EPSG:4326', scale=60))

    except Exception as e:
        logging.error(f"Could not process vegetation indices: {e}", exc_info=True)
        return enforce_band_schema(ee.Image.constant([0] * len(STANDARD_BANDS)).rename(STANDARD_BANDS)).clip(roi)


def process_roi_chunk(roi_chunk, start_date, end_date, chunk_id):
    """Process a single ROI chunk with memory optimization."""
    logging.info(f"Processing chunk {chunk_id}...")
    
    try:
        layers = []

        topo = get_dem_features_optimized(roi_chunk)
        layers.append(topo)

        veg = get_vegetation_indices_optimized(roi_chunk, start_date, end_date)
        layers.append(veg)

        precip = get_precipitation_optimized(roi_chunk, start_date, end_date)
        if precip: layers.append(precip)

        lc = get_landcover_optimized(roi_chunk)
        if lc: layers.append(lc)

        viirs = get_viirs_flood_proxy(roi_chunk, start_date, end_date)
        if viirs: layers.append(viirs)

        if len(layers) < 4:
            soil = get_soil_properties_optimized(roi_chunk)
            if soil: layers.append(soil)

        imerg_img = build_imerg_feature_image(start_date, end_date).clip(roi_chunk)
        smap_img = build_smap_feature_image(start_date, end_date).clip(roi_chunk)
        upa_img = MERIT_UPA_IMAGE.clip(roi_chunk)
        gsw_img = GSW_OCCURRENCE_IMAGE.clip(roi_chunk)
        glofas_img = build_glofas_feature_image(start_date, end_date).clip(roi_chunk)

        layers.append(imerg_img)
        layers.append(smap_img)
        layers.append(upa_img)
        layers.append(gsw_img)
        layers.append(glofas_img)

        if topo is None:
            topo_for_engineering = _empty_band("gee_slope_mean").addBands(_empty_band("gee_twi")).addBands(_empty_band("gee_aspect_mean"))
        else:
            topo_for_engineering = topo

        engineered = build_engineered_features_image(
            imerg_img,
            smap_img,
            topo_for_engineering,
            upa_img,
        ).clip(roi_chunk)

        layers.append(engineered)

        if not layers:
            logging.error(f"No valid layers for chunk {chunk_id}")
            # Return a well-formed empty image with all STANDARD_BANDS (masked)
            return ee.Image.constant([0] * len(STANDARD_BANDS)).rename(STANDARD_BANDS).selfMask()

        # Combine layers in a band-safe manner (per-band mosaic across layers)
        combined = combine_layers_to_schema(layers)
        return combined.reproject(crs='EPSG:4326', scale=30)
        
    except Exception as e:
        logging.error(f"Error processing chunk {chunk_id}: {e}", exc_info=True)
        return None


def get_comprehensive_environmental_data_safe(roi, start_date, end_date):
    """Memory-safe version of comprehensive environmental data collection."""
    logging.info("Assembling memory-optimized environmental dataset...")
    
    try:
        roi_chunks = create_memory_safe_roi(roi, max_area_km2=50)
        
        all_features = []
        successful_chunks = 0
        
        for i, chunk in enumerate(roi_chunks):
            try:
                chunk_data = process_roi_chunk(chunk, start_date, end_date, i)
                if chunk_data is not None:
                    # Enforce final schema so all chunks share identical band structure
                    all_features.append(enforce_band_schema(chunk_data))
                    successful_chunks += 1
                    logging.info(f"Successfully processed chunk {i+1}/{len(roi_chunks)}")
                else:
                    logging.warning(f"Failed to process chunk {i+1}/{len(roi_chunks)}")
                    
            except Exception as chunk_error:
                logging.error(f"Chunk {i+1} failed with error: {chunk_error}")
                continue
        
        if not all_features:
            logging.error("No chunks processed successfully")
            return None
        
        logging.info(f"Successfully processed {successful_chunks}/{len(roi_chunks)} chunks")
        
        if len(all_features) == 1:
            mosaic = all_features[0]
        else:
            mosaic = ee.ImageCollection(all_features).mosaic()

        # Final clipping and schema enforcement
        return enforce_band_schema(mosaic.clip(roi).reproject(crs='EPSG:4326', scale=30))
            
    except Exception as e:
        logging.error(f"Failed to assemble environmental dataset: {e}", exc_info=True)
        return None


def get_minimal_fallback_data(roi):
    """Minimal fallback data when GEE processing fails."""
    try:
        dem = (ee.ImageCollection("COPERNICUS/DEM/GLO30")
               .select('DEM')
               .filterBounds(roi)
               .mosaic()
               .clip(roi)
               .rename('Elevation'))
        
        slope = ee.Terrain.slope(dem).rename('Slope')
        
        return enforce_band_schema(dem.addBands(slope).reproject(crs='EPSG:4326', scale=90))
        
    except Exception as e:
        logging.error(f"Even fallback data failed: {e}")
        return None


def estimate_memory_usage(roi, start_date, end_date):
    """Rough estimate of memory requirements."""
    try:
        area_km2 = roi.area().divide(1000000).getInfo()
        days = ee.Date(end_date).difference(ee.Date(start_date), 'day').getInfo()
        
        estimated_mb = area_km2 * days * 0.1
        
        logging.info(f"Estimated memory usage: {estimated_mb:.1f} MB for {area_km2:.1f} km2 over {days} days")
        
        if estimated_mb > 500:
            return "high"
        elif estimated_mb > 200:
            return "medium" 
        else:
            return "low"
            
    except Exception as e:
        logging.warning(f"Could not estimate memory usage: {e}")
        return "unknown"

def get_pretrained_model_features(roi, start_date, end_date):
    """
    Extract features specifically for the pretrained XGBoost model.
    Based on the features used in the pretrained model from requirements.txt.
    """
    import pandas as pd
    import math
    
    try:
        start = ee.Date(start_date)
        end = ee.Date(end_date)
        
        # Initialize result dictionary
        features = {}
        
        # Geographic features (from coordinates)
        bounds = roi.bounds().getInfo()['coordinates'][0]
        center_lon = sum([p[0] for p in bounds]) / len(bounds)
        center_lat = sum([p[1] for p in bounds]) / len(bounds)
        features['latitude'] = center_lat
        features['longitude'] = center_lon
        
        # Temporal features
        event_date = pd.to_datetime(start_date)
        features['year'] = event_date.year
        features['month'] = event_date.month
        features['day_of_year'] = event_date.dayofyear
        features['month_sin'] = math.sin(2 * math.pi * event_date.month / 12.0)
        features['month_cos'] = math.cos(2 * math.pi * event_date.month / 12.0)
        features['doy_sin'] = math.sin(2 * math.pi * event_date.dayofyear / 365.0)
        features['doy_cos'] = math.cos(2 * math.pi * event_date.dayofyear / 365.0)
        
        # CHIRPS precipitation features
        try:
            chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
            
            # 1 day before
            precip_1d = chirps.filterDate(start.advance(-1, 'day'), start).sum()
            precip_1d_result = precip_1d.reduceRegion(
                reducer=ee.Reducer.mean(), geometry=roi, scale=5000, maxPixels=1e9
            ).getInfo()
            features['gee_chirps_precip_sum_1d_before'] = precip_1d_result.get('precipitation', 0) or 0
            
            # 3 days before
            precip_3d = chirps.filterDate(start.advance(-3, 'day'), start).sum()
            precip_3d_result = precip_3d.reduceRegion(
                reducer=ee.Reducer.mean(), geometry=roi, scale=5000, maxPixels=1e9
            ).getInfo()
            features['gee_chirps_precip_sum_3d_before'] = precip_3d_result.get('precipitation', 0) or 0
            
            # 7 days before
            precip_7d = chirps.filterDate(start.advance(-7, 'day'), start).sum()
            precip_7d_result = precip_7d.reduceRegion(
                reducer=ee.Reducer.mean(), geometry=roi, scale=5000, maxPixels=1e9
            ).getInfo()
            features['gee_chirps_precip_sum_7d_before'] = precip_7d_result.get('precipitation', 0) or 0
            
            # 30 days before
            precip_30d = chirps.filterDate(start.advance(-30, 'day'), start).sum()
            precip_30d_result = precip_30d.reduceRegion(
                reducer=ee.Reducer.mean(), geometry=roi, scale=5000, maxPixels=1e9
            ).getInfo()
            features['gee_chirps_precip_sum_30d_before'] = precip_30d_result.get('precipitation', 0) or 0
            
            # Max precipitation
            precip_max_3d = chirps.filterDate(start.advance(-3, 'day'), start).max()
            precip_max_3d_result = precip_max_3d.reduceRegion(
                reducer=ee.Reducer.mean(), geometry=roi, scale=5000, maxPixels=1e9
            ).getInfo()
            features['gee_chirps_precip_max_3d'] = precip_max_3d_result.get('precipitation', 0) or 0
            
            precip_max_7d = chirps.filterDate(start.advance(-7, 'day'), start).max()
            precip_max_7d_result = precip_max_7d.reduceRegion(
                reducer=ee.Reducer.mean(), geometry=roi, scale=5000, maxPixels=1e9
            ).getInfo()
            features['gee_chirps_precip_max_7d'] = precip_max_7d_result.get('precipitation', 0) or 0
            
            # Precipitation intensity
            if features['gee_chirps_precip_sum_7d_before'] > 0:
                features['gee_chirps_precip_intensity_7d_before'] = features['gee_chirps_precip_sum_7d_before'] / 7.0
            else:
                features['gee_chirps_precip_intensity_7d_before'] = 0
        except Exception as e:
            logging.warning(f"CHIRPS precipitation extraction failed: {e}")
            features['gee_chirps_precip_sum_1d_before'] = 0
            features['gee_chirps_precip_sum_3d_before'] = 0
            features['gee_chirps_precip_sum_7d_before'] = 0
            features['gee_chirps_precip_sum_30d_before'] = 0
            features['gee_chirps_precip_max_3d'] = 0
            features['gee_chirps_precip_max_7d'] = 0
            features['gee_chirps_precip_intensity_7d_before'] = 0
        
        # GPM precipitation features (with band compatibility)
        try:
            gpm = ee.ImageCollection("NASA/GPM_L3/IMERG_V07").select('precipitation')
            
            # 1 day before
            gpm_1d = gpm.filterDate(start.advance(-1, 'day'), start).sum()
            features['gee_gpm_precip_sum_1d_before'] = gpm_1d.reduceRegion(
                reducer=ee.Reducer.mean(), geometry=roi, scale=5000, maxPixels=1e9
            ).getInfo().get('precipitation', 0)
            
            # 7 days before
            gpm_7d = gpm.filterDate(start.advance(-7, 'day'), start).sum()
            features['gee_gpm_precip_sum_7d_before'] = gpm_7d.reduceRegion(
                reducer=ee.Reducer.mean(), geometry=roi, scale=5000, maxPixels=1e9
            ).getInfo().get('precipitation', 0)
            
            # Max rate and 6h accumulation on event day
            gpm_event = gpm.filterDate(start, end)
            gpm_max_rate = gpm_event.max()
            features['gee_gpm_precip_max_rate_event_mmhr'] = gpm_max_rate.reduceRegion(
                reducer=ee.Reducer.mean(), geometry=roi, scale=5000, maxPixels=1e9
            ).getInfo().get('precipitation', 0)
            
            # 6h accumulation (simplified)
            features['gee_gpm_precip_sum_max_6h_event_mm'] = features['gee_gpm_precip_max_rate_event_mmhr'] * 6.0
        except Exception as e:
            logging.warning(f"GPM precipitation extraction failed: {e}")
            features['gee_gpm_precip_sum_1d_before'] = 0
            features['gee_gpm_precip_sum_7d_before'] = 0
            features['gee_gpm_precip_max_rate_event_mmhr'] = 0
            features['gee_gpm_precip_sum_max_6h_event_mm'] = 0
        
        # Topographic features
        try:
            dem = ee.Image("USGS/SRTMGL1_003")
            slope = ee.Terrain.slope(dem)
            
            elevation_result = dem.reduceRegion(
                reducer=ee.Reducer.mean(), geometry=roi, scale=90, maxPixels=1e9
            ).getInfo()
            features['gee_elevation_mean'] = elevation_result.get('elevation', 0) or 0
            
            slope_result = slope.reduceRegion(
                reducer=ee.Reducer.mean(), geometry=roi, scale=90, maxPixels=1e9
            ).getInfo()
            features['gee_slope_mean'] = slope_result.get('slope', 0) or 0
        except Exception as e:
            logging.warning(f"Topographic features extraction failed: {e}")
            features['gee_elevation_mean'] = 0
            features['gee_slope_mean'] = 0
        
        # MERIT Hydro upstream area
        try:
            merit_hydro = ee.Image("MERIT/Hydro/v1_0_1").select("upa")
            upa_result = merit_hydro.reduceRegion(
                reducer=ee.Reducer.mean(), geometry=roi, scale=90, maxPixels=1e9
            ).getInfo()
            features['gee_merit_upa_mean'] = upa_result.get('upa', 0) or 0
        except Exception as e:
            logging.warning(f"MERIT Hydro extraction failed: {e}")
            features['gee_merit_upa_mean'] = 0
        
        # Surface water features
        try:
            gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
            occurrence_result = gsw.select("occurrence").reduceRegion(
                reducer=ee.Reducer.mean(), geometry=roi, scale=30, maxPixels=1e9
            ).getInfo()
            features['gee_gsw_occurrence_mean'] = occurrence_result.get('occurrence', 0) or 0
            
            seasonality_result = gsw.select("seasonality").reduceRegion(
                reducer=ee.Reducer.mean(), geometry=roi, scale=30, maxPixels=1e9
            ).getInfo()
            features['gee_gsw_seasonality_mean'] = seasonality_result.get('seasonality', 0) or 0
        except Exception as e:
            logging.warning(f"Surface water features extraction failed: {e}")
            features['gee_gsw_occurrence_mean'] = 0
            features['gee_gsw_seasonality_mean'] = 0
        
        # Vegetation features
        try:
            ndvi = ee.ImageCollection("MODIS/061/MOD13Q1").select("NDVI").filterDate(
                start.advance(-30, 'day'), start
            ).mean()
            ndvi_result = ndvi.reduceRegion(
                reducer=ee.Reducer.mean(), geometry=roi, scale=250, maxPixels=1e9
            ).getInfo()
            features['gee_ndvi_mean'] = ndvi_result.get('NDVI', 0) or 0
        except Exception as e:
            logging.warning(f"Vegetation features extraction failed: {e}")
            features['gee_ndvi_mean'] = 0
        
        # Engineered features (with None checks)
        elevation = features.get('gee_elevation_mean', 0)
        if elevation is None:
            elevation = 0
        
        features['snowmelt_risk_spring'] = 1 if (elevation > 1000 and 
                                                event_date.month in [3, 4, 5, 6]) else 0
        
        features['snow_region'] = 1 if (elevation > 800 and 
                                      abs(center_lat) > 35) else 0
        
        features['temp_elevation_interaction'] = 0  # Placeholder - would need temperature data
        
        features['is_snowmelt_season'] = 1 if event_date.month in [3, 4, 5, 6] else 0
        features['is_monsoon_season'] = 1 if event_date.month in [6, 7, 8, 9] else 0
        features['is_ice_season'] = 1 if event_date.month in [12, 1, 2, 3] else 0
        
        features['tropical_region'] = 1 if abs(center_lat) < 23.5 else 0
        features['temperate_region'] = 1 if (23.5 <= abs(center_lat) < 50) else 0
        features['northern_region'] = 1 if center_lat > 50 else 0
        
        # Additional features
        features['confidence_score'] = 1.0
        features['num_admin_regions'] = 1
        
        return features
        
    except Exception as e:
        logging.error(f"Pretrained model feature extraction failed: {e}")
        return None
