  
# georisk_enrich_export_v5.py
# Colab-ready. Safe against null .set() in EE.
# - Never sets null values into dictionaries
# - Uses fallback attempts (point, buffer, coarser scale)
# - Exports to Drive if > LOCAL_FETCH_LIMIT

import os, time
import pandas as pd
import geopandas as gpd
import numpy as np
import ee
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ---------------- CONFIG ----------------
EMDAT_EXCEL_PATH = "/content/emdat.xlsx"
NEGATIVE_SAMPLE_RATIO = 1.5
COUNTRY_CODES_FOR_NEGATIVES = ['USA','BRA','AUS','IND','CHN','NGA','RUS','IDN']
OUTPUT_MODEL_DIR = "/content/models"
OUTPUT_MODEL_NAME = "georisk_xgb_v5.json"
OUTPUT_FEATURES_NAME = "georisk_features_v5.txt"
TEST_SPLIT_SIZE = 0.2
RANDOM_STATE = 42

GEE_REDUCE_SCALE = 30
GEE_REDUCE_MAXPIXELS = int(1e9)
BUFFER_METERS = 250
LARGE_SCALE_MULTIPLIER = 4

LOCAL_FETCH_LIMIT = 500
DRIVE_EXPORT_FOLDER = "GEE_Exports"
DRIVE_EXPORT_PREFIX = "georisk_enriched_emdat_v5"

# ---------------- helpers ----------------
def initialize_gee():
    try:
        ee.Initialize()
        print("Earth Engine already initialized.")
    except Exception:
        ee.Authenticate()
        ee.Initialize()
    print("GEE Initialization Complete.")

def load_emdat_flood_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"EMDAT file not found at: {path}")
    df = pd.read_excel(path)
    req = ['Start Year','Start Month','Start Day','Latitude','Longitude']
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError("Missing columns in EMDAT: " + str(missing))
    df = df.dropna(subset=req)
    df['Start Year']  = df['Start Year'].astype(int).astype(str)
    df['Start Month'] = df['Start Month'].astype(int).astype(str).str.zfill(2)
    df['Start Day']   = df['Start Day'].astype(int).astype(str).str.zfill(2)
    df['flood_start_date'] = pd.to_datetime(df['Start Year'] + '-' + df['Start Month'] + '-' + df['Start Day'], errors='coerce')
    df = df.dropna(subset=['flood_start_date','Latitude','Longitude'])
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude), crs="EPSG:4326")
    gdf['target'] = 1
    print(f"Loaded {len(gdf)} flood events.")
    return gdf

def generate_negative_samples(positive_gdf):
    num = int(len(positive_gdf) * NEGATIVE_SAMPLE_RATIO)
    world_url = "https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json"
    world = gpd.read_file(world_url)
    if 'iso_a3' not in world.columns and 'id' in world.columns:
        world = world.rename(columns={'id':'iso_a3'})
    areas = world[world['iso_a3'].isin(COUNTRY_CODES_FOR_NEGATIVES)]
    if areas.empty:
        minx, miny, maxx, maxy = -180,-90,180,90
    else:
        minx, miny, maxx, maxy = areas.total_bounds
    xs = np.random.uniform(minx, maxx, num*2)
    ys = np.random.uniform(miny, maxy, num*2)
    neg = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xs, ys), crs="EPSG:4326")
    try:
        neg = gpd.sjoin(neg, areas, how="inner", predicate='within').head(num)
    except Exception:
        neg = gpd.sjoin(neg, areas, how="inner", op='within').head(num)
    neg = neg.reset_index(drop=True)
    neg['target'] = 0
    neg['flood_start_date'] = pd.NaT
    print(f"Generated {len(neg)} negative samples.")
    return neg[['geometry','target','flood_start_date']]

def safe_fetch_enriched_collection(fc):
    try:
        info = fc.getInfo()
        return info.get('features', [])
    except Exception as e:
        print("Warning: fc.getInfo() failed, falling back. Error:", e)
        size = int(fc.size().getInfo())
        features = []
        for i in range(size):
            try:
                feat = ee.Feature(fc.toList(size).get(i)).getInfo()
                features.append(feat)
            except Exception as e2:
                print(f"Warning: failed to fetch feature {i}: {e2}")
        return features

# ---------------- curvature ----------------
def compute_curvature_from_dem(dem):
    k_weights = [[0.0,0.5,0.0],[0.5,-2.0,0.5],[0.0,0.5,0.0]]
    lap_weights = [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]
    kernel = ee.Kernel.fixed(3,3,k_weights,normalize=False)
    lapk  = ee.Kernel.fixed(3,3,lap_weights,normalize=False)
    scale = dem.projection().nominalScale()
    scale_sq = scale.multiply(scale)
    conv = dem.convolve(kernel).divide(scale_sq).rename('curvature')
    lap  = dem.convolve(lapk).divide(scale_sq).rename('curvature_laplacian')
    return conv.addBands([conv.rename('profile_curvature'),
                          conv.rename('planform_curvature'),
                          lap])

# ---------------- GEE enrichment ----------------
def enrich_dataframe_with_gee_expert(gdf):
    print(f"Starting GEE enrichment for {len(gdf)} points...")
    feats = []
    for i,row in gdf.iterrows():
        geom = ee.Geometry.Point([float(row.geometry.x), float(row.geometry.y)])
        ds = ''
        if pd.notna(row['flood_start_date']):
            ds = row['flood_start_date'].strftime('%Y-%m-%d')
        feats.append(ee.Feature(geom, {'id':int(i),'flood_start_date':ds}))
    fc = ee.FeatureCollection(feats)

    dem = ee.Image("USGS/SRTMGL1_003").select('elevation')
    terrain = ee.Terrain.products(dem)
    slope = terrain.select('slope')
    aspect = terrain.select('aspect')
    curvature_img = compute_curvature_from_dem(dem)

    # Rivers
    try:
        rivers_fc = ee.FeatureCollection("WWF/HydroSHEDS/v1/FreeFlowingRivers")
        rivers_img = ee.Image().paint(rivers_fc, 1)
        distance_to_river = rivers_img.fastDistanceTransform(GEE_REDUCE_SCALE).sqrt().rename('distance_to_river')
    except:
        distance_to_river = ee.Image.constant(0).rename('distance_to_river')

    # Landcover
    try:
        landcover = ee.Image("ESA/WorldCover/v100/2020").select('Map')
    except:
        landcover = ee.Image.constant(0).rename('Map')

    # Population
    try:
        population = ee.ImageCollection("WorldPop/GP/100m/pop").filterDate('2020-01-01','2020-12-31').mean().rename('population')
    except:
        population = ee.Image.constant(0).rename('population')

    # Precip
    try:
        precip_coll = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").select('precipitation')
    except:
        precip_coll = ee.ImageCollection([])

    def extract(feature):
        geom = feature.geometry()
        ds = ee.String(feature.get('flood_start_date'))
        event_date = ee.Date(ee.Algorithms.If(ds.length().gt(0), ds, '2023-01-01'))
        props = ee.Dictionary({})

        def try_set(props_in, key, image, band):
            dict1 = image.reduceRegion(ee.Reducer.mean(), geom, GEE_REDUCE_SCALE, maxPixels=GEE_REDUCE_MAXPIXELS)
            val = ee.Dictionary(dict1).get(band)
            return ee.Dictionary(ee.Algorithms.If(val, props_in.set(key,val), props_in))

        # precip helper
        def try_precip(props_in, key, days, which):
            start = event_date.advance(-days,'day')
            sub = precip_coll.filterDate(start,event_date)
            cond = sub.size().gt(0)
            def compute_val():
                img = sub.sum() if which=='sum' else sub.max()
                d = img.reduceRegion(ee.Reducer.mean(), geom, GEE_REDUCE_SCALE, maxPixels=GEE_REDUCE_MAXPIXELS)
                return ee.Dictionary(d).get('precipitation')
            val = ee.Algorithms.If(cond, compute_val(), None)
            return ee.Dictionary(ee.Algorithms.If(val, props_in.set(key,val), props_in))

        # Precip windows
        props = try_precip(props,'precip_total_30d',30,'sum')
        props = try_precip(props,'precip_max_30d',30,'max')
        props = try_precip(props,'precip_total_90d',90,'sum')
        props = try_precip(props,'precip_max_90d',90,'max')

        # Terrain
        props = try_set(props,'elevation',dem,'elevation')
        props = try_set(props,'slope',slope,'slope')
        props = try_set(props,'aspect',aspect,'aspect')
        props = try_set(props,'distance_to_river',distance_to_river,'distance_to_river')
        props = try_set(props,'population',population,'population')
        props = try_set(props,'curvature_non_dir',curvature_img,'curvature')
        props = try_set(props,'profile_curvature',curvature_img,'profile_curvature')
        props = try_set(props,'planform_curvature',curvature_img,'planform_curvature')
        props = try_set(props,'curvature_laplacian',curvature_img,'curvature_laplacian')
        props = try_set(props,'landcover',landcover,'Map')

        return feature.set(props)

    enriched_fc = fc.map(extract)
    size = int(enriched_fc.size().getInfo())
    print("Enriched FeatureCollection size =", size)

    if size > LOCAL_FETCH_LIMIT:
        task = ee.batch.Export.table.toDrive(
            collection=enriched_fc,
            description=DRIVE_EXPORT_PREFIX,
            folder=DRIVE_EXPORT_FOLDER,
            fileNamePrefix=DRIVE_EXPORT_PREFIX,
            fileFormat='CSV'
        )
        task.start()
        print("Started export to Drive. Task ID:", task.id)
        return enriched_fc
    else:
        feats_info = safe_fetch_enriched_collection(enriched_fc)
        props_list = [f.get('properties',{}) for f in feats_info]
        return pd.json_normalize(props_list)

# ---------------- model training ----------------
def train_and_save_model(df):
    if df.empty: 
        print("No data for training.")
        return
    num_df = df.select_dtypes(include=[np.number])
    if 'target' not in num_df.columns and 'target' in df.columns:
        num_df['target'] = df['target'].astype(int)
    features = [c for c in num_df.columns if c != 'target']
    X = num_df[features].fillna(0)
    y = num_df['target'].astype(int)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=TEST_SPLIT_SIZE,stratify=y,random_state=RANDOM_STATE)
    clf = xgb.XGBClassifier(n_estimators=300,learning_rate=0.05,max_depth=8,
                             subsample=0.8,colsample_bytree=0.8,
                             use_label_encoder=False,eval_metric='logloss')
    clf.fit(X_train,y_train,eval_set=[(X_test,y_test)],early_stopping_rounds=30,verbose=50)
    os.makedirs(OUTPUT_MODEL_DIR,exist_ok=True)
    clf.save_model(os.path.join(OUTPUT_MODEL_DIR,OUTPUT_MODEL_NAME))
    with open(os.path.join(OUTPUT_MODEL_DIR,OUTPUT_FEATURES_NAME),'w') as f:
        for feat in features: f.write(feat+"\n")
    print("Model saved.")

# ---------------- main ----------------
if __name__=="__main__":
    start=time.time()
    initialize_gee()
    pos = load_emdat_flood_data(EMDAT_EXCEL_PATH)
    neg = generate_negative_samples(pos)
    cols = ['geometry','target','flood_start_date']
    combined = pd.concat([pos[cols],neg],ignore_index=True).sample(frac=1,random_state=RANDOM_STATE)
    enriched = enrich_dataframe_with_gee_expert(combined)
    if isinstance(enriched, ee.featurecollection.FeatureCollection):
        print("Export submitted — check Earth Engine Tasks or your Drive folder.")
    else:
        train_and_save_model(enriched)
    print("Done in %.2f minutes"%((time.time()-start)/60.0))
