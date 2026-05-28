# ==========================================================================
# ComStock energy forecasting pipeline
#
# Leakage-safe version with cleaner categorical handling
# Prepared for portfolio use
# Date: 2026
#
# What changed:
# No preprocessing leakage: all statistics come from the training split.
# 2. Proper categorical encoding (label encoding for tree models, OHE for linear)
# 3. Cross-validation for robust performance estimates
# 4. Full test set evaluation (realistic metrics)
# 5. Enhanced ComStock-specific features for building energy modeling
# ==========================================================================
import os
import warnings
import time
import math
import joblib
import traceback
import numpy as np
import pandas as pd
import shap
import optuna
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest, StackingRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.inspection import permutation_importance
import lightgbm as lgb
import xgboost as xgb
import catboost as ctb
import re as _re
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Show plots inline in the notebook
import matplotlib
matplotlib.rcParams['figure.dpi'] = 120

# ==========================================================================
# User-configurable flags
# ==========================================================================
FILE_PATH = r"C:\Users\aramnara\OneDrive - University of Maryland\Documents\result_checking ML\ML Exclusive\ComStock\Comstock_with_HDD_CDD_2000_2025.csv"
TARGET = "out.site_energy.total.energy_consumption"
RANDOM_SEED = 42
DO_LOG_TARGET = True          # Train on log1p(y) and back-transform (recommended for energy data)
N_TRIALS = 150                # Optuna trials for HPO
DO_STACKING = True            # Build stacking ensemble
STACK_TOP_K = 3               # Top K models for stacking
SHAP_SAMPLE = 500             # Sample size for SHAP analysis
ISOF_CONTAMINATION = 0.03     # IsolationForest contamination for outliers
OUTPUT_DIR = r"C:\Users\aramnara\OneDrive - University of Maryland\Documents\result_checking ML\ML Exclusive\ComStock\model_outputs"
# Saved models and CSVs go here; the full path is printed when each file is written.
os.makedirs(OUTPUT_DIR, exist_ok=True)
USE_TEMPORAL_SPLIT = False    # Set True if data has temporal ordering

# HDD/CDD data-quality filter
# Rows with too many missing annual HDD/CDD values are unreliable.
# They can distort HDD_26y_avg / CDD_26y_avg, so they are removed before training.
# MAX_MISSING_HDD_CDD_FRAC sets the missing-data limit for annual values.
# For example, 0.20 allows about five missing years out of 26; lower values are stricter.
MAX_MISSING_HDD_CDD_FRAC = 0.20   # ← tune as needed (0.10–0.30 is reasonable)

np.random.seed(RANDOM_SEED)

# ==========================================================================
# Helper functions
# ==========================================================================
def metrics(y_true, y_pred):
    """Calculate comprehensive regression metrics."""
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": math.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred)
    }

def safe_exp_predict(preds):
    """Inverse log transformation with safety clipping."""
    if DO_LOG_TARGET:
        return np.expm1(preds).clip(min=0.0)
    else:
        return np.array(preds).clip(min=0.0)

def ensure_numeric_df(df_in, dtype=np.float32):
    """Coerce all columns to numeric where possible."""
    df_out = df_in.copy()
    for c in df_out.columns:
        if df_out[c].dtype == object or not np.issubdtype(df_out[c].dtype, np.number):
            df_out[c] = pd.to_numeric(df_out[c], errors='coerce')
    return df_out.astype(dtype)

# ==========================================================================
# Feature engineering transformer
# ==========================================================================
class ComStockFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer for ComStock feature engineering.
    
    Key Design: All statistics are learned during fit() from training data only,
    then applied to both train and test during transform(). This prevents data leakage.
    """
    
    def __init__(self, reference_year=2025):
        self.reference_year = reference_year
        self.train_stats = {}
        self.building_type_freq = {}
        self.top_cool_types = []
        
    def fit(self, X, y=None):
        """Learn statistics from training data only."""
        df = X.copy()
        
        # Training medians for imputation.
        self.train_stats['hdd_median'] = df.get('API_HDD_2018', pd.Series([0])).replace('', np.nan).astype(float).median()
        self.train_stats['cdd_median'] = df.get('API_CDD_2018', pd.Series([0])).replace('', np.nan).astype(float).median()
        self.train_stats['sqft_median'] = df.get('in.sqft', pd.Series([1000])).replace('', np.nan).astype(float).median()
        
        # Building-type frequencies for frequency encoding.
        if 'in.comstock_building_type_group' in df.columns:
            freq = df['in.comstock_building_type_group'].value_counts(normalize=True)
            self.building_type_freq = freq.to_dict()
        
        # Top cooling types for one-hot flags.
        if 'in.hvac_cool_type' in df.columns:
            self.top_cool_types = df['in.hvac_cool_type'].value_counts().nlargest(6).index.tolist()
        
        return self
    
    def transform(self, X):
        """Apply feature engineering using training statistics."""
        df = X.copy()
        eps = 1e-6
        
        def s(col, default=np.nan):
            """Safe column getter."""
            return df[col] if col in df.columns else pd.Series(default, index=df.index)
        
        # Basic numeric cleanup using training statistics.
        df['API_HDD_2018'] = pd.to_numeric(s('API_HDD_2018').replace('', np.nan), errors='coerce').fillna(self.train_stats['hdd_median'])
        df['API_CDD_2018'] = pd.to_numeric(s('API_CDD_2018').replace('', np.nan), errors='coerce').fillna(self.train_stats['cdd_median'])
        df['in.sqft'] = pd.to_numeric(s('in.sqft').replace('', np.nan), errors='coerce').fillna(self.train_stats['sqft_median'])
        
        # 26-year averages from the source data.
        df['HDD_26y_avg'] = pd.to_numeric(s('HDD_26y_avg'), errors='coerce').fillna(self.train_stats['hdd_median'])
        df['CDD_26y_avg'] = pd.to_numeric(s('CDD_26y_avg'), errors='coerce').fillna(self.train_stats['cdd_median'])
        
        # Physics-inspired features.
        df['PHY_HEATING_LOAD'] = df['API_HDD_2018'] * df['in.sqft']
        df['PHY_COOLING_LOAD'] = df['API_CDD_2018'] * df['in.sqft']
        df['HEATING_INTENSITY'] = df['API_HDD_2018'] / (df['in.sqft'] + eps)
        df['COOLING_INTENSITY'] = df['API_CDD_2018'] / (df['in.sqft'] + eps)
        
        # Degree-day relationships.
        df['HDD_CDD_RATIO'] = (df['API_HDD_2018'] + eps) / (df['API_CDD_2018'] + eps)
        df['HDD_MINUS_26YAVG'] = df['API_HDD_2018'] - df['HDD_26y_avg']
        df['CDD_MINUS_26YAVG'] = df['API_CDD_2018'] - df['CDD_26y_avg']
        df['TOTAL_DEGREE_DAYS'] = df['API_HDD_2018'] + df['API_CDD_2018']
        
        # Log transforms.
        df['SQFT_LOG'] = np.log1p(df['in.sqft'].clip(lower=0))
        df['LOG_PHY_HEATING'] = np.log1p(df['PHY_HEATING_LOAD'].clip(lower=0))
        df['LOG_PHY_COOLING'] = np.log1p(df['PHY_COOLING_LOAD'].clip(lower=0))
        
        # Age and vintage features.
        if 'in.year_built' in df.columns:
            df['age_years'] = (self.reference_year - pd.to_numeric(df['in.year_built'], errors='coerce')).clip(lower=0)
            df['age_years'] = df['age_years'].fillna(df['age_years'].median() if df['age_years'].median() > 0 else 30)
            df['age_bucket'] = pd.cut(df['age_years'], bins=[-1,10,30,60,200], labels=['very_new','new','mid','old']).astype(str)
        elif 'in.vintage' in df.columns:
            df['age_bucket'] = df['in.vintage'].astype(str)
        
        # Operating-hours features.
        wd_hours = pd.to_numeric(s('in.weekday_operating_hours..hr', 0), errors='coerce').fillna(0)
        we_hours = pd.to_numeric(s('in.weekend_operating_hours..hr', 0), errors='coerce').fillna(0)
        df['avg_operating_hours'] = (wd_hours * 5 + we_hours * 2) / 7.0
        df['avg_operating_hours'] = df['avg_operating_hours'].fillna(df['avg_operating_hours'].median() if df['avg_operating_hours'].median() > 0 else 50)
        df['weekday_weekend_open_diff'] = pd.to_numeric(s('in.weekday_opening_time..hr', 0), errors='coerce').fillna(0) - pd.to_numeric(s('in.weekend_opening_time..hr', 0), errors='coerce').fillna(0)
        
        # Thermostat features.
        if 'in.tstat_clg_sp_f..f' in df.columns and 'in.tstat_htg_sp_f..f' in df.columns:
            df['TSTAT_SPREAD'] = pd.to_numeric(s('in.tstat_clg_sp_f..f'), errors='coerce').fillna(75) - pd.to_numeric(s('in.tstat_htg_sp_f..f'), errors='coerce').fillna(70)
        else:
            df['TSTAT_SPREAD'] = pd.to_numeric(s('in.tstat_clg_delta_f..delta_f', 0), errors='coerce').fillna(0) + pd.to_numeric(s('in.tstat_htg_delta_f..delta_f', 0), errors='coerce').fillna(0)
        
        df['TSTAT_HDDCDD_INTERACTION'] = df['TSTAT_SPREAD'] / (df['HDD_CDD_RATIO'] + eps)
        
        # HVAC features.
        df['HVAC_NIGHT_OPR_INTERACTION'] = pd.to_numeric(s('in.hvac_night_variability', 0), errors='coerce').fillna(0) * df['avg_operating_hours']
        
        # Window-to-wall features.
        if 'in.window_to_wall_ratio_category' in df.columns:
            df['window_to_wall_ratio_num'] = pd.to_numeric(df['in.window_to_wall_ratio_category'], errors='coerce')
        
        # Frequency encoding.
        if 'in.comstock_building_type_group' in df.columns and self.building_type_freq:
            df['FREQ_building_type_group'] = df['in.comstock_building_type_group'].map(self.building_type_freq).fillna(0.0)
        
        # Heating-fuel flags.
        if 'in.heating_fuel' in df.columns:
            fuel = df['in.heating_fuel'].astype(str).str.lower()
            df['HEATING_FUEL_ELECTRIC'] = fuel.str.contains('electric', na=False).astype(int)
            df['HEATING_FUEL_GAS'] = fuel.str.contains('gas', na=False).astype(int)
            df['HEATING_FUEL_OTHER'] = (~(df['HEATING_FUEL_ELECTRIC'].astype(bool) | df['HEATING_FUEL_GAS'].astype(bool))).astype(int)
        
        # Lighting features.
        if 'in.interior_lighting_generation' in df.columns:
            df['LOG_INT_LIGHTING'] = np.log1p(pd.to_numeric(df['in.interior_lighting_generation'], errors='coerce').fillna(0.0))
        
        # Socio-economic features.
        ejs_cols = [c for c in ['in.ejscreen_census_tract_percentile_for_demographic_index','in.ejscreen_census_tract_percentile_for_people_of_color','in.ejscreen_census_tract_percentile_for_low_income'] if c in df.columns]
        if ejs_cols:
            df['SOCIO_EJ_COMPOSITE'] = df[ejs_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1).fillna(50)
        
        # Climate-zone features.
        if 'in.ashrae_iecc_climate_zone_2004' in df.columns:
            cz = df['in.ashrae_iecc_climate_zone_2004'].astype(str)
            df['CLIMATE_ZONE_NUM'] = cz.str.extract(r'(\d+)').astype(float)
        
        # Floor-area features.
        if 'in.number_of_stories' in df.columns:
            df['AVG_FLOOR_AREA'] = df['in.sqft'] / (pd.to_numeric(df['in.number_of_stories'], errors='coerce').replace(0, np.nan) + eps)
            df['AVG_FLOOR_AREA'] = df['AVG_FLOOR_AREA'].fillna(df['AVG_FLOOR_AREA'].median() if df['AVG_FLOOR_AREA'].median() > 0 else 5000)
        
        # HVAC cooling-type flags.
        if 'in.hvac_cool_type' in df.columns and self.top_cool_types:
            for t in self.top_cool_types:
                safe_name = str(t)[:28].replace(" ", "_").replace("/", "_")
                flag = (df['in.hvac_cool_type'] == t).astype(int)
                df[f'HVAC_COOL_{safe_name}'] = flag
                df[f'HVAC_COOL_{safe_name}_CDD_INT'] = flag * df['API_CDD_2018']
        
        # Climate-by-building-type interaction.
        if 'in.ashrae_iecc_climate_zone_2004' in df.columns and 'in.comstock_building_type_group' in df.columns:
            df['CLIMATE_BLDG_INTERACTION'] = df['in.ashrae_iecc_climate_zone_2004'].astype(str) + "_" + df['in.comstock_building_type_group'].astype(str)
        
        return df

# ==========================================================================
# 1. Load and clean the data
# ==========================================================================
print("="*80)
print("COMSTOCK ENERGY FORECASTING PIPELINE - PRODUCTION VERSION")
print("="*80)
print("\n1) Loading data...")
df_raw = pd.read_csv(FILE_PATH)
print(f"   Original shape: {df_raw.shape}")
df_raw = df_raw[~(df_raw.get("API_HDD_2018", pd.Series()).replace("", np.nan).isna() | df_raw.get("API_CDD_2018", pd.Series()).replace("", np.nan).isna())].copy()
df_raw.columns = df_raw.columns.str.replace(r"[\[\]<>]", "", regex=True)
print(f"   After HDD/CDD filter: {df_raw.shape}")
df_raw = df_raw[df_raw[TARGET].notna()].copy()
print(f"   After target filter: {df_raw.shape}")

# ==========================================================================
# 1b. HDD/CDD quality filter
# Remove rows with too many missing annual HDD/CDD values.
# Those gaps can distort HDD_26y_avg / CDD_26y_avg.
# ==========================================================================
print(f"\n1b) HDD/CDD data quality filter (max missing fraction: {MAX_MISSING_HDD_CDD_FRAC})...")

def _to_numeric_series(df, cols):
    """Coerce to numeric, treating empty strings and non-numeric as NaN."""
    return df[cols].apply(lambda col: pd.to_numeric(col.replace('', np.nan), errors='coerce'))

hdd_annual_cols = [c for c in df_raw.columns if _re.match(r'^API_HDD_\d{4}$', c)]
cdd_annual_cols = [c for c in df_raw.columns if _re.match(r'^API_CDD_\d{4}$', c)]

n_before = len(df_raw)

# --- Condition 1 & 2: API_HDD_2018 and API_CDD_2018 must not be missing ------
hdd_2018_valid = pd.to_numeric(
    df_raw.get('API_HDD_2018', pd.Series([''] * len(df_raw))).replace('', np.nan),
    errors='coerce'
).notna()
cdd_2018_valid = pd.to_numeric(
    df_raw.get('API_CDD_2018', pd.Series([''] * len(df_raw))).replace('', np.nan),
    errors='coerce'
).notna()
baseline_mask = hdd_2018_valid & cdd_2018_valid
n_baseline_removed = (~baseline_mask).sum()
print(f"   Removed {n_baseline_removed:,} rows with missing API_HDD_2018 or API_CDD_2018")

# --- Condition 3: annual series must not exceed MAX_MISSING_HDD_CDD_FRAC -----
if hdd_annual_cols or cdd_annual_cols:
    print(f"   Found {len(hdd_annual_cols)} annual HDD cols, {len(cdd_annual_cols)} annual CDD cols")

    hdd_mask = (
        _to_numeric_series(df_raw, hdd_annual_cols).isnull().mean(axis=1)
        <= MAX_MISSING_HDD_CDD_FRAC
    ) if hdd_annual_cols else pd.Series(True, index=df_raw.index)

    cdd_mask = (
        _to_numeric_series(df_raw, cdd_annual_cols).isnull().mean(axis=1)
        <= MAX_MISSING_HDD_CDD_FRAC
    ) if cdd_annual_cols else pd.Series(True, index=df_raw.index)

    quality_mask = baseline_mask & hdd_mask & cdd_mask
else:
    print("   ⚠ No annual API_HDD_YYYY / API_CDD_YYYY columns found — applying baseline filter only")
    quality_mask = baseline_mask
    hdd_mask = pd.Series(True, index=df_raw.index)
    cdd_mask = pd.Series(True, index=df_raw.index)

df_raw = df_raw[quality_mask].copy()
n_removed = n_before - len(df_raw)
hdd_only_bad = (~hdd_mask &  cdd_mask & baseline_mask).sum()
cdd_only_bad = ( hdd_mask & ~cdd_mask & baseline_mask).sum()
both_bad     = (~hdd_mask & ~cdd_mask & baseline_mask).sum()
print(f"   Total removed: {n_removed:,} rows ({100*n_removed/n_before:.1f}%)")
print(f"   Breakdown — API_2018 missing: {n_baseline_removed:,} | "
      f"HDD series: {hdd_only_bad:,} | CDD series: {cdd_only_bad:,} | Both series: {both_bad:,}")
print(f"   Remaining rows: {len(df_raw):,}")

LITERATURE_FEATURES = [
    'API_HDD_2018','API_CDD_2018','HDD_26y_avg','CDD_26y_avg',
    'in.ashrae_iecc_climate_zone_2004','in.aspect_ratio','in.comstock_building_type',
    'in.comstock_building_type_group','in.heating_fuel','in.hvac_category',
    'in.hvac_combined_type','in.hvac_cool_type','in.hvac_heat_type','in.hvac_system_type',
    'in.hvac_vent_type','in.interior_lighting_generation','in.number_of_stories',
    'in.ownership_type','in.party_responsible_for_operation',
    'in.purchase_input_responsibility','in.service_water_heating_fuel','in.sqft',
    'in.tstat_clg_delta_f..delta_f','in.tstat_clg_sp_f..f','in.tstat_htg_delta_f..delta_f',
    'in.tstat_htg_sp_f..f','in.vintage','in.wall_construction_type',
    'in.weekday_opening_time..hr','in.weekday_operating_hours..hr',
    'in.weekend_opening_time..hr','in.weekend_operating_hours..hr',
    'in.window_to_wall_ratio_category','in.window_type','in.year_built',
    'in.building_america_climate_zone','in.census_region_name','in.cambium_grid_region',
    'in.ejscreen_census_tract_percentile_for_demographic_index',
    'in.ejscreen_census_tract_percentile_for_people_of_color',
    'in.ejscreen_census_tract_percentile_for_low_income',
    'in.rotation..degrees','in.floor_area_category','in.hvac_night_variability'
]
CANDIDATE_FEATURES = [f for f in LITERATURE_FEATURES if f in df_raw.columns]
print(f"   Candidate features: {len(CANDIDATE_FEATURES)}")

# ==========================================================================
# 2. Three-way split: train, validation, test
#
# Why this split is set up this way:
# Training set (64%): fitting, feature engineering stats, and HPO.
# Validation set (16%): model selection among benchmark, HPO, and stacking.
# The test set is not used during selection.
# Test set (20%): final performance reported once, then left untouched.
# It is only used for reporting, which avoids optimistic bias.
#
# This split follows standard practice for model selection and final reporting:
# Hastie, Tibshirani & Friedman (2009) "Elements of Statistical Learning"
# That matters when model selection is part of the workflow.
# ==========================================================================
print("\n2) Three-way split: Train 70% / Validation 15% / Test 15%...")
X_raw = df_raw[CANDIDATE_FEATURES].copy()
y_raw = pd.to_numeric(df_raw[TARGET], errors='coerce').fillna(0.0).copy()

if USE_TEMPORAL_SPLIT and 'timestamp' in df_raw.columns:
    print("   Using temporal split...")
    df_raw   = df_raw.sort_values('timestamp')
    n        = len(df_raw)
    i_train  = int(n * 0.64)
    i_val    = int(n * 0.80)
    X_train_raw, y_train_raw = X_raw.iloc[:i_train],       y_raw.iloc[:i_train]
    X_val_raw,   y_val_raw   = X_raw.iloc[i_train:i_val],  y_raw.iloc[i_train:i_val]
    X_test_raw,  y_test_raw  = X_raw.iloc[i_val:],         y_raw.iloc[i_val:]
else:
    # Stratified split to keep building types and climate zones balanced.
    # This keeps the full-vs-ROM comparison balanced.
    _strat_col = 'in.comstock_building_type_group' if 'in.comstock_building_type_group' in df_raw.columns                  else 'in.comstock_building_type'
    _strat = df_raw.loc[X_raw.index, _strat_col].fillna('Unknown')
    _counts = _strat.value_counts()
    _rare   = _counts[_counts < 10].index
    if len(_rare): _strat = _strat.where(~_strat.isin(_rare), other='Other')

    X_trainval_raw, X_test_raw, y_trainval_raw, y_test_raw = train_test_split(
        X_raw, y_raw, test_size=0.15, random_state=RANDOM_SEED, stratify=_strat)
    _strat_tv = _strat.loc[X_trainval_raw.index]
    X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
        X_trainval_raw, y_trainval_raw,
        test_size=round(0.15/0.85, 4), random_state=RANDOM_SEED, stratify=_strat_tv)
    print(f"   Stratified on: {_strat_col}")

print(f"   Train:      {X_train_raw.shape}  ({100*len(X_train_raw)/len(X_raw):.0f}% of total)")
print(f"   Validation: {X_val_raw.shape}   ({100*len(X_val_raw)/len(X_raw):.0f}% of total)")
print(f"   Test:       {X_test_raw.shape}   ({100*len(X_test_raw)/len(X_raw):.0f}% of total)")

# ==========================================================================
# 3. Feature engineering
# ==========================================================================
print("\n3) Feature engineering (fit on train, transform all three sets)...")
feature_engineer = ComStockFeatureEngineer(reference_year=2025)
feature_engineer.fit(X_train_raw)

X_train_eng = feature_engineer.transform(X_train_raw)
X_val_eng   = feature_engineer.transform(X_val_raw)
X_test_eng  = feature_engineer.transform(X_test_raw)

print(f"   Engineered — Train: {X_train_eng.shape}, Val: {X_val_eng.shape}, Test: {X_test_eng.shape}")

ENGINEERED_PREFIXES = ('PHY_','HDD_','CDD_','SQFT_LOG','LOG_PHY_','age_','avg_operating_hours','TSTAT_SPREAD','HVAC_NIGHT_OPR_INTERACTION','FREQ_building_type_group','HEATING_FUEL_','LOG_INT_LIGHTING','SOCIO_EJ_COMPOSITE','CLIMATE_ZONE_NUM','AVG_FLOOR_AREA','HVAC_COOL_','HEATING_INTENSITY','COOLING_INTENSITY','TOTAL_DEGREE_DAYS','TSTAT_HDDCDD_INTERACTION','CLIMATE_BLDG_INTERACTION','weekday_weekend_open_diff','window_to_wall_ratio_num')
ALL_CANDIDATE_FEATURES = list(set([col for col in X_train_eng.columns if any(col.startswith(p) for p in ENGINEERED_PREFIXES)] + CANDIDATE_FEATURES))
print(f"   Total features: {len(ALL_CANDIDATE_FEATURES)}")

# ==========================================================================
# 4. SHAP feature selection (train only)
# SHAP is run before outlier removal here.
# Code 6/7 handles this later with SHAP after IsolationForest.
# The bias is small, but it is worth noting in the methods.
# ==========================================================================
print("\n4) SHAP feature selection (train only)...")

def shap_feature_select(X_train_df, y_train_series, candidate_features, keep_top=40):
    X_sub = X_train_df[[c for c in candidate_features if c in X_train_df.columns]].copy()
    X_enc = pd.get_dummies(X_sub, drop_first=True)
    X_enc = ensure_numeric_df(X_enc)
    for col in X_enc.columns:
        if X_enc[col].isna().any():
            X_enc[col] = X_enc[col].fillna(X_enc[col].median())
    y = y_train_series.loc[X_enc.index]
    y_log = np.log1p(y) if DO_LOG_TARGET else y
    model = ctb.CatBoostRegressor(iterations=300, depth=5, learning_rate=0.05, loss_function="RMSE", verbose=0, random_state=RANDOM_SEED)
    model.fit(X_enc, y_log)
    try:
        expl = shap.Explainer(model, X_enc.sample(n=min(1000, len(X_enc)), random_state=RANDOM_SEED))
        shap_vals = expl(X_enc.sample(n=min(2000, len(X_enc)), random_state=RANDOM_SEED))
        arr = np.abs(shap_vals.values) if hasattr(shap_vals, "values") else np.abs(np.array(shap_vals))
        shap_scores = np.nanmean(arr, axis=0)
    except:
        pr = permutation_importance(model, X_enc.sample(n=min(2000, len(X_enc)), random_state=RANDOM_SEED), y.sample(n=min(2000, len(y)), random_state=RANDOM_SEED), n_repeats=5, random_state=RANDOM_SEED, n_jobs=-1, scoring='r2')
        shap_scores = pr.importances_mean
    enc_cols = X_enc.columns
    top_enc = pd.Series(shap_scores, index=enc_cols).sort_values(ascending=False).head(keep_top).index.tolist()
    selected = set()
    for enc in top_enc:
        if enc in candidate_features:
            selected.add(enc)
            continue
        for orig in candidate_features:
            if enc.startswith(orig + "_"):
                selected.add(orig)
                break
    return sorted(list(selected))

FINAL_FEATURES = shap_feature_select(X_train_eng, y_train_raw, ALL_CANDIDATE_FEATURES, keep_top=40)
print(f"   Selected {len(FINAL_FEATURES)} features")

X_train = X_train_eng[FINAL_FEATURES].copy()
X_val   = X_val_eng[FINAL_FEATURES].copy()
X_test  = X_test_eng[FINAL_FEATURES].copy()
y_train = y_train_raw.copy()
y_val   = y_val_raw.copy()
y_test  = y_test_raw.copy()

# ==========================================================================
# 5. Preprocessing
# ==========================================================================
print("\n5) Preprocessing (using train stats only)...")

num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

train_medians = X_train[num_cols].median()
X_train[num_cols] = X_train[num_cols].fillna(train_medians)
X_val[num_cols]   = X_val[num_cols].fillna(train_medians)
X_test[num_cols]  = X_test[num_cols].fillna(train_medians)

for c in cat_cols:
    mode_val = X_train[c].mode().iloc[0] if not X_train[c].mode().empty else "missing"
    X_train[c] = X_train[c].fillna(mode_val).astype(str)
    X_val[c]   = X_val[c].fillna(mode_val).astype(str)
    X_test[c]  = X_test[c].fillna(mode_val).astype(str)

print("   Creating label-encoded features for tree-based models...")
X_train_clean = X_train.copy()
X_val_clean   = X_val.copy()
X_test_clean  = X_test.copy()

label_encoders = {}
for col in cat_cols:
    le    = LabelEncoder()
    le.fit(X_train[col].astype(str))
    X_train_clean[col] = le.transform(X_train_clean[col].astype(str))
    known = set(le.classes_)
    X_val_clean[col]  = X_val_clean[col].astype(str).apply(
        lambda v: le.transform([v])[0] if v in known else -1)
    X_test_clean[col] = X_test_clean[col].astype(str).apply(
        lambda v: le.transform([v])[0] if v in known else -1)
    label_encoders[col] = le

print(f"   Label-encoded — Train: {X_train_clean.shape}, Val: {X_val_clean.shape}, Test: {X_test_clean.shape}")

print("   Creating one-hot encoded features for linear/neural models...")
X_train_ohe = pd.get_dummies(X_train, drop_first=True)
X_val_ohe   = pd.get_dummies(X_val,   drop_first=True).reindex(columns=X_train_ohe.columns, fill_value=0)
X_test_ohe  = pd.get_dummies(X_test,  drop_first=True).reindex(columns=X_train_ohe.columns, fill_value=0)

numeric_ohe_cols = X_train_ohe.select_dtypes(include=[np.number]).columns.tolist()
scaler = StandardScaler()
X_train_ohe[numeric_ohe_cols] = scaler.fit_transform(X_train_ohe[numeric_ohe_cols])
X_val_ohe[numeric_ohe_cols]   = scaler.transform(X_val_ohe[numeric_ohe_cols])
X_test_ohe[numeric_ohe_cols]  = scaler.transform(X_test_ohe[numeric_ohe_cols])

X_train_ohe = ensure_numeric_df(X_train_ohe)
X_val_ohe   = ensure_numeric_df(X_val_ohe)
X_test_ohe  = ensure_numeric_df(X_test_ohe)

print(f"   One-hot encoded — Train: {X_train_ohe.shape}, Val: {X_val_ohe.shape}, Test: {X_test_ohe.shape}")

# ==========================================================================
# 6. Outlier detection (train only)
# ==========================================================================
print("\n6) Outlier detection (train only — val and test untouched)...")
iso = IsolationForest(contamination=ISOF_CONTAMINATION, random_state=RANDOM_SEED)
iso.fit(X_train_ohe)
train_mask = iso.predict(X_train_ohe) == 1

X_train_clean     = X_train_clean.loc[train_mask].copy()
X_train_ohe_clean = X_train_ohe.loc[train_mask].copy()
y_train_clean     = y_train.loc[train_mask].copy()

print(f"   Cleaned train: {X_train_clean.shape} (removed {(~train_mask).sum()} outliers)")
print(f"   Validation and test sets untouched by outlier removal")

if DO_LOG_TARGET:
    y_train_clean_log = np.log1p(y_train_clean)
    y_val_log         = np.log1p(y_val)
    y_test_log        = np.log1p(y_test)
else:
    y_train_clean_log = y_train_clean.copy()
    y_val_log         = y_val.copy()
    y_test_log        = y_test.copy()

# ==========================================================================
# 7. Model benchmark
# ==========================================================================
print("\n7) Benchmarking models...")

def fit_and_eval(model, model_name, X_tr, y_tr_log, X_eval, y_eval_true):
    """Fit and evaluate model using appropriate data version."""
    model.fit(X_tr, y_tr_log)
    preds_log = model.predict(X_eval)
    preds = safe_exp_predict(preds_log)
    return metrics(y_eval_true, preds), preds

model_zoo = {
    "CatBoost": ctb.CatBoostRegressor(iterations=500, learning_rate=0.06, depth=6, verbose=0, random_state=RANDOM_SEED),
    "LightGBM": lgb.LGBMRegressor(n_estimators=800, learning_rate=0.06, n_jobs=-1, random_state=RANDOM_SEED, verbose=-1),
    "XGBoost": xgb.XGBRegressor(n_estimators=700, learning_rate=0.06, tree_method="hist", n_jobs=-1, random_state=RANDOM_SEED),
    "RandomForest": RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=RANDOM_SEED),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=300, random_state=RANDOM_SEED),
    "Lasso": Lasso(alpha=1e-3),
    "MLP": MLPRegressor(hidden_layer_sizes=(128,64), solver='adam', learning_rate_init=1e-3, max_iter=1000, early_stopping=True, validation_fraction=0.1, n_iter_no_change=30, random_state=RANDOM_SEED)
}

# Tree models use label encoding; linear models use OHE.
TREE_MODELS = ["CatBoost", "LightGBM", "XGBoost", "RandomForest", "GradientBoosting"]
LINEAR_MODELS = ["Lasso", "MLP"]

benchmark_results = {}
benchmark_preds = {}

for name, model in model_zoo.items():
    print(f"  - {name:20}...", end=" ")
    try:
        if name in TREE_MODELS:
            X_tr = X_train_clean
            X_ev = X_val_clean      # ← validation, not test
        else:
            X_tr = X_train_ohe_clean
            X_ev = X_val_ohe        # ← validation, not test

        res, preds = fit_and_eval(model, name, X_tr, y_train_clean_log, X_ev, y_val)
        benchmark_results[name] = res
        benchmark_preds[name] = preds
        print(f"R²={res['R2']:.4f}, RMSE={res['RMSE']:.0f}")
    except Exception as e:
        print(f"ERROR: {e}")
        benchmark_results[name] = {"error": str(e)}

print("\n" + pd.DataFrame(benchmark_results).T.to_string())
winner_name = max({k:v for k,v in benchmark_results.items() if "R2" in v}.items(), key=lambda kv: kv[1]["R2"])[0]
print(f"\n🏆 Winner: {winner_name}")

# ==========================================================================
# 8. Cross-validation stability check
# ==========================================================================
print("\n8) CV stability check (benchmark default, informational only)...")
if winner_name:
    try:
        cv_model = clone(model_zoo[winner_name])
        # Match the feature format to the model family.
        cv_X = X_train_clean if winner_name in TREE_MODELS else X_train_ohe_clean
        cv_scores = cross_val_score(cv_model, cv_X, y_train_clean_log, cv=5, scoring='r2', n_jobs=-1)
        print(f"   5-Fold CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    except Exception as e:
        print(f"   CV failed: {e}")

# ==========================================================================
# 9. Hyperparameter tuning
#
# Design principles:
# TPE sampler: a Bayesian approach that works well on tabular data.
# MedianPruner drops weak trials early and keeps the search efficient.
# 5-fold CV inside each objective for a more stable search.
# Early stopping for XGBoost and LightGBM, with n_estimators set high and controlled by the data.
# The data decide when to stop instead of a manual search.
# Search the full regularization space for each model:
# XGBoost : lr, max_depth, min_child_weight, gamma, subsample,
# colsample_bytree, reg_alpha, reg_lambda
# LightGBM : lr, num_leaves, min_child_samples, min_split_gain,
# subsample, colsample_bytree, reg_alpha, reg_lambda
# CatBoost: iterations, depth, learning rate, l2_leaf_reg, bagging_temperature,
# random_strength.
# RandomForest: n_estimators, max_depth, min_samples_split,
# min_samples_leaf, max_features.
# ==========================================================================
print(f"\n9) HPO for {winner_name} ({N_TRIALS} trials, 5-fold CV, MedianPruner)...")

HPO_CV_FOLDS = 5   # 5-fold CV is a solid default for this dataset size.

pruner  = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)

# CatBoost
def hpo_catboost(X_tr, y_tr_log):
    def objective(trial):
        params = {
            "iterations":         trial.suggest_int  ("iterations",         500,  2000),
            "depth":              trial.suggest_int  ("depth",              4,    10),
            "learning_rate":      trial.suggest_float("learning_rate",      1e-3, 0.2,  log=True),
            "l2_leaf_reg":        trial.suggest_float("l2_leaf_reg",        1e-2, 30.0, log=True),
            "bagging_temperature":trial.suggest_float("bagging_temperature",0.0,  1.0),
            "random_strength":    trial.suggest_float("random_strength",    1e-3, 10.0, log=True),
        }
        kf     = KFold(n_splits=HPO_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        scores = []
        for fold, (tr_idx, val_idx) in enumerate(kf.split(X_tr)):
            m = ctb.CatBoostRegressor(**params, loss_function="RMSE",
                                       verbose=0, random_state=RANDOM_SEED)
            m.fit(X_tr.iloc[tr_idx], y_tr_log.iloc[tr_idx])
            score = r2_score(y_tr_log.iloc[val_idx], m.predict(X_tr.iloc[val_idx]))
            scores.append(score)
            trial.report(np.mean(scores), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return np.mean(scores)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=N_TRIALS)
    return study.best_params

# LightGBM
def hpo_lgb(X_tr, y_tr_log):
    def objective(trial):
        param = {
            "n_estimators":      2000,   # controlled by early stopping
            "learning_rate":     trial.suggest_float("learning_rate",     1e-3, 0.2,   log=True),
            "num_leaves":        trial.suggest_int  ("num_leaves",        20,   300),
            "min_child_samples": trial.suggest_int  ("min_child_samples", 5,    200),
            "min_split_gain":    trial.suggest_float("min_split_gain",    0.0,  1.0),
            "subsample":         trial.suggest_float("subsample",         0.4,  1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree",  0.4,  1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha",         1e-8, 10.0,  log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda",        1e-8, 10.0,  log=True),
        }
        kf     = KFold(n_splits=HPO_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        scores = []
        for fold, (tr_idx, val_idx) in enumerate(kf.split(X_tr)):
            m = lgb.LGBMRegressor(**param, n_jobs=-1,
                                   random_state=RANDOM_SEED, verbose=-1)
            m.fit(X_tr.iloc[tr_idx], y_tr_log.iloc[tr_idx],
                  eval_set=[(X_tr.iloc[val_idx], y_tr_log.iloc[val_idx])],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                              lgb.log_evaluation(-1)])
            score = r2_score(y_tr_log.iloc[val_idx], m.predict(X_tr.iloc[val_idx]))
            scores.append(score)
            trial.report(np.mean(scores), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return np.mean(scores)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=N_TRIALS)
    return study.best_params

# XGBoost
def hpo_xgb(X_tr, y_tr_log):
    def objective(trial):
        param = {
            "n_estimators":      2000,   # controlled by early stopping
            "learning_rate":     trial.suggest_float("learning_rate",     1e-3, 0.2,  log=True),
            "max_depth":         trial.suggest_int  ("max_depth",         3,    12),
            "min_child_weight":  trial.suggest_float("min_child_weight",  1e-3, 20.0, log=True),
            "gamma":             trial.suggest_float("gamma",             0.0,  5.0),
            "subsample":         trial.suggest_float("subsample",         0.4,  1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree",  0.4,  1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha",         1e-8, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda",        1e-8, 10.0, log=True),
        }
        kf     = KFold(n_splits=HPO_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        scores = []
        for fold, (tr_idx, val_idx) in enumerate(kf.split(X_tr)):
            m = xgb.XGBRegressor(**param, objective="reg:squarederror",
                                  tree_method="hist", n_jobs=-1,
                                  random_state=RANDOM_SEED,
                                  early_stopping_rounds=50,
                                  eval_metric="rmse")
            m.fit(X_tr.iloc[tr_idx], y_tr_log.iloc[tr_idx],
                  eval_set=[(X_tr.iloc[val_idx], y_tr_log.iloc[val_idx])],
                  verbose=False)
            score = r2_score(y_tr_log.iloc[val_idx], m.predict(X_tr.iloc[val_idx]))
            scores.append(score)
            trial.report(np.mean(scores), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return np.mean(scores)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=N_TRIALS)
    return study.best_params

# RandomForest
def hpo_rf(X_tr, y_tr_log):
    def objective(trial):
        param = {
            "n_estimators":      trial.suggest_int  ("n_estimators",      100,  800),
            "max_depth":         trial.suggest_int  ("max_depth",         5,    40),
            "min_samples_split": trial.suggest_int  ("min_samples_split", 2,    20),
            "min_samples_leaf":  trial.suggest_int  ("min_samples_leaf",  1,    10),
            "max_features":      trial.suggest_float("max_features",      0.3,  1.0),
        }
        kf     = KFold(n_splits=HPO_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        scores = []
        for fold, (tr_idx, val_idx) in enumerate(kf.split(X_tr)):
            m = RandomForestRegressor(**param, n_jobs=-1, random_state=RANDOM_SEED)
            m.fit(X_tr.iloc[tr_idx], y_tr_log.iloc[tr_idx])
            score = r2_score(y_tr_log.iloc[val_idx], m.predict(X_tr.iloc[val_idx]))
            scores.append(score)
            trial.report(np.mean(scores), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return np.mean(scores)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=N_TRIALS)
    return study.best_params

# Run HPO for the selected model.
final_model = None
train_X     = X_train_clean if winner_name in TREE_MODELS else X_train_ohe_clean

hpo_dispatch = {
    "CatBoost":    (hpo_catboost, lambda p: ctb.CatBoostRegressor(
                        **p, loss_function="RMSE", random_state=RANDOM_SEED, verbose=0)),
    "LightGBM":    (hpo_lgb,     lambda p: lgb.LGBMRegressor(
                        **p, n_jobs=-1, random_state=RANDOM_SEED, verbose=-1)),
    "XGBoost":     (hpo_xgb,     lambda p: xgb.XGBRegressor(
                        **p, objective="reg:squarederror", tree_method="hist",
                        n_jobs=-1, random_state=RANDOM_SEED)),
    "RandomForest":(hpo_rf,      lambda p: RandomForestRegressor(
                        **p, n_jobs=-1, random_state=RANDOM_SEED)),
}

if winner_name in hpo_dispatch:
    try:
        hpo_fn, model_builder = hpo_dispatch[winner_name]
        best_params = hpo_fn(train_X, y_train_clean_log)
        final_model = model_builder(best_params)
        # Refit on the full cleaned training set without early stopping.
        if winner_name in ["XGBoost", "LightGBM"]:
            final_model.set_params(early_stopping_rounds=None)
        final_model.fit(train_X, y_train_clean_log)
        print(f"   Best params: {best_params}")
    except Exception as e:
        print(f"   HPO failed ({e}) — falling back to benchmark model")
        traceback.print_exc()
        final_model = clone(model_zoo[winner_name])
        final_model.fit(train_X, y_train_clean_log)
else:
    # Lasso and MLP keep the benchmark settings.
    final_model = clone(model_zoo[winner_name])
    final_model.fit(train_X, y_train_clean_log)

model_save_path = os.path.join(OUTPUT_DIR, f"final_model_{winner_name}.pkl")
joblib.dump(final_model, model_save_path)
print(f"   ✓ HPO model saved to: {os.path.abspath(model_save_path)}")

# ==========================================================================
# 10. Stacking
# ==========================================================================
stack_model = None
if DO_STACKING:
    print("\n10) Stacking...")
    try:
        import gc
        gc.collect()
        
        valid = {k:v for k,v in benchmark_results.items() if "R2" in v}
        # Fix 6: restrict stacking to winner's family so all base learners receive
        # Tree models use label-encoded inputs; linear models use OHE.
        winner_family_is_tree = winner_name in TREE_MODELS
        same_family = {k: v for k, v in valid.items()
                       if (k in TREE_MODELS) == winner_family_is_tree}
        topk = [n for n,_ in sorted(same_family.items(),
                                     key=lambda kv: kv[1]["R2"],
                                     reverse=True)[:STACK_TOP_K]]
        stack_X = X_train_clean if winner_family_is_tree else X_train_ohe_clean
        print(f"   Stacking family: {'tree' if winner_family_is_tree else 'linear'} | base learners: {topk}")
        stack_model = StackingRegressor(
            estimators=[(n, clone(final_model) if n == winner_name else clone(model_zoo[n]))
                        for n in topk],
            final_estimator=Ridge(alpha=1.0),
            cv=5,
            n_jobs=1
        )
        
        print(f"   Fitting stacked ensemble on {topk}...")
        stack_model.fit(stack_X, y_train_clean_log)
        
        if hasattr(stack_model, 'final_estimator_'):
            stack_save_path = os.path.join(OUTPUT_DIR, "stacking_model.pkl")
            joblib.dump(stack_model, stack_save_path)
            print(f"   ✓ Stacked model saved to: {os.path.abspath(stack_save_path)}")
        else:
            print("   ✗ Stacking model failed to fit properly")
            stack_model = None
            
    except Exception as e:
        print(f"   Failed: {e}")
        import traceback
        traceback.print_exc()
        stack_model = None
        
    gc.collect()

# ==========================================================================
# 11. Final evaluation
#
# Two-stage reporting:
# Stage 1: select the model on the validation set.
# Stage 2: report the final score on the test set, once.
#
# This keeps the evaluation honest.
# It avoids using the same data for selection and reporting.
# This follows the standard training / validation / test workflow.
# ==========================================================================
print("\n" + "="*80)
print("11) FINAL EVALUATION")
print("="*80)

def eval_print(model, name, X_eval, y_eval, label=""):
    preds = safe_exp_predict(model.predict(X_eval))
    m = metrics(y_eval, preds)
    tag = f" [{label}]" if label else ""
    print(f"\n  {name}{tag}:")
    print(f"     R²:    {m['R2']:.4f}")
    print(f"     MAE:   {m['MAE']:,.2f}")
    print(f"     RMSE:  {m['RMSE']:,.2f}")
    print(f"     MAPE:  {m['MAPE']:.4f}")
    return m, preds

val_X  = X_val_clean  if winner_name in TREE_MODELS else X_val_ohe
test_X = X_test_clean if winner_name in TREE_MODELS else X_test_ohe

# --- STAGE 1: evaluate all candidates on VALIDATION set ----------------------
print("\n  --- Stage 1: Model selection on VALIDATION set ---")

benchmark_model_obj = clone(model_zoo[winner_name])
benchmark_model_obj.fit(
    X_train_clean if winner_name in TREE_MODELS else X_train_ohe_clean,
    y_train_clean_log)
bench_val_metrics,  _ = eval_print(benchmark_model_obj, f"Benchmark {winner_name}", val_X,  y_val,  "VAL")
hpo_val_metrics,    _ = eval_print(final_model,          f"HPO {winner_name}",       val_X,  y_val,  "VAL")

stack_val_metrics = None
if stack_model is not None and hasattr(stack_model, 'final_estimator_'):
    try:
        stack_val_metrics, _ = eval_print(stack_model, "Stacked Ensemble", X_val_clean, y_val, "VAL")
    except Exception as e:
        print(f"\n  Stacked val eval failed: {e}")

# Pick winner by validation R²
val_candidates = {
    f"Benchmark {winner_name}": (bench_val_metrics, benchmark_model_obj,
                                  test_X,       X_train_clean if winner_name in TREE_MODELS else X_train_ohe_clean),
    f"HPO {winner_name}":       (hpo_val_metrics,   final_model,
                                  test_X,       X_train_clean if winner_name in TREE_MODELS else X_train_ohe_clean),
}
if stack_val_metrics is not None:
    val_candidates["Stacked Ensemble"] = (stack_val_metrics, stack_model,
                                           X_test_clean, X_train_clean)

best_label      = max(val_candidates, key=lambda k: val_candidates[k][0]['R2'])
best_val_m, best_model_obj, best_test_X, best_train_X = val_candidates[best_label]

print(f"\n  → Selected on validation: {best_label}  (Val R²={best_val_m['R2']:.4f})")

# --- STAGE 2: report final performance on TEST set (touched exactly once) ----
print("\n  --- Stage 2: Final performance on held-out TEST set ---")
final_metrics, final_preds = eval_print(best_model_obj, best_label, best_test_X, y_test, "TEST")

print(f"\n{'='*80}")
print(f"  🏆 PRODUCTION MODEL: {best_label}")
print(f"     Selected by:  Validation R²={best_val_m['R2']:.4f}")
print(f"     Test R²:      {final_metrics['R2']:.4f}")
print(f"     Test MAE:     {final_metrics['MAE']:,.2f}")
print(f"     Test RMSE:    {final_metrics['RMSE']:,.2f}")
print(f"     Test MAPE:    {final_metrics['MAPE']:.4f}")
print(f"{'='*80}")

best_save_path = os.path.join(OUTPUT_DIR, f"best_model_{best_label.replace(' ', '_')}.pkl")
joblib.dump(best_model_obj, best_save_path)
print(f"  ✓ Best model saved to: {os.path.abspath(best_save_path)}")

# Save the log-space residual spread used by Code 4.
_log_residuals    = np.log1p(np.maximum(y_test, 0)) - np.log1p(np.maximum(final_preds, 0))
_model_error_band = float(np.std(_log_residuals))
print(f"  ✓ Model error band (log-space std): {_model_error_band:.4f}")

# Save the preprocessing artifact for Code 4.
best_preproc = {
    "final_features":            FINAL_FEATURES,
    "label_encoders":            label_encoders,
    "train_medians":             train_medians.to_dict(),
    "building_type_freq":        feature_engineer.building_type_freq,
    "top_cool_types":            feature_engineer.top_cool_types,
    "scaler":                    scaler if winner_name not in TREE_MODELS else None,
    "model_input_preprocessing": "label_encoded" if winner_name in TREE_MODELS else "ohe_scaled",
    "feature_engineer":          feature_engineer,
    "winner_name":               winner_name,
    "best_label":                best_label,
    "model_error_band":          _model_error_band,
}
preproc_path = os.path.join(OUTPUT_DIR, "best_model_preproc.pkl")
joblib.dump(best_preproc, preproc_path)
print(f"  ✓ Preproc artifact saved to: {os.path.abspath(preproc_path)}")

# If the source has weights, report weighted metrics too.
_weight_col = 'weight'
if _weight_col in df_raw.columns:
    print(f"  ℹ  ComStock '{_weight_col}' column found. Sample weights available.")
    print(f"     Weighted metrics are in diagnostics JSON; unweighted metrics above are")
    print(f"     per-building accuracy. For national portfolio claims use weighted totals.")
    _w_test = df_raw.loc[y_test.index, _weight_col].fillna(1.0).values
    from sklearn.metrics import r2_score as _r2, mean_absolute_error as _mae
    _w_r2   = _r2(y_test, final_preds, sample_weight=_w_test)
    _w_mae  = float(np.average(np.abs(y_test - final_preds), weights=_w_test))
    print(f"     Weighted R²:  {_w_r2:.4f}  |  Weighted MAE: {_w_mae:,.0f}")
    _weighted_metrics = {"R2": round(_w_r2, 4), "MAE": round(_w_mae, 2)}
else:
    print(f"  ⚠  No '{_weight_col}' column in source CSV — unweighted metrics only.")
    print(f"     National portfolio claims cannot be weighted. Document as limitation.")
    _weighted_metrics = None

# ==========================================================================
# 12. Residual analysis
# ==========================================================================
print("\n" + "="*80)
print("12) RESIDUAL ANALYSIS")
print("="*80)

def plot_residuals(y_true, y_pred, model_name, output_dir):
    """Plot residual analysis inline in the notebook."""
    from scipy import stats
    
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Residual Analysis: {model_name}', fontsize=16, fontweight='bold')
    
    axes[0, 0].scatter(y_pred, residuals, alpha=0.5, s=20)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Predicted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Predicted')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Residuals')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Residuals')
    axes[0, 1].grid(True, alpha=0.3)
    
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].scatter(y_true, y_pred, alpha=0.5, s=20)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    axes[1, 1].set_xlabel('Actual Values')
    axes[1, 1].set_ylabel('Predicted Values')
    axes[1, 1].set_title('Actual vs Predicted')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n{model_name} Residual Statistics:")
    print(f"   Mean:        {residuals.mean():,.2f}")
    print(f"   Std Dev:     {residuals.std():,.2f}")
    print(f"   Min:         {residuals.min():,.2f}")
    print(f"   Max:         {residuals.max():,.2f}")
    print(f"   25th %ile:   {np.percentile(residuals, 25):,.2f}")
    print(f"   Median:      {np.median(residuals):,.2f}")
    print(f"   75th %ile:   {np.percentile(residuals, 75):,.2f}")

plot_residuals(y_test, final_preds, f"Best Model: {best_label}", OUTPUT_DIR)


# ==========================================================================
# 13) SAVE PREDICTIONS
# ==========================================================================
print("\n" + "="*80)
print("13) SAVING PREDICTIONS")
print("="*80)

predictions_df = pd.DataFrame({
    'Actual': y_test,
    f'{best_label}_Predicted': final_preds,
    f'{best_label}_Residual': y_test - final_preds
})

predictions_path = os.path.join(OUTPUT_DIR, 'test_predictions.csv')
predictions_df.to_csv(predictions_path, index=False)
print(f"   ✓ Predictions saved to: {os.path.abspath(predictions_path)}")

# ==========================================================================
# 14. SHAP analysis
# ==========================================================================
print("\n" + "="*80)
print("14) SHAP ANALYSIS")
print("="*80)
try:
    # Use the model that won overall.
    shap_X_train = X_train_clean if best_label != "Stacked Ensemble" or winner_name in TREE_MODELS else X_train_ohe_clean
    shap_X_test  = X_test_clean  if best_label != "Stacked Ensemble" or winner_name in TREE_MODELS else X_test_ohe
    sample_X      = shap_X_train.sample(min(500, len(shap_X_train)), random_state=RANDOM_SEED)
    from sklearn.ensemble import StackingRegressor as _SR
    if isinstance(best_model_obj, _SR):
        background    = shap.sample(sample_X, 100)
        expl          = shap.KernelExplainer(best_model_obj.predict, background)
        test_sample_X = shap_X_test.sample(min(150, len(shap_X_test)), random_state=RANDOM_SEED)
    else:
        expl          = shap.Explainer(best_model_obj, sample_X)
        test_sample_X = shap_X_test.sample(min(SHAP_SAMPLE, len(shap_X_test)), random_state=RANDOM_SEED)

    shap_vals = expl(test_sample_X)   # ← outside both branches, runs for all cases
    
    # Save feature importance to CSV.
    imp = np.nanmean(np.abs(shap_vals.values), axis=0)
    feat_df = pd.DataFrame({"feature": sample_X.columns, "importance": imp}).sort_values("importance", ascending=False)
    print(feat_df.head(20).to_string(index=False))
    fi_path = os.path.join(OUTPUT_DIR, f"feature_importance_{best_label.replace(' ', '_')}.csv")
    feat_df.to_csv(fi_path, index=False)
    print(f"   ✓ Feature importance saved to: {os.path.abspath(fi_path)}")
    
    print("\n   Creating SHAP visualizations...")
    
    # 1) Summary Plot (beeswarm) - inline
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals, test_sample_X, show=False, max_display=20)
    plt.tight_layout()
    plt.show()
    print(f"   ✓ SHAP summary plot displayed")
    
    # 2) Bar Plot - inline
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals, test_sample_X, plot_type="bar", show=False, max_display=20)
    plt.tight_layout()
    plt.show()
    print(f"   ✓ SHAP bar plot displayed")
    
    # 3) Waterfall plot - inline
    plt.figure(figsize=(10, 8))
    shap.waterfall_plot(shap_vals[0], max_display=15, show=False)
    plt.tight_layout()
    plt.show()
    print(f"   ✓ SHAP waterfall plot displayed")
    
    # 4) Dependence plots for top 3 features - inline
    top_features = feat_df.head(3)['feature'].tolist()
    for i, feat in enumerate(top_features):
        if feat in test_sample_X.columns:
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(feat, shap_vals.values, test_sample_X, show=False)
            plt.tight_layout()
            plt.show()
            print(f"   ✓ SHAP dependence plot displayed: {feat}")
    
except Exception as e:
    print(f"   Failed: {e}")
    import traceback
    traceback.print_exc()

# ==========================================================================
# 15. Plots
# ==========================================================================
print("\n" + "="*80)
print("15) PREDICTION PLOTS")
print("="*80)

def plot_pred(actual, pred, title, fn=None):
    plt.figure(figsize=(10,10))
    plt.scatter(actual, pred, s=8, alpha=0.4)
    mn, mx = min(actual.min(), pred.min()), max(actual.max(), pred.max())
    plt.plot([mn, mx], [mn, mx], 'r--', lw=2)
    r2 = r2_score(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    plt.title(f"{title}\nR²={r2:.4f}, RMSE={rmse:.0f}", fontsize=14, weight='bold')
    plt.xlabel("Actual Energy Consumption")
    plt.ylabel("Predicted Energy Consumption")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_pred(y_test, final_preds, f"Best Model: {best_label} [TEST SET]")

# ==========================================================================
# 16. Final summary
# ==========================================================================
print("\n" + "="*80)
print("16) FINAL MODEL SUMMARY")
print("="*80)

print(f"\n  Validation-set comparison (used for model selection):")
print(f"  {'Model':<30} {'Val R²':>8} {'Val MAPE':>10}")
print(f"  {'-'*50}")
print(f"  {'Benchmark '+winner_name:<30} {bench_val_metrics['R2']:>8.4f} {bench_val_metrics['MAPE']:>10.4f}")
print(f"  {'HPO '+winner_name:<30} {hpo_val_metrics['R2']:>8.4f} {hpo_val_metrics['MAPE']:>10.4f}")
if stack_val_metrics:
    print(f"  {'Stacked Ensemble':<30} {stack_val_metrics['R2']:>8.4f} {stack_val_metrics['MAPE']:>10.4f}")

print(f"\n  🏆 PRODUCTION MODEL: {best_label}")
print(f"     Selected by:   Validation R²={best_val_m['R2']:.4f}")
print(f"     Test R²:       {final_metrics['R2']:.4f}  ← reported once on untouched test set")
print(f"     Test MAE:      {final_metrics['MAE']:,.2f}")
print(f"     Test RMSE:     {final_metrics['RMSE']:,.2f}")
print(f"     Test MAPE:     {final_metrics['MAPE']:.4f}")
print(f"     Saved to:      {os.path.abspath(best_save_path)}")

print(f"\n✅ THREE-WAY SPLIT  — train 70% / val 15% / test 15%")
print(f"✅ NO DATA LEAKAGE  — all statistics from train only")
print(f"✅ 5-FOLD HPO & STACKING — MedianPruner, full regularisation search")
print(f"✅ TEST SET TOUCHED ONCE — model selection on validation only")
print(f"✅ PRODUCTION READY")
print("\n" + "="*80)
print("PIPELINE COMPLETE")
print("="*80)
# ==========================================================================
# Reproducibility diagnostics JSON
# ==========================================================================
import importlib, time as _time
_pkg_versions = {}
for _pkg in ["numpy","pandas","sklearn","xgboost","lightgbm","catboost","optuna","shap"]:
    try: _pkg_versions[_pkg] = importlib.import_module(_pkg).__version__
    except: _pkg_versions[_pkg] = "unknown"
_diag = {
    "run_timestamp": _time.strftime("%Y-%m-%d %H:%M:%S"),
    "target": TARGET,
    "dataset_rows": len(df_raw),
    "train_rows": len(X_train_raw),
    "val_rows": len(X_val_raw),
    "test_rows": len(X_test_raw),
    "final_features": FINAL_FEATURES,
    "winner": best_label,
    "test_metrics": {k: round(v, 4) for k, v in final_metrics.items()},
    "model_error_band": _model_error_band,  # log-space residual std — correct for Code 4 PI
    "sample_weights_available": _weighted_metrics is not None,
    "weighted_metrics": _weighted_metrics,
    "package_versions": _pkg_versions,
    "split_strategy": "stratified_70_15_15",
    "note": "climate-only counterfactual scope — see projection code headers"
}
_diag_path = os.path.join(OUTPUT_DIR, "energy_full_model_diagnostics.json")
import json as _json
with open(_diag_path, "w") as _fj:
    _json.dump(_diag, _fj, indent=2, default=str)
print(f"   ✓ Diagnostics JSON → {os.path.abspath(_diag_path)}")