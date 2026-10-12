"""Marine Pollution Impact Score (0-100) & Biodiversity Loss Prediction.

Multi-modal AI Fusion Engine combining:
- Plastic waste detection & coverage
- Fish behavioral anomaly index
- Coral reef bleaching severity index
- Ocean water quality parameters (temp, pH, DO, salinity, turbidity)

Trained with XGBoost to predict the Marine Pollution Impact Score (0-100) and a
projected biodiversity loss %.

IMPORTANT - what this model is (and is not):
There is no public dataset that pairs plastic / fish / coral / water measurements with
measured biodiversity loss, so the training targets are produced by an explicit,
hand-specified expert formula (see generate_synthetic_marine_training_data) applied to
synthetic survey data. XGBoost therefore learns that formula; it is a documented expert
scoring model, not one fitted to observed ecological outcomes. The direction of each
effect is motivated by published work, e.g.
  - Lamb et al. 2018, Science 359:460 - plastic contact raises coral disease likelihood
  - Hughes et al. 2017, Nature 543:373 - heat stress drives mass coral bleaching
  - Jambeck et al. 2015, Science 347:768 - scale of plastic inputs to the ocean
but the weights themselves are assumptions. Replacing them with weights fitted to real
survey outcomes (e.g. OBIS / Reef Life Survey) is future work.
"""
from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from src.config import PROJECT_ROOT

IMPACT_MODEL_DIR = PROJECT_ROOT / "models" / "impact_model"
IMPACT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_FILE = IMPACT_MODEL_DIR / "xgboost_mpis_model.joblib"
BIODIV_MODEL_FILE = IMPACT_MODEL_DIR / "xgboost_biodiv_model.joblib"

FEATURE_NAMES = [
    "plastic_coverage_percentage",
    "plastic_count",
    "debris_count",
    "fish_behavior_anomaly_index",
    "fish_count",
    "coral_present",
    "coral_bleaching_severity_index",
    "coral_cover_percentage",
    "water_quality_index",
    "temperature_c",
    "ph",
    "dissolved_oxygen_mg_l",
    "salinity_psu",
    "turbidity_ntu",
]


# Share of synthetic sites without coral (e.g. deep-sea / soft-bottom, like TrashCan).
NON_REEF_FRACTION = 0.35

# Expert weights of the impact score. Reef sites: plastic, fish stress, coral
# bleaching and poor water quality. Non-reef sites have no coral term, so its
# weight is spread over the other three (no coral != healthy coral).
MPIS_WEIGHTS_REEF = {"plastic": 0.32, "fish": 0.22, "coral": 0.24, "water": 0.22}
MPIS_WEIGHTS_NON_REEF = {"plastic": 0.42, "fish": 0.29, "coral": 0.0, "water": 0.29}


def generate_synthetic_marine_training_data(n_samples: int = 1500, random_seed: int = 42) -> pd.DataFrame:
    """Synthetic survey data + expert-formula targets (see module docstring).

    Input distributions are plausible ranges, not fitted to a real survey dataset.
    """
    rng = np.random.default_rng(random_seed)

    # 1. Plastic variables
    plastic_cov = rng.exponential(scale=3.5, size=n_samples)
    plastic_cov = np.clip(plastic_cov, 0.0, 45.0)
    plastic_count = np.clip(rng.poisson(lam=plastic_cov * 0.8 + 0.5), 0, 15)
    debris_count = plastic_count + rng.poisson(lam=1.2, size=n_samples)

    # 2. Water Quality variables
    temp = rng.normal(loc=27.2, scale=2.1, size=n_samples)
    temp = np.clip(temp, 21.0, 33.0)

    ph = rng.normal(loc=8.15 - (plastic_cov * 0.006), scale=0.18, size=n_samples)
    ph = np.clip(ph, 7.55, 8.40)

    do = rng.normal(loc=6.9 - (plastic_cov * 0.07), scale=1.1, size=n_samples)
    do = np.clip(do, 2.2, 9.2)

    sal = rng.normal(loc=35.2, scale=1.4, size=n_samples)
    sal = np.clip(sal, 29.0, 39.0)

    turb = rng.exponential(scale=1.8 + plastic_cov * 0.2, size=n_samples)
    turb = np.clip(turb, 0.2, 18.0)

    # Compute WQI from parameters
    wqi = (
        np.clip(100 - np.maximum(0, temp - 27.8) * 20, 0, 100) * 0.25
        + np.clip(100 - np.abs(ph - 8.2) * 150, 0, 100) * 0.20
        + np.clip((do / 7.0) * 100, 0, 100) * 0.30
        + np.clip(100 - np.maximum(0, turb - 2.0) * 8, 0, 100) * 0.15
        + np.clip(100 - np.abs(sal - 35.0) * 12, 0, 100) * 0.10
    )
    wqi = np.clip(wqi, 10.0, 98.0)

    # 3. Coral variables (zero at non-reef sites)
    coral_present = (rng.random(n_samples) >= NON_REEF_FRACTION).astype(float)
    coral_cover = np.clip(rng.normal(loc=45.0 - plastic_cov * 0.8, scale=18.0), 5.0, 85.0) * coral_present
    temp_bleach_risk = np.maximum(0, temp - 28.2) * 18.0
    bsi = np.clip(
        temp_bleach_risk + (plastic_cov * 1.2) + ((100 - wqi) * 0.35) + rng.normal(0, 5, n_samples),
        0.0,
        100.0,
    ) * coral_present

    # 4. Fish variables
    fish_count = np.clip(rng.poisson(lam=np.maximum(1.0, 8.0 - plastic_cov * 0.25)), 0, 25)
    fish_bai = np.clip(
        20.0 + (plastic_cov * 2.1) + ((100 - wqi) * 0.4) + np.maximum(0, 5.5 - do) * 8.0 + rng.normal(0, 6, n_samples),
        5.0,
        98.0,
    )

    # True Marine Pollution Impact Score (MPIS, 0-100)
    # Ground truth formula incorporating multi-modal interactions + non-linear saturation
    def weight(key):
        return np.where(coral_present > 0, MPIS_WEIGHTS_REEF[key], MPIS_WEIGHTS_NON_REEF[key])

    mpis = (
        weight("plastic") * (np.clip(plastic_cov * 3.8 + plastic_count * 3.0, 0, 100))
        + weight("fish") * fish_bai
        + weight("coral") * bsi
        + weight("water") * (100.0 - wqi)
        + rng.normal(0, 2.5, n_samples)
    )
    mpis = np.clip(mpis, 0.0, 100.0)

    # Predicted Biodiversity Loss (% species richness decline)
    biodiv_loss = np.clip(
        (mpis * 0.72) + (bsi * 0.18) + (np.maximum(0, 4.5 - do) * 5.0) + rng.normal(0, 3.0, n_samples),
        0.0,
        96.0,
    )

    df = pd.DataFrame(
        {
            "plastic_coverage_percentage": plastic_cov,
            "plastic_count": plastic_count,
            "debris_count": debris_count,
            "fish_behavior_anomaly_index": fish_bai,
            "fish_count": fish_count,
            "coral_present": coral_present,
            "coral_bleaching_severity_index": bsi,
            "coral_cover_percentage": coral_cover,
            "water_quality_index": wqi,
            "temperature_c": temp,
            "ph": ph,
            "dissolved_oxygen_mg_l": do,
            "salinity_psu": sal,
            "turbidity_ntu": turb,
            "marine_pollution_impact_score": mpis,
            "biodiversity_loss_pct": biodiv_loss,
        }
    )
    return df


class MarineImpactScoringEngine:
    """Predicts Marine Pollution Impact Score and Biodiversity Loss using trained XGBoost models."""

    def __init__(self, force_retrain: bool = False):
        self.mpis_model: XGBRegressor | None = None
        self.biodiv_model: XGBRegressor | None = None
        self.training_df: pd.DataFrame | None = None
        self.load_or_train_models(force_retrain=force_retrain)

    def load_or_train_models(self, force_retrain: bool = False):
        # Deterministic (seeded), so regenerating it on load gives the same data the
        # saved models were trained on -- SHAP needs it as background data.
        df = generate_synthetic_marine_training_data(n_samples=2500, random_seed=42)
        self.training_df = df

        if not force_retrain and MODEL_FILE.exists() and BIODIV_MODEL_FILE.exists():
            try:
                mpis_model = joblib.load(MODEL_FILE)
                biodiv_model = joblib.load(BIODIV_MODEL_FILE)
                # Saved models from an older feature set are retrained below
                if list(getattr(mpis_model, "feature_names_in_", [])) == FEATURE_NAMES:
                    self.mpis_model, self.biodiv_model = mpis_model, biodiv_model
                    return
            except Exception:
                pass

        # Train new XGBoost models (held-out split only to report how well the formula is learned)
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        X = train_df[FEATURE_NAMES]
        y_mpis = train_df["marine_pollution_impact_score"]
        y_biodiv = train_df["biodiversity_loss_pct"]

        self.mpis_model = XGBRegressor(
            n_estimators=160,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
        )
        self.mpis_model.fit(X, y_mpis)

        self.biodiv_model = XGBRegressor(
            n_estimators=140,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
        )
        self.biodiv_model.fit(X, y_biodiv)

        X_test = test_df[FEATURE_NAMES]
        self.fit_report = {
            "note": "held-out split of the SYNTHETIC data: measures how well XGBoost reproduces the expert formula",
            "mpis_r2": round(float(r2_score(test_df["marine_pollution_impact_score"], self.mpis_model.predict(X_test))), 3),
            "mpis_mae": round(float(mean_absolute_error(test_df["marine_pollution_impact_score"], self.mpis_model.predict(X_test))), 2),
            "biodiv_r2": round(float(r2_score(test_df["biodiversity_loss_pct"], self.biodiv_model.predict(X_test))), 3),
        }
        (IMPACT_MODEL_DIR / "fit_report.json").write_text(json.dumps(self.fit_report, indent=2))

        joblib.dump(self.mpis_model, MODEL_FILE)
        joblib.dump(self.biodiv_model, BIODIV_MODEL_FILE)

    def predict(
        self,
        plastic_stats: dict,
        fish_stats: dict,
        coral_stats: dict,
        water_quality_stats: dict,
    ) -> dict:
        """Computes the Marine Pollution Impact Score and Biodiversity Loss."""
        features = {
            "plastic_coverage_percentage": float(plastic_stats.get("plastic_coverage_percentage", 0.0)),
            "plastic_count": float(plastic_stats.get("plastic_count", 0)),
            "debris_count": float(plastic_stats.get("debris_count", 0)),
            "fish_behavior_anomaly_index": float(fish_stats.get("behavioral_anomaly_index", 20.0)),
            "fish_count": float(fish_stats.get("fish_count", 0)),
            "coral_present": float(coral_stats.get("coral_present", True)),
            "coral_bleaching_severity_index": float(coral_stats.get("bleaching_severity_index", 15.0)),
            "coral_cover_percentage": float(coral_stats.get("coral_cover_percentage", 35.0)),
            "water_quality_index": float(water_quality_stats.get("water_quality_index", 85.0)),
            "temperature_c": float(water_quality_stats.get("temperature_c", 26.5)),
            "ph": float(water_quality_stats.get("ph", 8.2)),
            "dissolved_oxygen_mg_l": float(water_quality_stats.get("dissolved_oxygen_mg_l", 7.2)),
            "salinity_psu": float(water_quality_stats.get("salinity_psu", 35.0)),
            "turbidity_ntu": float(water_quality_stats.get("turbidity_ntu", 1.0)),
        }

        X_input = pd.DataFrame([features])[FEATURE_NAMES]
        mpis_raw = float(self.mpis_model.predict(X_input)[0])
        mpis = float(np.clip(mpis_raw, 0.0, 100.0))

        biodiv_raw = float(self.biodiv_model.predict(X_input)[0])
        biodiv_loss = float(np.clip(biodiv_raw, 0.0, 100.0))

        # Classify risk level
        if mpis < 25.0:
            risk_tier = "Low Impact (Pristine / Healthy Ecosystem)"
            risk_color = "#2ecc71"
            urgency = "Routine Monitoring"
        elif mpis < 50.0:
            risk_tier = "Moderate Impact (Vulnerable / Early Stress)"
            risk_color = "#f39c12"
            urgency = "Enhanced Surveillance & Targeted Waste Cleanup"
        elif mpis < 75.0:
            risk_tier = "High Impact (Threatened Habitat / Acute Degradation)"
            risk_color = "#e67e22"
            urgency = "Active Intervention & Debris Containment"
        else:
            risk_tier = "Critical Impact (Severe Ecosystem Collapse Hazard)"
            risk_color = "#e74c3c"
            urgency = "Emergency Ecological Recovery Action"

        # Projected biodiversity indicators
        biomass_loss = float(np.clip(biodiv_loss * 0.88 + (100 - water_quality_stats.get("water_quality_index", 85)) * 0.1, 0, 100))
        shannon_drop = float(np.clip(biodiv_loss * 0.035, 0.0, 3.2))

        return {
            "marine_pollution_impact_score": round(mpis, 1),
            "risk_tier": risk_tier,
            "risk_color": risk_color,
            "urgency": urgency,
            "predicted_biodiversity_loss_pct": round(biodiv_loss, 1),
            "predicted_biomass_decline_pct": round(biomass_loss, 1),
            "projected_shannon_diversity_drop": round(shannon_drop, 2),
            "features": features,
            "feature_df": X_input,
        }


def plot_impact_score_gauge(mpis: float, risk_tier: str, save_path: str | None = None) -> plt.Figure:
    """Generates an aesthetic gauge / meter visual for the Marine Pollution Impact Score."""
    fig, ax = plt.subplots(figsize=(6, 3.6))

    # Gradient background bands
    ax.barh([0], [25], color="#2ecc71", alpha=0.35, label="Low (0-25)")
    ax.barh([0], [25], left=[25], color="#f1c40f", alpha=0.35, label="Moderate (26-50)")
    ax.barh([0], [25], left=[50], color="#e67e22", alpha=0.35, label="High (51-75)")
    ax.barh([0], [25], left=[75], color="#e74c3c", alpha=0.35, label="Critical (76-100)")

    # Score pointer
    color = "#2ecc71" if mpis < 25 else "#f39c12" if mpis < 50 else "#e67e22" if mpis < 75 else "#e74c3c"
    ax.scatter([mpis], [0], color=color, s=260, zorder=5, edgecolor="black", linewidth=2)
    ax.axvline(mpis, color=color, linewidth=2.5, linestyle="-")

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel("Marine Pollution Impact Score (0 - 100)", fontsize=10)
    ax.set_title(f"Impact Score: {mpis:.1f} / 100\n{risk_tier}", fontsize=11, fontweight="bold", pad=12)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=4, fontsize=8)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
