from typing import List, Dict, Any
import pandas as pd
from app.repositories.detection_repository import detection_repo
from app.schemas.schemas import AnalyticsSummary, AlertItem

class AnalyticsService:
    @staticmethod
    def _number(record: Dict[str, Any], key: str, default: float = 0.0) -> float:
        try:
            value = record.get(key, default)
            return float(value) if value is not None else default
        except (TypeError, ValueError):
            return default

    @staticmethod
    def get_summary() -> AnalyticsSummary:
        raw_items = detection_repo.get_raw_list()
        total = len(raw_items)
        industrial_assoc = sum(
            1 for r in raw_items
            if AnalyticsService._number(r, "dist_to_industrial", 99) < 1.0
            or AnalyticsService._number(r, "industrial_count_5km") > 0
        )
        active_thermal = sum(1 for r in raw_items if r.get("true_class") in ["industrial_fire", "gas_flare"])
        persistent = sum(1 for r in raw_items if AnalyticsService._number(r, "persistence_score") >= 0.1)
        high_conf = sum(1 for r in raw_items if AnalyticsService._number(r, "confidence") >= 90.0)
        high_risk = sum(1 for r in raw_items if AnalyticsService._number(r, "frp") >= 50.0 or (
            r.get("true_class") == "industrial_fire"
            and AnalyticsService._number(r, "classification_confidence") >= 0.66
        ))

        return AnalyticsSummary(
            total_detections=total,
            industrial_associated_detections=industrial_assoc,
            active_thermal_sources=active_thermal,
            persistent_sources=persistent,
            high_confidence_events=high_conf,
            high_risk_events=high_risk
        )

    @staticmethod
    def get_temporal_trends() -> List[Dict[str, Any]]:
        raw_items = detection_repo.get_raw_list()
        if not raw_items:
            return []
        df = pd.DataFrame(raw_items)
        if "acq_date" not in df.columns:
            return []
        frp = df["frp"] if "frp" in df.columns else pd.Series(0.0, index=df.index)
        df["frp"] = pd.to_numeric(frp, errors="coerce").fillna(0)
        if "true_class" not in df.columns:
            df["true_class"] = ""

        grouped = df.groupby("acq_date").agg(
            total_events=("acq_date", "count"),
            avg_frp=("frp", "mean"),
            industrial_events=("true_class", lambda x: int((x.isin(["industrial_fire", "gas_flare"])).sum()))
        ).reset_index()

        grouped["avg_frp"] = grouped["avg_frp"].round(2)
        return grouped.sort_values("acq_date").to_dict(orient="records")

    @staticmethod
    def get_classification_breakdown() -> List[Dict[str, Any]]:
        raw_items = detection_repo.get_raw_list()
        if not raw_items:
            return []
        df = pd.DataFrame(raw_items)
        if "true_class" not in df.columns:
            return []

        # Map classifications to expected frontend classes
        class_mapping = {
            'agricultural_burning': 'agricultural_burning',
            'industrial_fire': 'industrial_fire',
            'gas_flare': 'gas_flare',
            'mining_activity': 'mining_activity'
        }

        df['mapped_class'] = df['true_class'].fillna('').map(class_mapping).fillna('other_thermal_anomaly')
        counts = df["mapped_class"].value_counts().reset_index()
        counts.columns = ["classification", "count"]
        total = len(df)
        counts["percentage"] = (counts["count"] / total * 100).round(2)
        return counts.to_dict(orient="records")

    @staticmethod
    def get_regional_breakdown() -> List[Dict[str, Any]]:
        raw_items = detection_repo.get_raw_list()
        if not raw_items:
            return []
        df = pd.DataFrame(raw_items)

        # Create regions based on geographic coordinates if not present
        if "region" not in df.columns:
            df["region"] = "Unknown"
        else:
            df["region"] = df["region"].fillna("Unknown").astype(str)

        # Normalize optional persistence values from datasets that may omit them.
        persistence = df.get("persistence_score", pd.Series(0.0, index=df.index))
        df["persistence_score"] = pd.to_numeric(persistence, errors="coerce").fillna(0)
        frp = df.get("frp", pd.Series(0.0, index=df.index))
        df["frp"] = pd.to_numeric(frp, errors="coerce").fillna(0)
        if "id" not in df.columns:
            df["id"] = range(len(df))

        industrial_fire = df.get("true_class", pd.Series("", index=df.index)).eq("industrial_fire")
        model_confidence = df.get("classification_confidence", pd.Series(0.0, index=df.index))
        model_confidence = pd.to_numeric(model_confidence, errors="coerce").fillna(0)
        df["high_risk"] = (df["frp"] >= 50) | (industrial_fire & (model_confidence >= 0.66))

        grouped = df.groupby("region").agg(
            total_detections=("id", "count"),
            avg_persistence=("persistence_score", "mean"),
            high_risk_count=("high_risk", "sum")
        ).reset_index()

        grouped["avg_persistence"] = grouped["avg_persistence"].round(3)
        return grouped.sort_values("total_detections", ascending=False).to_dict(orient="records")

    @staticmethod
    def get_persistence_clusters() -> List[Dict[str, Any]]:
        raw_items = detection_repo.get_raw_list()
        # Persistence scores are normalized to [0, 1] by the repository, but
        # tolerate legacy data with strings or missing values.
        persistent_items = []
        for record in raw_items:
            score = AnalyticsService._number(record, "persistence_score")
            if score >= 0.1:
                persistent_items.append((score, record))
        persistent_items.sort(key=lambda item: item[0], reverse=True)

        clusters = []
        for score, r in persistent_items[:20]: # Top persistent anomalies
            clusters.append({
                "detection_id": r.get("id", "unknown"),
                "region": r.get("region", "South Asia"),
                "latitude": r.get("latitude", 0),
                "longitude": r.get("longitude", 0),
                "persistence_score": score,
                "nearest_facility": r.get("nearest_facility_type", "unknown"),
                "detection_count_30d": r.get("detection_count_30d", 1),
                "status": "ACTIVE_OPERATIONAL_FLARE" if score > 0.8 else "RECURRING_INDUSTRIAL"
            })
        return clusters

class AlertService:
    @staticmethod
    def get_alerts() -> List[AlertItem]:
        raw_items = detection_repo.get_raw_list()
        alerts = []
        alert_id = 1

        def number(record: Dict[str, Any], key: str, default: float = 0.0) -> float:
            try:
                return float(record.get(key, default) or default)
            except (TypeError, ValueError):
                return default

        for r in raw_items:
            model_conf = number(r, "classification_confidence")
            persistence_score = number(r, "persistence_score")
            frp = number(r, "frp")
            distance = number(r, "dist_to_industrial")
            facility = r.get("nearest_facility_type") or "unknown facility"

            # Rule 1: High-confidence industrial fire (model classification)
            if r.get("true_class") == "industrial_fire" and model_conf >= 0.66:
                alerts.append(AlertItem(
                    id=f"ALT-2025-{alert_id:04d}",
                    detection_id=str(r.get("id", "unknown")),
                    timestamp=f"{r.get('acq_date', '2024-01-01')} {r.get('acq_time', '00:00')} UTC",
                    latitude=r.get("latitude", 0),
                    longitude=r.get("longitude", 0),
                    severity="CRITICAL",
                    type="HIGH_CONFIDENCE_INDUSTRIAL_FIRE",
                    explanation=(
                        f"Industrial fire classified with {model_conf * 100:.1f}% model confidence "
                        f"within {distance:.2f} km of {facility}. "
                        f"FRP: {frp:g} MW."
                    ),
                    region=r.get("region", "South Asia")
                ))
                alert_id += 1

            # Rule 2: Persistent flare / recurring thermal source
            elif r.get("true_class") in ("gas_flare", "mining_activity") and persistence_score >= 0.1:
                alerts.append(AlertItem(
                    id=f"ALT-2025-{alert_id:04d}",
                    detection_id=str(r.get("id", "unknown")),
                    timestamp=f"{r.get('acq_date', '2024-01-01')} {r.get('acq_time', '00:00')} UTC",
                    latitude=r.get("latitude", 0),
                    longitude=r.get("longitude", 0),
                    severity="MEDIUM",
                    type="PERSISTENT_THERMAL_SOURCE",
                    explanation=(
                        f"Recurring {r.get('true_class', 'thermal').replace('_', ' ')} detected "
                        f"({r.get('detection_count_30d', 0)} prior observations at this location)."
                    ),
                    region=r.get("region", "South Asia")
                ))
                alert_id += 1

            # Rule 3: Unusually high FRP event
            elif frp >= 50.0:
                alerts.append(AlertItem(
                    id=f"ALT-2025-{alert_id:04d}",
                    detection_id=str(r.get("id", "unknown")),
                    timestamp=f"{r.get('acq_date', '2024-01-01')} {r.get('acq_time', '00:00')} UTC",
                    latitude=r.get("latitude", 0),
                    longitude=r.get("longitude", 0),
                    severity="HIGH",
                    type="HIGH_FRP_EVENT",
                    explanation=f"Thermal anomaly exceeding 50 MW Fire Radiative Power (FRP: {frp:g} MW).",
                    region=r.get("region", "South Asia")
                ))
                alert_id += 1

        # Sort CRITICAL first, then HIGH, then MEDIUM
        severity_rank = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        alerts.sort(key=lambda x: severity_rank.get(x.severity, 4))
        return alerts
