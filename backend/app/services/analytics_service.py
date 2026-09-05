from typing import List, Dict, Any
import pandas as pd
from app.repositories.detection_repository import detection_repo
from app.schemas.schemas import AnalyticsSummary, AlertItem

class AnalyticsService:
    @staticmethod
    def get_summary() -> AnalyticsSummary:
        raw_items = detection_repo.get_raw_list()
        total = len(raw_items)
        industrial_assoc = sum(1 for r in raw_items if r.get("is_industrial") == 1 or r.get("dist_to_industrial", 99) < 1.0)
        active_thermal = sum(1 for r in raw_items if r.get("true_class") == "industrial_thermal_source")
        persistent = sum(1 for r in raw_items if r.get("persistence_score", 0) >= 0.70)
        high_conf = sum(1 for r in raw_items if r.get("confidence", 0) >= 90.0)
        high_risk = sum(1 for r in raw_items if r.get("frp", 0) >= 50.0 or (r.get("true_class") == "industrial_fire" and r.get("confidence", 0) >= 80))

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

        grouped = df.groupby("acq_date").agg(
            total_events=("id", "count"),
            avg_frp=("frp", "mean"),
            industrial_events=("is_industrial", lambda x: int((x == 1).sum()))
        ).reset_index()

        grouped["avg_frp"] = grouped["avg_frp"].round(2)
        return grouped.sort_values("acq_date").to_dict(orient="records")

    @staticmethod
    def get_classification_breakdown() -> List[Dict[str, Any]]:
        raw_items = detection_repo.get_raw_list()
        if not raw_items:
            return []
        df = pd.DataFrame(raw_items)
        counts = df["true_class"].value_counts().reset_index()
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
        grouped = df.groupby("region").agg(
            total_detections=("id", "count"),
            avg_persistence=("persistence_score", "mean"),
            high_risk_count=("frp", lambda x: int((x > 50).sum()))
        ).reset_index()

        grouped["avg_persistence"] = grouped["avg_persistence"].round(3)
        return grouped.sort_values("total_detections", ascending=False).to_dict(orient="records")

    @staticmethod
    def get_persistence_clusters() -> List[Dict[str, Any]]:
        raw_items = detection_repo.get_raw_list()
        persistent_items = [r for r in raw_items if r.get("persistence_score", 0) >= 0.60]

        clusters = []
        for r in persistent_items[:20]: # Top persistent anomalies
            clusters.append({
                "detection_id": r["id"],
                "region": r["region"],
                "latitude": r["latitude"],
                "longitude": r["longitude"],
                "persistence_score": r["persistence_score"],
                "nearest_facility": r.get("nearest_facility_type", "unknown"),
                "detection_count_30d": r.get("detection_count_30d", 1),
                "status": "ACTIVE_OPERATIONAL_FLARE" if r.get("persistence_score", 0) > 0.8 else "RECURRING_INDUSTRIAL"
            })
        return clusters

class AlertService:
    @staticmethod
    def get_alerts() -> List[AlertItem]:
        raw_items = detection_repo.get_raw_list()
        alerts = []
        alert_id = 1

        for r in raw_items:
            # Rule 1: High Confidence Industrial Fire
            if r.get("true_class") == "industrial_fire" and r.get("confidence", 0) >= 80:
                alerts.append(AlertItem(
                    id=f"ALT-2025-{alert_id:04d}",
                    detection_id=r["id"],
                    timestamp=f"{r['acq_date']} {r['acq_time']} UTC",
                    latitude=r["latitude"],
                    longitude=r["longitude"],
                    severity="CRITICAL",
                    type="HIGH_CONFIDENCE_INDUSTRIAL_FIRE",
                    explanation=f"Industrial fire detected within {r.get('dist_to_industrial', 0):.2f} km of {r.get('nearest_facility_type', 'facility')}. High FRP ({r.get('frp')} MW).",
                    region=r["region"]
                ))
                alert_id += 1

            # Rule 2: Persistent Flare/Thermal Source Anomaly
            elif r.get("true_class") == "industrial_thermal_source" and r.get("persistence_score", 0) >= 0.85:
                alerts.append(AlertItem(
                    id=f"ALT-2025-{alert_id:04d}",
                    detection_id=r["id"],
                    timestamp=f"{r['acq_date']} {r['acq_time']} UTC",
                    latitude=r["latitude"],
                    longitude=r["longitude"],
                    severity="MEDIUM",
                    type="PERSISTENT_THERMAL_SOURCE",
                    explanation=f"Continuous persistent flare/thermal emission detected ({r.get('detection_count_30d')} observations in trailing 30 days).",
                    region=r["region"]
                ))
                alert_id += 1

            # Rule 3: Unusually High FRP Event
            elif r.get("frp", 0) >= 100.0:
                alerts.append(AlertItem(
                    id=f"ALT-2025-{alert_id:04d}",
                    detection_id=r["id"],
                    timestamp=f"{r['acq_date']} {r['acq_time']} UTC",
                    latitude=r["latitude"],
                    longitude=r["longitude"],
                    severity="HIGH",
                    type="HIGH_FRP_EVENT",
                    explanation=f"Thermal anomaly exceeding 100 MW Fire Radiative Power (FRP: {r.get('frp')} MW).",
                    region=r["region"]
                ))
                alert_id += 1

        # Sort CRITICAL first, then HIGH, then MEDIUM
        severity_rank = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        alerts.sort(key=lambda x: severity_rank.get(x.severity, 4))
        return alerts
