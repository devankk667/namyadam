import json
import os
import pandas as pd
from typing import List, Dict, Any, Optional
from app.core.config import settings

class ThermalDetectionRepository:
    """
    Decoupled data access repository. Serves detection records from demo JSON
    or processed CSV data based on settings.DEMO_MODE without requiring database dependencies.
    """
    def __init__(self):
        self._cache: List[Dict[str, Any]] = []
        self.reload()

    def reload(self):
        if settings.DEMO_MODE or not os.path.exists(settings.PROCESSED_DATA_PATH):
            data_file = settings.DEMO_DATA_PATH
            if not os.path.exists(data_file):
                from ml.scripts.generate_demo_data import main as gen_main
                gen_main()
            with open(data_file, "r") as f:
                self._cache = json.load(f)
        else:
            df = pd.read_csv(settings.PROCESSED_DATA_PATH)
            self._cache = df.to_dict(orient="records")

    def get_all(self) -> List[Dict[str, Any]]:
        return self._cache

    def get_filtered(
        self,
        min_confidence: Optional[float] = None,
        min_frp: Optional[float] = None,
        classification: Optional[str] = None,
        region: Optional[str] = None,
        page: int = 1,
        page_size: int = 50
    ) -> Dict[str, Any]:
        records = self._cache
        if min_confidence is not None:
            records = [r for r in records if r.get("confidence", 0) >= min_confidence]
        if min_frp is not None:
            records = [r for r in records if r.get("frp", 0) >= min_frp]
        if classification:
            records = [r for r in records if r.get("true_class") == classification]
        if region:
            records = [r for r in records if region.lower() in r.get("region", "").lower()]

        total = len(records)
        start = (page - 1) * page_size
        end = start + page_size
        paged_items = records[start:end]

        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "items": paged_items
        }

    def get_by_id(self, detection_id: str) -> Optional[Dict[str, Any]]:
        for r in self._cache:
            if r.get("id") == detection_id:
                return r
        return None

    def get_raw_list(self) -> List[Dict[str, Any]]:
        return self._cache

detection_repo = ThermalDetectionRepository()
