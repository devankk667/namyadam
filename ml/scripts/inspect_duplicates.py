import pandas as pd

df = pd.read_csv(
    "data/processed/firms_noaa20_sp_bbox_68_8_97_37_2024-06-01_to_2026-05-31.csv"
)

key = ["latitude", "longitude", "acq_date", "acq_time", "satellite", "instrument"]
loose_mask = df.duplicated(subset=key, keep=False)  # keep=False shows ALL members
dups = df.loc[loose_mask, key + ["bright_ti4", "bright_ti5", "frp",
                                 "confidence", "daynight", "source_file"]]
print(dups.to_string())