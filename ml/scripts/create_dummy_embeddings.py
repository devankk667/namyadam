import numpy as np
import pandas as pd


INPUT_FILE = "data/processed/thermal_detections_processed.csv"
OUTPUT_FILE = "data/processed/sentinel_embeddings_demo.csv"

EMBEDDING_DIM = 512
RANDOM_SEED = 42


def main():
    df = pd.read_csv(INPUT_FILE)

    rng = np.random.default_rng(RANDOM_SEED)

    embeddings = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(len(df), EMBEDDING_DIM)
    )

    embedding_df = pd.DataFrame(
        embeddings,
        columns=[f"embedding_{i}" for i in range(EMBEDDING_DIM)]
    )

    embedding_df.insert(0, "id", df["id"].values)

    embedding_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Created {len(embedding_df)} dummy image embeddings")
    print(f"Embedding dimension: {EMBEDDING_DIM}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()