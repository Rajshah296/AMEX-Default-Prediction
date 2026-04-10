import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import gc
import ctypes
import pyarrow as pa
import pyarrow.parquet as pq

print(f"The Pandas version: {pd.__version__}")
import polars as pl
print(f"The Polars version: {pl.__version__}")
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows', None)

def release_memory():
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)


def write_parquet_chunks(df: pl.DataFrame, output_dir: str, chunk_size: int = 50_000):
    """
    Polars equivalent of the original pandas write_parquet_chunks.
    Writes the DataFrame in row-chunks as snappy-compressed parquet files.
    """
    os.makedirs(output_dir, exist_ok=True)
    prefix = output_dir.split("/")[-2].split("_")[0]
    n = len(df)

    for i, start in enumerate(range(0, n, chunk_size)):
        chunk = df.slice(start, chunk_size)  # zero-copy slice, no .copy() needed
        chunk.write_parquet(
            f"{output_dir}{prefix}_part_{i:04d}.parquet",
            compression="snappy",
        )
        del chunk
        release_memory()


# =============================
# HELPERS
# =============================


# =============================
# LOAD + SORT
# =============================
# Polars reads parquet natively — no pandas index concept,
# customer_ID comes in as a regular column automatically.
# Nullable Int8/Int16 are handled natively — no conversion needed.
X_test = pl.read_parquet(
    "/kaggle/input/datasets/analyticgentleman/silver-data/test_data"
)

X_test = X_test.sort(
    ["customer_ID", "year", "month", "day"],
    maintain_order=True,   # stable sort — preserves row order for identical timestamps
)

release_memory()
print("Load and sort completed.")
print(f"Shape: {X_test.shape}")

# =============================
# FEATURE LISTS
# =============================
cat_features = [
    "B_30","B_38","D_114","D_116","D_117",
    "D_120","D_126","D_63","D_64","D_66","D_68",
]

num_features = [
    c for c in X_test.columns
    if c not in ["customer_ID", "year", "month", "day"] + cat_features
]

# =============================
# BASIC NUM AGG
# =============================
# Build all aggregation expressions dynamically.
# Polars casts to Float32 at expression level — no post-hoc .astype() needed.
num_agg_exprs = []
for col in num_features:
    num_agg_exprs.extend([
        pl.col(col).mean().cast(pl.Float32).alias(f"{col}_mean"),
        pl.col(col).min().cast(pl.Float32).alias(f"{col}_min"),
        pl.col(col).max().cast(pl.Float32).alias(f"{col}_max"),
        pl.col(col).std().cast(pl.Float32).alias(f"{col}_std"),
        pl.col(col).last().cast(pl.Float32).alias(f"{col}_last"),
        pl.col(col).sum().cast(pl.Float32).alias(f"{col}_sum"),
    ])

X = (
    X_test
    .lazy()
    .group_by("customer_ID", maintain_order=False)
    .agg(num_agg_exprs)
    .collect()
)

release_memory()
print(f"num_agg executed — X shape: {X.shape}")

# =============================
# RECORD COUNT
# =============================
count_agg = (
    X_test
    .lazy()
    .group_by("customer_ID", maintain_order=False)
    .agg(pl.len().cast(pl.Float32).alias("record_count"))
    .collect()
)

X = X.join(count_agg, on="customer_ID", how="left")
del count_agg
release_memory()
print("count features created!")

# =============================
# LAST - FIRST
# =============================
last_first_exprs = []
for col in num_features:
    last_first_exprs.append(
        (pl.col(col).last() - pl.col(col).first())
        .cast(pl.Float32)
        .alias(f"{col}_last_first_diff")
    )

last_first_diff = (
    X_test
    .lazy()
    .group_by("customer_ID", maintain_order=False)
    .agg(last_first_exprs)
    .collect()
)

X = X.join(last_first_diff, on="customer_ID", how="left")
del last_first_diff
release_memory()
print("last-first created")

# =============================
# LAST3 / LAST6
# =============================
# group_by(...).tail(n) gives last n rows per group efficiently.
# maintain_order=True on the outer group_by preserves date sort order
# so tail() correctly picks the most recent statements.

last3_exprs = []
last6_exprs = []
for col in num_features:
    last3_exprs.extend([
        pl.col(col).mean().cast(pl.Float32).alias(f"{col}_last3_mean"),
        pl.col(col).std().cast(pl.Float32).alias(f"{col}_last3_std"),
        pl.col(col).sum().cast(pl.Float32).alias(f"{col}_last3_sum"),
    ])
    last6_exprs.extend([
        pl.col(col).mean().cast(pl.Float32).alias(f"{col}_last6_mean"),
        pl.col(col).std().cast(pl.Float32).alias(f"{col}_last6_std"),
        pl.col(col).sum().cast(pl.Float32).alias(f"{col}_last6_sum"),
    ])

last3 = (
    X_test
    .lazy()
    .group_by("customer_ID", maintain_order=True)
    .tail(3)
    .group_by("customer_ID", maintain_order=False)
    .agg(last3_exprs)
    .collect()
)

X = X.join(last3, on="customer_ID", how="left")
del last3
release_memory()

last6 = (
    X_test
    .lazy()
    .group_by("customer_ID", maintain_order=True)
    .tail(6)
    .group_by("customer_ID", maintain_order=False)
    .agg(last6_exprs)
    .collect()
)

X = X.join(last6, on="customer_ID", how="left")
del last6
release_memory()
print("last3/last6 created")

# =============================
# CATEGORICAL AGG
# =============================
cat_agg_exprs = []
for col in cat_features:
    cat_agg_exprs.extend([
        pl.col(col).last().cast(pl.Float32).alias(f"{col}_last"),
        pl.col(col).n_unique().cast(pl.Float32).alias(f"{col}_nunique"),
        pl.col(col).count().cast(pl.Float32).alias(f"{col}_count"),
    ])

cat_agg = (
    X_test
    .lazy()
    .group_by("customer_ID", maintain_order=False)
    .agg(cat_agg_exprs)
    .collect()
)

X = X.join(cat_agg, on="customer_ID", how="left")
del cat_agg
release_memory()
print("categorical handled")

# =============================
# RANK FEATURES
# =============================
# This is the key advantage of Polars over pandas for this step.
# rank().over() computes per-group rank lazily without materializing
# a full float64 copy of the dataset. The result is computed and
# immediately cast to float32 within the query plan.
#
# We process in chunks to keep the with_columns expressions
# manageable, then take the last row per customer.

CHUNK = 15  # Polars handles rank far more efficiently — larger chunks are safe

rank_exprs_chunks = [
    [
        (pl.col(col).rank(method="average").over("customer_ID") /
        pl.col(col).count().over("customer_ID"))
        .cast(pl.Float32)
        .alias(f"{col}_rank")
        for col in num_features[i : i + CHUNK]
    ]
    for i in range(0, len(num_features), CHUNK)
]

rank_result = (
    X_test
    .lazy()
    .select(["customer_ID"] + num_features)
)

# Apply rank expressions chunk by chunk in the lazy plan
for chunk_exprs in rank_exprs_chunks:
    rank_result = rank_result.with_columns(chunk_exprs)

# Take last row per customer (already sorted by date) and select only rank cols
rank_cols = [f"{col}_rank" for col in num_features]
rank_agg = (
    rank_result
    .group_by("customer_ID", maintain_order=True)
    .tail(1)
    .select(["customer_ID"] + rank_cols)
    .collect()
)

del rank_result
release_memory()

X = X.join(rank_agg, on="customer_ID", how="left")
del rank_agg
release_memory()
print("rank features created")

# =============================
# YM RANK FEATURES
# =============================
# Rank each row relative to all rows in the same year-month period,
# then take the last row per customer.
# Purpose: adjusts for macroeconomic seasonality — a spending increase
# that is average for the month is less risky than one that puts the
# customer in the top 5% of spenders for that month.

ym_rank_exprs_chunks = [
    [
        (pl.col(col).rank(method="average").over("ym") /
        pl.col(col).count().over("ym"))
        .cast(pl.Float32)
        .alias(f"{col}_ym_rank")
        for col in num_features[i : i + CHUNK]
    ]
    for i in range(0, len(num_features), CHUNK)
]

ym_rank_result = (
    X_test
    .lazy()
    .with_columns(
        (pl.col("year").cast(pl.String) + "-" + pl.col("month").cast(pl.String))
        .alias("ym")
    )
    .select(["customer_ID", "ym"] + num_features)
)

for chunk_exprs in ym_rank_exprs_chunks:
    ym_rank_result = ym_rank_result.with_columns(chunk_exprs)

ym_rank_cols = [f"{col}_ym_rank" for col in num_features]
ym_rank_agg = (
    ym_rank_result
    .group_by("customer_ID", maintain_order=True)
    .tail(1)
    .select(["customer_ID"] + ym_rank_cols)
    .collect()
)

del ym_rank_result
release_memory()

X = X.join(ym_rank_agg, on="customer_ID", how="left")
del ym_rank_agg
release_memory()
print("ym rank created")

# =============================
# FREE ORIGINAL DATA
# =============================
del X_test
release_memory()

print("TEST FINAL SHAPE:", X.shape)

write_parquet_chunks(X, "/kaggle/working/test_data/")

del X
release_memory()


# =============================
# LOAD + SORT
# =============================
# Polars reads parquet natively — no pandas index concept,
# customer_ID comes in as a regular column automatically.
# Nullable Int8/Int16 are handled natively — no conversion needed.
X_train = pl.read_parquet(
    "/kaggle/input/datasets/analyticgentleman/silver-data/train_data"
)

X_train = X_train.sort(
    ["customer_ID", "year", "month", "day"],
    maintain_order=True,   # stable sort — preserves row order for identical timestamps
)

release_memory()
print("Load and sort completed.")
print(f"Shape: {X_train.shape}")

# =============================
# FEATURE LISTS
# =============================
cat_features = [
    "B_30","B_38","D_114","D_116","D_117",
    "D_120","D_126","D_63","D_64","D_66","D_68",
]

num_features = [
    c for c in X_train.columns
    if c not in ["customer_ID", "year", "month", "day"] + cat_features
]

# =============================
# BASIC NUM AGG
# =============================
# Build all aggregation expressions dynamically.
# Polars casts to Float32 at expression level — no post-hoc .astype() needed.
num_agg_exprs = []
for col in num_features:
    num_agg_exprs.extend([
        pl.col(col).mean().cast(pl.Float32).alias(f"{col}_mean"),
        pl.col(col).min().cast(pl.Float32).alias(f"{col}_min"),
        pl.col(col).max().cast(pl.Float32).alias(f"{col}_max"),
        pl.col(col).std().cast(pl.Float32).alias(f"{col}_std"),
        pl.col(col).last().cast(pl.Float32).alias(f"{col}_last"),
        pl.col(col).sum().cast(pl.Float32).alias(f"{col}_sum"),
    ])

X = (
    X_train
    .lazy()
    .group_by("customer_ID", maintain_order=False)
    .agg(num_agg_exprs)
    .collect()
)

release_memory()
print(f"num_agg executed — X shape: {X.shape}")

# =============================
# RECORD COUNT
# =============================
count_agg = (
    X_train
    .lazy()
    .group_by("customer_ID", maintain_order=False)
    .agg(pl.len().cast(pl.Float32).alias("record_count"))
    .collect()
)

X = X.join(count_agg, on="customer_ID", how="left")
del count_agg
release_memory()
print("count features created!")

# =============================
# LAST - FIRST
# =============================
last_first_exprs = []
for col in num_features:
    last_first_exprs.append(
        (pl.col(col).last() - pl.col(col).first())
        .cast(pl.Float32)
        .alias(f"{col}_last_first_diff")
    )

last_first_diff = (
    X_train
    .lazy()
    .group_by("customer_ID", maintain_order=False)
    .agg(last_first_exprs)
    .collect()
)

X = X.join(last_first_diff, on="customer_ID", how="left")
del last_first_diff
release_memory()
print("last-first created")

# =============================
# LAST3 / LAST6
# =============================
# group_by(...).tail(n) gives last n rows per group efficiently.
# maintain_order=True on the outer group_by preserves date sort order
# so tail() correctly picks the most recent statements.

last3_exprs = []
last6_exprs = []
for col in num_features:
    last3_exprs.extend([
        pl.col(col).mean().cast(pl.Float32).alias(f"{col}_last3_mean"),
        pl.col(col).std().cast(pl.Float32).alias(f"{col}_last3_std"),
        pl.col(col).sum().cast(pl.Float32).alias(f"{col}_last3_sum"),
    ])
    last6_exprs.extend([
        pl.col(col).mean().cast(pl.Float32).alias(f"{col}_last6_mean"),
        pl.col(col).std().cast(pl.Float32).alias(f"{col}_last6_std"),
        pl.col(col).sum().cast(pl.Float32).alias(f"{col}_last6_sum"),
    ])

last3 = (
    X_train
    .lazy()
    .group_by("customer_ID", maintain_order=True)
    .tail(3)
    .group_by("customer_ID", maintain_order=False)
    .agg(last3_exprs)
    .collect()
)

X = X.join(last3, on="customer_ID", how="left")
del last3
release_memory()

last6 = (
    X_train
    .lazy()
    .group_by("customer_ID", maintain_order=True)
    .tail(6)
    .group_by("customer_ID", maintain_order=False)
    .agg(last6_exprs)
    .collect()
)

X = X.join(last6, on="customer_ID", how="left")
del last6
release_memory()
print("last3/last6 created")

# =============================
# CATEGORICAL AGG
# =============================
cat_agg_exprs = []
for col in cat_features:
    cat_agg_exprs.extend([
        pl.col(col).last().cast(pl.Float32).alias(f"{col}_last"),
        pl.col(col).n_unique().cast(pl.Float32).alias(f"{col}_nunique"),
        pl.col(col).count().cast(pl.Float32).alias(f"{col}_count"),
    ])

cat_agg = (
    X_train
    .lazy()
    .group_by("customer_ID", maintain_order=False)
    .agg(cat_agg_exprs)
    .collect()
)

X = X.join(cat_agg, on="customer_ID", how="left")
del cat_agg
release_memory()
print("categorical handled")

# =============================
# RANK FEATURES
# =============================
# This is the key advantage of Polars over pandas for this step.
# rank().over() computes per-group rank lazily without materializing
# a full float64 copy of the dataset. The result is computed and
# immediately cast to float32 within the query plan.
#
# We process in chunks to keep the with_columns expressions
# manageable, then take the last row per customer.

CHUNK = 15  # Polars handles rank far more efficiently — larger chunks are safe

rank_exprs_chunks = [
    [
        (pl.col(col).rank(method="average").over("customer_ID") /
         pl.col(col).count().over("customer_ID"))
        .cast(pl.Float32)
        .alias(f"{col}_rank")
        for col in num_features[i : i + CHUNK]
    ]
    for i in range(0, len(num_features), CHUNK)
]

rank_result = (
    X_train
    .lazy()
    .select(["customer_ID"] + num_features)
)

# Apply rank expressions chunk by chunk in the lazy plan
for chunk_exprs in rank_exprs_chunks:
    rank_result = rank_result.with_columns(chunk_exprs)

# Take last row per customer (already sorted by date) and select only rank cols
rank_cols = [f"{col}_rank" for col in num_features]
rank_agg = (
    rank_result
    .group_by("customer_ID", maintain_order=True)
    .tail(1)
    .select(["customer_ID"] + rank_cols)
    .collect()
)

del rank_result
release_memory()

X = X.join(rank_agg, on="customer_ID", how="left")
del rank_agg
release_memory()
print("rank features created")

# =============================
# YM RANK FEATURES
# =============================
# Rank each row relative to all rows in the same year-month period,
# then take the last row per customer.
# Purpose: adjusts for macroeconomic seasonality — a spending increase
# that is average for the month is less risky than one that puts the
# customer in the top 5% of spenders for that month.

ym_rank_exprs_chunks = [
    [
        (pl.col(col).rank(method="average").over("ym") /
         pl.col(col).count().over("ym"))
        .cast(pl.Float32)
        .alias(f"{col}_ym_rank")
        for col in num_features[i : i + CHUNK]
    ]
    for i in range(0, len(num_features), CHUNK)
]

ym_rank_result = (
    X_train
    .lazy()
    .with_columns(
        (pl.col("year").cast(pl.String) + "-" + pl.col("month").cast(pl.String))
        .alias("ym")
    )
    .select(["customer_ID", "ym"] + num_features)
)

for chunk_exprs in ym_rank_exprs_chunks:
    ym_rank_result = ym_rank_result.with_columns(chunk_exprs)

ym_rank_cols = [f"{col}_ym_rank" for col in num_features]
ym_rank_agg = (
    ym_rank_result
    .group_by("customer_ID", maintain_order=True)
    .tail(1)
    .select(["customer_ID"] + ym_rank_cols)
    .collect()
)

del ym_rank_result
release_memory()

X = X.join(ym_rank_agg, on="customer_ID", how="left")
del ym_rank_agg
release_memory()
print("ym rank created")

# =============================
# FREE ORIGINAL DATA
# =============================
del X_train
release_memory()

# =============================
# JOIN TRAIN LABELS
# =============================
train_labels = pl.read_csv(
    "/kaggle/input/datasets/analyticgentleman/train-labels/train_labels.csv"
)

# ensure same dtype
train_labels = train_labels.with_columns(
    pl.col("customer_ID").cast(X["customer_ID"].dtype)
)

X = X.join(train_labels, on="customer_ID", how="left")

del train_labels
release_memory()

print("train labels joined")
print("FINAL SHAPE:", X.shape)

write_parquet_chunks(X, "/kaggle/working/train_data/")

del X
release_memory()

## Pandas version - Test (Doesn't run successfully, fails at rank feature creation, highly optimized, yet comes nowhere near Polars performance).

# # =============================
# # LOAD + SORT
# # =============================
# X_test = pd.read_parquet(
#     "/kaggle/input/datasets/analyticgentleman/silver-data/test_data"
# ).reset_index()

# if X_test["customer_ID"].dtype != "string":
#     X_test["customer_ID"] = X_test["customer_ID"].astype("string")

# # Convert nullable Int8/Int16 → float16 one column at a time
# # to avoid bulk-conversion memory spike
# nullable_int_cols = tuple(c for c in X_test.columns if pd.api.types.is_extension_array_dtype(X_test[c])
#     and c != "customer_ID")

# for c in nullable_int_cols:
#     X_test[c] = X_test[c].astype("float16")


# X_test = X_test.sort_values(
#     ["customer_ID", "year", "month", "day"],
#     ignore_index=True,
#     kind="mergesort"
# )
# release_memory()
# print("Load and sort completed.")

# # =============================
# # FEATURE LISTS
# # =============================
# cat_features = [
#     "B_30","B_38","D_114","D_116","D_117",
#     "D_120","D_126","D_63","D_64","D_66","D_68"
# ]

# num_features = [
#     c for c in X_test.columns
#     if c not in ["customer_ID","year","month","day"] + cat_features
# ]

# # =============================
# # BASIC NUM AGG
# # =============================
# num_agg = (
#     X_test
#     .groupby("customer_ID", sort=False, observed=True)[num_features]
#     .agg(["mean","min","max","std","last","sum"])
# ).astype("float32")

# num_agg.columns = num_agg.columns.map("_".join)
# print("num_agg executed")

# X = num_agg
# X.index = X.index.astype("string")
# del num_agg
# release_memory()

# # =============================
# # RECORD COUNT
# # =============================
# count_agg = (
#     X_test
#     .groupby("customer_ID", sort=False, observed=True)
#     .size()
#     .to_frame("record_count")
# ).astype("float32")

# X[count_agg.columns] = count_agg
# del count_agg
# for col in X.columns:
#     X[col] = X[col].copy()
# release_memory()
# print("count features created!")

# # =============================
# # LAST - FIRST
# # =============================
# first_last = (
#     X_test
#     .groupby("customer_ID", sort=False, observed=True)[num_features]
#     .agg(["first","last"])
# )

# last_first_diff = (
#     first_last.xs("last", level=1, axis=1)
#     .sub(first_last.xs("first", level=1, axis=1))
#     .add_suffix("_last_first_diff")
# ).astype("float32")

# del first_last
# X[last_first_diff.columns] = last_first_diff
# del last_first_diff
# print("last-first created")
# release_memory()

# # =============================
# # LAST3 / LAST6
# # =============================
# # X_test[::-1] returns a view, not a copy — no extra memory allocated
# rev = X_test[::-1]

# cc = (
#     rev
#     .groupby("customer_ID", sort=False, observed=True)
#     .cumcount()
# )

# mask3 = cc < 3
# mask6 = cc < 6

# last3 = (
#     rev.loc[mask3]
#     .groupby("customer_ID", observed=True)[num_features]
#     .agg(["mean","std","sum"])
# ).astype("float32")

# last6 = (
#     rev.loc[mask6]
#     .groupby("customer_ID", observed=True)[num_features]
#     .agg(["mean","std","sum"])
# ).astype("float32")

# last3.columns = last3.columns.map("{0[0]}_last3_{0[1]}".format)
# last6.columns = last6.columns.map("{0[0]}_last6_{0[1]}".format)

# X[last3.columns] = last3
# X[last6.columns] = last6

# del last3, last6, rev, cc, mask3, mask6
# print("last3/last6 created")
# print("Releasing memory...")
# for col in X.columns:
#     X[col] = X[col].copy()
# release_memory()
# print("Memory released")
# # =============================
# # CATEGORICAL AGG
# # =============================
# cat_agg = (
#     X_test
#     .groupby("customer_ID", sort=False, observed=True)[cat_features]
#     .agg(["last","nunique","count"])
# ).astype("float32")

# cat_agg.columns = cat_agg.columns.map("_".join)
# X[cat_agg.columns] = cat_agg
# del cat_agg
# print("categorical handled")
# print("Releasing memory...")
# for col in X.columns:
#     X[col] = X[col].copy()
# release_memory()
# print("Memory released")

# # =============================
# # RANK FEATURES (chunked)
# # =============================
# # rank(pct=True) internally upcasts to float64 regardless of input dtype.
# # Processing in small column chunks keeps the float64 spike to:
# #   CHUNK cols × N rows × 8 bytes  (e.g. 5 × 5.5M × 8 ≈ 220 MB per chunk)
# # vs the original full-column explosion of ~4 GB in one shot.

# CHUNK = 5

# # Precompute customer_ID lookup once outside the loop
# # instead of hitting X_test.loc inside every iteration
# last_idx = (
#     X_test
#     .groupby("customer_ID", sort=False, observed=True)
#     .tail(1)
#     .index
# )
# customer_ids_at_last = X_test.loc[last_idx, "customer_ID"].values

# for i in range(0, len(num_features), CHUNK):
#     chunk_feats = num_features[i : i + CHUNK]
#     print(f"Processing chunk-{i//CHUNK}...")
#     # Step 1: compute ranks, extract last row immediately, cast — all separate
#     # so each intermediate is deleted before the next is created
#     ranked = (
#         X_test[chunk_feats]
#         .groupby(X_test["customer_ID"], sort=False, observed=True)
#         .rank(pct=True)          # float64, full rows
#     )

#     # Step 2: extract last rows using precomputed index — no second groupby
#     ranked = ranked.loc[last_idx]

#     # Step 3: cast to float16 immediately — halves memory vs float32
#     ranked = ranked.astype("float32")

#     # Step 4: assign index and suffix
#     ranked.index = customer_ids_at_last
#     ranked.columns = ranked.columns + "_rank"

#     X[ranked.columns] = ranked
#     del ranked
#     release_memory()

# print("rank features created")
# print("Releasing memory...")
# del last_idx, customer_ids_at_last
# for col in X.columns:
#     X[col] = X[col].copy()
# release_memory()
# print("Memory released")

# # =============================
# # YM RANK FEATURES (chunked)
# # =============================

# X_test["ym"] = (
#     X_test["year"].astype(str) + "-" + X_test["month"].astype(str)
# )

# # Precompute last row index per ym group once outside the loop
# ym_last_idx = (
#     X_test
#     .groupby("ym", sort=False, observed=True)
#     .tail(1)
#     .index
# )
# customer_ids_at_ym_last = X_test.loc[ym_last_idx, "customer_ID"].values

# for i in range(0, len(num_features), CHUNK):
#     chunk_feats = num_features[i : i + CHUNK]
#     print(f"Processing chunk-{i//CHUNK}...")
#     # Step 1: compute ym ranks — float64, full rows
#     ym_ranked = (
#         X_test[chunk_feats]
#         .groupby(X_test["ym"], sort=False, observed=True)
#         .rank(pct=True)
#     )

#     # Step 2: extract last rows per ym group using precomputed index
#     ym_ranked = ym_ranked.loc[ym_last_idx]

#     # Step 3: cast to float32
#     ym_ranked = ym_ranked.astype("float32")

#     # Step 4: assign index and suffix
#     ym_ranked.index = customer_ids_at_ym_last
#     ym_ranked.columns = ym_ranked.columns + "_ym_rank"

#     X[ym_ranked.columns] = ym_ranked
#     del ym_ranked
#     release_memory()

# del ym_last_idx, customer_ids_at_ym_last
# for col in X.columns:
#     X[col] = X[col].copy()
# release_memory()
# print("ym rank created")

# # =============================
# # FREE ORIGINAL DATA
# # =============================
# del X_test
# release_memory()

# X = X.copy()
# release_memory()

# print("FINAL SHAPE:", X.shape)
# print(X.info())
# write_parquet_chunks(X, "/kaggle/working/test_data/")

# del X
# release_memory()