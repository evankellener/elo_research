import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("interleaved_cleaned.csv", low_memory=False)

# Check if MOV columns exist and have data
mov_columns = ['ko', 'kod', 'subw', 'subwd', 'udec', 'udecd', 'mdec', 'mdecd', 'sdec', 'sdecd']

print("="*60)
print("MOV COLUMN ANALYSIS")
print("="*60)

for col in mov_columns:
    if col in df.columns:
        non_zero = (df[col] == 1).sum()
        total = len(df)
        pct = (non_zero / total * 100) if total > 0 else 0
        print(f"{col:10s}: {non_zero:6d} / {total:6d} ({pct:5.2f}%)")
    else:
        print(f"{col:10s}: COLUMN NOT FOUND")

print("\n" + "="*60)
print("SAMPLE ROWS WITH MOV FLAGS")
print("="*60)

# Show some sample rows
sample_cols = ['FIGHTER', 'opp_FIGHTER', 'result'] + [c for c in mov_columns if c in df.columns]
if len(sample_cols) > 3:
    print(df[sample_cols].head(20).to_string())

print("\n" + "="*60)
print("TESTING method_of_victory_scale FUNCTION")
print("="*60)

def method_of_victory_scale(row,
                            w_ko=1.4,
                            w_sub=1.3,
                            w_udec=1.0,
                            w_mdec=0.9,
                            w_sdec=0.7):
    """Return a multiplier for K based on method of victory."""
    if row["ko"] == 1 or row["kod"] == 1:
        return w_ko
    if row["subw"] == 1 or row["subwd"] == 1:
        return w_sub
    if row["udec"] == 1 or row["udecd"] == 1:
        return w_udec
    if row["mdec"] == 1 or row["mdecd"] == 1:
        return w_mdec
    if row["sdec"] == 1 or row["sdecd"] == 1:
        return w_sdec
    return 1.0

# Test on first 100 rows
mov_values = []
for idx, row in df.head(100).iterrows():
    mov_val = method_of_victory_scale(row)
    mov_values.append(mov_val)

mov_values = np.array(mov_values)
print(f"First 100 rows MOV scale distribution:")
print(f"  Mean: {mov_values.mean():.4f}")
print(f"  Min: {mov_values.min():.4f}")
print(f"  Max: {mov_values.max():.4f}")
print(f"  Unique values: {np.unique(mov_values)}")
print(f"  Count of 1.0: {(mov_values == 1.0).sum()}")
print(f"  Count != 1.0: {(mov_values != 1.0).sum()}")

# Check full dataset
print(f"\nFull dataset MOV scale distribution:")
all_mov_values = []
for idx, row in df.iterrows():
    try:
        mov_val = method_of_victory_scale(row)
        all_mov_values.append(mov_val)
    except:
        pass

all_mov_values = np.array(all_mov_values)
print(f"  Total rows: {len(all_mov_values)}")
print(f"  Mean: {all_mov_values.mean():.4f}")
print(f"  Min: {all_mov_values.min():.4f}")
print(f"  Max: {all_mov_values.max():.4f}")
print(f"  Unique values: {sorted(np.unique(all_mov_values))}")
print(f"  Count of 1.0: {(all_mov_values == 1.0).sum()} ({100*(all_mov_values == 1.0).sum()/len(all_mov_values):.2f}%)")
print(f"  Count != 1.0: {(all_mov_values != 1.0).sum()} ({100*(all_mov_values != 1.0).sum()/len(all_mov_values):.2f}%)")

# Check if there are any differences in ratings when MOV is applied
print("\n" + "="*60)
print("TESTING RATING DIFFERENCES")
print("="*60)

from optimal_k_with_mov import run_basic_elo

# Run a small test with first 1000 rows
test_df = df.head(1000).copy()
test_df["result"] = pd.to_numeric(test_df["result"], errors="coerce")
test_df["DATE"] = pd.to_datetime(test_df["DATE"])

# Run with MOV
df_mov = run_basic_elo(test_df.copy(), k=100, use_mov=True)
# Run without MOV
df_no_mov = run_basic_elo(test_df.copy(), k=100, use_mov=False)

# Compare final ratings
print("Comparing final postcomp_elo ratings:")
print(f"  Rows compared: {len(df_mov)}")
print(f"  Identical ratings: {(df_mov['postcomp_elo'] == df_no_mov['postcomp_elo']).sum()}")
print(f"  Different ratings: {(df_mov['postcomp_elo'] != df_no_mov['postcomp_elo']).sum()}")
if (df_mov['postcomp_elo'] != df_no_mov['postcomp_elo']).sum() > 0:
    diff = df_mov['postcomp_elo'] - df_no_mov['postcomp_elo']
    print(f"  Max difference: {diff.abs().max():.6f}")
    print(f"  Mean absolute difference: {diff.abs().mean():.6f}")
    print(f"  Sample differences (first 10 non-zero):")
    non_zero_diffs = diff[diff != 0].head(10)
    print(non_zero_diffs.to_string())

