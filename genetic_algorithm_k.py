import pandas as pd

def run_basic_elo(df, k=32, base_elo=1500, denominator=400):
    """
    Run a simple Elo update over the dataframe in date order.
    Adds:
      precomp_elo, postcomp_elo, opp_precomp_elo, opp_postcomp_elo
    """
    df = df.copy()
    ratings = {}
    pre, post, opp_pre, opp_post = [], [], [], []

    for _, row in df.iterrows():
        f1, f2, res = row['FIGHTER'], row['opp_FIGHTER'], row['result']
        r1 = ratings.get(f1, base_elo)
        r2 = ratings.get(f2, base_elo)

        e1 = 1 / (1 + 10 ** ((r2 - r1) / denominator))
        e2 = 1 / (1 + 10 ** ((r1 - r2) / denominator))

        r1_new = r1 + k * (res - e1)
        r2_new = r2 + k * ((1 - res) - e2)

        ratings[f1], ratings[f2] = r1_new, r2_new
        pre.append(r1)
        post.append(r1_new)
        opp_pre.append(r2)
        opp_post.append(r2_new)

    df['precomp_elo'] = pre
    df['postcomp_elo'] = post
    df['opp_precomp_elo'] = opp_pre
    df['opp_postcomp_elo'] = opp_post
    return df


def add_bout_counts(df):
    """
    Adds two columns:
      precomp_boutcount
      opp_precomp_boutcount

    These count how many fights each fighter has had BEFORE this fight.
    """
    df = df.sort_values("DATE").copy()

    bout_counts = {}
    pre_counts = []
    opp_pre_counts = []

    for _, r in df.iterrows():
        f1 = r["FIGHTER"]
        f2 = r["opp_FIGHTER"]

        c1 = bout_counts.get(f1, 0)
        c2 = bout_counts.get(f2, 0)

        pre_counts.append(c1)
        opp_pre_counts.append(c2)

        bout_counts[f1] = c1 + 1
        bout_counts[f2] = c2 + 1

    df["precomp_boutcount"] = pre_counts
    df["opp_precomp_boutcount"] = opp_pre_counts
    return df


def compute_fight_predictions(df):
    """
    Build per fight prediction rows from a df that already has
    precomp_elo, opp_precomp_elo, and boutcount columns.

    Only count fights where:
      - result is 0 or 1
      - both fighters have at least 1 prior fight
      - Elo ratings are not tied
    """
    rows = []

    for _, r in df.iterrows():
        d = r['DATE']
        res = r['result']

        if pd.isna(d) or res not in (0, 1):
            continue

        # require each fighter to have at least 1 prior fight
        if r.get("precomp_boutcount", 0) < 1:
            continue
        if r.get("opp_precomp_boutcount", 0) < 1:
            continue

        if r['precomp_elo'] == r['opp_precomp_elo']:
            continue

        pred = int(r['precomp_elo'] > r['opp_precomp_elo'])
        correct = int(pred == int(res))

        rows.append(
            {
                'date': d,
                'prediction': pred,
                'result': int(res),
                'correct': correct,
            }
        )

    return pd.DataFrame(rows)


def elo_accuracy(df, cutoff_date=None):
    """
    Returns:
      acc_all, acc_future, n_future
    All using the boutcount filter (both sides at least 1 prior fight).
    """
    preds = compute_fight_predictions(df)
    if preds.empty:
        return None, 0, 0

    acc_all = preds['correct'].mean()

    if cutoff_date is None:
        return acc_all, None, 0

    future = preds[preds['date'] > cutoff_date]
    if future.empty:
        return acc_all, None, 0

    acc_future = future['correct'].mean()
    return acc_all, acc_future, len(future)


def genetic_algorithm_k(df, k_range=range(10, 500, 10),
                        base_elo=1500, denominator=400):
    """
    Simple grid search over k values.
    Uses future accuracy (last 20 percent of data) as the score.
    Only counts fights where both fighters had at least 1 prior bout.
    """
    best_k = None
    best_future = -1.0
    cutoff = df['DATE'].quantile(0.8)

    for k in k_range:
        trial = run_basic_elo(df.copy(), k, base_elo, denominator)
        acc_all, acc_future, n_future = elo_accuracy(trial, cutoff)
        print(f"k={k}, all={acc_all:.4f}, future={acc_future:.4f} (n={n_future})")
        if acc_future is not None and acc_future > best_future:
            best_k = k
            best_future = acc_future

    return best_k, best_future


def latest_ratings_from_trained_df(df, base_elo=1500):
    """
    Build a dict:
        {fighter_name: latest_post_fight_elo}
    from a df that already has pre or post Elo columns.
    Uses both FIGHTER and opp_FIGHTER sides.
    """
    ratings = {}

    for _, r in df.iterrows():
        ratings[r['FIGHTER']] = r.get('postcomp_elo', base_elo)
        ratings[r['opp_FIGHTER']] = r.get('opp_postcomp_elo', base_elo)

    return ratings


def test_out_of_sample_accuracy(df_trained_with_elo,
                                test_df,
                                best_k,
                                base_elo=1500,
                                verbose=False,
                                gap_threshold=None):
    """
    Evaluate out of sample on test_df using frozen ratings from df_trained_with_elo.

    Rules:
      - Only count fights where both fighters have appeared in history
        (so they have at least 1 prior fight).
      - If gap_threshold is set, also compute accuracy on the subset where
        abs(Elo diff) >= gap_threshold.
    """
    rating_lookup = latest_ratings_from_trained_df(df_trained_with_elo, base_elo=base_elo)

    tdf = test_df.copy()
    tdf['result'] = pd.to_numeric(tdf['result'], errors='coerce')
    tdf['DATE'] = pd.to_datetime(tdf['date'])

    hits_all, total_all = 0, 0
    hits_gap, total_gap = 0, 0

    for _, row in tdf.iterrows():
        f1 = row['fighter']
        f2 = row['opp_fighter']
        res = row['result']

        if res not in (0, 1):
            continue

        # require both fighters to have at least 1 prior fight in training
        if f1 not in rating_lookup or f2 not in rating_lookup:
            continue

        r1 = rating_lookup.get(f1, base_elo)
        r2 = rating_lookup.get(f2, base_elo)
        pred = int(r1 > r2)
        correct = int(pred == int(res))

        hits_all += correct
        total_all += 1

        if gap_threshold is not None:
            if abs(r1 - r2) >= gap_threshold:
                hits_gap += correct
                total_gap += 1

        if verbose:
            print(f"{row['DATE'].date()} | {f1}({r1:.1f}) vs {f2}({r2:.1f}) -> pred={pred} res={int(res)}")

    acc_all = hits_all / total_all if total_all else None
    acc_gap = hits_gap / total_gap if total_gap else None

    if gap_threshold is not None:
        print(f"\nGAP FILTER ACTIVE: abs(Elo diff) >= {gap_threshold}")
        print(f"gap fights counted: {total_gap}")
        print(f"gap accuracy: {acc_gap}")

    return acc_all


if __name__ == "__main__":
    # load data
    df = pd.read_csv('interleaved_cleaned.csv', low_memory=False)
    test_df = pd.read_csv('past3_events.csv', low_memory=False)

    # basic cleanup
    df['result'] = pd.to_numeric(df['result'], errors='coerce')
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.sort_values('DATE').reset_index(drop=True)

    # add bout counts so every k run can use them in accuracy
    df = add_bout_counts(df)

    cut = df['DATE'].quantile(0.8)
    print("Training end date:", cut)
    print(test_df['date'].min(), test_df['date'].max())

    # 1) search best k on the historical data
    best_k, best_future_acc = genetic_algorithm_k(df)
    print("best k value:", best_k)
    print("best future accuracy:", best_future_acc)

    # 2) run Elo once with best k to get final trained ratings on history
    df_trained = run_basic_elo(df.copy(), k=best_k)

    # 3) evaluate true out of sample events using the frozen rating snapshot
    oos_acc = test_out_of_sample_accuracy(
        df_trained,
        test_df,
        best_k,
        verbose=True,
        gap_threshold=75  # can set to None if you do not want the gap split
    )
    print("Overall OOS:", oos_acc)

    # 4) optional full OOS sweep across all k values
    print("\n=== FULL OOS SWEEP (boutcount rule applied) ===")
    oos_results = []

    for k in range(10, 500, 10):
        trained_df_k = run_basic_elo(df.copy(), k)
        acc_oos_k = test_out_of_sample_accuracy(
            trained_df_k,
            test_df,
            k,
            verbose=False,
            gap_threshold=None
        )
        print(f"k={k}, OOS={acc_oos_k:.4f}" if acc_oos_k is not None else f"k={k}, OOS=None")
        oos_results.append((k, acc_oos_k))

    # if you want, you can dump the OOS sweep to a csv
    # pd.DataFrame(oos_results, columns=['k', 'oos_acc']).to_csv('oos_k_sweep.csv', index=False)
