import pandas as pd


def method_of_victory_scale(row,
                            w_ko=1.4,
                            w_sub=1.3,
                            w_udec=1.0,
                            w_mdec=0.9,
                            w_sdec=0.7):
    """
    Return a multiplier for K based on method of victory.

    We look at the row from the FIGHTER perspective, but the flags
    (ko, kod, subw, subwd, etc) encode outcome for both sides.
    For the Elo update we just care about how decisive the fight was,
    not who got it, so we collapse to categories.
    """
    # KO or TKO
    if row["ko"] == 1 or row["kod"] == 1:
        return w_ko
    # Submission
    if row["subw"] == 1 or row["subwd"] == 1:
        return w_sub
    # Unanimous decision
    if row["udec"] == 1 or row["udecd"] == 1:
        return w_udec
    # Majority decision
    if row["mdec"] == 1 or row["mdecd"] == 1:
        return w_mdec
    # Split decision
    if row["sdec"] == 1 or row["sdecd"] == 1:
        return w_sdec
    # Fallback when we do not recognize the type
    return 1.0


def run_basic_elo(df, k=32, base_elo=1500, denominator=400):
    """
    Core Elo loop, now with method of victory scaling the K for each fight.
    """
    df = df.copy()
    ratings = {}
    pre, post, opp_pre, opp_post = [], [], [], []

    for _, row in df.iterrows():
        f1, f2, res = row["FIGHTER"], row["opp_FIGHTER"], row["result"]
        r1 = ratings.get(f1, base_elo)
        r2 = ratings.get(f2, base_elo)

        # logistic expectation
        e1 = 1 / (1 + 10 ** ((r2 - r1) / denominator))
        e2 = 1 / (1 + 10 ** ((r1 - r2) / denominator))

        # method of victory multiplier
        mov_scale = method_of_victory_scale(row)
        k_eff = k * mov_scale

        # update
        r1_new = r1 + k_eff * (res - e1)
        r2_new = r2 + k_eff * ((1 - res) - e2)

        ratings[f1], ratings[f2] = r1_new, r2_new
        pre.append(r1)
        post.append(r1_new)
        opp_pre.append(r2)
        opp_post.append(r2_new)

    df["precomp_elo"] = pre
    df["postcomp_elo"] = post
    df["opp_precomp_elo"] = opp_pre
    df["opp_postcomp_elo"] = opp_post
    return df


def build_fighter_history(df):
    a = df[["DATE", "FIGHTER", "precomp_elo", "postcomp_elo"]].rename(
        columns={
            "FIGHTER": "fighter",
            "precomp_elo": "pre_elo",
            "postcomp_elo": "post_elo",
        }
    )
    b = df[["DATE", "opp_FIGHTER", "opp_precomp_elo", "opp_postcomp_elo"]].rename(
        columns={
            "opp_FIGHTER": "fighter",
            "opp_precomp_elo": "pre_elo",
            "opp_postcomp_elo": "post_elo",
        }
    )
    hist = pd.concat([a, b], ignore_index=True)
    return (
        hist.sort_values("DATE")
        .rename(columns={"DATE": "date"})
        .reset_index(drop=True)
    )


def has_prior_history(first_dates, fighter, date):
    d0 = first_dates.get(fighter)
    return bool(d0 and date > d0)


def add_bout_counts(df):
    """
    Adds:
        precomp_boutcount
        opp_precomp_boutcount

    Each is how many bouts that fighter had BEFORE this bout.
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
    Compute in sample predictions from Elo columns.

    Rules:
      - valid date
      - result in {0,1}
      - different precomp elos
      - both fighters have boutcount >= 1
      - both fighters have prior history date wise
    """
    hist = build_fighter_history(df)
    first_dates = hist.groupby("fighter")["date"].min().to_dict()
    out = []

    for _, r in df.iterrows():
        d = r["DATE"]
        if pd.isna(d) or r["result"] not in (0, 1):
            continue

        # skip fights where ratings are equal
        if r["precomp_elo"] == r["opp_precomp_elo"]:
            continue

        # boutcount filter: both must have at least one prior fight
        if r["precomp_boutcount"] < 1 or r["opp_precomp_boutcount"] < 1:
            continue

        # extra date based prior history check
        if not has_prior_history(first_dates, r["FIGHTER"], d):
            continue
        if not has_prior_history(first_dates, r["opp_FIGHTER"], d):
            continue

        pred = int(r["precomp_elo"] > r["opp_precomp_elo"])
        out.append(
            {
                "date": d,
                "prediction": pred,
                "result": int(r["result"]),
                "correct": int(pred == r["result"]),
            }
        )

    return pd.DataFrame(out)


def elo_accuracy(df, cutoff_date=None):
    preds = compute_fight_predictions(df)
    if preds.empty:
        return None, 0, 0
    acc_all = preds["correct"].mean()
    if cutoff_date is None:
        return acc_all, None, 0
    future = preds[preds["date"] > cutoff_date]
    return (
        acc_all,
        None if future.empty else future["correct"].mean(),
        len(future),
    )


def genetic_algorithm_k(df, k_range=range(10, 500, 10), base_elo=1500, denominator=400):
    """
    Hyperparameter sweep for k.

    Uses method of victory aware Elo via run_basic_elo.
    Accuracy is computed only on fights where both fighters had prior bouts.
    """
    best_k, best_future = None, -1.0
    cutoff = df["DATE"].quantile(0.8)

    for k in k_range:
        trial = run_basic_elo(df.copy(), k=k, base_elo=base_elo, denominator=denominator)
        acc_all, acc_future, n_future = elo_accuracy(trial, cutoff)
        print(f"k={k}, all={acc_all:.4f}, future={acc_future:.4f} (n={n_future})")
        if acc_future is not None and acc_future > best_future:
            best_k, best_future = k, acc_future

    return best_k, best_future


def latest_ratings_from_trained_df(df, base_elo=1500):
    """
    Build dict {fighter: latest_post_fight_elo} from a df that already
    has pre and post Elo columns for both sides.
    """
    ratings = {}
    for _, r in df.iterrows():
        ratings[r["FIGHTER"]] = r["postcomp_elo"]
        ratings[r["opp_FIGHTER"]] = r["opp_postcomp_elo"]
    return ratings


def test_out_of_sample_accuracy(
    df_trained_with_elo,
    test_df,
    best_k,
    base_elo=1500,
    verbose=False,
    gap_threshold=None,
):
    """
    Use frozen ratings from df_trained_with_elo to predict test_df fights.

    gap_threshold:
        if set, we also report accuracy on fights with abs(Elo difference)
        at least that large.
    """
    rating_lookup = latest_ratings_from_trained_df(df_trained_with_elo, base_elo=base_elo)

    tdf = test_df.copy()
    tdf["result"] = pd.to_numeric(tdf["result"], errors="coerce")
    tdf["DATE"] = pd.to_datetime(tdf["date"])

    hits_all, total_all = 0, 0
    hits_gap, total_gap = 0, 0

    for _, row in tdf.iterrows():
        f1 = row["fighter"]
        f2 = row["opp_fighter"]
        res = row["result"]

        if res not in (0, 1):
            continue

        r1 = rating_lookup.get(f1, base_elo)
        r2 = rating_lookup.get(f2, base_elo)
        pred = int(r1 > r2)
        correct = int(pred == res)

        hits_all += correct
        total_all += 1

        if gap_threshold is not None and abs(r1 - r2) >= gap_threshold:
            hits_gap += correct
            total_gap += 1

        if verbose:
            print(
                f"{row['DATE'].date()} | {f1}({r1:.1f}) vs {f2}({r2:.1f}) -> pred={pred} res={int(res)}"
            )

    acc_all = hits_all / total_all if total_all else None
    acc_gap = hits_gap / total_gap if total_gap else None

    if gap_threshold is not None:
        print(f"\nGAP FILTER ACTIVE: abs(Elo diff) >= {gap_threshold}")
        print(f"gap fights counted: {total_gap}")
        print(f"gap accuracy: {acc_gap}")

    return acc_all


def full_oos_sweep(df, test_df, k_range, base_elo=1500):
    """
    For debugging intuition:
      train Elo with each k, then check OOS accuracy on the 3 event file.
      Uses the boutcount rule and NO gap filter.
    """
    print("\n=== FULL OOS SWEEP (boutcount rule applied) ===")
    for k in k_range:
        df_trained_k = run_basic_elo(df.copy(), k=k, base_elo=base_elo)
        acc_k = test_out_of_sample_accuracy(
            df_trained_k,
            test_df,
            best_k=k,
            base_elo=base_elo,
            verbose=False,
            gap_threshold=None,
        )
        print(f"k={k}, OOS={acc_k:.4f}")


if __name__ == "__main__":
    df = pd.read_csv("interleaved_cleaned.csv", low_memory=False)
    test_df = pd.read_csv("past3_events.csv", low_memory=False)

    df["result"] = pd.to_numeric(df["result"], errors="coerce")
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.sort_values("DATE").reset_index(drop=True)

    # boutcounts first, so prediction filters can use them
    df = add_bout_counts(df)

    cut = df["DATE"].quantile(0.8)
    print("Training end date:", cut)
    print(test_df["date"].min(), test_df["date"].max())

    # 1) sweep k on historical data (with method of victory aware Elo)
    best_k, best_accuracy = genetic_algorithm_k(df)
    print("best k value:", best_k)
    print("best future accuracy:", best_accuracy)

    # 2) train once with best k to get final ratings
    df_trained = run_basic_elo(df.copy(), k=best_k)

    # 3) evaluate OOS on the three known events
    oos_acc = test_out_of_sample_accuracy(
        df_trained,
        test_df,
        best_k=best_k,
        verbose=True,
        gap_threshold=75,  # can change or set to None
    )
    print("Overall OOS:", oos_acc)

    # 4) debug OOS across all k values
    full_oos_sweep(df, test_df, k_range=range(10, 500, 10))
