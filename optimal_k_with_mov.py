import pandas as pd
import matplotlib.pyplot as plt


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
    
    Note: Handles both string and numeric values for MOV flags.
    """
    # Helper to safely check if a value indicates the event occurred
    def is_true(val):
        if pd.isna(val):
            return False
        # Handle string values - check if it's '1' or the column name itself
        if isinstance(val, str):
            return val.strip() == '1' or val.strip().lower() in ['true', '1']
        # Handle numeric values
        return val == 1 or val == True
    
    # KO or TKO
    if is_true(row.get("ko")) or is_true(row.get("kod")):
        return w_ko
    # Submission
    if is_true(row.get("subw")) or is_true(row.get("subwd")):
        return w_sub
    # Unanimous decision
    if is_true(row.get("udec")) or is_true(row.get("udecd")):
        return w_udec
    # Majority decision
    if is_true(row.get("mdec")) or is_true(row.get("mdecd")):
        return w_mdec
    # Split decision
    if is_true(row.get("sdec")) or is_true(row.get("sdecd")):
        return w_sdec
    # Fallback when we do not recognize the type
    return 1.0


def run_basic_elo(df, k=32, base_elo=1500, denominator=400, use_mov=True):
    """
    Core Elo loop, with optional method of victory scaling the K for each fight.
    
    Args:
        use_mov: If True, applies method of victory scaling to K. If False, uses constant K.
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

        # method of victory multiplier (if enabled)
        if use_mov:
            mov_scale = method_of_victory_scale(row)
            k_eff = k * mov_scale
        else:
            k_eff = k

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


def genetic_algorithm_k(df, k_range=range(10, 500, 10), base_elo=1500, denominator=400, 
                        return_all_results=False, use_mov=True, verbose=True):
    """
    Hyperparameter sweep for k.

    Uses method of victory aware Elo via run_basic_elo (if use_mov=True).
    Accuracy is computed only on fights where both fighters had prior bouts.
    
    Args:
        use_mov: If True, uses method of victory scaling. If False, uses constant K.
        return_all_results: If True, returns a DataFrame with all k values and their accuracies.
        verbose: If True, prints progress for each k value.
    """
    best_k, best_future = None, -1.0
    cutoff = df["DATE"].quantile(0.8)
    
    results = []

    for k in k_range:
        trial = run_basic_elo(df.copy(), k=k, base_elo=base_elo, denominator=denominator, use_mov=use_mov)
        acc_all, acc_future, n_future = elo_accuracy(trial, cutoff)
        if verbose:
            mov_label = "MOV" if use_mov else "No MOV"
            print(f"k={k} ({mov_label}), all={acc_all:.4f}, future={acc_future:.4f} (n={n_future})")
        
        if return_all_results:
            results.append({
                'k': k,
                'overall_accuracy': acc_all,
                'test_accuracy': acc_future,
                'n_future': n_future
            })
        
        if acc_future is not None and acc_future > best_future:
            best_k, best_future = k, acc_future

    if return_all_results:
        return best_k, best_future, pd.DataFrame(results)
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


def full_oos_sweep(df, test_df, k_range, base_elo=1500, return_results=False, use_mov=True, verbose=True):
    """
    For debugging intuition:
      train Elo with each k, then check OOS accuracy on the 3 event file.
      Uses the boutcount rule and NO gap filter.
    
    Args:
        use_mov: If True, uses method of victory scaling. If False, uses constant K.
        return_results: If True, returns a DataFrame with all k values and their OOS accuracies.
        verbose: If True, prints progress for each k value.
    """
    mov_label = "MOV" if use_mov else "No MOV"
    if verbose:
        print(f"\n=== FULL OOS SWEEP ({mov_label}, boutcount rule applied) ===")
    results = []
    
    for k in k_range:
        df_trained_k = run_basic_elo(df.copy(), k=k, base_elo=base_elo, use_mov=use_mov)
        acc_k = test_out_of_sample_accuracy(
            df_trained_k,
            test_df,
            best_k=k,
            base_elo=base_elo,
            verbose=False,
            gap_threshold=None,
        )
        if verbose:
            print(f"k={k}, OOS={acc_k:.4f}")
        
        if return_results:
            results.append({
                'k': k,
                'oos_accuracy': acc_k
            })
    
    if return_results:
        return pd.DataFrame(results)


def plot_k_optimization_results(train_results_df, oos_results_df, save_path=None):
    """
    Plot overall accuracy, test accuracy, and OOS accuracy vs k value.
    
    Args:
        train_results_df: DataFrame with columns ['k', 'overall_accuracy', 'test_accuracy']
        oos_results_df: DataFrame with columns ['k', 'oos_accuracy']
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot training accuracies
    ax.plot(train_results_df['k'], train_results_df['overall_accuracy'], 
            'o-', label='Overall Accuracy', linewidth=2, markersize=4, alpha=0.7)
    ax.plot(train_results_df['k'], train_results_df['test_accuracy'], 
            's-', label='Test Accuracy (Future)', linewidth=2, markersize=4, alpha=0.7)
    
    # Plot OOS accuracy
    ax.plot(oos_results_df['k'], oos_results_df['oos_accuracy'], 
            '^-', label='Out-of-Sample Accuracy', linewidth=2, markersize=4, alpha=0.7, color='red')
    
    ax.set_xlabel('K Value', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Elo K Parameter Optimization: Accuracy Metrics', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_mov_comparison(train_mov_df, train_no_mov_df, oos_mov_df, oos_no_mov_df, save_path=None):
    """
    Plot comparison of Elo with and without Method of Victory weights.
    
    Args:
        train_mov_df: Training results with MOV, columns ['k', 'overall_accuracy', 'test_accuracy']
        train_no_mov_df: Training results without MOV, columns ['k', 'overall_accuracy', 'test_accuracy']
        oos_mov_df: OOS results with MOV, columns ['k', 'oos_accuracy']
        oos_no_mov_df: OOS results without MOV, columns ['k', 'oos_accuracy']
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Overall Accuracy comparison
    ax1 = axes[0, 0]
    ax1.plot(train_mov_df['k'], train_mov_df['overall_accuracy'], 
             'o-', label='With MOV', linewidth=2, markersize=4, alpha=0.7, color='blue')
    ax1.plot(train_no_mov_df['k'], train_no_mov_df['overall_accuracy'], 
             's--', label='Without MOV', linewidth=2, markersize=4, alpha=0.7, color='orange')
    ax1.set_xlabel('K Value', fontsize=11)
    ax1.set_ylabel('Accuracy', fontsize=11)
    ax1.set_title('Overall Accuracy: MOV vs No MOV', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Test Accuracy comparison
    ax2 = axes[0, 1]
    ax2.plot(train_mov_df['k'], train_mov_df['test_accuracy'], 
             'o-', label='With MOV', linewidth=2, markersize=4, alpha=0.7, color='blue')
    ax2.plot(train_no_mov_df['k'], train_no_mov_df['test_accuracy'], 
             's--', label='Without MOV', linewidth=2, markersize=4, alpha=0.7, color='orange')
    ax2.set_xlabel('K Value', fontsize=11)
    ax2.set_ylabel('Accuracy', fontsize=11)
    ax2.set_title('Test Accuracy (Future): MOV vs No MOV', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # OOS Accuracy comparison
    ax3 = axes[1, 0]
    ax3.plot(oos_mov_df['k'], oos_mov_df['oos_accuracy'], 
             '^-', label='With MOV', linewidth=2, markersize=4, alpha=0.7, color='blue')
    ax3.plot(oos_no_mov_df['k'], oos_no_mov_df['oos_accuracy'], 
             'v--', label='Without MOV', linewidth=2, markersize=4, alpha=0.7, color='orange')
    ax3.set_xlabel('K Value', fontsize=11)
    ax3.set_ylabel('Accuracy', fontsize=11)
    ax3.set_title('Out-of-Sample Accuracy: MOV vs No MOV', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    
    # Combined view - all metrics
    ax4 = axes[1, 1]
    ax4.plot(train_mov_df['k'], train_mov_df['overall_accuracy'], 
             'o-', label='Overall (MOV)', linewidth=1.5, markersize=3, alpha=0.6, color='blue')
    ax4.plot(train_mov_df['k'], train_mov_df['test_accuracy'], 
             's-', label='Test (MOV)', linewidth=1.5, markersize=3, alpha=0.6, color='cyan')
    ax4.plot(oos_mov_df['k'], oos_mov_df['oos_accuracy'], 
             '^-', label='OOS (MOV)', linewidth=1.5, markersize=3, alpha=0.6, color='darkblue')
    ax4.plot(train_no_mov_df['k'], train_no_mov_df['overall_accuracy'], 
             'o--', label='Overall (No MOV)', linewidth=1.5, markersize=3, alpha=0.6, color='orange')
    ax4.plot(train_no_mov_df['k'], train_no_mov_df['test_accuracy'], 
             's--', label='Test (No MOV)', linewidth=1.5, markersize=3, alpha=0.6, color='yellow')
    ax4.plot(oos_no_mov_df['k'], oos_no_mov_df['oos_accuracy'], 
             'v--', label='OOS (No MOV)', linewidth=1.5, markersize=3, alpha=0.6, color='red')
    ax4.set_xlabel('K Value', fontsize=11)
    ax4.set_ylabel('Accuracy', fontsize=11)
    ax4.set_title('All Metrics: MOV vs No MOV', fontsize=12, fontweight='bold')
    ax4.legend(loc='best', fontsize=8, ncol=2)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])
    
    plt.suptitle('Elo Rating System: Method of Victory (MOV) Impact Comparison', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv("interleaved_cleaned.csv", low_memory=False)
    test_df = pd.read_csv("past3_events.csv", low_memory=False)

    df["result"] = pd.to_numeric(df["result"], errors="coerce")
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.sort_values("DATE").reset_index(drop=True)

    # boutcounts first, so prediction filters can use them
    df = add_bout_counts(df)

    # Training end date
    cut = df["DATE"].quantile(0.8)
    print("Training end date:", cut)
    print(test_df["date"].min(), test_df["date"].max())

    k_range = range(10, 500, 10)
    
    # ========== WITH MOV ==========
    print("\n" + "="*60)
    print("OPTIMIZING WITH METHOD OF VICTORY (MOV) WEIGHTS")
    print("="*60)
    
    # 1) sweep k on historical data (with method of victory aware Elo)
    best_k_mov, best_accuracy_mov, train_results_mov_df = genetic_algorithm_k(
        df, k_range=k_range, return_all_results=True, use_mov=True, verbose=True
    )
    print(f"\nBest k value (MOV): {best_k_mov}")
    print(f"Best future accuracy (MOV): {best_accuracy_mov:.4f}")

    # 2) train once with best k to get final ratings
    df_trained_mov = run_basic_elo(df.copy(), k=best_k_mov, use_mov=True)

    # 3) evaluate OOS on the three known events
    oos_acc_mov = test_out_of_sample_accuracy(
        df_trained_mov,
        test_df,
        best_k=best_k_mov,
        verbose=False,
        gap_threshold=75,  # can change or set to None
    )
    print(f"Overall OOS (MOV): {oos_acc_mov:.4f}")

    # 4) debug OOS across all k values
    oos_results_mov_df = full_oos_sweep(
        df, test_df, k_range=k_range, return_results=True, use_mov=True, verbose=True
    )
    
    # ========== WITHOUT MOV ==========
    print("\n" + "="*60)
    print("OPTIMIZING WITHOUT METHOD OF VICTORY (MOV) WEIGHTS")
    print("="*60)
    
    # 1) sweep k on historical data (without method of victory)
    best_k_no_mov, best_accuracy_no_mov, train_results_no_mov_df = genetic_algorithm_k(
        df, k_range=k_range, return_all_results=True, use_mov=False, verbose=True
    )
    print(f"\nBest k value (No MOV): {best_k_no_mov}")
    print(f"Best future accuracy (No MOV): {best_accuracy_no_mov:.4f}")

    # 2) train once with best k to get final ratings
    df_trained_no_mov = run_basic_elo(df.copy(), k=best_k_no_mov, use_mov=False)

    # 3) evaluate OOS on the three known events
    oos_acc_no_mov = test_out_of_sample_accuracy(
        df_trained_no_mov,
        test_df,
        best_k=best_k_no_mov,
        verbose=False,
        gap_threshold=75,  # can change or set to None
    )
    print(f"Overall OOS (No MOV): {oos_acc_no_mov:.4f}")

    # 4) debug OOS across all k values
    oos_results_no_mov_df = full_oos_sweep(
        df, test_df, k_range=k_range, return_results=True, use_mov=False, verbose=True
    )
    
    # ========== SUMMARY ==========
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    print(f"\nWITH MOV:")
    print(f"  Best K: {best_k_mov}")
    print(f"  Best Test Accuracy: {best_accuracy_mov:.4f}")
    print(f"  OOS Accuracy (at best K): {oos_acc_mov:.4f}")
    print(f"\nWITHOUT MOV:")
    print(f"  Best K: {best_k_no_mov}")
    print(f"  Best Test Accuracy: {best_accuracy_no_mov:.4f}")
    print(f"  OOS Accuracy (at best K): {oos_acc_no_mov:.4f}")
    print(f"\nMOV Improvement:")
    print(f"  Test Accuracy: {best_accuracy_mov - best_accuracy_no_mov:+.4f}")
    print(f"  OOS Accuracy: {oos_acc_mov - oos_acc_no_mov:+.4f}")
    
    # 5) Plot comparison
    print("\n=== Generating comparison plots ===")
    plot_mov_comparison(
        train_results_mov_df,
        train_results_no_mov_df,
        oos_results_mov_df,
        oos_results_no_mov_df,
        save_path='mov_comparison_plot.png'
    )
