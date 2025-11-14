import random
import math
import pandas as pd


# =========================
# Elo with Method of Victory
# =========================

def mov_factor(row, params):
    """
    Compute method of victory multiplier for this fight.
    params is a dict with keys:
        w_ko, w_sub, w_udec, w_sdec, w_mdec
    We treat win and loss sides symmetrically for the K boost.
    """
    w_ko   = params["w_ko"]
    w_sub  = params["w_sub"]
    w_udec = params["w_udec"]
    w_sdec = params["w_sdec"]
    w_mdec = params["w_mdec"]

    # If a KO happened (either fighter)
    if (row.get("ko", 0) == 1) or (row.get("kod", 0) == 1):
        return w_ko
    # If a sub happened (either fighter)
    if (row.get("subw", 0) == 1) or (row.get("subwd", 0) == 1):
        return w_sub
    # Unanimous decision
    if (row.get("udec", 0) == 1) or (row.get("udecd", 0) == 1):
        return w_udec
    # Split decision
    if (row.get("sdec", 0) == 1) or (row.get("sdecd", 0) == 1):
        return w_sdec
    # Majority decision
    if (row.get("mdec", 0) == 1) or (row.get("mdecd", 0) == 1):
        return w_mdec

    # Fallback if none of the flags are set
    return 1.0


def run_basic_elo(df, k=32, base_elo=1500, denominator=400, mov_params=None):
    """
    Core Elo engine, with optional method of victory multipliers.
    mov_params is a dict with the MoV weights, or None for no adjustment.
    """
    df = df.copy()
    ratings = {}
    pre, post, opp_pre, opp_post = [], [], [], []

    for _, row in df.iterrows():
        f1, f2, res = row["FIGHTER"], row["opp_FIGHTER"], row["result"]
        r1 = ratings.get(f1, base_elo)
        r2 = ratings.get(f2, base_elo)

        # Base expected scores
        e1 = 1.0 / (1.0 + 10.0 ** ((r2 - r1) / denominator))
        e2 = 1.0 / (1.0 + 10.0 ** ((r1 - r2) / denominator))

        # Method of victory factor
        if mov_params is not None:
            factor = mov_factor(row, mov_params)
        else:
            factor = 1.0

        k_eff = k * factor

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


# =========================
# Accuracy helpers
# =========================

def build_fighter_history(df):
    a = df[["DATE", "FIGHTER", "precomp_elo", "postcomp_elo"]].rename(
        columns={"FIGHTER": "fighter",
                 "precomp_elo": "pre_elo",
                 "postcomp_elo": "post_elo"}
    )
    b = df[["DATE", "opp_FIGHTER", "opp_precomp_elo", "opp_postcomp_elo"]].rename(
        columns={"opp_FIGHTER": "fighter",
                 "opp_precomp_elo": "pre_elo",
                 "opp_postcomp_elo": "post_elo"}
    )
    hist = pd.concat([a, b], ignore_index=True)
    hist = hist.sort_values("DATE").rename(columns={"DATE": "date"}).reset_index(drop=True)
    return hist


def has_prior_history(first_dates, fighter, date):
    """
    Returns True if fighter has at least one fight before this date.
    This enforces your rule:
    both fighters must have 1 or more prior fights.
    """
    d0 = first_dates.get(fighter)
    return bool(d0 and date > d0)


def compute_fight_predictions(df):
    """
    Build predictions from pre-fight Elo.
    Only counts fights where:
      - result is 0 or 1
      - precomp_elo != opp_precomp_elo
      - both fighters have at least one prior fight
    """
    hist = build_fighter_history(df)
    first_dates = hist.groupby("fighter")["date"].min().to_dict()
    out = []

    for _, r in df.iterrows():
        d = r["DATE"]
        if pd.isna(d) or r["result"] not in (0, 1):
            continue
        if r["precomp_elo"] == r["opp_precomp_elo"]:
            continue
        if not has_prior_history(first_dates, r["FIGHTER"], d):
            continue
        if not has_prior_history(first_dates, r["opp_FIGHTER"], d):
            continue

        pred = int(r["precomp_elo"] > r["opp_precomp_elo"])
        out.append({
            "date": d,
            "prediction": pred,
            "result": int(r["result"]),
            "correct": int(pred == r["result"])
        })

    return pd.DataFrame(out)


def elo_accuracy(df, cutoff_date=None):
    preds = compute_fight_predictions(df)
    if preds.empty:
        return None, 0.0, 0

    acc_all = preds["correct"].mean()

    if cutoff_date is None:
        return acc_all, None, 0

    future = preds[preds["date"] > cutoff_date]
    if future.empty:
        return acc_all, None, 0

    acc_future = future["correct"].mean()
    return acc_all, acc_future, len(future)


# =========================
# OOS helpers
# =========================

def latest_ratings_from_trained_df(df, base_elo=1500):
    """
    Build {fighter_name: latest_post_fight_elo} from df with Elo columns.
    Uses both FIGHTER and opp_FIGHTER sides.
    """
    ratings = {}
    for _, r in df.iterrows():
        ratings[r["FIGHTER"]] = r["postcomp_elo"]
        ratings[r["opp_FIGHTER"]] = r["opp_postcomp_elo"]
    return ratings


def test_out_of_sample_accuracy(df_trained_with_elo,
                                test_df,
                                base_elo=1500,
                                verbose=False,
                                gap_threshold=None):
    """
    Evaluate OOS accuracy using frozen ratings from df_trained_with_elo.
    Does not update ratings during OOS.
    gap_threshold: if not None, also compute accuracy only on fights
                   where abs(Elo diff) >= gap_threshold.
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


# =========================
# Genetic algorithm over k and MOV weights
# =========================

PARAM_BOUNDS = {
    "k":     (10.0, 500.0),
    "w_ko":  (1.0, 2.0),
    "w_sub": (1.0, 2.0),
    "w_udec": (0.8, 1.2),
    "w_sdec": (0.5, 1.1),
    "w_mdec": (0.7, 1.2),
}


def random_param_value(key):
    lo, hi = PARAM_BOUNDS[key]
    return random.uniform(lo, hi)


def random_params():
    return {
        "k": random_param_value("k"),
        "w_ko": random_param_value("w_ko"),
        "w_sub": random_param_value("w_sub"),
        "w_udec": random_param_value("w_udec"),
        "w_sdec": random_param_value("w_sdec"),
        "w_mdec": random_param_value("w_mdec"),
    }


def clip_param(key, value):
    lo, hi = PARAM_BOUNDS[key]
    return max(lo, min(hi, value))


def evaluate_params(df, cutoff_date, params):
    """
    Run Elo with given params and return future accuracy.
    params: dict with k and mov weights
    """
    mov_params = {
        "w_ko": params["w_ko"],
        "w_sub": params["w_sub"],
        "w_udec": params["w_udec"],
        "w_sdec": params["w_sdec"],
        "w_mdec": params["w_mdec"],
    }
    k = params["k"]

    trial = run_basic_elo(df, k=k, mov_params=mov_params)
    acc_all, acc_future, n_future = elo_accuracy(trial, cutoff_date)

    if acc_future is None:
        return 0.0
    return acc_future


def tournament_select(population, k_tour=3):
    """
    Tournament selection: pick k individuals and return the best.
    population is a list of dicts with keys: params, fitness.
    """
    contenders = random.sample(population, k_tour)
    contenders.sort(key=lambda ind: ind["fitness"], reverse=True)
    return contenders[0]


def crossover(parent1, parent2, crossover_rate=0.5):
    """
    Simple uniform or averaging crossover.
    For each param, with probability 0.5 take average, else take one parent.
    """
    child_params = {}
    for key in parent1["params"].keys():
        v1 = parent1["params"][key]
        v2 = parent2["params"][key]
        if random.random() < crossover_rate:
            child_params[key] = 0.5 * (v1 + v2)
        else:
            child_params[key] = random.choice([v1, v2])
    return child_params


def mutate(params, mutation_rate=0.3, mutation_scale=0.1):
    """
    Gaussian mutation: for each param, with probability mutation_rate,
    add noise proportional to range * mutation_scale.
    """
    new_params = params.copy()
    for key in new_params.keys():
        if random.random() < mutation_rate:
            lo, hi = PARAM_BOUNDS[key]
            span = hi - lo
            noise = random.gauss(0.0, mutation_scale * span)
            new_val = new_params[key] + noise
            new_params[key] = clip_param(key, new_val)
    return new_params


def ga_search_params(df,
                     population_size=30,
                     generations=30,
                     cutoff_quantile=0.8,
                     seed=42):
    """
    Full GA search over k and method of victory weights.
    Returns best_params, best_fitness.
    """
    random.seed(seed)

    cutoff_date = df["DATE"].quantile(cutoff_quantile)
    print("Training end date:", cutoff_date)

    # Initialize population
    population = []
    for _ in range(population_size):
        p = random_params()
        fitness = evaluate_params(df, cutoff_date, p)
        population.append({"params": p, "fitness": fitness})

    best_ind = max(population, key=lambda ind: ind["fitness"])
    print(f"Initial best fitness: {best_ind['fitness']:.4f}, params: {best_ind['params']}")

    # GA loop
    for gen in range(generations):
        new_population = []

        # Always keep the best (elitism)
        population.sort(key=lambda ind: ind["fitness"], reverse=True)
        elite = population[0]
        new_population.append(elite)

        # Fill the rest
        while len(new_population) < population_size:
            parent1 = tournament_select(population)
            parent2 = tournament_select(population)
            child_params = crossover(parent1, parent2)
            child_params = mutate(child_params)

            fitness = evaluate_params(df, cutoff_date, child_params)
            new_population.append({"params": child_params, "fitness": fitness})

        population = new_population
        gen_best = max(population, key=lambda ind: ind["fitness"])
        if gen_best["fitness"] > best_ind["fitness"]:
            best_ind = gen_best

        print(f"Gen {gen + 1:02d}, best future acc = {gen_best['fitness']:.4f}, params = {gen_best['params']}")

    return best_ind["params"], best_ind["fitness"], cutoff_date


# =========================
# Main
# =========================

if __name__ == "__main__":
    df = pd.read_csv("interleaved_cleaned.csv", low_memory=False)
    test_df = pd.read_csv("past3_events.csv", low_memory=False)

    df["result"] = pd.to_numeric(df["result"], errors="coerce")
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.sort_values("DATE").reset_index(drop=True)

    print(test_df["date"].min(), test_df["date"].max())

    # Run GA search over k and MoV weights
    best_params, best_future_acc, cutoff = ga_search_params(
        df,
        population_size=30,
        generations=30,
        cutoff_quantile=0.8,
        seed=42
    )

    print("\n=== GA best params ===")
    print(best_params)
    print(f"Best future accuracy on training window: {best_future_acc:.4f}")

    # Train final Elo with best params on full training df
    mov_params = {
        "w_ko": best_params["w_ko"],
        "w_sub": best_params["w_sub"],
        "w_udec": best_params["w_udec"],
        "w_sdec": best_params["w_sdec"],
        "w_mdec": best_params["w_mdec"],
    }
    best_k = best_params["k"]

    df_trained = run_basic_elo(df.copy(), k=best_k, mov_params=mov_params)

    # OOS evaluation on past 3 events
    print("\n=== OOS evaluation on past3_events.csv ===")
    oos_acc = test_out_of_sample_accuracy(
        df_trained,
        test_df,
        verbose=True,
        gap_threshold=75
    )
    print("Overall OOS accuracy:", oos_acc)

    # Optional: full OOS sweep printing for each k is no longer needed here
    # because GA already searched k, but you can still add similar sweeps
    # if you want to debug or sanity check the GA results.
