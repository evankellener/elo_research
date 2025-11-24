import random
import pandas as pd
import unicodedata
from multiprocessing import Pool, cpu_count
from functools import partial
from elo_utils import mov_factor, build_fighter_history, has_prior_history, add_bout_counts


def normalize_name(name):
    """
    Normalize fighter name for matching (handles special characters, middle names, etc.)
    Converts special characters like 'ę' to 'e' by normalizing unicode and removing combining marks.
    """
    if pd.isna(name):
        return ""
    # Normalize unicode (decomposes characters like 'ę' into 'e' + combining mark)
    name = unicodedata.normalize('NFKD', str(name))
    # Filter out combining characters (like the combining mark after decomposed 'ę')
    name = ''.join(c for c in name if not unicodedata.combining(c))
    # Convert to lowercase and remove extra spaces
    name = ' '.join(name.lower().split())
    return name


def find_fighter_match(test_name, training_names):
    """
    Find the best matching fighter name from training data.
    Handles cases like:
    - "Jose Miguel Delgado" vs "Jose Delgado"
    - "Mateusz Rębecki" vs "Mateusz Rebecki"
    """
    test_normalized = normalize_name(test_name)
    
    # First try exact match
    for train_name in training_names:
        if normalize_name(train_name) == test_normalized:
            return train_name
    
    # Try matching by last name and first name (handles middle name variations)
    test_parts = test_normalized.split()
    if len(test_parts) >= 2:
        test_first = test_parts[0]
        test_last = test_parts[-1]
        for train_name in training_names:
            train_normalized = normalize_name(train_name)
            train_parts = train_normalized.split()
            if len(train_parts) >= 2:
                train_first = train_parts[0]
                train_last = train_parts[-1]
                # Match if first and last names match (ignoring middle names)
                if test_first == train_first and test_last == train_last:
                    return train_name
    
    return None


# =========================
# Elo with Method of Victory
# =========================


def run_basic_elo(df, k=32, base_elo=1500, denominator=400, mov_params=None, draw_k_factor=0.5):
    """
    Core Elo engine, with optional method of victory multipliers.
    mov_params is a dict with the MoV weights, or None for no adjustment.
    draw_k_factor: Multiplier for K-factor in draws (default 0.5, meaning draws have half the impact)
    
    Note: Assumes df is sorted by DATE. If not, results may be incorrect.
    """
    df = df.copy()
    # Ensure dataframe is sorted by date for correct chronological processing
    if "DATE" in df.columns:
        df = df.sort_values("DATE").reset_index(drop=True)
    ratings = {}
    pre, post, opp_pre, opp_post = [], [], [], []

    for _, row in df.iterrows():
        f1, f2, res = row["FIGHTER"], row["opp_FIGHTER"], row["result"]
        
        # Check if this is a draw (both win and loss are 0)
        # Note: Default value of 1 means if columns don't exist, it won't be detected as a draw
        # This assumes "win" and "loss" columns always exist in the data
        is_draw = (row.get("win", 1) == 0) and (row.get("loss", 1) == 0)
        
        # For draws, use 0.5 as the result (half win for each fighter)
        if is_draw:
            res = 0.5
        
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
        
        # Reduce K-factor for draws (draws are less decisive)
        if is_draw:
            k_eff = k_eff * draw_k_factor

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

        # Use precomp_elo for predictions (rating before the fight)
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

def latest_ratings_from_trained_df(df, base_elo=1500, as_of_date=None):
    """
    Build {fighter_name: latest_post_fight_elo} from df with Elo columns.
    Uses both FIGHTER and opp_FIGHTER sides.
    
    If as_of_date is provided, only uses fights before that date to get ratings.
    Returns the postcomp_elo from each fighter's LAST fight before as_of_date.
    """
    if as_of_date is not None:
        df = df[df["DATE"] < as_of_date].copy()
    
    # Sort by date to ensure we get the latest rating for each fighter
    if "DATE" in df.columns:
        df = df.sort_values("DATE").reset_index(drop=True)
    
    ratings = {}
    for _, r in df.iterrows():
        # Use postcomp_elo (rating AFTER the fight)
        # Later fights will overwrite earlier ones, giving us the latest rating
        ratings[r["FIGHTER"]] = r["postcomp_elo"]
        ratings[r["opp_FIGHTER"]] = r["opp_postcomp_elo"]
    return ratings


def build_training_bout_counts(df):
    """
    Count how many bouts each fighter has in the training data.
    This is used to enforce the "must have at least N prior fights" rule
    on the OOS evaluation.
    """
    counts = {}
    for _, r in df.iterrows():
        f1 = r["FIGHTER"]
        f2 = r["opp_FIGHTER"]
        counts[f1] = counts.get(f1, 0) + 1
        counts[f2] = counts.get(f2, 0) + 1
    return counts


def test_out_of_sample_accuracy(
    df_trained_with_elo,
    test_df,
    base_elo=1500,
    verbose=False,
    gap_threshold=None,
    min_train_bouts=1,
):
    """
    Evaluate OOS accuracy using frozen ratings from df_trained_with_elo.
    Does not update ratings during OOS.

    Only counts OOS fights where BOTH fighters have at least min_train_bouts
    fights in the training data.
    """
    bout_counts = build_training_bout_counts(df_trained_with_elo)
    
    # Build set of all fighter names in training data for name matching
    all_training_fighters = set()
    for _, r in df_trained_with_elo.iterrows():
        all_training_fighters.add(r["FIGHTER"])
        all_training_fighters.add(r["opp_FIGHTER"])

    tdf = test_df.copy()
    tdf["result"] = pd.to_numeric(tdf["result"], errors="coerce")
    tdf["DATE"] = pd.to_datetime(tdf["date"])

    hits_all, total_all = 0, 0
    hits_gap, total_gap = 0, 0

    raw_count = 0
    used_count = 0
    skipped_boutcount = 0

    for _, row in tdf.iterrows():
        f1_test = row["fighter"]
        f2_test = row["opp_fighter"]
        res = row["result"]
        test_date = row["DATE"]

        if res not in (0, 1):
            continue

        raw_count += 1

        # Try to match fighter names (handles name variations)
        f1 = find_fighter_match(f1_test, all_training_fighters) or f1_test
        f2 = find_fighter_match(f2_test, all_training_fighters) or f2_test

        # Get total bout counts for both fighters from training data (for filtering)
        f1_bouts = bout_counts.get(f1, 0)
        f2_bouts = bout_counts.get(f2, 0)
        
        # Calculate precomp_boutcount for this specific test fight date
        # (how many fights each fighter had BEFORE this test date)
        f1_precomp = len(df_trained_with_elo[
            (df_trained_with_elo["DATE"] < test_date) & 
            ((df_trained_with_elo["FIGHTER"] == f1) | (df_trained_with_elo["opp_FIGHTER"] == f1))
        ])
        f2_precomp = len(df_trained_with_elo[
            (df_trained_with_elo["DATE"] < test_date) & 
            ((df_trained_with_elo["FIGHTER"] == f2) | (df_trained_with_elo["opp_FIGHTER"] == f2))
        ])

        # Bout count rule: both fighters must have at least min_train_bouts fights in training
        if f1_bouts < min_train_bouts or f2_bouts < min_train_bouts:
            skipped_boutcount += 1
            if verbose:
                name_note = f" (matched: {f1}/{f2})" if (f1 != f1_test or f2 != f2_test) else ""
                print(
                    f"{row['DATE'].date()} | {f1_test} vs {f2_test}{name_note} | "
                    f"precomp_boutcount: {f1_precomp}/{f2_precomp} | "
                    f"train_bouts: {f1_bouts}/{f2_bouts} -> SKIPPED (train_bouts < {min_train_bouts})"
                )
            continue

        used_count += 1

        # Get postcomp_elo ratings as of this test date (ratings AFTER their last fight before test date)
        rating_lookup = latest_ratings_from_trained_df(df_trained_with_elo, base_elo=base_elo, as_of_date=test_date)
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
            name_note = f" (matched: {f1}/{f2})" if (f1 != f1_test or f2 != f2_test) else ""
            print(
                f"{row['DATE'].date()} | {f1_test} vs {f2_test}{name_note} | "
                f"precomp_boutcount: {f1_precomp}/{f2_precomp} | "
                f"train_bouts: {f1_bouts}/{f2_bouts} | "
                f"postcomp_elo: {r1:.1f}/{r2:.1f} -> pred={pred} res={int(res)} {'✓' if correct else '✗'}"
            )

    acc_all = hits_all / total_all if total_all else None
    acc_gap = hits_gap / total_gap if total_gap else None

    if verbose:
        print(f"\nOOS fights with labels: {raw_count}")
        print(f"OOS used after boutcount rule (min_train_bouts={min_train_bouts}): {used_count}")
        print(f"OOS skipped by boutcount rule: {skipped_boutcount}")

        if gap_threshold is not None:
            print(f"\nGAP FILTER ACTIVE: abs(Elo diff) >= {gap_threshold}")
            print(f"gap fights counted: {total_gap}")
            print(f"gap accuracy: {acc_gap}")

    return acc_all


# =========================
# Genetic algorithm over k and MOV weights
# =========================

def _calculate_oos_accuracy_worker(args):
    """Module-level worker function for parallel OOS calculation (must be at module level for multiprocessing)"""
    params_dict, train_data, test_data = args
    params = params_dict["params"]
    mov_params = {
        "w_ko": params["w_ko"],
        "w_sub": params["w_sub"],
        "w_udec": params["w_udec"],
        "w_sdec": params["w_sdec"],
        "w_mdec": params["w_mdec"],
    }
    
    # Train on all data before test dates
    df_trained = run_basic_elo(train_data.copy(), k=params["k"], mov_params=mov_params)
    
    # Calculate OOS accuracy
    oos_acc = test_out_of_sample_accuracy(
        df_trained,
        test_data,
        base_elo=1500,
        verbose=False,
        min_train_bouts=1,
    )
    return params_dict["index"], oos_acc

PARAM_BOUNDS = {
    "k":      (10.0, 500.0),
    "w_ko":   (1.0, 2.0),
    "w_sub":  (1.0, 2.0),
    "w_udec": (0.8, 1.2),
    "w_sdec": (0.5, 1.1),
    "w_mdec": (0.7, 1.2),
}


def random_param_value(key):
    lo, hi = PARAM_BOUNDS[key]
    return random.uniform(lo, hi)


def random_params():
    return {
        "k":      random_param_value("k"),
        "w_ko":   random_param_value("w_ko"),
        "w_sub":  random_param_value("w_sub"),
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
    Training accuracy already uses boutcount rule through compute_fight_predictions.
    """
    mov_params = {
        "w_ko":   params["w_ko"],
        "w_sub":  params["w_sub"],
        "w_udec": params["w_udec"],
        "w_sdec": params["w_sdec"],
        "w_mdec": params["w_mdec"],
    }
    k = params["k"]

    trial = run_basic_elo(df, k=k, mov_params=mov_params)
    _, acc_future, _ = elo_accuracy(trial, cutoff_date)

    if acc_future is None:
        return 0.0
    return acc_future


def tournament_select(population, k_tour=3):
    """
    Tournament selection: pick k individuals and return the best.
    population is a list of dicts with keys: params, fitness.
    
    If k_tour > len(population), uses len(population) instead.
    """
    k_tour = min(k_tour, len(population))
    contenders = random.sample(population, k_tour)
    contenders.sort(key=lambda ind: ind["fitness"], reverse=True)
    return contenders[0]


def crossover(parent1, parent2, crossover_rate=0.5):
    """
    Simple crossover.
    For each param, with probability crossover_rate take average, else take one parent.
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
    mutated = False
    for key in new_params.keys():
        if random.random() < mutation_rate:
            lo, hi = PARAM_BOUNDS[key]
            span = hi - lo
            noise = random.gauss(0.0, mutation_scale * span)
            new_val = new_params[key] + noise
            new_params[key] = clip_param(key, new_val)
            mutated = True
    # Ensure at least one parameter is mutated to maintain diversity
    if not mutated and random.random() < 0.5:
        key = random.choice(list(new_params.keys()))
        lo, hi = PARAM_BOUNDS[key]
        span = hi - lo
        noise = random.gauss(0.0, mutation_scale * span)
        new_val = new_params[key] + noise
        new_params[key] = clip_param(key, new_val)
    return new_params


def ga_search_params(
    df,
    test_df=None,
    population_size=30,
    generations=30,
    cutoff_quantile=0.8,
    seed=42,
    return_all_results=False,
    verbose=True,
):
    """
    Full GA search over k and method of victory weights.
    
    Args:
        df: Training data
        test_df: Optional test data for OOS evaluation tracking (does not affect fitness)
        return_all_results: If True, returns a list of all generation results
        verbose: If True, prints progress for each generation
    
    Returns:
        If return_all_results=False: best_params, best_fitness, cutoff_date
        If return_all_results=True: best_params, best_fitness, cutoff_date, all_results
    """
    if seed is not None:
        random.seed(seed)

    cutoff_date = df["DATE"].quantile(cutoff_quantile)
    if verbose:
        print("Training end date:", cutoff_date)
    
    # Prepare test data for OOS evaluation if provided
    oos_test_start_date = None
    df_train_all_for_oos = None
    if test_df is not None:
        test_df_copy = test_df.copy()
        test_df_copy["DATE"] = pd.to_datetime(test_df_copy["date"])
        oos_test_start_date = test_df_copy["DATE"].min()
        df_train_all_for_oos = df[df["DATE"] < oos_test_start_date].copy()
        if verbose:
            print(f"OOS test start date: {oos_test_start_date}")

    all_results = []


    # Helper function to calculate OOS accuracy for a set of params (single-threaded fallback)
    def calculate_oos_accuracy(params):
        """Calculate OOS accuracy for given params (does not affect fitness)"""
        if test_df is None or df_train_all_for_oos is None:
            return None
        
        mov_params = {
            "w_ko": params["w_ko"],
            "w_sub": params["w_sub"],
            "w_udec": params["w_udec"],
            "w_sdec": params["w_sdec"],
            "w_mdec": params["w_mdec"],
        }
        
        # Train on all data before test dates
        df_trained = run_basic_elo(df_train_all_for_oos.copy(), k=params["k"], mov_params=mov_params)
        
        # Calculate OOS accuracy
        oos_acc = test_out_of_sample_accuracy(
            df_trained,
            test_df,
            base_elo=1500,
            verbose=False,
            min_train_bouts=1,
        )
        return oos_acc

    def calculate_oos_accuracy_batch(params_list, use_parallel=True):
        """Calculate OOS accuracy for a batch of params, optionally in parallel"""
        if test_df is None or df_train_all_for_oos is None:
            return [None] * len(params_list)
        
        if use_parallel and len(params_list) > 1:
            # Use multiprocessing for parallel evaluation
            num_workers = min(cpu_count(), len(params_list))
            worker_inputs = [
                ({"params": p, "index": i}, df_train_all_for_oos, test_df)
                for i, p in enumerate(params_list)
            ]
            
            with Pool(processes=num_workers) as pool:
                results = pool.map(_calculate_oos_accuracy_worker, worker_inputs)
            
            # Sort by index and extract OOS accuracies
            results.sort(key=lambda x: x[0])
            return [acc for _, acc in results]
        else:
            # Sequential evaluation
            return [calculate_oos_accuracy(p) for p in params_list]

    # Initialize population with diverse parameters
    population = []
    params_list = []
    for i in range(population_size):
        p = random_params()
        fitness = evaluate_params(df, cutoff_date, p)
        population.append({"params": p, "fitness": fitness, "oos_accuracy": None})
        params_list.append(p)
    
    # Calculate OOS accuracy in parallel for initial population
    if test_df is not None:
        if verbose:
            print(f"Calculating OOS accuracy for initial population of {population_size} individuals (parallel)...")
        oos_accs = calculate_oos_accuracy_batch(params_list, use_parallel=True)
        for i, oos_acc in enumerate(oos_accs):
            population[i]["oos_accuracy"] = oos_acc

    best_ind = max(population, key=lambda ind: ind["fitness"])
    if verbose:
        fitnesses = [ind["fitness"] for ind in population]
        avg_fitness = sum(fitnesses) / len(fitnesses)
        oos_accs = [ind["oos_accuracy"] for ind in population if ind["oos_accuracy"] is not None]
        oos_str = f", OOS={best_ind['oos_accuracy']:.4f}" if best_ind["oos_accuracy"] is not None else ""
        if oos_accs:
            avg_oos = sum(oos_accs) / len(oos_accs)
            oos_str += f" (avg={avg_oos:.4f})"
        print(
            f"Initial population: best={best_ind['fitness']:.4f}{oos_str}, "
            f"avg={avg_fitness:.4f}, "
            f"params: {best_ind['params']}"
        )

    # GA loop
    for gen in range(generations):
        new_population = []

        # Elitism - keep top individual (OOS already calculated)
        population.sort(key=lambda ind: ind["fitness"], reverse=True)
        elite = population[0].copy()  # Make a copy to avoid reference issues
        new_population.append(elite)

        # Fill the rest
        children_needed = population_size - len(new_population)
        child_params_list = []
        child_fitness_list = []
        
        while len(new_population) < population_size:
            parent1 = tournament_select(population)
            parent2 = tournament_select(population)
            child_params = crossover(parent1, parent2)
            child_params = mutate(child_params)
            
            fitness = evaluate_params(df, cutoff_date, child_params)
            child_params_list.append(child_params)
            child_fitness_list.append(fitness)
            new_population.append({"params": child_params, "fitness": fitness, "oos_accuracy": None})
        
        # Calculate OOS accuracy in parallel for new children
        if test_df is not None and children_needed > 0:
            if verbose:
                print(f"  Calculating OOS for {children_needed} new children in generation {gen + 1} (parallel)...")
            oos_accs = calculate_oos_accuracy_batch(child_params_list, use_parallel=True)
            # Update OOS accuracies (skip the elite which is already at index 0)
            for i, oos_acc in enumerate(oos_accs):
                new_population[i + 1]["oos_accuracy"] = oos_acc

        population = new_population
        gen_best = max(population, key=lambda ind: ind["fitness"])
        if gen_best["fitness"] > best_ind["fitness"]:
            best_ind = gen_best

        if verbose:
            # Calculate population diversity metrics
            fitnesses = [ind["fitness"] for ind in population]
            avg_fitness = sum(fitnesses) / len(fitnesses)
            min_fitness = min(fitnesses)
            max_fitness = max(fitnesses)
            fitness_std = (sum((f - avg_fitness) ** 2 for f in fitnesses) / len(fitnesses)) ** 0.5
            
            # Calculate OOS metrics
            oos_accs = [ind["oos_accuracy"] for ind in population if ind["oos_accuracy"] is not None]
            oos_str = ""
            if oos_accs:
                best_oos = max(oos_accs)
                avg_oos = sum(oos_accs) / len(oos_accs)
                min_oos = min(oos_accs)
                max_oos = max(oos_accs)
                oos_str = f", OOS: best={best_oos:.4f}, avg={avg_oos:.4f}, range=[{min_oos:.4f}, {max_oos:.4f}]"
            
            # Calculate parameter diversity (sample a few params to check)
            sample_k_values = [ind["params"]["k"] for ind in population[:5]]
            k_range = max(sample_k_values) - min(sample_k_values)
            
            print(
                f"Gen {gen + 1:02d}, best={gen_best['fitness']:.4f}, "
                f"avg={avg_fitness:.4f}, std={fitness_std:.4f}, "
                f"range=[{min_fitness:.4f}, {max_fitness:.4f}], "
                f"k_range={k_range:.1f}{oos_str}"
            )
            if gen_best["fitness"] > best_ind["fitness"]:
                oos_note = f", OOS={gen_best['oos_accuracy']:.4f}" if gen_best["oos_accuracy"] is not None else ""
                print(f"  *** NEW BEST: {gen_best['params']}{oos_note}")

        if return_all_results:
            # Store generation summary
            oos_accs = [ind['oos_accuracy'] for ind in population if ind['oos_accuracy'] is not None]
            gen_summary = {
                'generation': gen + 1,
                'best_fitness': gen_best['fitness'],
                'best_params': gen_best['params'].copy(),
                'population_avg_fitness': sum(ind['fitness'] for ind in population) / len(population),
                'population_max_fitness': gen_best['fitness'],
                'population_min_fitness': min(ind['fitness'] for ind in population),
            }
            if oos_accs:
                gen_summary['best_oos_accuracy'] = max(oos_accs)
                gen_summary['avg_oos_accuracy'] = sum(oos_accs) / len(oos_accs)
                gen_summary['min_oos_accuracy'] = min(oos_accs)
                gen_summary['max_oos_accuracy'] = max(oos_accs)
            all_results.append(gen_summary)

    if return_all_results:
        return best_ind["params"], best_ind["fitness"], cutoff_date, all_results
    return best_ind["params"], best_ind["fitness"], cutoff_date


# =========================
# Main
# =========================

if __name__ == "__main__":
    df = pd.read_csv("data/interleaved_cleaned.csv", low_memory=False)
    test_df = pd.read_csv("data/past3_events.csv", low_memory=False)

    df["result"] = pd.to_numeric(df["result"], errors="coerce")
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.sort_values("DATE").reset_index(drop=True)

    # boutcounts first, so prediction filters can use them
    df = add_bout_counts(df)
    # Ensure boutcount columns are numeric (they may be strings from CSV)
    if "precomp_boutcount" in df.columns:
        df["precomp_boutcount"] = pd.to_numeric(df["precomp_boutcount"], errors="coerce")
    if "opp_precomp_boutcount" in df.columns:
        df["opp_precomp_boutcount"] = pd.to_numeric(df["opp_precomp_boutcount"], errors="coerce")

    print(test_df["date"].min(), test_df["date"].max())

    # Run GA search over k and MoV weights
    # Set seed=None for different results on each run, or use a fixed seed for reproducibility
    best_params, best_future_acc, cutoff = ga_search_params(
        df,
        test_df=test_df,  # Pass test_df for OOS evaluation tracking
        population_size=30,
        generations=30,
        cutoff_quantile=0.8,
        seed=None,  # Set to None for random runs, or an integer for reproducibility
    )

    print("\n=== GA best params ===")
    print(best_params)
    print(f"Best future accuracy on training window: {best_future_acc:.4f}")

    # Train final Elo with best params on ALL DATA before test dates
    # The cutoff was only for GA optimization (train/validation split)
    # For OOS, we use all historical data before the test events
    mov_params = {
        "w_ko":   best_params["w_ko"],
        "w_sub":  best_params["w_sub"],
        "w_udec": best_params["w_udec"],
        "w_sdec": best_params["w_sdec"],
        "w_mdec": best_params["w_mdec"],
    }
    best_k = best_params["k"]

    # Use ALL data before the test dates for final training
    test_start_date = pd.to_datetime(test_df["date"]).min()
    df_train_all = df[df["DATE"] < test_start_date].copy()
    df_trained = run_basic_elo(df_train_all, k=best_k, mov_params=mov_params)

    # OOS evaluation on past 3 events
    print("\n=== OOS evaluation on data/past3_events.csv ===")
    oos_acc = test_out_of_sample_accuracy(
        df_trained,
        test_df,
        verbose=True,
        gap_threshold=75,
        min_train_bouts=1,  # bump this to 2 or 3 if you want "trusted Elo only"
    )
    print("Overall OOS accuracy:", oos_acc)
