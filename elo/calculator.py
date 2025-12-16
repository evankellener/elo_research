import pandas as pd
from .elo_utils import method_of_victory_scale

def run_basic_elo(df, k=32, base_elo=1500, denominator=400, draw_k_factor=0.5):
    ratings = {}
    pre_elo, post_elos = [], []
    opp_pre_elos, opp_post_elos = [], []

    for _, row in df.iterrows():
        f1, f2, result = row['FIGHTER'], row['opp_FIGHTER'], row['result']
        
        is_draw = (row.get('win', 1) == 0) and (row.get('loss', 1) == 0)
        if is_draw:
            result = 0.5

        ratings.setdefault(f1, base_elo)
        ratings.setdefault(f2, base_elo)

        f1_pre, f2_pre = ratings[f1], ratings[f2]

        expected_f1 = 1/(1+10.0**((f2_pre-f1_pre)/denominator))
        expected_f2 = 1/(1+10.0**((f1_pre-f2_pre)/denominator))

        k_eff = k * (draw_k_factor if is_draw else 1.0)

        new_f1_rating = f1_pre + k_eff * (result - expected_f1)
        new_f2_rating = f2_pre + k_eff * ((1 - result) - expected_f2)

        ratings[f1] = new_f1_rating
        ratings[f2] = new_f2_rating

        pre_elo.append(f1_pre)
        post_elos.append(new_f1_rating)
        opp_pre_elos.append(f2_pre)
        opp_post_elos.append(new_f2_rating)

    df['precomp_elo'] = pre_elo
    df['postcomp_elo'] = post_elos
    df['opp_precomp_elo'] = opp_pre_elos
    df['opp_postcomp_elo'] = opp_post_elos

    return df

def run_basic_elo_with_mov(df, k=167.19618191211478, base_elo=1500, denominator=400, draw_k_factor=0.5, 
                           w_ko=None, w_sub=None, w_udec=None, w_sdec=None, w_mdec=None):
    ratings = {}
    pre_elo, post_elos = [], []
    opp_pre_elos, opp_post_elos = [], []

    for _, row in df.iterrows():
        f1, f2, result = row['FIGHTER'], row['opp_FIGHTER'], row['result']
        
        is_draw = (row.get('win', 1) == 0) and (row.get('loss', 1) == 0)
        if is_draw:
            result = 0.5

        ratings.setdefault(f1, base_elo)
        ratings.setdefault(f2, base_elo)

        f1_pre, f2_pre = ratings[f1], ratings[f2]

        expected_f1 = 1/(1+10.0**((f2_pre-f1_pre)/denominator))
        expected_f2 = 1/(1+10.0**((f1_pre-f2_pre)/denominator))

        if w_ko is not None or w_sub is not None or w_udec is not None or w_sdec is not None or w_mdec is not None:
            mov_scale = method_of_victory_scale(row, 
                                               w_ko=w_ko if w_ko is not None else 1.4,
                                               w_sub=w_sub if w_sub is not None else 1.3,
                                               w_udec=w_udec if w_udec is not None else 1.0,
                                               w_sdec=w_sdec if w_sdec is not None else 0.7,
                                               w_mdec=w_mdec if w_mdec is not None else 0.9)
        else:
            mov_scale = method_of_victory_scale(row)
        k_eff = k * mov_scale
        
        if is_draw:
            k_eff = k_eff * draw_k_factor

        new_f1_rating = f1_pre + k_eff * (result - expected_f1)
        new_f2_rating = f2_pre + k_eff * ((1 - result) - expected_f2)

        ratings[f1] = new_f1_rating
        ratings[f2] = new_f2_rating

        pre_elo.append(f1_pre)
        post_elos.append(new_f1_rating)
        opp_pre_elos.append(f2_pre)
        opp_post_elos.append(new_f2_rating)

    df['precomp_elo'] = pre_elo
    df['postcomp_elo'] = post_elos
    df['opp_precomp_elo'] = opp_pre_elos
    df['opp_postcomp_elo'] = opp_post_elos

    return df
