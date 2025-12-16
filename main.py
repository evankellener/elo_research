import pandas as pd
from elo.calculator import run_basic_elo, run_basic_elo_with_mov
from elo.visualization import display_top_n_elos, most_recent_elo, graph_fighter_elo_history

def main():
    df = pd.read_csv('data/interleaved_cleaned.csv', low_memory=False)
    df['result'] = pd.to_numeric(df['result'], errors='coerce')
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.sort_values('DATE').reset_index(drop=True)
    
    print("\n" + "="*60)
    print("ELO RATINGS - Basic System")
    print("="*60)
    df_basic = run_basic_elo(df.copy(), k=32)
    display_top_n_elos(df_basic, n=10)
    most_recent_elo(df_basic, n=10)
    
    print("\n" + "="*60)
    print("ELO RATINGS - With Method of Victory")
    print("="*60)
    df_mov = run_basic_elo_with_mov(
        df.copy(),
        k=135.5024295855922,
        w_ko=1.304905948911245,
        w_sub=2.0,
        w_udec=0.8150291233970635,
        w_sdec=0.6277860597403592,
        w_mdec=1.0229206801627735
    )
    display_top_n_elos(df_mov, n=10)
    most_recent_elo(df_mov, n=10)
    
    fighter = input("\nEnter fighter name to view Elo history (or press Enter to skip): ").strip()
    if fighter:
        graph_fighter_elo_history(df_mov, fighter)

if __name__ == "__main__":
    main()
