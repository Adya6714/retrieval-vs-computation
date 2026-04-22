import pandas as pd
import os

results_files = ['results/contamination_triage.csv', 'results/behavioral_sweep.csv', 'results/RvC Mechanistic Sweep.csv']
for f in results_files:
    if os.path.exists(f):
        try:
            df = pd.read_csv(f)
            if 'notes' not in df.columns:
                df['notes'] = ''
            mask = df['problem_id'].astype(str).str.startswith('BW_E')
            if mask.sum() > 0:
                df.loc[mask, 'notes'] = df.loc[mask, 'notes'].astype(str) + ' [NEW_BATCH]'
                df.to_csv(f, index=False)
                print(f'Flagged {f}')
            else:
                print(f'No BW_E records in {f}')
        except Exception as e:
             print(f'Error reading {f}')
