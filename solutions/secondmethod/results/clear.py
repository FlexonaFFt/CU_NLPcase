import pandas as pd
df = pd.read_csv('smalltrainer_res/submission52.csv')
df_new = df[['pred_label']].rename(columns={'pred_label': 'category'})
df_new.to_csv('submission.csv', index=False)