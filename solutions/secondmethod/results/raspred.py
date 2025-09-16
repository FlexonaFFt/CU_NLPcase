import pandas as pd
df = pd.read_csv('smalltrainer_res/submission52.csv')
class_distribution = df['pred_label'].value_counts()
print("Распределение классов по колонке 'label':")
print(class_distribution)