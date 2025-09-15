import pandas as pd
df = pd.read_csv('train_labeled-2.csv')
class_distribution = df['label'].value_counts()
print("Распределение классов по колонке 'label':")
print(class_distribution)