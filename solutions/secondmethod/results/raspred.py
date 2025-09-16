import pandas as pd
df = pd.read_csv('submission.csv')
class_distribution = df['category'].value_counts()
print("Распределение классов по колонке 'category':")
print(class_distribution)