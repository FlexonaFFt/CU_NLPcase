import pandas as pd
import matplotlib.pyplot as plt
import os

file1 = 'submission.csv'
file2 = 'submission_n.csv'
def get_label_column(df):
    for col in ['category', 'label', 'pred_label']:
        if col in df.columns:
            return col
    raise ValueError('Нет подходящей колонки с классами!')


df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
col1 = get_label_column(df1)
col2 = get_label_column(df2)
dist1 = df1[col1].value_counts()
dist2 = df2[col2].value_counts()

all_classes = sorted(set(dist1.index).union(dist2.index))
result_df = pd.DataFrame({
    os.path.basename(file1): [dist1.get(cls, 0) for cls in all_classes],
    os.path.basename(file2): [dist2.get(cls, 0) for cls in all_classes]
}, index=all_classes)

colors = ['#f4acb7', '#fcbf49']  
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.4
y = range(len(all_classes))

ax.barh(
    [i - bar_width/2 for i in y],
    result_df.iloc[:, 0],
    height=bar_width,
    color=colors[0],
    label=result_df.columns[0]
)
ax.barh(
    [i + bar_width/2 for i in y],
    result_df.iloc[:, 1],
    height=bar_width,
    color=colors[1],
    label=result_df.columns[1]
)

ax.set_yticks(y)
ax.set_yticklabels(all_classes)
ax.set_xlabel('Количество')
ax.set_ylabel('Класс')
ax.set_title('Сравнение распределения классов по двум файлам')
ax.legend()
plt.tight_layout()
plt.show()
