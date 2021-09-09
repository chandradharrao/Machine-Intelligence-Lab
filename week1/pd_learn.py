import pandas as pd

df = pd.read_csv("example.csv")
print(df.columns.values[-1])