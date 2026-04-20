from pandas import pandas as pd

DATA_PATH = "Fake_Real_Data.csv"
df = pd.read_csv(DATA_PATH)

# pd.read_csv(DATA_PATH).head()
df.columns = df.columns.str.lower()


df = df[["text", "label"]]
