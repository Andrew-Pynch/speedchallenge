import pandas as pd

from SpeedDataset import SpeedDataset

dataset = SpeedDataset("vidCaps/")

df = pd.DataFrame(columns=["image", "label"])
df.head()

for i, data in enumerate(dataset):
    df.loc[i] = data["fname"], data["label"]

df.to_csv("labels.csv", index=False)
