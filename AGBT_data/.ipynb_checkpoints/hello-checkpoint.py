import pandas as pd
file_path = "GenerationByFuelType-2016-01-01T00_00_00.000Z-2026-03-10T12_30_00.000Z.csv"
df = pd.read_csv(file_path)
df.head()
