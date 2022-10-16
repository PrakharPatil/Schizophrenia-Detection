import os
import pandas as pd
demographic = pd.read_csv("Data\demographic.csv")
for i, t in enumerate(list(demographic[" group"])):
    if t:
        print(f"{i} - Schizophrenia")
    else:
        print(f"{i} - HEALTHY")