"""
Name   Age   Gender city
Alaa    20    Male  mansoura
Sara         Female Giza
Omar   25    Male   Cairo
"""
import pandas as pd
data = pd.read_excel("MTHS114-19951971.xlsx")

# Show first 5 rows
print(data.head())