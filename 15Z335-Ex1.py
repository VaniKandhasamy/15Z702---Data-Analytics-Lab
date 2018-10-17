import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('EmployeeAttrition.csv')
print(df.shape)
df.head(5)
df.describe()
print(len(df))
df.isnull().sum()
df.boxplot('WorkLifeBalance')
plt.scatter(df.TotalWorkingYears,df.JobLevel,s=df.Age)

