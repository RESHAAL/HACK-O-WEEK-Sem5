import pandas as pd

#dataframe
data={
"Name":["A","B","C"],
"Age":[20,21,22]
}

df=pd.DataFrame(data)
print(df)

#working on dataset
df=pd.read_csv("students.csv")
df["Age"]
df[["Name","Age"]]

#cleaning
df.isnull() #missing values 
df.drop_duplicates() #remove duplicates

#sorting
df.sort_values("Age",ascending=False)

#merging
pd.merge(student,marks,on="ID")

#groupBy
df.groupby("Gender")["Marks"].mean()

