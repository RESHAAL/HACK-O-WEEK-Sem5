import matplotlib.pyplot as plt
import seaborn as sns

#line plot
x=[1,2,3,4]
y=[2,5,3,8]
plt.plot(x,y)
plt.show()

#bar plot
names=["A","B","C"]
marks=[90,80,95]
plt.bar(names,marks)
plt.show()

#histogram
marks=[50,60,70,80,90,95]
plt.hist(marks)
plt.show()

#scatter plot
plt.scatter(x,y)
plt.show()

#pie chart 
sizes=[40,35,25]
labels=["Python","Java","C++"]
plt.pie(sizes,labels=labels,autopct="%1.1f%%")
plt.show()

#seaborn plot
sns.countplot(x="Gender",data=df)
plt.show()

#heatmap
sns.heatmap(df.corr(),annot=True)
plt.show()