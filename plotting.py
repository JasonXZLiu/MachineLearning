# Import necessary libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Load iris data
iris = sns.load_dataset("iris")

# Construct iris plot
sns.swarmplot(x="sepal_length", y="petal_length", data=iris)

# Show plot
plt.show()