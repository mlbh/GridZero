#MAKING CORRELATION HEATMAP
from load_and_merge import fully_merged_data
import seaborn as sns
import matplotlib.pyplot as plt
ordered_data = fully_merged_data.sort_values('time')
ordered_data.describe()
correlation_matrix = fully_merged_data.corr()
# Plot the heatmap
plt.figure(figsize=(15, 15))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of the Dataset')
plt.show()
