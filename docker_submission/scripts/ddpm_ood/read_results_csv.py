# functions to read and visualize the results of the reconstrcution error saved as CSV 

# %%
import pandas as pd

result_dir = '/home/bme001/shared/mood/code/ddpm-ood/checkpoints/mood_ddpm_07_07/ood/results_brain_toy.csv'
df = pd.read_csv(result_dir)
df

# %%
df['subject'] = df['filename'].apply(lambda x: '_'.join(x.split('_')[:2]))
df


# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Create the first scatter plot: t vs mse with hue as subjects
plt.figure(figsize=(12, 6))
sns.stripplot(data=df, x='t', y='mse', hue='subject')
plt.xlabel('t')
plt.ylabel('MSE')
plt.title('Scatter Plot: t vs MSE')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Create the second scatter plot: t vs perceptual_difference with hue as subjects
plt.figure(figsize=(12, 6))
sns.stripplot(data=df, x='t', y='perceptual_difference', hue='subject')
plt.xlabel('t')
plt.ylabel('Perceptual Difference')
plt.title('Scatter Plot: t vs Perceptual Difference')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %%
import pandas as pd
import matplotlib.pyplot as plt

# Find the mean value for each subject at each t
df_mean = df.groupby(['subject', 't']).mean().reset_index()

# Plot the mean values
plt.figure(figsize=(12, 6))
for subject in df_mean['subject'].unique():
    df_subject = df_mean[df_mean['subject'] == subject]
    plt.plot(df_subject['t'], df_subject['mse'], label=subject)

# Add labels and title
plt.xlabel('t')
plt.ylabel('Mean MSE')
plt.title('Mean MSE for Each Subject at Each t')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Display the plot
plt.show()
# %%
