import matplotlib.pyplot as plt

# Sample data for the horizontal bar chart
categories = ['GPT-4 w/ CoD', 'GPT-4 zero-shot', 'Fine-tuned ChatGPT']
values = [8.03, 6.54, 7.65]

plt.figure(figsize=(8, 4))  # Set the figure size
bars = plt.barh(categories, values, color=['skyblue', 'lightcoral', 'lightgreen'])   # Use plt.barh() for horizontal bar chart

# Add labels and title
plt.xlabel('Score (1-10)', loc='center')
plt.title('Automated evaluation of summaries', pad=20)

# Remove spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Save the chart as a PNG file
plt.savefig('score.png', dpi=300, bbox_inches='tight')

# Show the chart (optional)
plt.show()