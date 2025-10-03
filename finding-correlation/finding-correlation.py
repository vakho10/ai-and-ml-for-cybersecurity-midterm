import matplotlib.pyplot as plt
import numpy as np


# Main function
def main():
    # Step 1: Data Extraction
    # The data points were extracted from the JavaScript code in the HTML source
    x_data = np.array([-5, -5, -3, -1, 1, 3, 5, 7])
    y_data = np.array([-5, 2, -1, 1, -2, 1, -3, -2])

    # Step 2: Calculate Pearson's Correlation Coefficient
    # np.corrcoef returns a 2x2 matrix. The correlation coefficient is at [0, 1] or [1, 0].
    correlation_matrix = np.corrcoef(x_data, y_data)
    pearson_r = correlation_matrix[0, 1]

    print(f"Extracted X data: {x_data}")
    print(f"Extracted Y data: {y_data}")
    print("-" * 30)
    print(f"Calculated Pearson's Correlation Coefficient (r): {pearson_r:.4f}")

    # Step 3: Create the Visualization
    plt.style.use('seaborn-v0_8-whitegrid')  # Use a nice style for the plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create the scatter plot
    ax.scatter(x_data, y_data, color='blue', label='Data Points', s=100, alpha=0.7)

    # Calculate the line of best fit (y = mx + b)
    m, b = np.polyfit(x_data, y_data, 1)
    ax.plot(x_data, m * x_data + b, color='red', linestyle='--',
            label=f'Line of Best Fit (y = {m:.2f}x + {b:.2f})')

    # Add titles and labels for clarity
    ax.set_title('Scatter Plot of X vs. Y with Line of Best Fit', fontsize=16)
    ax.set_xlabel('X Variable', fontsize=12)
    ax.set_ylabel('Y Variable', fontsize=12)

    # Add a legend to identify the plot elements
    ax.legend()

    # Add grid for better readability
    ax.grid(True)

    # Display the plot
    plt.tight_layout()
    plt.savefig('correlation_plot.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
