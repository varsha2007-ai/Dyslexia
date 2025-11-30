# Validation with corrected data
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load original data
fixations = pd.read_csv(r"D:\PREDICTION OF DYSLEXIA\Backend\final-data\data\Subject_1350_T4_Meaningful_Text_fixations.csv")

# Load corrected synthetic data for comparison
synthetic_data_corrected = pd.read_csv(r"D:\PREDICTION OF DYSLEXIA\Backend\final-data\data\Subject_2350_T4_Meaningful_Text_fixations.csv")

features_to_compare = ['fix_x', 'fix_y', 'duration_ms', 'start_ms', 'end_ms']

# Remove any NaNs
real_data_clean = fixations[features_to_compare].dropna()
synthetic_data_clean = synthetic_data_corrected[features_to_compare].dropna()

# Ensure equal lengths
min_len = min(len(real_data_clean), len(synthetic_data_clean))
real_data_clean = real_data_clean.iloc[:min_len]
synthetic_data_clean = synthetic_data_clean.iloc[:min_len]

print("\n" + "=" * 60)
print("CORRECTED VALIDATION RESULTS (fix_y unchanged)")
print("=" * 60)

# KS-Test
print("\nKS-Test p-values (>0.05 indicates similar distributions):")
for feature in features_to_compare:
    p_value = ks_2samp(real_data_clean[feature], synthetic_data_clean[feature]).pvalue
    status = "✓ PASS" if p_value > 0.05 else "✗ FAIL"
    print(f"  {feature}: {p_value:.4f} {status}")

# MAE
print("\nMean Absolute Error (lower is better):")
for feature in features_to_compare:
    mae = mean_absolute_error(real_data_clean[feature], synthetic_data_clean[feature])
    print(f"  {feature}: {mae:.2f}")

# Correlation
print("\nPearson Correlation (closer to 1 is better):")
for feature in features_to_compare:
    corr = np.corrcoef(real_data_clean[feature], synthetic_data_clean[feature])[0, 1]
    print(f"  {feature}: {corr:.4f}")

# Special validation for fix_y (should be identical)
print("\n" + "=" * 60)
print("SPECIAL VALIDATION FOR fix_y (should be identical)")
print("=" * 60)

fix_y_identical = np.array_equal(real_data_clean['fix_y'].values, synthetic_data_clean['fix_y'].values)
print(f"fix_y values identical: {fix_y_identical}")
if fix_y_identical:
    print("✓ SUCCESS: fix_y values are perfectly preserved!")
else:
    print("✗ ERROR: fix_y values have been modified!")

# Check number of identical fix_y values
identical_count = np.sum(real_data_clean['fix_y'].values == synthetic_data_clean['fix_y'].values)
total_count = len(real_data_clean)
print(f"Identical fix_y values: {identical_count}/{total_count} ({identical_count / total_count * 100:.1f}%)")

# Statistical Summary Comparison
print("\n" + "=" * 60)
print("STATISTICAL SUMMARY COMPARISON")
print("=" * 60)

for feature in features_to_compare:
    real_mean, real_std = real_data_clean[feature].mean(), real_data_clean[feature].std()
    synth_mean, synth_std = synthetic_data_clean[feature].mean(), synthetic_data_clean[feature].std()

    mean_diff = abs(real_mean - synth_mean) / real_mean * 100 if real_mean != 0 else 0
    std_diff = abs(real_std - synth_std) / real_std * 100 if real_std != 0 else 0

    print(f"\n{feature}:")
    print(f"  Real:      μ={real_mean:.2f}, σ={real_std:.2f}")
    print(f"  Synthetic: μ={synth_mean:.2f}, σ={synth_std:.2f}")
    print(f"  Difference: μ={mean_diff:.1f}%, σ={std_diff:.1f}%")

# Improved visualization
print("\nGenerating corrected distribution plots...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, feature in enumerate(features_to_compare):
    ax = axes[i]

    # Plot distributions
    sns.histplot(real_data_clean[feature], alpha=0.6, label='Real', color='blue',
                 kde=True, stat='density', ax=ax)
    sns.histplot(synthetic_data_clean[feature], alpha=0.6, label='Synthetic', color='orange',
                 kde=True, stat='density', ax=ax)

    ax.set_title(f'Distribution of {feature}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Special annotation for fix_y
    if feature == 'fix_y':
        ax.text(0.05, 0.95, 'fix_y unchanged\n(identical values)',
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
                verticalalignment='top')

plt.tight_layout()
plt.show()

# Time series comparison for temporal features
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot first 50 fixations for temporal comparison
n_plot = min(50, len(real_data_clean))

axes[0].plot(real_data_clean['start_ms'].iloc[:n_plot], 'b-', label='Real', alpha=0.7)
axes[0].plot(synthetic_data_clean['start_ms'].iloc[:n_plot], 'r--', label='Synthetic', alpha=0.7)
axes[0].set_title('Start Times Comparison')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(real_data_clean['duration_ms'].iloc[:n_plot], 'b-', label='Real', alpha=0.7)
axes[1].plot(synthetic_data_clean['duration_ms'].iloc[:n_plot], 'r--', label='Synthetic', alpha=0.7)
axes[1].set_title('Duration Comparison')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Scatter plot for spatial comparison - highlighting fix_y preservation
axes[2].scatter(real_data_clean['fix_x'].iloc[:n_plot], real_data_clean['fix_y'].iloc[:n_plot],
                alpha=0.6, label='Real', color='blue', s=30)
axes[2].scatter(synthetic_data_clean['fix_x'].iloc[:n_plot], synthetic_data_clean['fix_y'].iloc[:n_plot],
                alpha=0.6, label='Synthetic', color='red', s=30, marker='x')
axes[2].set_title('Spatial Fixation Pattern\n(fix_y unchanged)')
axes[2].set_xlabel('fix_x (perturbed)')
axes[2].set_ylabel('fix_y (unchanged)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Additional validation: Show fix_x perturbation statistics
print("\n" + "=" * 60)
print("fix_x PERTURBATION ANALYSIS")
print("=" * 60)

fix_x_diff = synthetic_data_clean['fix_x'].values - real_data_clean['fix_x'].values
print(f"fix_x perturbation statistics:")
print(f"  Mean perturbation: {np.mean(fix_x_diff):.3f} pixels")
print(f"  Std perturbation: {np.std(fix_x_diff):.3f} pixels")
print(f"  Max perturbation: {np.max(np.abs(fix_x_diff)):.3f} pixels")
print(f"  Min perturbation: {np.min(np.abs(fix_x_diff)):.3f} pixels")

# Add heatmap visualization at the end of the existing code
print("\nGenerating heatmap visualization for fixation distributions...")

# Create heatmaps for fixation distributions
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Real data heatmap
sns.kdeplot(x=real_data_clean['fix_x'], y=real_data_clean['fix_y'],
            cmap='Blues', fill=True, bw_adjust=0.5, ax=axes[0])
axes[0].set_title('Fixation Distribution - Real Data', fontsize=14, fontweight='bold')
axes[0].set_xlabel('fix_x (pixels)')
axes[0].set_ylabel('fix_y (pixels)')
axes[0].grid(True, alpha=0.3)

# Synthetic data heatmap
sns.kdeplot(x=synthetic_data_clean['fix_x'], y=synthetic_data_clean['fix_y'],
            cmap='Oranges', fill=True, bw_adjust=0.5, ax=axes[1])
axes[1].set_title('Fixation Distribution - Synthetic Data', fontsize=14, fontweight='bold')
axes[1].set_xlabel('fix_x (pixels)')
axes[1].set_ylabel('fix_y (pixels)')
axes[1].grid(True, alpha=0.3)

# Overlay comparison
sns.kdeplot(x=real_data_clean['fix_x'], y=real_data_clean['fix_y'],
            color='blue', alpha=0.6, label='Real', ax=axes[2])
sns.kdeplot(x=synthetic_data_clean['fix_x'], y=synthetic_data_clean['fix_y'],
            color='orange', alpha=0.6, label='Synthetic', ax=axes[2])
axes[2].set_title('Fixation Distribution - Overlay Comparison', fontsize=14, fontweight='bold')
axes[2].set_xlabel('fix_x (pixels)')
axes[2].set_ylabel('fix_y (pixels)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Additional 2D histogram comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Real data 2D histogram
h1 = axes[0].hist2d(real_data_clean['fix_x'], real_data_clean['fix_y'],
                    bins=30, cmap='Blues', alpha=0.8)
axes[0].set_title('2D Histogram - Real Data', fontsize=14, fontweight='bold')
axes[0].set_xlabel('fix_x (pixels)')
axes[0].set_ylabel('fix_y (pixels)')
plt.colorbar(h1[3], ax=axes[0], label='Frequency')

# Synthetic data 2D histogram
h2 = axes[1].hist2d(synthetic_data_clean['fix_x'], synthetic_data_clean['fix_y'],
                    bins=30, cmap='Oranges', alpha=0.8)
axes[1].set_title('2D Histogram - Synthetic Data', fontsize=14, fontweight='bold')
axes[1].set_xlabel('fix_x (pixels)')
axes[1].set_ylabel('fix_y (pixels)')
plt.colorbar(h2[3], ax=axes[1], label='Frequency')

plt.tight_layout()
plt.show()

# Quantitative comparison of spatial distributions
print("\n" + "=" * 60)
print("SPATIAL DISTRIBUTION ANALYSIS")
print("=" * 60)

# Calculate spatial statistics
real_x_range = real_data_clean['fix_x'].max() - real_data_clean['fix_x'].min()
real_y_range = real_data_clean['fix_y'].max() - real_data_clean['fix_y'].min()
synth_x_range = synthetic_data_clean['fix_x'].max() - synthetic_data_clean['fix_x'].min()
synth_y_range = synthetic_data_clean['fix_y'].max() - synthetic_data_clean['fix_y'].min()

print(f"Spatial coverage comparison:")
print(f"  Real data - X range: {real_x_range:.1f} pixels, Y range: {real_y_range:.1f} pixels")
print(f"  Synthetic data - X range: {synth_x_range:.1f} pixels, Y range: {synth_y_range:.1f} pixels")
print(f"  X range difference: {abs(real_x_range - synth_x_range):.1f} pixels")
print(f"  Y range difference: {abs(real_y_range - synth_y_range):.1f} pixels")

# Calculate center of mass for each distribution
real_centroid_x = real_data_clean['fix_x'].mean()
real_centroid_y = real_data_clean['fix_y'].mean()
synth_centroid_x = synthetic_data_clean['fix_x'].mean()
synth_centroid_y = synthetic_data_clean['fix_y'].mean()

print(f"\nCentroid comparison:")
print(f"  Real data centroid: ({real_centroid_x:.1f}, {real_centroid_y:.1f})")
print(f"  Synthetic data centroid: ({synth_centroid_x:.1f}, {synth_centroid_y:.1f})")
print(f"  Centroid shift: ({synth_centroid_x - real_centroid_x:.1f}, {synth_centroid_y - real_centroid_y:.1f})")

print("\nHeatmap visualization complete!")
