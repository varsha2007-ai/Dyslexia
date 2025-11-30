# this is to cmpare  the real and sythetically generated files and check if there are no exactly same entries in both the files.

import pandas as pd
import numpy as np


def find_exact_matches(real_file_path, synthetic_file_path):
    """
    Compare real and synthetic eye-tracking data to find exact matches

    Parameters:
    real_file_path: str - path to real data CSV
    synthetic_file_path: str - path to synthetic data CSV

    Returns:
    dict with match information
    """

    # Load the data
    try:
        real_data = pd.read_csv(real_file_path)
        synthetic_data = pd.read_csv(synthetic_file_path)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return None

    # Columns to compare for exact matches
    cols_to_compare = ['fix_x', 'fix_y', 'duration_ms', 'start_ms', 'end_ms']

    # Ensure all columns exist in both datasets
    missing_cols_real = [col for col in cols_to_compare if col not in real_data.columns]
    missing_cols_synth = [col for col in cols_to_compare if col not in synthetic_data.columns]

    if missing_cols_real or missing_cols_synth:
        print(f"Missing columns in real data: {missing_cols_real}")
        print(f"Missing columns in synthetic data: {missing_cols_synth}")
        return None

    # Find exact matches
    matches = []
    for i, real_row in real_data[cols_to_compare].iterrows():
        for j, synth_row in synthetic_data[cols_to_compare].iterrows():
            if real_row.equals(synth_row):
                matches.append((i, j))

    # Create detailed results
    results = {
        'total_matches': len(matches),
        'match_pairs': matches,
        'real_data_rows': len(real_data),
        'synthetic_data_rows': len(synthetic_data),
        'match_percentage_real': (len(matches) / len(real_data)) * 100 if len(real_data) > 0 else 0,
        'match_percentage_synthetic': (len(matches) / len(synthetic_data)) * 100 if len(synthetic_data) > 0 else 0
    }

    return results


def find_approximate_matches(real_file_path, synthetic_file_path, tolerance=1e-6):
    """
    Find approximate matches with a small tolerance for floating point comparison

    Parameters:
    tolerance: float - tolerance for floating point comparison
    """

    # Load the data
    try:
        real_data = pd.read_csv(real_file_path)
        synthetic_data = pd.read_csv(synthetic_file_path)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return None

    cols_to_compare = ['fix_x', 'fix_y', 'duration_ms', 'start_ms', 'end_ms']

    matches = []
    for i, real_row in real_data[cols_to_compare].iterrows():
        for j, synth_row in synthetic_data[cols_to_compare].iterrows():
            # Check if all values are within tolerance
            if all(abs(real_row[col] - synth_row[col]) <= tolerance for col in cols_to_compare):
                matches.append((i, j))

    return {
        'total_approximate_matches': len(matches),
        'match_pairs': matches,
        'tolerance_used': tolerance
    }


# Example usage with your file paths
real_file = r"D:\PREDICTION OF DYSLEXIA\Backend\final-data\data\Subject_1350_T4_Meaningful_Text_fixations.csv"
synthetic_file = r"D:\PREDICTION OF DYSLEXIA\Backend\final-data\data\Subject_2350_T4_Meaningful_Text_fixations.csv"

# Find exact matches
print("=== EXACT MATCH ANALYSIS ===")
exact_results = find_exact_matches(real_file, synthetic_file)

if exact_results:
    print(f"Total exact matches found: {exact_results['total_matches']}")
    print(f"Real data has {exact_results['real_data_rows']} rows")
    print(f"Synthetic data has {exact_results['synthetic_data_rows']} rows")
    print(f"Match percentage (real): {exact_results['match_percentage_real']:.2f}%")
    print(f"Match percentage (synthetic): {exact_results['match_percentage_synthetic']:.2f}%")

    if exact_results['match_pairs']:
        print("\nExact matching row pairs (real_row, synthetic_row):")
        for real_idx, synth_idx in exact_results['match_pairs']:
            print(f"  Real row {real_idx} matches Synthetic row {synth_idx}")
    else:
        print("No exact matches found - this is expected for good synthetic data!")

# Find approximate matches (useful for floating point data)
print("\n=== APPROXIMATE MATCH ANALYSIS ===")
approx_results = find_approximate_matches(real_file, synthetic_file, tolerance=0.01)

if approx_results:
    print(f"Total approximate matches (tolerance=0.01): {approx_results['total_approximate_matches']}")

    if approx_results['match_pairs']:
        print("Approximate matching row pairs:")
        for real_idx, synth_idx in approx_results['match_pairs'][:10]:  # Show first 10
            print(f"  Real row {real_idx} ≈ Synthetic row {synth_idx}")


# Additional analysis: Check for identical values in individual columns
def column_wise_analysis(real_file_path, synthetic_file_path):
    """
    Analyze matches column by column
    """
    try:
        real_data = pd.read_csv(real_file_path)
        synthetic_data = pd.read_csv(synthetic_file_path)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    cols_to_compare = ['fix_x', 'fix_y', 'duration_ms', 'start_ms', 'end_ms']

    print("\n=== COLUMN-WISE IDENTICAL VALUES ===")
    for col in cols_to_compare:
        if col in real_data.columns and col in synthetic_data.columns:
            # Find identical values in each column
            real_values = set(real_data[col].values)
            synth_values = set(synthetic_data[col].values)
            common_values = real_values.intersection(synth_values)

            print(f"{col}:")
            print(f"  Unique values in real: {len(real_values)}")
            print(f"  Unique values in synthetic: {len(synth_values)}")
            print(f"  Common identical values: {len(common_values)}")
            print(f"  Overlap percentage: {(len(common_values) / len(real_values)) * 100:.1f}%")


# Run column-wise analysis
column_wise_analysis(real_file, synthetic_file)


# Quality check: If too many exact matches, synthetic data may not be diverse enough
def quality_assessment(exact_matches, total_real_rows):
    """
    Assess the quality of synthetic data based on match patterns
    """
    print("\n=== SYNTHETIC DATA QUALITY ASSESSMENT ===")

    if exact_matches == 0:
        print("✓ EXCELLENT: No exact matches found - synthetic data is properly diversified")
    elif exact_matches < total_real_rows * 0.05:  # Less than 5%
        print("✓ GOOD: Very few exact matches - acceptable level of variation")
    elif exact_matches < total_real_rows * 0.20:  # Less than 20%
        print("⚠ WARNING: Some exact matches found - consider increasing perturbation")
    else:
        print("✗ POOR: Too many exact matches - synthetic data lacks sufficient variation")

    return exact_matches / total_real_rows if total_real_rows > 0 else 0


# Run quality assessment
if exact_results:
    match_ratio = quality_assessment(exact_results['total_matches'], exact_results['real_data_rows'])
    print(f"Match ratio: {match_ratio:.4f}")