import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')


def generate_synthetic_fixations_corrected(original_fixations, subject_id, perturbation_factor=0.05):
    """
    Generate synthetic fixations with proper temporal alignment, fix_y unchanged
    """
    synthetic_fix = original_fixations.copy()

    # Perturb only fix_x positions (maintaining reading pattern)
    spatial_std_x = np.std(original_fixations['fix_x'].values) * perturbation_factor
    position_noise_x = np.random.normal(0, spatial_std_x, len(synthetic_fix))

    synthetic_fix['fix_x'] += position_noise_x
    # fix_y remains unchanged - no perturbation applied

    # Perturb durations more conservatively
    duration_multiplier = 1 + np.random.normal(0, 0.05, len(synthetic_fix))  # Reduced to 5%
    duration_multiplier = np.clip(duration_multiplier, 0.8, 1.2)  # Clamp to reasonable range
    synthetic_fix['duration_ms'] *= duration_multiplier

    # **CRITICAL FIX**: Maintain original temporal structure
    # Instead of regenerating timestamps, adjust them proportionally
    original_total_duration = original_fixations['end_ms'].iloc[-1] - original_fixations['start_ms'].iloc[0]

    # Recalculate timestamps while preserving sequence timing
    cumulative_time = original_fixations['start_ms'].iloc[0]  # Start from original first timestamp

    for i in range(len(synthetic_fix)):
        synthetic_fix.iloc[i, synthetic_fix.columns.get_loc('start_ms')] = cumulative_time
        synthetic_fix.iloc[i, synthetic_fix.columns.get_loc('end_ms')] = cumulative_time + synthetic_fix.iloc[i][
            'duration_ms']

        # Add inter-fixation interval (saccade time) based on original pattern
        if i < len(synthetic_fix) - 1:
            original_gap = original_fixations['start_ms'].iloc[i + 1] - original_fixations['end_ms'].iloc[i]
            # Add small perturbation to gap but keep it realistic
            perturbed_gap = original_gap * (1 + np.random.normal(0, 0.03))
            cumulative_time = synthetic_fix.iloc[i]['end_ms'] + max(perturbed_gap, 10)  # Minimum 10ms gap

    # Update subject ID
    synthetic_fix['sid'] = subject_id

    # Perturb displacement values more conservatively
    if 'disp_x' in synthetic_fix.columns:
        synthetic_fix['disp_x'] += np.random.normal(0, 0.05, len(synthetic_fix))
    if 'disp_y' in synthetic_fix.columns:
        # Keep disp_y unchanged to align with fix_y unchanged
        pass

    return synthetic_fix


# Load original data
fixations = pd.read_csv(r"D:\PREDICTION OF DYSLEXIA\DATASETS\13332134\data\data\Subject_1996_T4_Meaningful_Text_fixations.csv")

# Generate corrected synthetic data
print("Generating corrected synthetic data with fix_y unchanged...")
synthetic_fix_2003 = generate_synthetic_fixations_corrected(fixations, 2003, perturbation_factor=0.03)
synthetic_fix_3996 = generate_synthetic_fixations_corrected(fixations, 3996, perturbation_factor=0.05)

# Save corrected data
synthetic_fix_2003.to_csv('Subject_2003_T4_Meaningful_Text_fixations.csv', index=False)
synthetic_fix_3996.to_csv('Subject_3996_T4_Meaningful_Text_fixations.csv', index=False)

print("Corrected synthetic data saved!")