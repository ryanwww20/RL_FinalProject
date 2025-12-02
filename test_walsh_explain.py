"""
Explain how generate_walsh_matrix(n) works step by step.
"""

import numpy as np
from scipy.linalg import hadamard


def generate_walsh_matrix_explained(n):
    """
    Generate a Walsh matrix of size n x n.
    
    Walsh matrix = Hadamard matrix with rows reordered by "sequency"
    (sequency = number of sign changes in a row)
    """
    
    print(f"Step 1: Generate Hadamard matrix of size {n}x{n}")
    print("=" * 50)
    
    # Hadamard matrix: 只有 +1 和 -1，每行/每列互相正交
    H = hadamard(n)
    
    print("Hadamard matrix H (before reordering):")
    print(H)
    print(f"\nProperties:")
    print(f"  - All elements are +1 or -1")
    print(f"  - Rows are orthogonal: H @ H.T = {n} * I")
    
    # 驗證正交性
    orthogonal_check = H @ H.T
    print(f"\nH @ H.T = ")
    print(orthogonal_check)
    
    print(f"\n\nStep 2: Count 'sequency' for each row")
    print("=" * 50)
    print("Sequency = number of sign changes in a row")
    print("(Similar to 'frequency' in Fourier transform)\n")
    
    def count_sign_changes(row):
        # 計算相鄰元素符號變化的次數
        return np.sum(np.abs(np.diff(row)) > 0)
    
    print("Row | Pattern          | Sign changes (sequency)")
    print("-" * 50)
    
    sequency = []
    for i in range(n):
        row = H[i]
        seq = count_sign_changes(row)
        sequency.append(seq)
        
        # 視覺化 pattern
        pattern = "".join(['+' if x > 0 else '-' for x in row])
        print(f"  {i} | {pattern:16s} | {seq}")
    
    print(f"\nStep 3: Sort rows by sequency (low to high)")
    print("=" * 50)
    
    sorted_indices = np.argsort(sequency)
    print(f"Sorted indices: {sorted_indices}")
    print("(Row 0 has lowest sequency, row {n-1} has highest)")
    
    # Reorder rows
    walsh = H[sorted_indices]
    
    print(f"\n\nStep 4: Create Walsh matrix (reordered Hadamard)")
    print("=" * 50)
    print("Walsh matrix W (after reordering by sequency):")
    
    print("\nRow | Pattern          | Sequency | Visual")
    print("-" * 60)
    
    for i in range(n):
        row = walsh[i]
        seq = count_sign_changes(row)
        pattern = "".join(['+' if x > 0 else '-' for x in row])
        visual = "".join(['█' if x > 0 else '░' for x in row])
        print(f"  {i} | {pattern:16s} | {seq:8d} | {visual}")
    
    print(f"\n\nStep 5: Normalize (optional, for orthonormal matrix)")
    print("=" * 50)
    
    walsh_normalized = walsh / np.sqrt(n)
    
    print(f"Divide by sqrt({n}) = {np.sqrt(n):.4f}")
    print(f"Now: W @ W.T = I (identity matrix)")
    
    # 驗證
    identity_check = walsh_normalized @ walsh_normalized.T
    print(f"\nW_normalized @ W_normalized.T =")
    print(np.round(identity_check, 4))
    
    return walsh_normalized


def visualize_walsh_as_frequencies():
    """Show Walsh basis as spatial frequencies."""
    
    print("\n" + "=" * 60)
    print("Walsh Basis = Spatial Frequencies (like Fourier, but binary)")
    print("=" * 60)
    
    n = 16
    walsh = generate_simple_walsh(n)
    
    print(f"\nWalsh matrix {n}x{n} - Each row is a 'frequency':")
    print("-" * 60)
    print("Row | Sequency | Pattern (█=+1, ░=-1)")
    print("-" * 60)
    
    for i in range(n):
        row = walsh[i]
        seq = np.sum(np.abs(np.diff(row)) > 0)
        visual = "".join(['█' if x > 0 else '░' for x in row])
        
        if seq == 0:
            freq_name = "DC (constant)"
        elif seq <= 3:
            freq_name = "Low freq"
        elif seq <= 7:
            freq_name = "Mid freq"
        else:
            freq_name = "High freq"
        
        print(f" {i:2d} | {seq:8d} | {visual}  ({freq_name})")
    
    print("""
Interpretation:
- Sequency 0 (Row 0): DC component - all same sign (█████████████████)
- Low sequency: Slow changes - represents large-scale structure
- High sequency: Fast changes - represents fine details

This is similar to Fourier basis, but:
- Fourier: sin/cos waves (continuous)
- Walsh: square waves (binary +1/-1)

For your PIC design:
- Low sequency basis → Large Si/SiO2 blocks
- High sequency basis → Fine-grained patterns
""")


def generate_simple_walsh(n):
    """Simple Walsh matrix generation."""
    H = hadamard(n)
    
    def count_sign_changes(row):
        return np.sum(np.abs(np.diff(row)) > 0)
    
    sequency = [count_sign_changes(H[i]) for i in range(n)]
    sorted_indices = np.argsort(sequency)
    
    return H[sorted_indices]


def show_basis_selection_meaning():
    """Explain what basis selection means."""
    
    print("\n" + "=" * 60)
    print("What does 'selecting basis [0,4,8,12]' mean?")
    print("=" * 60)
    
    n = 16
    walsh = generate_simple_walsh(n)
    
    print("""
Full Walsh matrix has 16 rows (basis vectors).
Your action has 4 dimensions.

When you select basis [0, 4, 8, 12]:
""")
    
    selected = [0, 4, 8, 12]
    print("Selected basis vectors:")
    print("-" * 50)
    
    for idx in selected:
        row = walsh[idx]
        seq = np.sum(np.abs(np.diff(row)) > 0)
        visual = "".join(['█' if x > 0 else '░' for x in row])
        print(f"  Basis {idx:2d} (seq={seq:2d}): {visual}")
    
    print("""
Your 4-dim action controls the 'weight' of each basis:
  action = [a0, a1, a2, a3]
  
  output = a0 * basis_0 + a1 * basis_4 + a2 * basis_8 + a3 * basis_12

Example:
  action = [1, 0, 0, 0]  → Uses only basis_0 (DC)       → All same
  action = [0, 1, 0, 0]  → Uses only basis_4 (mid freq) → Some pattern
  action = [0.5, 0.5, 0, 0] → Mix of DC and mid freq   → Blended pattern
""")
    
    # Demonstrate
    print("\nDemonstration:")
    print("-" * 50)
    
    W_selected = walsh[selected, :].T  # (16, 4)
    
    test_actions = [
        ([1, 0, 0, 0], "Only DC"),
        ([0, 1, 0, 0], "Only basis_4"),
        ([0, 0, 1, 0], "Only basis_8"),
        ([1, 1, 0, 0], "DC + basis_4"),
        ([1, 0.5, 0.3, 0.1], "Weighted mix"),
    ]
    
    for action, desc in test_actions:
        action = np.array(action)
        output = W_selected @ action
        binary = (output > 0).astype(int)
        visual = "".join(['█' if x == 1 else '░' for x in binary])
        print(f"  {str(action):20s} ({desc:15s}) → {visual}")


if __name__ == "__main__":
    # Explain with small example (n=8)
    print("=" * 60)
    print("WALSH MATRIX EXPLAINED (using n=8 for clarity)")
    print("=" * 60)
    print()
    
    walsh = generate_walsh_matrix_explained(8)
    
    # Show as frequencies
    visualize_walsh_as_frequencies()
    
    # Explain basis selection
    show_basis_selection_meaning()

