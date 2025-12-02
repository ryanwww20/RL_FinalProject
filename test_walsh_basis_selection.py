"""
Test different Walsh basis selection strategies.

Question: If action is 4-dim, which 4 basis vectors should we use?
- Option A: First 4 basis (0, 1, 2, 3) - lowest frequencies
- Option B: Every 4th basis (0, 4, 8, 12) - spread across frequencies

Transform: (1, 4) @ (4, 16) = (1, 16)
"""

import numpy as np
from scipy.linalg import hadamard
import matplotlib.pyplot as plt


def generate_walsh_matrix(n):
    """Generate Walsh matrix of size n x n (n must be power of 2)."""
    H = hadamard(n)
    
    def count_sign_changes(row):
        return np.sum(np.abs(np.diff(row)) > 0)
    
    sequency = [count_sign_changes(H[i]) for i in range(n)]
    sorted_indices = np.argsort(sequency)
    walsh = H[sorted_indices]
    
    return walsh / np.sqrt(n)


def get_transform_matrix_select_basis(basis_indices, high_dim=16):
    """
    Get transform matrix using only selected basis vectors.
    
    Args:
        basis_indices: List of basis indices to use (e.g., [0, 4, 8, 12])
        high_dim: Output dimension (16)
    
    Returns:
        Transform matrix of shape (high_dim, len(basis_indices))
    """
    walsh_full = generate_walsh_matrix(high_dim)
    
    # Select only the specified basis vectors (columns in transpose view)
    # Walsh matrix rows are the basis vectors
    # We want: output = sum of (action[i] * basis[i])
    # So transform matrix columns should be the selected basis vectors
    transform = walsh_full[basis_indices, :].T  # Shape: (high_dim, num_basis)
    
    return transform


def test_basis_selection():
    """Compare different basis selection strategies."""
    
    print("=" * 70)
    print("Walsh Basis Selection Test")
    print("=" * 70)
    
    high_dim = 16
    low_dim = 4
    
    # Full Walsh matrix for reference
    W_full = generate_walsh_matrix(high_dim)
    
    print("\nFull Walsh Matrix - All 16 Basis Vectors:")
    print("-" * 70)
    for i in range(high_dim):
        row = W_full[i]
        binary = (row > 0).astype(int)
        row_str = "".join(["#" if x == 1 else "." for x in binary])
        sequency = np.sum(np.abs(np.diff(row)) > 0)
        print(f"  Basis {i:2d}: {row_str}  (sequency: {sequency:2d})")
    
    # Strategy A: First 4 basis (lowest frequencies)
    print("\n" + "=" * 70)
    print("Strategy A: First 4 basis [0, 1, 2, 3] - Lowest frequencies")
    print("=" * 70)
    
    basis_A = [0, 1, 2, 3]
    W_A = get_transform_matrix_select_basis(basis_A, high_dim)
    print(f"Transform matrix shape: {W_A.shape}  (16 x 4)")
    
    print("\nSelected basis vectors:")
    for i, idx in enumerate(basis_A):
        row = W_full[idx]
        binary = (row > 0).astype(int)
        row_str = "".join(["#" if x == 1 else "." for x in binary])
        print(f"  Basis {idx}: {row_str}")
    
    # Test with random action
    np.random.seed(42)
    action = np.random.uniform(-1, 1, low_dim)
    print(f"\nTest action: {action}")
    
    continuous = W_A @ action
    binary = (continuous > 0).astype(int)
    print(f"Output (continuous): {continuous}")
    print(f"Output (binary):     {binary}")
    print(f"Output as string:    {''.join(['#' if x == 1 else '.' for x in binary])}")
    
    # Strategy B: Every 4th basis [0, 4, 8, 12]
    print("\n" + "=" * 70)
    print("Strategy B: Every 4th basis [0, 4, 8, 12] - Spread frequencies")
    print("=" * 70)
    
    basis_B = [0, 4, 8, 12]
    W_B = get_transform_matrix_select_basis(basis_B, high_dim)
    print(f"Transform matrix shape: {W_B.shape}  (16 x 4)")
    
    print("\nSelected basis vectors:")
    for i, idx in enumerate(basis_B):
        row = W_full[idx]
        binary = (row > 0).astype(int)
        row_str = "".join(["#" if x == 1 else "." for x in binary])
        sequency = np.sum(np.abs(np.diff(row)) > 0)
        print(f"  Basis {idx}: {row_str}  (sequency: {sequency})")
    
    continuous = W_B @ action
    binary = (continuous > 0).astype(int)
    print(f"\nTest action: {action}")
    print(f"Output (continuous): {continuous}")
    print(f"Output (binary):     {binary}")
    print(f"Output as string:    {''.join(['#' if x == 1 else '.' for x in binary])}")
    
    # Strategy C: Power of 2 basis [0, 1, 2, 4] (your suggestion, similar concept)
    print("\n" + "=" * 70)
    print("Strategy C: Mixed [0, 1, 4, 8] - DC + some frequencies")
    print("=" * 70)
    
    basis_C = [0, 1, 4, 8]
    W_C = get_transform_matrix_select_basis(basis_C, high_dim)
    
    print("\nSelected basis vectors:")
    for idx in basis_C:
        row = W_full[idx]
        binary = (row > 0).astype(int)
        row_str = "".join(["#" if x == 1 else "." for x in binary])
        print(f"  Basis {idx}: {row_str}")
    
    continuous = W_C @ action
    binary = (continuous > 0).astype(int)
    print(f"\nOutput as string:    {''.join(['#' if x == 1 else '.' for x in binary])}")


def test_pattern_generation():
    """Generate patterns with different strategies."""
    
    print("\n" + "=" * 70)
    print("Pattern Generation Comparison (16x16 design)")
    print("=" * 70)
    
    high_dim = 16
    low_dim = 4
    
    strategies = {
        'A: First 4 [0,1,2,3]': [0, 1, 2, 3],
        'B: Every 4th [0,4,8,12]': [0, 4, 8, 12],
        'C: Mixed [0,1,4,8]': [0, 1, 4, 8],
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    
    np.random.seed(789)
    
    for ax, (name, basis_indices) in zip(axes, strategies.items()):
        W = get_transform_matrix_select_basis(basis_indices, high_dim)
        
        # Generate 16x16 design (one row per step)
        design = np.zeros((high_dim, high_dim), dtype=int)
        
        for row in range(high_dim):
            action = np.random.uniform(-1, 1, low_dim)
            continuous = W @ action
            binary = (continuous > 0).astype(int)
            design[row] = binary
        
        ax.imshow(design, cmap='binary', interpolation='nearest')
        ax.set_title(f'{name}\nSi ratio: {np.mean(design):.1%}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y (row)')
        
        # Print design
        print(f"\n{name}:")
        for row in design:
            print("  " + "".join(["#" if x == 1 else "." for x in row]))
    
    plt.tight_layout()
    plt.savefig('/Users/ryan/NTUEE_Local/114-1/RL_FinalPJ/RL_FinalProject/sample_img/basis_comparison.png', 
                dpi=150, bbox_inches='tight')
    print("\nSaved comparison image to sample_img/basis_comparison.png")
    plt.close()


def test_controllability():
    """Test how controllable each strategy is."""
    
    print("\n" + "=" * 70)
    print("Controllability Test: What patterns can each strategy produce?")
    print("=" * 70)
    
    high_dim = 16
    low_dim = 4
    
    strategies = {
        'A [0,1,2,3]': [0, 1, 2, 3],
        'B [0,4,8,12]': [0, 4, 8, 12],
    }
    
    for name, basis_indices in strategies.items():
        print(f"\n--- Strategy {name} ---")
        W = get_transform_matrix_select_basis(basis_indices, high_dim)
        
        # Test specific actions
        test_actions = {
            '[1,0,0,0]': np.array([1, 0, 0, 0]),
            '[0,1,0,0]': np.array([0, 1, 0, 0]),
            '[0,0,1,0]': np.array([0, 0, 1, 0]),
            '[0,0,0,1]': np.array([0, 0, 0, 1]),
            '[1,1,1,1]': np.array([1, 1, 1, 1]),
            '[-1,-1,-1,-1]': np.array([-1, -1, -1, -1]),
        }
        
        for action_name, action in test_actions.items():
            continuous = W @ action
            binary = (continuous > 0).astype(int)
            row_str = "".join(["#" if x == 1 else "." for x in binary])
            print(f"  {action_name:15s} -> {row_str}  (ones: {np.sum(binary):2d})")


def demonstrate_math():
    """Demonstrate the matrix multiplication."""
    
    print("\n" + "=" * 70)
    print("Matrix Multiplication Demonstration")
    print("=" * 70)
    
    print("""
Your understanding is correct!

If action is 4-dimensional:
  action shape: (1, 4) or (4,)
  
Select 4 basis vectors from 16x16 Walsh matrix:
  W_selected shape: (4, 16) - each row is one basis vector
  
Transform:
  W_transform = W_selected.T  -> shape (16, 4)
  
  output = W_transform @ action
         = (16, 4) @ (4,) 
         = (16,)  ✓

The key question is: WHICH 4 basis to select?
""")
    
    high_dim = 16
    W_full = generate_walsh_matrix(high_dim)
    
    print("Option: basis % 4 == 0  ->  [0, 4, 8, 12]")
    print("-" * 50)
    
    basis_indices = [0, 4, 8, 12]
    
    print(f"W_full shape: {W_full.shape}")
    print(f"Selected basis indices: {basis_indices}")
    
    # W_selected: rows are the basis vectors
    W_selected = W_full[basis_indices, :]  # (4, 16)
    print(f"W_selected shape: {W_selected.shape}  (4 basis × 16 elements)")
    
    # W_transform: for matrix multiplication
    W_transform = W_selected.T  # (16, 4)
    print(f"W_transform shape: {W_transform.shape}  (16 output × 4 input)")
    
    action = np.array([0.5, -0.3, 0.8, -0.2])
    print(f"\nAction: {action}  shape: {action.shape}")
    
    output = W_transform @ action
    print(f"Output shape: {output.shape}")
    print(f"Output (continuous): {output}")
    
    binary = (output > 0).astype(int)
    print(f"Output (binary): {''.join(['#' if x == 1 else '.' for x in binary])}")


if __name__ == "__main__":
    demonstrate_math()
    test_basis_selection()
    test_controllability()
    test_pattern_generation()
    
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print("""
For 4-dim action -> 16-dim output:

✅ Strategy B: basis [0, 4, 8, 12] (every 4th)
   - Covers different spatial frequencies
   - More diverse patterns possible
   - Your suggestion (basis % 4 == 0) is good!

Also consider:
   - Strategy A [0,1,2,3]: Smoother, low-frequency patterns
   - Strategy C [0,1,4,8]: Mix of smooth + some detail

The math:
  (1, 4) @ (4, 16) = (1, 16)  ✓
  
  Or equivalently:
  (16, 4) @ (4,) = (16,)  ✓
""")

