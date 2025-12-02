"""
Test Walsh matrix transformation for dimensionality reduction in RL action space.

Goal: Agent learns a low-dimensional vector, which is then multiplied by Walsh matrix
to produce a 20-dimensional binary vector (0 or 1 representing SiO2 or Si).
"""

import numpy as np
from scipy.linalg import hadamard


def generate_walsh_matrix(n):
    """
    Generate a Walsh matrix of size n x n.
    Walsh matrix is derived from Hadamard matrix with rows reordered by sequency.
    
    Args:
        n: Size of the matrix (must be a power of 2)
    
    Returns:
        Walsh matrix of size n x n
    """
    # Hadamard matrix (size must be power of 2)
    H = hadamard(n)
    
    # Convert to Walsh ordering (by sequency - number of sign changes)
    def count_sign_changes(row):
        return np.sum(np.abs(np.diff(row)) > 0)
    
    # Sort rows by number of sign changes (sequency)
    sequency = [count_sign_changes(H[i]) for i in range(n)]
    sorted_indices = np.argsort(sequency)
    walsh = H[sorted_indices]
    
    return walsh / np.sqrt(n)  # Normalize


def get_transform_matrix(low_dim, high_dim):
    """
    Get a transformation matrix from low_dim to high_dim using Walsh basis.
    
    Args:
        low_dim: Input dimension (agent's action space)
        high_dim: Output dimension (design space, e.g., 20)
    
    Returns:
        Transform matrix of shape (high_dim, low_dim)
    """
    # Find the smallest power of 2 >= high_dim
    n = 1
    while n < high_dim:
        n *= 2
    
    # Generate full Walsh matrix
    walsh_full = generate_walsh_matrix(n)
    
    # Take the first low_dim columns and first high_dim rows
    # This gives us the lowest-frequency Walsh basis vectors
    transform = walsh_full[:high_dim, :low_dim]
    
    return transform


def low_dim_to_binary(low_dim_action, transform_matrix, threshold=0.0):
    """
    Convert low-dimensional continuous action to high-dimensional binary vector.
    
    Args:
        low_dim_action: Low-dimensional vector from agent (continuous values)
        transform_matrix: Walsh transform matrix
        threshold: Threshold for binarization
    
    Returns:
        Binary vector (0 or 1)
    """
    # Transform to high dimension
    high_dim = transform_matrix @ low_dim_action
    
    # Binarize using threshold
    binary = (high_dim > threshold).astype(int)
    
    return high_dim, binary


def test_walsh_transform():
    """Test the Walsh transform with different low dimensions."""
    
    high_dim = 20  # Target dimension (design pixels per row)
    
    print("=" * 60)
    print("Walsh Matrix Transform Test for PIC Inverse Design")
    print("=" * 60)
    print(f"Target output dimension: {high_dim}")
    print()
    
    # Test different low dimensions
    for low_dim in [4, 8, 10, 16]:
        print(f"\n{'='*60}")
        print(f"Low dimension: {low_dim} -> High dimension: {high_dim}")
        print(f"Compression ratio: {high_dim/low_dim:.2f}x")
        print("-" * 60)
        
        # Get transform matrix
        W = get_transform_matrix(low_dim, high_dim)
        print(f"Transform matrix shape: {W.shape}")
        
        # Test with random continuous input (simulating agent output)
        np.random.seed(42)
        
        # Test 1: Random continuous values in [-1, 1]
        low_action = np.random.uniform(-1, 1, low_dim)
        print(f"\nTest 1 - Random input: {low_action[:5]}... (showing first 5)")
        
        continuous, binary = low_dim_to_binary(low_action, W)
        print(f"Continuous output (first 10): {continuous[:10]}")
        print(f"Binary output: {binary}")
        print(f"Ones ratio: {np.mean(binary):.2%}")
        
        # Test 2: All zeros -> should give roughly 50% ones
        low_action_zeros = np.zeros(low_dim)
        continuous, binary = low_dim_to_binary(low_action_zeros, W)
        print(f"\nTest 2 - All zeros input:")
        print(f"Binary output: {binary}")
        print(f"Ones ratio: {np.mean(binary):.2%}")
        
        # Test 3: All positive -> should give more ones
        low_action_pos = np.ones(low_dim) * 0.5
        continuous, binary = low_dim_to_binary(low_action_pos, W)
        print(f"\nTest 3 - All positive (0.5) input:")
        print(f"Binary output: {binary}")
        print(f"Ones ratio: {np.mean(binary):.2%}")
        
        # Test 4: All negative -> should give fewer ones
        low_action_neg = np.ones(low_dim) * -0.5
        continuous, binary = low_dim_to_binary(low_action_neg, W)
        print(f"\nTest 4 - All negative (-0.5) input:")
        print(f"Binary output: {binary}")
        print(f"Ones ratio: {np.mean(binary):.2%}")


def test_2d_design():
    """Test for 2D design area (20x20 pixels)."""
    
    print("\n" + "=" * 60)
    print("2D Design Area Test (20x20 pixels)")
    print("=" * 60)
    
    design_size = 20
    total_pixels = design_size * design_size  # 400 pixels
    
    # For 2D, we can either:
    # Option 1: Apply Walsh transform to each row independently
    # Option 2: Apply 2D Walsh transform
    
    print(f"\nDesign area: {design_size}x{design_size} = {total_pixels} pixels")
    
    # Option 1: Row-wise transform
    low_dim_per_row = 8  # Each row compressed to 8 dimensions
    total_low_dim = low_dim_per_row * design_size  # 8 * 20 = 160 dimensions
    
    print(f"\nOption 1: Row-wise Walsh transform")
    print(f"  - Low dim per row: {low_dim_per_row}")
    print(f"  - Total agent action dim: {total_low_dim}")
    print(f"  - Compression ratio: {total_pixels/total_low_dim:.2f}x")
    
    W_row = get_transform_matrix(low_dim_per_row, design_size)
    
    # Simulate agent output
    np.random.seed(123)
    agent_action = np.random.uniform(-1, 1, (design_size, low_dim_per_row))
    
    # Transform each row
    design = np.zeros((design_size, design_size), dtype=int)
    for i in range(design_size):
        _, binary = low_dim_to_binary(agent_action[i], W_row)
        design[i] = binary
    
    print(f"\n  Generated design (20x20):")
    print(f"  Si (1) ratio: {np.mean(design):.2%}")
    
    # Visualize as text
    print("\n  Design visualization (. = SiO2, # = Si):")
    for row in design:
        print("  " + "".join(["#" if x == 1 else "." for x in row]))
    
    # Option 2: Even lower dimension with 2D Walsh
    print(f"\n\nOption 2: 2D Walsh transform")
    low_dim_2d = 8  # 8x8 = 64 dimensional latent space
    total_low_dim_2d = low_dim_2d * low_dim_2d
    print(f"  - Latent space: {low_dim_2d}x{low_dim_2d} = {total_low_dim_2d} dimensions")
    print(f"  - Compression ratio: {total_pixels/total_low_dim_2d:.2f}x")
    
    # For 2D Walsh, we need Walsh matrices for both dimensions
    W_row_2d = get_transform_matrix(low_dim_2d, design_size)  # 20x8
    W_col_2d = get_transform_matrix(low_dim_2d, design_size)  # 20x8
    
    # Simulate agent output (8x8 latent)
    latent = np.random.uniform(-1, 1, (low_dim_2d, low_dim_2d))
    
    # 2D transform: design = W_row @ latent @ W_col.T
    continuous_2d = W_row_2d @ latent @ W_col_2d.T
    design_2d = (continuous_2d > 0).astype(int)
    
    print(f"\n  Generated design (20x20):")
    print(f"  Si (1) ratio: {np.mean(design_2d):.2%}")
    
    print("\n  Design visualization (. = SiO2, # = Si):")
    for row in design_2d:
        print("  " + "".join(["#" if x == 1 else "." for x in row]))


if __name__ == "__main__":
    test_walsh_transform()
    test_2d_design()
    
    print("\n" + "=" * 60)
    print("Summary: Walsh transform works for dimensionality reduction!")
    print("=" * 60)
    print("""
Recommendations for your RL setup:
1. For 20x20 design area (400 pixels total):
   
   Option A: Row-wise transform
   - Agent action: 20 rows × 8 dims/row = 160 dimensions
   - Each row uses 8-dim Walsh transform to 20-dim
   
   Option B: 2D Walsh transform  
   - Agent action: 8×8 = 64 dimensions (more compact!)
   - Use 2D Walsh: design = W_row @ latent @ W_col.T
   
2. Agent outputs continuous values in [-1, 1]
3. After Walsh transform, threshold at 0 to get binary (Si/SiO2)
""")

