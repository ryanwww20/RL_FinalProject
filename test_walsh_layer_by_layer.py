"""
Test layer-by-layer generation for PIC Inverse Design with Walsh transform.

Design area: 16x16 pixels (16 = 2^4, perfect for Walsh/Hadamard matrix)
Agent generates one row at a time:
- Input: low-dimensional vector (e.g., 4, 8 dims)
- Output: 16-dimensional binary vector (one row of the design)
- Total steps per episode: 16 (one for each row)
"""

import numpy as np
from scipy.linalg import hadamard
import matplotlib.pyplot as plt


def generate_walsh_matrix(n):
    """
    Generate a Walsh matrix of size n x n.
    Walsh matrix is derived from Hadamard matrix with rows reordered by sequency.
    
    For n = power of 2, this is exact and optimal.
    """
    H = hadamard(n)
    
    def count_sign_changes(row):
        return np.sum(np.abs(np.diff(row)) > 0)
    
    sequency = [count_sign_changes(H[i]) for i in range(n)]
    sorted_indices = np.argsort(sequency)
    walsh = H[sorted_indices]
    
    return walsh / np.sqrt(n)


def get_transform_matrix(low_dim, high_dim):
    """
    Get a transformation matrix from low_dim to high_dim using Walsh basis.
    
    For 16x16 design with high_dim=16 (power of 2), we get exact Walsh matrix.
    """
    n = 1
    while n < high_dim:
        n *= 2
    
    walsh_full = generate_walsh_matrix(n)
    transform = walsh_full[:high_dim, :low_dim]
    
    return transform


class LayerByLayerDesignEnv:
    """
    Environment for layer-by-layer design generation.
    
    Design: 16x16 pixels
    Episode: 16 steps (one row per step)
    Action: low_dim continuous vector -> Walsh transform -> 16 binary pixels
    """
    
    def __init__(self, design_size=16, low_dim=8):
        self.design_size = design_size
        self.low_dim = low_dim
        
        # Pre-compute Walsh transform matrix
        self.W = get_transform_matrix(low_dim, design_size)
        
        # Action space: continuous [-1, 1] for each low_dim
        self.action_dim = low_dim
        
        # State will include current design progress
        self.reset()
    
    def reset(self):
        """Reset the environment for a new episode."""
        self.current_row = 0
        self.design = np.zeros((self.design_size, self.design_size), dtype=np.float32)
        self.binary_design = np.zeros((self.design_size, self.design_size), dtype=np.int32)
        return self._get_state()
    
    def _get_state(self):
        """Get current state."""
        state = {
            'current_row': self.current_row,
            'partial_design': self.binary_design.copy(),
            'progress': self.current_row / self.design_size
        }
        return state
    
    def action_to_row(self, action):
        """
        Convert low-dimensional action to binary row using Walsh transform.
        
        Args:
            action: Low-dimensional vector of shape (low_dim,) with values in [-1, 1]
        
        Returns:
            binary_row: Binary vector of shape (design_size,) with values 0 or 1
            continuous_row: Continuous values before thresholding
        """
        action = np.clip(action, -1, 1)
        continuous_row = self.W @ action
        binary_row = (continuous_row > 0).astype(np.int32)
        
        return binary_row, continuous_row
    
    def step(self, action):
        """Take a step: generate the next row."""
        if self.current_row >= self.design_size:
            raise ValueError("Episode already finished!")
        
        binary_row, continuous_row = self.action_to_row(action)
        
        self.design[self.current_row] = continuous_row
        self.binary_design[self.current_row] = binary_row
        
        self.current_row += 1
        done = (self.current_row >= self.design_size)
        reward = 0.0
        
        info = {
            'row_generated': self.current_row - 1,
            'binary_row': binary_row,
            'continuous_row': continuous_row
        }
        
        return self._get_state(), reward, done, info
    
    def render(self, mode='text'):
        """Render current design state."""
        if mode == 'text':
            print(f"\nCurrent design (row {self.current_row}/{self.design_size}):")
            for i in range(self.design_size):
                if i < self.current_row:
                    row_str = "".join(["#" if x == 1 else "." for x in self.binary_design[i]])
                else:
                    row_str = " " * self.design_size
                print(f"  Row {i:2d}: {row_str}")
        elif mode == 'array':
            return self.binary_design.copy()


def test_layer_by_layer():
    """Test the layer-by-layer generation."""
    
    print("=" * 60)
    print("Layer-by-Layer Design Generation Test (16x16)")
    print("=" * 60)
    
    design_size = 16  # 2^4 - perfect for Walsh matrix!
    low_dim = 8       # Agent outputs 8-dim vector per row
    
    print(f"\nConfiguration:")
    print(f"  Design size: {design_size}x{design_size} (= 256 pixels)")
    print(f"  Action dim per step: {low_dim}")
    print(f"  Total steps per episode: {design_size}")
    print(f"  Compression ratio: {design_size/low_dim:.1f}x per row")
    print(f"  Note: 16 = 2^4, perfect for Walsh/Hadamard matrix!")
    
    env = LayerByLayerDesignEnv(design_size=design_size, low_dim=low_dim)
    
    print("\n" + "-" * 60)
    print("Simulating one episode with random actions...")
    print("-" * 60)
    
    state = env.reset()
    np.random.seed(42)
    
    done = False
    step = 0
    
    while not done:
        action = np.random.uniform(-1, 1, low_dim)
        state, reward, done, info = env.step(action)
        
        if step % 4 == 0 or done:
            print(f"\nStep {step + 1}: Generated row {info['row_generated']}")
            print(f"  Action (first 4): {action[:4]}")
            print(f"  Binary row: {info['binary_row']}")
            print(f"  Ones ratio: {np.mean(info['binary_row']):.1%}")
        
        step += 1
    
    print("\n" + "=" * 60)
    print("Final Design (16x16):")
    print("=" * 60)
    env.render(mode='text')
    
    final_design = env.binary_design
    print(f"\nDesign Statistics:")
    print(f"  Total Si (1) pixels: {np.sum(final_design)}")
    print(f"  Total SiO2 (0) pixels: {np.sum(1 - final_design)}")
    print(f"  Si ratio: {np.mean(final_design):.1%}")
    
    return env


def test_different_low_dims():
    """Test different low dimension settings."""
    
    print("\n" + "=" * 60)
    print("Testing Different Low Dimensions for 16x16 Design")
    print("=" * 60)
    
    design_size = 16
    
    for low_dim in [4, 8, 12, 16]:
        print(f"\n--- Low dim: {low_dim} (compression: {design_size/low_dim:.2f}x) ---")
        
        env = LayerByLayerDesignEnv(design_size=design_size, low_dim=low_dim)
        state = env.reset()
        np.random.seed(123)
        
        for _ in range(design_size):
            action = np.random.uniform(-1, 1, low_dim)
            state, _, _, _ = env.step(action)
        
        print(f"Si ratio: {np.mean(env.binary_design):.1%}")
        print("Design:")
        for row in env.binary_design:
            print("  " + "".join(["#" if x == 1 else "." for x in row]))


def test_controlled_patterns():
    """Test generating controlled patterns."""
    
    print("\n" + "=" * 60)
    print("Controlled Pattern Generation Test (16x16)")
    print("=" * 60)
    
    design_size = 16
    low_dim = 8
    
    env = LayerByLayerDesignEnv(design_size=design_size, low_dim=low_dim)
    
    # Pattern 1: All positive
    print("\n--- Pattern 1: All positive actions ---")
    state = env.reset()
    for _ in range(design_size):
        action = np.ones(low_dim) * 0.5
        state, _, _, _ = env.step(action)
    
    print(f"Si ratio: {np.mean(env.binary_design):.1%}")
    for row in env.binary_design:
        print("  " + "".join(["#" if x == 1 else "." for x in row]))
    
    # Pattern 2: Alternating
    print("\n--- Pattern 2: Alternating actions (stripes) ---")
    state = env.reset()
    for row in range(design_size):
        if row % 2 == 0:
            action = np.ones(low_dim) * 0.5
        else:
            action = np.ones(low_dim) * -0.5
        state, _, _, _ = env.step(action)
    
    print(f"Si ratio: {np.mean(env.binary_design):.1%}")
    for row in env.binary_design:
        print("  " + "".join(["#" if x == 1 else "." for x in row]))
    
    # Pattern 3: Gradient
    print("\n--- Pattern 3: Gradient ---")
    state = env.reset()
    for row in range(design_size):
        val = -1 + 2 * row / (design_size - 1)
        action = np.ones(low_dim) * val
        state, _, _, _ = env.step(action)
    
    print(f"Si ratio: {np.mean(env.binary_design):.1%}")
    for row in env.binary_design:
        print("  " + "".join(["#" if x == 1 else "." for x in row]))


def visualize_walsh_basis():
    """Visualize what each Walsh basis vector produces."""
    
    print("\n" + "=" * 60)
    print("Walsh Basis Visualization (16x16)")
    print("=" * 60)
    
    design_size = 16
    low_dim = 8
    
    W = get_transform_matrix(low_dim, design_size)
    
    print(f"\nWalsh transform matrix shape: {W.shape}")
    print(f"Each column is one Walsh basis vector\n")
    
    for basis_idx in range(low_dim):
        action = np.zeros(low_dim)
        action[basis_idx] = 1.0
        
        continuous = W @ action
        binary = (continuous > 0).astype(int)
        
        row_str = "".join(["#" if x == 1 else "." for x in binary])
        print(f"Basis {basis_idx}: {row_str}  (ones: {np.sum(binary):2d})")
    
    print("\nNote: Walsh basis vectors have different spatial frequencies")
    print("  - Basis 0: DC component (all same)")
    print("  - Higher basis: Higher frequency patterns")


def plot_design(design, title="Design", save_path=None):
    """Plot the design as an image."""
    plt.figure(figsize=(6, 6))
    plt.imshow(design, cmap='binary', interpolation='nearest')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y (row/layer)')
    plt.colorbar(label='Material (0=SiO2, 1=Si)')
    
    # Add grid
    ax = plt.gca()
    ax.set_xticks(np.arange(-0.5, design.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, design.shape[0], 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    # Test basic layer-by-layer generation
    env = test_layer_by_layer()
    
    # Test different low dimensions
    test_different_low_dims()
    
    # Test controlled patterns
    test_controlled_patterns()
    
    # Visualize Walsh basis
    visualize_walsh_basis()
    
    # Save sample design images
    print("\n" + "=" * 60)
    print("Saving sample design images...")
    print("=" * 60)
    
    # Random design
    env2 = LayerByLayerDesignEnv(design_size=16, low_dim=8)
    state = env2.reset()
    np.random.seed(456)
    
    for _ in range(16):
        action = np.random.uniform(-1, 1, 8)
        state, _, _, _ = env2.step(action)
    
    plot_design(
        env2.binary_design, 
        title="Layer-by-Layer Generated Design\n(16x16, 8-dim action per row)",
        save_path="/Users/ryan/NTUEE_Local/114-1/RL_FinalPJ/RL_FinalProject/sample_img/layer_by_layer_16x16.png"
    )
    
    print("\n" + "=" * 60)
    print("Summary: Layer-by-Layer Generation (16x16)")
    print("=" * 60)
    print("""
Setup for your RL:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Design:       16x16 pixels (256 total)
  Episode:      16 steps (one row per step)
  Action:       8-dim continuous [-1, 1]
  Transform:    Walsh matrix (8 -> 16)
  Output:       Binary (0=SiO2, 1=Si)
  Compression:  2x per row
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Why 16x16 is better:
- 16 = 2^4 -> Exact Walsh/Hadamard matrix
- No truncation or padding needed
- More efficient basis representation

Agent workflow per step:
1. Observe: Current partial design (rows 0 to i-1)
2. Output:  8-dim action vector
3. Transform: W @ action (8 -> 16)
4. Binarize: threshold at 0
5. Append row to design
""")
