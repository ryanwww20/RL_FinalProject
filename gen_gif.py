import os
from PIL import Image

# --- CONFIG ---
input_folder = "ppo_model_logs/flux_images"        # folder containing PNGs
output_path = "flux.gif"     # where to save gif
duration = 100                 # ms per frame
# --------------

# Get all PNG files in the folder, sorted by filename
files = sorted([
    os.path.join(input_folder, f)
    for f in os.listdir(input_folder)
    if f.lower().endswith(".png")
])

if not files:
    raise ValueError("No PNG files found in folder.")

# Load images
frames = [Image.open(f) for f in files]

# Save GIF
frames[0].save(
    output_path,
    save_all=True,
    append_images=frames[1:],
    duration=duration,
    loop=0
)

print(f"GIF saved to {output_path}")