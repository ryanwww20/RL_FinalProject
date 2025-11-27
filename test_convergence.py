import numpy as np
import matplotlib.pyplot as plt
from envs.meep_simulation import WaveguideSimulation
from config import config

def test_time_convergence():
    # 測試不同的模擬時間
    times = [100, 200, 300, 400, 500, 1000, 1500, 2000]
    transmissions = []  # 改存 ratio
    
    np.random.seed(42)
    material_matrix = np.random.randint(0, 2, (config.simulation.pixel_num_x, config.simulation.pixel_num_y))
    
    print("開始收斂測試 (Ratio)...")
    
    for t in times:
        print(f"Testing simulation_time = {t}...", end="", flush=True)
        config.simulation.simulation_time = t
        
        sim = WaveguideSimulation()
        # 獲取 input 和 output
        input_flux, out1, out2, _, _ = sim.calculate_flux(material_matrix)
        
        # 計算穿透率 (避免除以零)
        if input_flux == 0:
            ratio = 0
        else:
            ratio = (out1 + out2) / input_flux
            
        transmissions.append(ratio)
        print(f" Transmission: {ratio:.6f}")

    # 畫圖
    plt.figure(figsize=(10, 6))
    plt.plot(times, transmissions, 'o-', linewidth=2)
    
    baseline = transmissions[-1]
    plt.axhline(y=baseline, color='r', linestyle='--', label=f'Baseline (t=2000): {baseline:.4f}')
    plt.fill_between(times, baseline*0.99, baseline*1.01, color='green', alpha=0.1, label='1% Error Margin')
    
    plt.xlabel('Simulation Time')
    plt.ylabel('Transmission (Output/Input)')
    plt.title('Convergence Test: Transmission Ratio vs Time')
    plt.grid(True)
    plt.legend()
    plt.ylim(0, 1.1) # 限制 y 軸在合理範圍
    plt.savefig('convergence_test_ratio.png')
    plt.show()

if __name__ == "__main__":
    test_time_convergence()
