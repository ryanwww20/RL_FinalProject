### train_ppo.py

```python
train_ppo()

- model.learn(): callback - save_fig at each step(50 layers)

- test_model()


```

### Each step

```
continuous_gym

take action: add layer

get input/output flux

calculate reward

```

### Every 50 layers (at termination)

```
plot 3 images: reward(saved in csv, later plotted manually), distribution, design
ask meep_simulation to plot distribution and design
reward = cur_score - last_score, last_score = 0 at reset
```

### meep_simulation

```python
calculate_flux(material_matrix) {
    return input_flux_value, output_flux_value_1, output_flux_value_2, output_all_flux, ez_data
}

plot_distribution(output_all_flux)
plot_design(material_matrix)
```
