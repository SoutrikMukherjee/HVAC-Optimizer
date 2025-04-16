# HVAC-Optimizer

A **Machine Learning-Based HVAC Optimization System** designed to enhance energy efficiency in commercial buildings.  
This system uses historical sensor data and a trained Random Forest model to predict energy consumption and dynamically optimize HVAC settings (temperature & fan speed) in real time.

---

## 🧠 How It Works

1. **Train a model** using past HVAC sensor data.
2. **Generate valid combinations** of temperature setpoints and fan speeds.
3. **Predict energy consumption** for all combinations using the trained model.
4. **Select the best setting** based on minimum predicted energy usage within comfort constraints.
5. **Visualize** the energy landscape using a heatmap.

---

## 📊 Visualization Output

After successful optimization, the following heatmap is generated to illustrate predicted energy consumption across various settings:

![Energy Heatmap](images/hvac_optimization_heatmap.png)

🟦 **Blue** = lower energy usage (good)  
🟥 **Red** = higher energy usage (bad)  
💡 The optimizer **automatically selects** the best zone (lowest energy in comfort range)

---

## ✅ Example Output

```bash
Initializing HVAC optimizer...

Starting model training...
Processing features: outside_temp, outside_humidity, inside_temp, inside_humidity, hour, day_of_week, month, temperature_setpoint, fan_speed
Training Random Forest model...
Model training completed successfully

Starting optimization process...
Evaluating 55 combinations...

Found optimal settings:
Temperature: 22.5°C
Fan Speed: 0.5
Predicted Energy Usage: 58.73 units

Optimization completed successfully!
