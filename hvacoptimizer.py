import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import datetime
import logging
from typing import Dict, List, Tuple

class HVACOptimizer:
    def __init__(self, building_config: Dict):
        self.building_config = building_config
        self.is_trained = False
        self.feature_names = None
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)

    def _prepare_optimization_input(self, conditions: Dict) -> np.ndarray:
        """Prepare current conditions for model input."""
        if not self.feature_names:
            raise ValueError("Feature names not initialized. Model must be trained first.")

        print("\nPreparing input features...")

        # Create DataFrame with current conditions
        df = pd.DataFrame([conditions])
        processed = self.preprocess_sensor_data(df)

        # Add placeholder values for settings (will be replaced during optimization)
        processed['temperature_setpoint'] = processed['inside_temp']  # Initial placeholder
        processed['fan_speed'] = 0.8  # Initial placeholder

        # Ensure all required features are present and in correct order
        if not all(f in processed.columns for f in self.feature_names):
            missing = [f for f in self.feature_names if f not in processed.columns]
            raise ValueError(f"Missing required features: {missing}")

        # Select features in the same order as training
        processed = processed[self.feature_names]
        print("Using features:", list(processed.columns))

        # Scale numeric features
        numeric_cols = processed.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            processed[numeric_cols] = self.scaler.transform(processed[numeric_cols])

        return processed.values

    def _generate_setting_combinations(self, constraints: Dict) -> List[Dict]:
        """Generate possible HVAC setting combinations."""
        combinations = []
        temp_range = np.arange(constraints['min_temp'], constraints['max_temp'] + 0.5, 0.5)
        fan_speeds = np.arange(0.4, 1.1, 0.1)

        print(f"\nGenerating combinations for {len(temp_range)} temperatures and {len(fan_speeds)} fan speeds...")

        for temp in temp_range:
            for speed in fan_speeds:
                combinations.append({
                    'temperature_setpoint': round(temp, 1),
                    'fan_speed': round(speed, 1)
                })

        print(f"Generated {len(combinations)} possible combinations")
        return combinations

    def _meets_comfort_constraints(self, settings: Dict, constraints: Dict) -> bool:
        """Check if settings meet comfort constraints."""
        return (constraints['min_temp'] <= settings['temperature_setpoint'] <= constraints['max_temp'] and
                0.4 <= settings['fan_speed'] <= 1.0)

    def _find_optimal_settings(self, input_data: np.ndarray, setting_combinations: List[Dict],
                             constraints: Dict) -> Dict:
        """Find optimal settings that minimize energy consumption."""
        best_energy = float('inf')
        best_settings = None

        print(f"\nEvaluating {len(setting_combinations)} combinations...")

        # Ensure input_data is 2D
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)

        # Find indices for settings in feature names
        temp_idx = self.feature_names.index('temperature_setpoint')
        fan_idx = self.feature_names.index('fan_speed')

        for settings in setting_combinations:
            # Create input by modifying the settings columns
            prediction_input = input_data.copy()
            prediction_input[0, temp_idx] = settings['temperature_setpoint']
            prediction_input[0, fan_idx] = settings['fan_speed']

            try:
                # Predict energy consumption
                energy_consumption = self.model.predict(prediction_input)[0]

                # Update best settings if better
                if (energy_consumption < best_energy and
                    self._meets_comfort_constraints(settings, constraints)):
                    best_energy = energy_consumption
                    best_settings = settings.copy()
                    best_settings['predicted_energy'] = float(energy_consumption)

            except Exception as e:
                print(f"Warning: Prediction failed for settings {settings}: {str(e)}")
                continue

        if best_settings:
            print(f"\nFound optimal settings:")
            print(f"Temperature: {best_settings['temperature_setpoint']}°C")
            print(f"Fan Speed: {best_settings['fan_speed']}")
            print(f"Predicted Energy Usage: {best_settings['predicted_energy']:.2f} units")
        else:
            print("\nNo valid settings found within constraints")

        return best_settings

    def preprocess_sensor_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess sensor data for the model."""
        processed_data = raw_data.copy()

        if 'timestamp' in processed_data.columns:
            timestamp_series = pd.to_datetime(processed_data['timestamp'])
            processed_data['hour'] = timestamp_series.dt.hour
            processed_data['day_of_week'] = timestamp_series.dt.dayofweek
            processed_data['month'] = timestamp_series.dt.month
            processed_data = processed_data.drop('timestamp', axis=1)

        return processed_data

    def train_model(self, training_data: pd.DataFrame) -> bool:
        """Train the model with historical data."""
        try:
            print("\nStarting model training...")

            # Create a copy to avoid modifying the original data
            df = training_data.copy()

            # Preprocess and add settings columns
            processed = self.preprocess_sensor_data(df)
            processed['temperature_setpoint'] = processed['inside_temp']
            processed['fan_speed'] = 0.8

            # Define features in correct order
            base_features = [
                'outside_temp', 'outside_humidity', 'inside_temp', 'inside_humidity',
                'hour', 'day_of_week', 'month'
            ]
            control_features = ['temperature_setpoint', 'fan_speed']
            self.feature_names = base_features + control_features

            # Ensure all features exist
            for feature in self.feature_names:
                if feature not in processed.columns:
                    raise ValueError(f"Missing required feature: {feature}")

            print(f"Processing features: {', '.join(self.feature_names)}")

            # Prepare training data
            X = processed[self.feature_names]
            y = df['energy_consumption']

            # Scale features
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            X_scaled = X.copy()
            X_scaled[numeric_cols] = self.scaler.fit_transform(X[numeric_cols])

            # Train model
            print("Training Random Forest model...")
            self.model.fit(X_scaled.to_numpy(), y.to_numpy())

            self.is_trained = True
            print("Model training completed successfully")
            return True

        except Exception as e:
            print(f"Error during training: {str(e)}")
            self.is_trained = False
            raise

    def optimize_settings(self, current_conditions: Dict, comfort_constraints: Dict) -> Dict:
        """Optimize HVAC settings based on current conditions."""
        try:
            if not self.is_trained:
                raise ValueError("Model has not been trained yet")

            print("\nStarting optimization process...")

            # Prepare input data
            input_data = self._prepare_optimization_input(current_conditions)

            # Generate and evaluate combinations
            setting_combinations = self._generate_setting_combinations(comfort_constraints)
            best_settings = self._find_optimal_settings(input_data, setting_combinations,
                                                      comfort_constraints)

            if not best_settings:
                raise ValueError("No valid settings found within constraints")

            return best_settings

        except Exception as e:
            print(f"Optimization failed: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    try:
        # Initialize optimizer
        building_config = {
            "zones": ["Zone1", "Zone2", "Zone3"],
            "sensors": {
                "Zone1": {"temp": "sensor1", "humidity": "sensor2"},
                "Zone2": {"temp": "sensor3", "humidity": "sensor4"},
                "Zone3": {"temp": "sensor5", "humidity": "sensor6"}
            }
        }

        print("\nInitializing HVAC optimizer...")
        optimizer = HVACOptimizer(building_config)

        # Create training data
        dates = pd.date_range(start="2023-01-01", periods=1000, freq="h")
        historical_data = pd.DataFrame({
            "timestamp": dates,
            "outside_temp": np.random.uniform(10, 35, 1000),
            "outside_humidity": np.random.uniform(30, 80, 1000),
            "inside_temp": np.random.uniform(20, 25, 1000),
            "inside_humidity": np.random.uniform(40, 60, 1000),
            "energy_consumption": np.random.uniform(50, 200, 1000)
        })

        # Train model
        optimizer.train_model(historical_data)

        # Define optimization parameters
        comfort_constraints = {
            "min_temp": 20.0,
            "max_temp": 24.0,
            "max_humidity": 60.0
        }

        current_conditions = {
            "timestamp": pd.Timestamp.now(),
            "outside_temp": 30.0,
            "outside_humidity": 65.0,
            "inside_temp": 23.0,
            "inside_humidity": 55.0
        }

        # Optimize settings
        optimal_settings = optimizer.optimize_settings(current_conditions, comfort_constraints)
        print("\nOptimization completed successfully!")

    except Exception as e:
        print(f"\nError in main process: {str(e)}")
    finally:
        print("\nProcess completed.")
