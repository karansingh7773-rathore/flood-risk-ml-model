# Complete Flood Risk Prediction System with Model Training
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class FloodRiskModelTrainer:
    def __init__(self):
        """Initialize the model trainer"""
        self.model = None
        self.categorical_features = ['Infrastructure', 'Land_Cover', 'Soil_Type']
        self.numerical_features = [
            'Temperature_°C', 'Humidity_', 'Pressure_hPa', 'Wind_Speed_kmh',
            'Rainfall_mm', 'Elevation_m', 'Distance_to_River_km', 'Population_Density',
            'Water_Level_m', 'River_Discharge_ms', 'Historical_Floods',
            'Latitude', 'Longitude'
        ]
        
    def create_training_data(self, n_samples=1000):
        """Generate synthetic training data for flood prediction"""
        np.random.seed(42)
        
        data = []
        for i in range(n_samples):
            # Generate base features
            temp = np.random.normal(25, 8)  # Temperature
            humidity = np.random.uniform(30, 100)  # Humidity
            pressure = np.random.normal(1013, 20)  # Pressure
            wind_speed = np.random.exponential(10)  # Wind speed
            rainfall = np.random.exponential(5)  # Rainfall (exponential for realistic distribution)
            elevation = np.random.uniform(0, 1000)  # Elevation
            distance_to_river = np.random.exponential(2)  # Distance to river
            population_density = np.random.uniform(100, 15000)  # Population density
            water_level = np.random.uniform(0.5, 8)  # Water level
            river_discharge = np.random.uniform(10, 500)  # River discharge
            historical_floods = np.random.randint(0, 10)  # Historical floods count
            
            # Location (India coordinates)
            latitude = np.random.uniform(8, 35)
            longitude = np.random.uniform(68, 97)
            
            # Categorical features
            infrastructure = np.random.choice(['Poor', 'Medium', 'Good'], p=[0.3, 0.5, 0.2])
            land_cover = np.random.choice(['Urban', 'Forest', 'Agricultural', 'Water'], p=[0.4, 0.2, 0.3, 0.1])
            soil_type = np.random.choice(['Clay', 'Sand', 'Loam', 'Rock'], p=[0.3, 0.25, 0.35, 0.1])
            
            # Create flood risk based on realistic conditions
            flood_score = 0
            
            # High rainfall increases flood risk
            if rainfall > 15: flood_score += 3
            elif rainfall > 8: flood_score += 2
            elif rainfall > 3: flood_score += 1
            
            # High water level increases risk
            if water_level > 5: flood_score += 3
            elif water_level > 3: flood_score += 2
            elif water_level > 2: flood_score += 1
            
            # High river discharge increases risk
            if river_discharge > 300: flood_score += 2
            elif river_discharge > 200: flood_score += 1
            
            # Low elevation increases risk
            if elevation < 50: flood_score += 2
            elif elevation < 100: flood_score += 1
            
            # Close to river increases risk
            if distance_to_river < 0.5: flood_score += 3
            elif distance_to_river < 1: flood_score += 2
            elif distance_to_river < 2: flood_score += 1
            
            # Historical floods increase risk
            if historical_floods > 5: flood_score += 2
            elif historical_floods > 2: flood_score += 1
            
            # Infrastructure affects risk
            if infrastructure == 'Poor': flood_score += 2
            elif infrastructure == 'Medium': flood_score += 1
            
            # Soil type affects risk
            if soil_type == 'Clay': flood_score += 1  # Clay retains water
            elif soil_type == 'Sand': flood_score -= 1  # Sand drains well
            
            # High humidity with high temperature can increase risk
            if humidity > 80 and temp > 30: flood_score += 1
            
            # Create binary flood risk (threshold can be adjusted)
            flood_risk = 1 if flood_score >= 6 else 0
            
            # Add some randomness to make it more realistic
            if np.random.random() < 0.1:  # 10% random flip
                flood_risk = 1 - flood_risk
            
            data.append({
                'Temperature_°C': round(temp, 2),
                'Humidity_': round(humidity, 2),
                'Pressure_hPa': round(pressure, 2),
                'Wind_Speed_kmh': round(wind_speed, 2),
                'Rainfall_mm': round(rainfall, 2),
                'Elevation_m': round(elevation, 2),
                'Distance_to_River_km': round(distance_to_river, 2),
                'Population_Density': round(population_density, 2),
                'Water_Level_m': round(water_level, 2),
                'River_Discharge_ms': round(river_discharge, 2),
                'Historical_Floods': historical_floods,
                'Latitude': round(latitude, 4),
                'Longitude': round(longitude, 4),
                'Infrastructure': infrastructure,
                'Land_Cover': land_cover,
                'Soil_Type': soil_type,
                'Flood_Risk': flood_risk
            })
        
        return pd.DataFrame(data)
    
    def train_model(self, df):
        """Train the flood prediction model with proper preprocessing"""
        print("Training flood prediction model...")
        
        # Prepare features and target
        X = df.drop('Flood_Risk', axis=1)
        y = df['Flood_Risk']
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_features),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), self.categorical_features)
            ])
        
        # Create the full pipeline
        self.model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2
            ))
        ])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model trained successfully!")
        print(f"Training Accuracy: {accuracy:.3f}")
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save the model
        joblib.dump(self.model, 'flood_risk_model_with_preprocessing.pkl')
        print("Model saved as 'flood_risk_model_with_preprocessing.pkl'")
        
        return self.model

class FloodRiskPredictor:
    def __init__(self, model_path="flood_risk_model_with_preprocessing.pkl"):
        """Initialize the predictor with a trained model"""
        try:
            self.model = joblib.load(model_path)
            print(f"Model loaded successfully from {model_path}")
        except FileNotFoundError:
            print(f"Model file not found. Please train the model first.")
            self.model = None
            
        # Define expected features
        self.numerical_features = [
            'Temperature_°C', 'Humidity_', 'Pressure_hPa', 'Wind_Speed_kmh',
            'Rainfall_mm', 'Elevation_m', 'Distance_to_River_km', 'Population_Density',
            'Water_Level_m', 'River_Discharge_ms', 'Historical_Floods',
            'Latitude', 'Longitude'
        ]
        self.categorical_features = ['Infrastructure', 'Land_Cover', 'Soil_Type']
        self.all_features = self.numerical_features + self.categorical_features

    def predict_from_dict(self, data_dict):
        """Make prediction from a dictionary of current conditions"""
        if self.model is None:
            print("Model not loaded. Cannot make predictions.")
            return None
            
        try:
            # Create DataFrame from input
            df = pd.DataFrame([data_dict])
            
            # Ensure all required columns are present with proper defaults
            for feature in self.all_features:
                if feature not in df.columns:
                    if feature in self.numerical_features:
                        # Set reasonable defaults for numerical features
                        defaults = {
                            'Temperature_°C': 25.0,
                            'Humidity_': 70.0,
                            'Pressure_hPa': 1013.0,
                            'Wind_Speed_kmh': 10.0,
                            'Rainfall_mm': 0.0,
                            'Elevation_m': 100.0,
                            'Distance_to_River_km': 2.0,
                            'Population_Density': 1000.0,
                            'Water_Level_m': 2.0,
                            'River_Discharge_ms': 100.0,
                            'Historical_Floods': 1,
                            'Latitude': 20.0,
                            'Longitude': 77.0
                        }
                        df[feature] = defaults.get(feature, 0.0)
                    else:
                        # Set defaults for categorical features
                        defaults = {
                            'Infrastructure': 'Medium',
                            'Land_Cover': 'Urban',
                            'Soil_Type': 'Clay'
                        }
                        df[feature] = defaults.get(feature, 'Medium')
            
            # Select only the features the model expects
            df = df[self.all_features]
            
            # Make prediction
            prediction = self.model.predict(df)[0]
            probability = self.model.predict_proba(df)[0, 1]
            
            return {
                'flood_risk': bool(prediction),
                'flood_probability': float(probability),
                'risk_level': self._get_risk_level(probability),
                'confidence': float(max(self.model.predict_proba(df)[0]))
            }
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None

    def predict_from_csv(self, csv_path):
        """Make predictions from a CSV file"""
        if self.model is None:
            print("Model not loaded. Cannot make predictions.")
            return None
            
        try:
            # Load data
            df = pd.read_csv(csv_path)
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Ensure all required features are present
            for feature in self.all_features:
                if feature not in df.columns:
                    if feature in self.numerical_features:
                        df[feature] = 0.0  # Default for numerical
                    else:
                        df[feature] = 'Medium'  # Default for categorical
            
            # Select only model features
            features_df = df[self.all_features]
            
            # Make predictions
            predictions = self.model.predict(features_df)
            probabilities = self.model.predict_proba(features_df)[:, 1]
            
            # Add results to original dataframe
            df['Flood_Risk_Prediction'] = predictions
            df['Flood_Probability'] = probabilities
            df['Risk_Level'] = [self._get_risk_level(p) for p in probabilities]
            
            return df
            
        except Exception as e:
            print(f"Error processing CSV: {e}")
            return None

    def _get_risk_level(self, probability):
        """Convert probability to risk level"""
        if probability < 0.3:
            return "Low"
        elif probability < 0.7:
            return "Medium"
        else:
            return "High"

class WeatherDataFetcher:
    def __init__(self, api_key="d47cbe0c596d4fdab72165430250909"):
        self.api_key = api_key
        self.base_url = "http://api.weatherapi.com/v1/current.json"

    def get_current_weather(self, city_name):
        """Fetch current weather data from API"""
        try:
            params = {
                'key': self.api_key,
                'q': city_name,
                'aqi': 'no'
            }
            
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            if response.status_code == 200:
                current = data['current']
                location = data['location']
                
                weather_data = {
                    'Temperature_°C': current['temp_c'],
                    'Humidity_': current['humidity'],
                    'Pressure_hPa': current['pressure_mb'],
                    'Wind_Speed_kmh': current['wind_kph'],
                    'Rainfall_mm': current.get('precip_mm', 0),
                    'Latitude': location['lat'],
                    'Longitude': location['lon']
                }
                return weather_data
            else:
                print(f"API Error: {data.get('error', {}).get('message', 'Unknown error')}")
                return None
                
        except Exception as e:
            print(f"Error fetching weather data: {e}")
            return None

# Example usage functions
def train_new_model():
    """Train a new flood prediction model"""
    trainer = FloodRiskModelTrainer()
    
    # Generate training data
    print("Generating synthetic training data...")
    training_data = trainer.create_training_data(n_samples=2000)
    
    # Save training data for reference
    training_data.to_csv('flood_training_data.csv', index=False)
    print("Training data saved as 'flood_training_data.csv'")
    
    # Train the model
    model = trainer.train_model(training_data)
    
    return model

def test_predictions():
    """Test the trained model with sample data"""
    predictor = FloodRiskPredictor()
    
    if predictor.model is None:
        print("Please train the model first using train_new_model()")
        return
    
    # Test with manual input
    print("\n--- Testing Manual Input ---")
    test_conditions = {
        'Temperature_°C': 32.0,
        'Humidity_': 85.0,
        'Pressure_hPa': 1008.0,
        'Wind_Speed_kmh': 25.0,
        'Rainfall_mm': 45.0,
        'Elevation_m': 15.0,
        'Distance_to_River_km': 0.8,
        'Population_Density': 5000.0,
        'Water_Level_m': 4.2,
        'River_Discharge_ms': 250.0,
        'Historical_Floods': 3,
        'Latitude': 19.0760,
        'Longitude': 72.8777,
        'Infrastructure': 'Poor',
        'Land_Cover': 'Urban',
        'Soil_Type': 'Clay'
    }
    
    result = predictor.predict_from_dict(test_conditions)
    if result:
        print(f"Flood Risk: {result['risk_level']}")
        print(f"Probability: {result['flood_probability']:.2%}")
        print(f"Confidence: {result['confidence']:.2%}")

def test_weather_api_predictions():
    """Test predictions using live weather data"""
    predictor = FloodRiskPredictor()
    weather_fetcher = WeatherDataFetcher()
    
    if predictor.model is None:
        print("Please train the model first using train_new_model()")
        return
    
    cities = ["Mumbai", "Chennai", "Delhi"]
    
    for city in cities:
        print(f"\n--- {city} Flood Risk Assessment ---")
        
        weather_data = weather_fetcher.get_current_weather(city)
        
        if weather_data:
            # Add additional features with realistic defaults for Indian cities
            city_defaults = {
                'Mumbai': {
                    'Elevation_m': 11.0,
                    'Distance_to_River_km': 0.5,
                    'Population_Density': 20482.0,
                    'Water_Level_m': 2.5,
                    'River_Discharge_ms': 150.0,
                    'Historical_Floods': 4,
                    'Infrastructure': 'Medium',
                    'Land_Cover': 'Urban',
                    'Soil_Type': 'Clay'
                },
                'Chennai': {
                    'Elevation_m': 6.0,
                    'Distance_to_River_km': 1.2,
                    'Population_Density': 26903.0,
                    'Water_Level_m': 2.0,
                    'River_Discharge_ms': 80.0,
                    'Historical_Floods': 5,
                    'Infrastructure': 'Medium',
                    'Land_Cover': 'Urban',
                    'Soil_Type': 'Sand'
                },
                'Delhi': {
                    'Elevation_m': 216.0,
                    'Distance_to_River_km': 2.0,
                    'Population_Density': 11320.0,
                    'Water_Level_m': 1.8,
                    'River_Discharge_ms': 120.0,
                    'Historical_Floods': 2,
                    'Infrastructure': 'Good',
                    'Land_Cover': 'Urban',
                    'Soil_Type': 'Loam'
                }
            }
            
            # Add city-specific defaults
            weather_data.update(city_defaults.get(city, city_defaults['Mumbai']))
            
            result = predictor.predict_from_dict(weather_data)
            
            if result:
                print(f"Risk Level: {result['risk_level']}")
                print(f"Probability: {result['flood_probability']:.2%}")
                print(f"Current Temperature: {weather_data['Temperature_°C']}°C")
                print(f"Current Rainfall: {weather_data['Rainfall_mm']}mm")

if __name__ == "__main__":
    print("=== Flood Risk Prediction System ===\n")
    
    # Step 1: Train a new model from your CSV data
    print("Step 1: Training new model from your data...")
    
    # Option 1: Specify your CSV file path directly
    # model = train_new_model("your_flood_data.csv")
    
    # Option 2: Interactive mode - asks for CSV path
    model = train_new_model()
    
    if model:
        # Step 2: Test predictions
        print("\nStep 2: Testing predictions...")
        test_predictions()
        
        # Step 3: Test with weather API
        print("\nStep 3: Testing with weather API...")
        test_weather_api_predictions()
    else:
        print("Model training failed. Please check your CSV file and try again.")