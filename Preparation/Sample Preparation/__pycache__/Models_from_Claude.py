# Predictive Maintenance Audio Classification System
# This script handles audio data import, feature extraction, and ML classification

import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class AudioPredictiveMaintenance:
    def __init__(self, data_path):
        self.data_path = data_path
        self.audio_data = []
        self.labels = []
        self.features_df = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}
        
    def load_audio_data(self, sample_rate=22050, duration=None):
        """
        Load audio files from organized directory structure
        Expected structure: data_path/condition_name/*.mp3
        """
        print("Loading audio data...")
        
        for condition_folder in os.listdir(self.data_path):
            condition_path = os.path.join(self.data_path, condition_folder)
            
            if not os.path.isdir(condition_path):
                continue
                
            print(f"Processing condition: {condition_folder}")
            
            for audio_file in os.listdir(condition_path):
                if audio_file.lower().endswith(('.mp3', '.wav', '.flac')):
                    file_path = os.path.join(condition_path, audio_file)
                    
                    try:
                        # Load audio file
                        y, sr = librosa.load(file_path, sr=sample_rate, duration=duration)
                        
                        self.audio_data.append({
                            'audio': y,
                            'sample_rate': sr,
                            'filename': audio_file,
                            'condition': condition_folder
                        })
                        self.labels.append(condition_folder)
                        
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
        
        print(f"Loaded {len(self.audio_data)} audio samples")
        return len(self.audio_data)
    
    def extract_time_domain_features(self, audio_signal):
        """Extract time-domain features from audio signal"""
        features = {}
        
        # Basic statistical features
        features['mean'] = np.mean(audio_signal)
        features['std'] = np.std(audio_signal)
        features['var'] = np.var(audio_signal)
        features['rms'] = np.sqrt(np.mean(audio_signal**2))
        features['max'] = np.max(audio_signal)
        features['min'] = np.min(audio_signal)
        features['range'] = features['max'] - features['min']
        
        # Advanced statistical features
        features['skewness'] = stats.skew(audio_signal)
        features['kurtosis'] = stats.kurtosis(audio_signal)
        features['peak_factor'] = features['max'] / features['rms']
        features['crest_factor'] = features['max'] / np.mean(np.abs(audio_signal))
        
        # Zero crossing rate
        features['zcr'] = np.sum(np.diff(np.sign(audio_signal)) != 0) / (2 * len(audio_signal))
        
        return features
    
    def extract_frequency_domain_features(self, audio_signal, sr):
        """Extract frequency-domain features from audio signal"""
        features = {}
        
        # FFT-based features
        fft = np.fft.fft(audio_signal)
        magnitude = np.abs(fft)
        freqs = np.fft.fftfreq(len(fft), 1/sr)
        
        # Power spectral density
        psd = magnitude**2 / len(magnitude)
        
        # Spectral features
        features['spectral_centroid'] = np.sum(freqs[:len(freqs)//2] * psd[:len(psd)//2]) / np.sum(psd[:len(psd)//2])
        features['spectral_bandwidth'] = np.sqrt(np.sum(((freqs[:len(freqs)//2] - features['spectral_centroid'])**2) * psd[:len(psd)//2]) / np.sum(psd[:len(psd)//2]))
        features['spectral_rolloff'] = freqs[np.where(np.cumsum(psd) >= 0.85 * np.sum(psd))[0][0]] if len(np.where(np.cumsum(psd) >= 0.85 * np.sum(psd))[0]) > 0 else 0
        
        # Energy in different frequency bands
        features['low_freq_energy'] = np.sum(psd[freqs <= 1000])
        features['mid_freq_energy'] = np.sum(psd[(freqs > 1000) & (freqs <= 5000)])
        features['high_freq_energy'] = np.sum(psd[freqs > 5000])
        
        return features
    
    def extract_mfcc_features(self, audio_signal, sr, n_mfcc=13):
        """Extract MFCC features"""
        mfccs = librosa.feature.mfcc(y=audio_signal, sr=sr, n_mfcc=n_mfcc)
        
        features = {}
        for i in range(n_mfcc):
            features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i}_std'] = np.std(mfccs[i])
            features[f'mfcc_{i}_max'] = np.max(mfccs[i])
            features[f'mfcc_{i}_min'] = np.min(mfccs[i])
        
        return features
    
    def extract_advanced_features(self, audio_signal, sr):
        """Extract advanced audio features using librosa"""
        features = {}
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_signal, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_signal, sr=sr)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_signal, sr=sr)[0]
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=audio_signal, sr=sr)
        features['chroma_mean'] = np.mean(chroma)
        features['chroma_std'] = np.std(chroma)
        
        # Tonnetz features
        tonnetz = librosa.feature.tonnetz(y=audio_signal, sr=sr)
        features['tonnetz_mean'] = np.mean(tonnetz)
        features['tonnetz_std'] = np.std(tonnetz)
        
        # Temporal features
        tempo, beats = librosa.beat.beat_track(y=audio_signal, sr=sr)
        features['tempo'] = tempo
        
        return features
    
    def extract_all_features(self):
        """Extract all features from loaded audio data"""
        print("Extracting features...")
        
        all_features = []
        
        for i, audio_data in enumerate(self.audio_data):
            print(f"Processing {i+1}/{len(self.audio_data)}", end='\r')
            
            audio_signal = audio_data['audio']
            sr = audio_data['sample_rate']
            condition = audio_data['condition']
            
            # Combine all feature types
            features = {}
            
            # Time domain features
            time_features = self.extract_time_domain_features(audio_signal)
            features.update(time_features)
            
            # Frequency domain features
            freq_features = self.extract_frequency_domain_features(audio_signal, sr)
            features.update(freq_features)
            
            # MFCC features
            mfcc_features = self.extract_mfcc_features(audio_signal, sr)
            features.update(mfcc_features)
            
            # Advanced features
            advanced_features = self.extract_advanced_features(audio_signal, sr)
            features.update(advanced_features)
            
            # Add metadata
            features['condition'] = condition
            features['filename'] = audio_data['filename']
            
            all_features.append(features)
        
        # Convert to DataFrame
        self.features_df = pd.DataFrame(all_features)
        
        # Handle any NaN or infinite values
        self.features_df = self.features_df.replace([np.inf, -np.inf], np.nan)
        self.features_df = self.features_df.fillna(self.features_df.mean())
        
        print(f"\nExtracted {len(self.features_df.columns)-2} features from {len(self.features_df)} samples")
        return self.features_df
    
    def prepare_data_for_ml(self, test_size=0.2, random_state=42):
        """Prepare data for machine learning"""
        if self.features_df is None:
            raise ValueError("Features not extracted. Run extract_all_features() first.")
        
        # Separate features and labels
        feature_columns = [col for col in self.features_df.columns if col not in ['condition', 'filename']]
        X = self.features_df[feature_columns]
        y = self.features_df['condition']
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, feature_columns
    
    def train_models(self, X_train, y_train):
        """Train multiple ML models"""
        print("Training models...")
        
        # Define models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        }
        
        # Train and evaluate models
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            print(f"{name} - CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            self.models[name] = model
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate trained models"""
        print("\nModel Evaluation:")
        print("="*50)
        
        results = {}
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"\n{name}:")
            print(f"Accuracy: {accuracy:.4f}")
            
            # Detailed classification report
            class_names = self.label_encoder.classes_
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=class_names))
            
            results[name] = {
                'accuracy': accuracy,
                'predictions': y_pred,
                'model': model
            }
        
        return results
    
    def plot_confusion_matrices(self, X_test, y_test):
        """Plot confusion matrices for all models"""
        class_names = self.label_encoder.classes_
        n_models = len(self.models)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, (name, model) in enumerate(self.models.items()):
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names, ax=axes[i])
            axes[i].set_title(f'{name} - Confusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.show()
    
    def feature_importance_analysis(self, feature_names):
        """Analyze feature importance for tree-based models"""
        tree_models = ['Random Forest', 'Gradient Boosting']
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        for i, model_name in enumerate(tree_models):
            if model_name in self.models:
                model = self.models[model_name]
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1][:20]  # Top 20 features
                
                axes[i].figure(figsize=(10, 6))
                axes[i].barh(range(20), importances[indices][::-1])
                axes[i].set_yticks(range(20))
                axes[i].set_yticklabels([feature_names[i] for i in indices[::-1]])
                axes[i].set_title(f'{model_name} - Top 20 Feature Importances')
                axes[i].set_xlabel('Importance')
        
        plt.tight_layout()
        plt.show()
    
    def run_complete_pipeline(self, sample_rate=22050, duration=None):
        """Run the complete predictive maintenance pipeline"""
        # Load data
        self.load_audio_data(sample_rate=sample_rate, duration=duration)
        
        if len(self.audio_data) == 0:
            print("No audio data loaded. Please check your data path and file structure.")
            return
        
        # Extract features
        self.extract_all_features()
        
        # Prepare data for ML
        X_train, X_test, y_train, y_test, feature_names = self.prepare_data_for_ml()
        
        # Train models
        self.train_models(X_train, y_train)
        
        # Evaluate models
        results = self.evaluate_models(X_test, y_test)
        
        # Visualizations
        self.plot_confusion_matrices(X_test, y_test)
        self.feature_importance_analysis(feature_names)
        
        return results
    
    def predict_condition(self, audio_file_path, model_name='Random Forest'):
        """Predict condition for a new audio file"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained.")
        
        # Load audio
        y, sr = librosa.load(audio_file_path, sr=22050)
        
        # Extract features
        features = {}
        features.update(self.extract_time_domain_features(y))
        features.update(self.extract_frequency_domain_features(y, sr))
        features.update(self.extract_mfcc_features(y, sr))
        features.update(self.extract_advanced_features(y, sr))
        
        # Create feature vector
        feature_vector = pd.DataFrame([features])
        feature_vector = feature_vector.reindex(columns=self.features_df.drop(['condition', 'filename'], axis=1).columns, fill_value=0)
        feature_vector = feature_vector.replace([np.inf, -np.inf], np.nan)
        feature_vector = feature_vector.fillna(0)
        
        # Scale features
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        # Predict
        prediction = self.models[model_name].predict(feature_vector_scaled)[0]
        prediction_proba = self.models[model_name].predict_proba(feature_vector_scaled)[0]
        
        # Decode prediction
        condition = self.label_encoder.inverse_transform([prediction])[0]
        
        return {
            'predicted_condition': condition,
            'confidence': max(prediction_proba),
            'all_probabilities': dict(zip(self.label_encoder.classes_, prediction_proba))
        }

# Example usage
if __name__ == "__main__":
    # Initialize the system
    data_path = "path/to/your/audio/data"  # Update this path
    
    # Create instance
    pm_system = AudioPredictiveMaintenance(data_path)
    
    # Run complete pipeline
    # results = pm_system.run_complete_pipeline(sample_rate=22050, duration=10)
    
    # For prediction on new audio file:
    # prediction = pm_system.predict_condition("path/to/new/audio/file.mp3")
    # print(f"Predicted condition: {prediction['predicted_condition']}")
    # print(f"Confidence: {prediction['confidence']:.4f}")