"""Live audio inspection for continuous predictive maintenance monitoring.

This module provides functions to simulate real-time audio streaming and continuous
model inference for predictive maintenance scenarios.
"""
import warnings
# Comprehensive librosa warning suppression
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')
warnings.filterwarnings('ignore', category=FutureWarning, module='librosa')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='librosa')
warnings.filterwarnings('ignore', message='.*PySoundFile failed.*')
warnings.filterwarnings('ignore', message='.*librosa.*')
warnings.filterwarnings('ignore', message='.*audioread.*')
warnings.filterwarnings('ignore', message='.*soundfile.*')

import numpy as np
import threading
import time
import queue
from typing import Optional, Callable, Dict, Any, List
from pathlib import Path
import sounddevice as sd
from collections import deque
import pandas as pd

# Import feature extraction components
try:
    import sys
    from pathlib import Path
    repo_root = Path(__file__).parents[3]  # Go up to repo root
    pipeline_path = repo_root / 'Preparation' / 'Sample Preparation' / 'Feature_extraction_pipeline'
    sys.path.append(str(pipeline_path))
    
    from splitters import segment_region
    from rich_features import extract_features_for_list
    from features_extractor import extract_features_for_list as basic_extract_features_for_list
except ImportError as e:
    print(f"Warning: Could not import feature extraction components: {e}")
    segment_region = None
    extract_features_for_list = None
    basic_extract_features_for_list = None

# Target sample rate for consistency with training
TARGET_SAMPLE_RATE = 40000


class LiveAudioInspector:
    """Real-time audio inspector for continuous predictive maintenance monitoring."""
    
    def __init__(self, 
                 model,
                 scaler,
                 segment_seconds: float = 2.0,
                 overlap: float = 0.5,
                 feature_level: str = 'standard',
                 feature_names: Optional[List[str]] = None,
                 buffer_duration: float = 10.0,
                 device: Optional[int] = None):
        """
        Initialize the live audio inspector.
        
        Args:
            model: Trained sklearn model for prediction
            scaler: Fitted StandardScaler for feature normalization
            segment_seconds: Length of each analysis segment
            overlap: Overlap fraction between segments (0.0 - 1.0)
            feature_level: Feature extraction level ('raw', 'basic', 'standard', 'advanced')
            feature_names: List of expected feature names from training
            buffer_duration: Duration of audio buffer to maintain (seconds)
            device: Audio input device index (None for default)
        """
        self.model = model
        self.scaler = scaler
        self.segment_seconds = segment_seconds
        self.overlap = overlap
        self.feature_level = feature_level
        self.feature_names = feature_names
        self.device = device
        
        # Audio buffer parameters
        self.sample_rate = TARGET_SAMPLE_RATE
        self.buffer_size = int(buffer_duration * self.sample_rate)
        self.segment_samples = int(segment_seconds * self.sample_rate)
        self.hop_samples = int(self.segment_samples * (1 - overlap))
        
        # Threading and data structures
        self.audio_buffer = deque(maxlen=self.buffer_size)
        self.prediction_queue = queue.Queue()
        self.is_running = False
        self.audio_thread = None
        self.analysis_thread = None
        
        # Results tracking
        self.results_history = []
        self.callback_function = None
        
    def set_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Set callback function to receive prediction results."""
        self.callback_function = callback
        
    def _audio_callback(self, indata, frames, time, status):
        """Callback for audio input stream."""
        if status:
            print(f"Audio callback status: {status}")
        
        # Convert to mono if multi-channel
        if indata.shape[1] > 1:
            mono_data = np.mean(indata, axis=1)
        else:
            mono_data = indata.flatten()
            
        # Add to buffer
        self.audio_buffer.extend(mono_data)
        
    def _analysis_worker(self):
        """Worker thread for continuous audio analysis."""
        last_analysis_time = 0
        hop_duration = self.hop_samples / self.sample_rate
        
        while self.is_running:
            current_time = time.time()
            
            # Check if it's time for next analysis
            if current_time - last_analysis_time >= hop_duration:
                if len(self.audio_buffer) >= self.segment_samples:
                    # Extract current segment
                    audio_segment = np.array(list(self.audio_buffer)[-self.segment_samples:])
                    
                    try:
                        # Extract features
                        features = self._extract_features(audio_segment)
                        
                        if features is not None and features.size > 0:
                            # Scale features
                            features_scaled = self.scaler.transform(features.reshape(1, -1))
                            
                            # Make prediction
                            prediction = self.model.predict(features_scaled)[0]
                            
                            # Get prediction probabilities if available
                            if hasattr(self.model, 'predict_proba'):
                                probabilities = self.model.predict_proba(features_scaled)[0]
                                confidence = np.max(probabilities)
                            else:
                                probabilities = None
                                confidence = 1.0
                            
                            # Create result
                            result = {
                                'timestamp': current_time,
                                'prediction': prediction,
                                'confidence': confidence,
                                'probabilities': probabilities,
                                'segment_duration': self.segment_seconds,
                                'buffer_length': len(self.audio_buffer)
                            }
                            
                            # Store result
                            self.results_history.append(result)
                            
                            # Call callback if set
                            if self.callback_function:
                                self.callback_function(result)
                                
                    except Exception as e:
                        print(f"Analysis error: {e}")
                        
                    last_analysis_time = current_time
                    
            time.sleep(0.01)  # Small sleep to prevent busy waiting
            
    def _extract_features(self, audio_segment: np.ndarray) -> Optional[np.ndarray]:
        """Extract features from audio segment."""
        import warnings
        
        try:
            # Suppress librosa warnings during feature extraction
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning, module='librosa')
                warnings.filterwarnings('ignore', category=FutureWarning, module='librosa')
                warnings.filterwarnings('ignore', message='.*PySoundFile failed.*')
                
                # Use appropriate feature extractor based on level
                if self.feature_level in ['raw', 'basic', 'standard', 'advanced'] and extract_features_for_list is not None:
                    features, names = extract_features_for_list([audio_segment], self.sample_rate, level=self.feature_level)
                    if features.shape[0] > 0:
                        feature_vector = features[0]
                    else:
                        return None
                elif basic_extract_features_for_list is not None:
                    features, names = basic_extract_features_for_list([audio_segment], self.sample_rate)
                    if features.shape[0] > 0:
                        feature_vector = features[0]
                    else:
                        return None
                else:
                    # Fallback to basic time-domain features if imports failed
                    feature_vector = self._basic_features(audio_segment)
                    
                # Ensure feature alignment with training data
                if self.feature_names and len(feature_vector) != len(self.feature_names):
                    print(f"Warning: Feature count mismatch. Expected {len(self.feature_names)}, got {len(feature_vector)}")
                    
                return feature_vector
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None
            
    def _basic_features(self, audio_segment: np.ndarray) -> np.ndarray:
        """Fallback basic feature extraction."""
        features = []
        
        # Time domain statistics
        features.extend([
            np.mean(audio_segment),
            np.std(audio_segment),
            np.var(audio_segment),
            np.median(audio_segment),
            np.min(audio_segment),
            np.max(audio_segment),
            np.sqrt(np.mean(audio_segment**2)),  # RMS
            np.sum(audio_segment**2),  # Energy
        ])
        
        # Zero crossing rate
        zcr = np.sum(np.diff(np.sign(audio_segment)) != 0) / (2 * len(audio_segment))
        features.append(zcr)
        
        return np.array(features)
        
    def start(self):
        """Start live audio inspection."""
        if self.is_running:
            print("Inspector is already running")
            return
            
        print(f"Starting live audio inspection...")
        print(f"Device: {self.device}, Sample rate: {self.sample_rate} Hz")
        print(f"Segment: {self.segment_seconds}s, Overlap: {self.overlap}")
        print(f"Feature level: {self.feature_level}")
        
        self.is_running = True
        
        # Start analysis thread first
        self.analysis_thread = threading.Thread(target=self._analysis_worker)
        self.analysis_thread.start()
        
        # Start audio stream
        try:
            self.stream = sd.InputStream(
                callback=self._audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=1024,
                device=self.device
            )
            self.stream.start()
            print("Audio stream started successfully")
            
        except Exception as e:
            print(f"Failed to start audio stream: {e}")
            self.is_running = False
            if self.analysis_thread:
                self.analysis_thread.join()
            raise
            
    def stop(self):
        """Stop live audio inspection."""
        if not self.is_running:
            print("Inspector is not running")
            return
            
        print("Stopping live audio inspection...")
        self.is_running = False
        
        # Stop audio stream
        if hasattr(self, 'stream') and self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                print(f"Error stopping audio stream: {e}")
                
        # Wait for analysis thread to finish
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.analysis_thread.join(timeout=2.0)
            
        print("Live audio inspection stopped")
        
    def get_results_dataframe(self) -> pd.DataFrame:
        """Get results as a pandas DataFrame."""
        if not self.results_history:
            return pd.DataFrame()
            
        df_data = []
        for result in self.results_history:
            row = {
                'timestamp': result['timestamp'],
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'buffer_length': result['buffer_length']
            }
            
            # Add probability columns if available
            if result['probabilities'] is not None:
                classes = getattr(self.model, 'classes_', None)
                if classes is not None:
                    for i, cls in enumerate(classes):
                        row[f'prob_{cls}'] = result['probabilities'][i]
                        
            df_data.append(row)
            
        return pd.DataFrame(df_data)
        
    def clear_history(self):
        """Clear prediction history."""
        self.results_history.clear()


def create_live_inspector_ui(inspector: LiveAudioInspector):
    """Create advanced industrial-grade UI for live audio inspection with threshold monitoring."""
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    import matplotlib.pyplot as plt
    import threading
    import time
    import warnings
    import numpy as np
    
    # Enhanced UI styling - Industrial theme
    header_style = {
        'background-color': '#1e3a8a',
        'color': 'white',
        'padding': '15px',
        'border-radius': '8px',
        'box-shadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
        'text-align': 'center',
        'font-family': 'Arial, sans-serif'
    }
    
    panel_style = {
        'background-color': '#f8fafc',
        'border': '2px solid #e2e8f0',
        'border-radius': '8px',
        'padding': '15px',
        'margin': '10px 0',
        'box-shadow': '0 2px 4px rgba(0, 0, 0, 0.1)'
    }
    
    # Enhanced control buttons with professional styling
    start_button = widgets.Button(
        description='▶️ START MONITORING',
        button_style='success',
        layout=widgets.Layout(width='180px', height='45px'),
        style={'font_weight': 'bold', 'font_size': '14px'}
    )
    
    stop_button = widgets.Button(
        description='⏹️ STOP MONITORING',
        button_style='danger',
        layout=widgets.Layout(width='180px', height='45px'),
        style={'font_weight': 'bold', 'font_size': '14px'}
    )
    
    clear_button = widgets.Button(
        description='🗑️ CLEAR DATA',
        button_style='warning',
        layout=widgets.Layout(width='150px', height='45px'),
        style={'font_weight': 'bold', 'font_size': '14px'}
    )
    
    # Emergency stop button
    emergency_button = widgets.Button(
        description='� EMERGENCY STOP',
        button_style='danger',
        layout=widgets.Layout(width='180px', height='45px'),
        style={'font_weight': 'bold', 'font_size': '14px', 'button_color': '#dc2626'}
    )
    
    # Confidence threshold controls with industrial styling
    threshold_slider = widgets.FloatSlider(
        value=0.70,
        min=0.0,
        max=1.0,
        step=0.01,
        description='Confidence Threshold:',
        style={'description_width': '150px', 'handle_color': '#1e3a8a'},
        layout=widgets.Layout(width='400px', height='40px'),
        readout_format='.3f'
    )
    
    threshold_display = widgets.HTML(
        value="<div style='font-size: 16px; font-weight: bold; color: #1e3a8a;'>Current Threshold: 0.70</div>"
    )
    
    # Status indicators with LED-like styling
    system_status = widgets.HTML(
        value="<div style='font-size: 18px; font-weight: bold; color: #dc2626;'>🔴 SYSTEM OFFLINE</div>"
    )
    
    prediction_display = widgets.HTML(
        value="<div style='font-size: 16px; font-weight: bold; color: #64748b;'>Current Prediction: <span style='color: #475569;'>STANDBY</span></div>"
    )
    
    confidence_display = widgets.HTML(
        value="<div style='font-size: 16px; font-weight: bold; color: #64748b;'>Confidence Level: <span style='color: #475569;'>N/A</span></div>"
    )
    
    # Alert system
    alert_display = widgets.HTML(
        value="",
        layout=widgets.Layout(margin='5px 0', min_height='40px')
    )
    
    # Metrics counters
    metrics_display = widgets.HTML(
        value="""
        <div style='background-color: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 6px; padding: 10px; margin: 5px 0;'>
            <div style='display: flex; justify-content: space-between; font-family: monospace;'>
                <span><b>Total Predictions:</b> 0</span>
                <span><b>Low Confidence Alerts:</b> 0</span>
                <span><b>Uptime:</b> 00:00:00</span>
            </div>
        </div>
        """
    )
    
    output_area = widgets.Output()
    plot_output = widgets.Output()
    
    # Enhanced state tracking
    monitoring_state = {
        "is_running": False,
        "start_time": None,
        "total_predictions": 0,
        "low_confidence_count": 0,
        "current_prediction": {"value": "STANDBY", "confidence": 0.0},
        "last_alert_time": 0
    }
    
    plot_update_thread = None
    stop_plotting = False
    metrics_update_thread = None
    stop_metrics = False
    
    def update_threshold_display():
        """Update threshold display when slider changes."""
        threshold_value = threshold_slider.value
        threshold_display.value = f"<div style='font-size: 16px; font-weight: bold; color: #1e3a8a;'>Current Threshold: {threshold_value:.3f}</div>"
    
    def update_metrics():
        """Update system metrics display."""
        while not stop_metrics and monitoring_state["is_running"]:
            if monitoring_state["start_time"]:
                uptime_seconds = int(time.time() - monitoring_state["start_time"])
                hours = uptime_seconds // 3600
                minutes = (uptime_seconds % 3600) // 60
                seconds = uptime_seconds % 60
                uptime_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                
                metrics_display.value = f"""
                <div style='background-color: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 6px; padding: 10px; margin: 5px 0;'>
                    <div style='display: flex; justify-content: space-between; font-family: monospace; font-size: 14px;'>
                        <span><b>Total Predictions:</b> {monitoring_state['total_predictions']}</span>
                        <span><b>Low Confidence Alerts:</b> {monitoring_state['low_confidence_count']}</span>
                        <span><b>Uptime:</b> {uptime_str}</span>
                    </div>
                </div>
                """
            time.sleep(1.0)
    
    def update_display():
        """Update the enhanced real-time display with threshold indicators."""
        with plot_output:
            clear_output(wait=True)
            
            df = inspector.get_results_dataframe()
            if not df.empty and len(df) > 1:
                # Suppress matplotlib warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    # Enhanced plotting with industrial styling
                    plt.style.use('default')
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
                    fig.patch.set_facecolor('#f8fafc')
                    
                    # Convert timestamp to relative time
                    df['relative_time'] = df['timestamp'] - df['timestamp'].iloc[0]
                    threshold_val = threshold_slider.value
                    
                    # Plot predictions over time with enhanced color scheme
                    unique_preds = df['prediction'].unique()
                    
                    # Industrial color mapping
                    color_map = {}
                    fault_colors = ['#dc2626', '#ea580c', '#7c2d12', '#991b1b', '#450a0a']
                    fault_idx = 0
                    
                    for pred in unique_preds:
                        if pred.lower() in ['good', 'normal', 'healthy', 'ok']:
                            color_map[pred] = '#16a34a'  # Professional green
                        else:
                            color_map[pred] = fault_colors[fault_idx % len(fault_colors)]
                            fault_idx += 1
                    
                    # Enhanced scatter plot with threshold zones
                    for pred in unique_preds:
                        mask = df['prediction'] == pred
                        confidence_values = df[mask]['confidence']
                        time_values = df[mask]['relative_time']
                        
                        # Separate points above and below threshold
                        above_threshold = confidence_values >= threshold_val
                        below_threshold = confidence_values < threshold_val
                        
                        if above_threshold.any():
                            ax1.scatter(time_values[above_threshold], confidence_values[above_threshold], 
                                      c=color_map[pred], label=f'{pred} (Above Threshold)', 
                                      s=40, alpha=0.8, edgecolors='white', linewidth=1)
                        
                        if below_threshold.any():
                            ax1.scatter(time_values[below_threshold], confidence_values[below_threshold], 
                                      c=color_map[pred], label=f'{pred} (Below Threshold)', 
                                      s=40, alpha=0.8, marker='x', linewidth=2)
                    
                    # Add threshold line and zones
                    ax1.axhline(y=threshold_val, color='#dc2626', linestyle='--', linewidth=3, 
                               label=f'Threshold ({threshold_val:.3f})', alpha=0.9)
                    
                    # Color zones
                    ax1.fill_between(df['relative_time'], 0, threshold_val, 
                                   alpha=0.1, color='red', label='Critical Zone')
                    ax1.fill_between(df['relative_time'], threshold_val, 1, 
                                   alpha=0.1, color='green', label='Safe Zone')
                    
                    ax1.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
                    ax1.set_ylabel('Confidence Level', fontsize=12, fontweight='bold')
                    ax1.set_title('🔍 Real-Time Prediction Timeline with Threshold Monitoring', 
                                fontsize=14, fontweight='bold', color='#1e3a8a')
                    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
                    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                    ax1.set_ylim(-0.05, 1.05)
                    ax1.set_facecolor('#ffffff')
                    
                    # Enhanced confidence trend with moving average
                    conf_values = df['confidence'].values
                    time_values = df['relative_time'].values
                    
                    # Calculate moving average for trend
                    window_size = min(5, len(conf_values))
                    if len(conf_values) >= window_size:
                        moving_avg = np.convolve(conf_values, np.ones(window_size)/window_size, mode='valid')
                        moving_time = time_values[window_size-1:]
                        ax2.plot(moving_time, moving_avg, 'b-', linewidth=2, label='Confidence Trend')
                    
                    ax2.plot(time_values, conf_values, 'lightblue', linewidth=1, alpha=0.7, label='Raw Confidence')
                    ax2.fill_between(time_values, conf_values, alpha=0.2, color='blue')
                    
                    # Add threshold line to confidence plot
                    ax2.axhline(y=threshold_val, color='#dc2626', linestyle='--', linewidth=3, 
                               label=f'Threshold ({threshold_val:.3f})', alpha=0.9)
                    
                    # Highlight low confidence regions
                    below_threshold_mask = conf_values < threshold_val
                    if np.any(below_threshold_mask):
                        ax2.fill_between(time_values, 0, conf_values, 
                                       where=below_threshold_mask, alpha=0.3, color='red',
                                       label='Low Confidence Periods')
                    
                    ax2.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
                    ax2.set_ylabel('Confidence Level', fontsize=12, fontweight='bold')
                    ax2.set_title('📊 Confidence Trend Analysis with Alert Zones', 
                                fontsize=14, fontweight='bold', color='#1e3a8a')
                    ax2.legend(fontsize=10)
                    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                    ax2.set_ylim(-0.05, 1.05)
                    ax2.set_facecolor('#ffffff')
                    
                    # Add latest value annotation with enhanced styling
                    if not df.empty:
                        latest = df.iloc[-1]
                        status_color = '#16a34a' if latest["confidence"] >= threshold_val else '#dc2626'
                        status_text = 'NORMAL' if latest["confidence"] >= threshold_val else 'ALERT'
                        
                        ax1.annotate(f'LATEST: {latest["prediction"]}\nConfidence: {latest["confidence"]:.3f}\nStatus: {status_text}',
                                   xy=(latest['relative_time'], latest['confidence']),
                                   xytext=(20, 20), textcoords='offset points',
                                   bbox=dict(boxstyle='round,pad=0.5', facecolor=status_color, alpha=0.8, edgecolor='white'),
                                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', 
                                                 color=status_color, linewidth=2),
                                   fontsize=10, fontweight='bold', color='white')
                    
                    plt.tight_layout()
                    plt.subplots_adjust(right=0.85)
                    plt.show()
    
    def continuous_plot_updater():
        """Continuously update plots while inspection is running."""
        while not stop_plotting and inspector.is_running:
            if len(inspector.results_history) > 0:
                update_display()
            time.sleep(1.5)  # Faster updates for industrial monitoring
                
    def prediction_callback(result):
        """Enhanced callback for new predictions with threshold monitoring."""
        monitoring_state["current_prediction"]["value"] = result['prediction']
        monitoring_state["current_prediction"]["confidence"] = result['confidence']
        monitoring_state["total_predictions"] += 1
        
        threshold_val = threshold_slider.value
        confidence = result['confidence']
        prediction = result['prediction']
        
        # Determine status colors and alerts
        if confidence >= threshold_val:
            pred_color = '#16a34a'  # Green for good confidence
            status_icon = '🟢'
            status_text = 'NORMAL'
        else:
            pred_color = '#dc2626'  # Red for low confidence
            status_icon = '🔴'
            status_text = 'ALERT'
            monitoring_state["low_confidence_count"] += 1
            
            # Generate alert message
            current_time = time.time()
            if current_time - monitoring_state["last_alert_time"] > 2.0:  # Prevent alert spam
                alert_display.value = f"""
                <div style='background-color: #fef2f2; border: 2px solid #dc2626; border-radius: 8px; padding: 15px; margin: 10px 0; animation: pulse 2s infinite;'>
                    <div style='color: #dc2626; font-weight: bold; font-size: 16px; text-align: center;'>
                        🚨 LOW CONFIDENCE ALERT 🚨<br/>
                        <span style='font-size: 14px;'>Confidence: {confidence:.3f} &lt; Threshold: {threshold_val:.3f}</span><br/>
                        <span style='font-size: 14px; color: #991b1b;'>⚠️ Low confidence prediction - Please check hardware or model ⚠️</span>
                    </div>
                </div>
                """
                monitoring_state["last_alert_time"] = current_time
        
        # Update status displays
        prediction_display.value = f"""
        <div style='font-size: 16px; font-weight: bold; color: #1e40af;'>
            Current Prediction: <span style='color: {pred_color}; font-size: 18px;'>{status_icon} {prediction}</span>
        </div>
        """
        
        confidence_display.value = f"""
        <div style='font-size: 16px; font-weight: bold; color: #1e40af;'>
            Confidence Level: <span style='color: {pred_color}; font-size: 18px;'>{confidence:.3f}</span>
            <span style='color: #64748b; font-size: 14px;'>({status_text})</span>
        </div>
        """
    
    # Event handlers with enhanced functionality
    def on_threshold_change(change):
        """Handle threshold slider changes."""
        update_threshold_display()
        alert_display.value = ""  # Clear alerts when threshold changes
    
    def on_start_clicked(b):
        nonlocal plot_update_thread, stop_plotting, metrics_update_thread, stop_metrics
        
        try:
            with output_area:
                print("🚀 Initializing Industrial Monitoring System...")
                print("🔧 Starting audio capture and analysis pipeline...")
            
            # Reset monitoring state
            monitoring_state["is_running"] = True
            monitoring_state["start_time"] = time.time()
            monitoring_state["total_predictions"] = 0
            monitoring_state["low_confidence_count"] = 0
            monitoring_state["last_alert_time"] = 0
            alert_display.value = ""
            
            inspector.set_callback(prediction_callback)
            inspector.start()
            
            system_status.value = "<div style='font-size: 18px; font-weight: bold; color: #16a34a;'>🟢 SYSTEM ONLINE</div>"
            start_button.disabled = True
            stop_button.disabled = False
            emergency_button.disabled = False
            
            # Start enhanced monitoring threads
            stop_plotting = False
            stop_metrics = False
            
            plot_update_thread = threading.Thread(target=continuous_plot_updater)
            plot_update_thread.daemon = True
            plot_update_thread.start()
            
            metrics_update_thread = threading.Thread(target=update_metrics)
            metrics_update_thread.daemon = True
            metrics_update_thread.start()
            
            with output_area:
                print("✅ Industrial Monitoring System ONLINE!")
                print("📡 Real-time audio analysis active")
                print("🎯 Threshold monitoring enabled")
                print("📊 Metrics tracking initiated")
                
        except Exception as e:
            with output_area:
                print(f"❌ SYSTEM STARTUP FAILED: {e}")
                print("🔧 TROUBLESHOOTING GUIDE:")
                print("  • Verify microphone connection and permissions")
                print("  • Check audio device availability in system settings")
                print("  • Ensure no other applications are using the audio device")
                print("  • Try file-based simulation for testing")
            system_status.value = "<div style='font-size: 18px; font-weight: bold; color: #dc2626;'>🔴 SYSTEM ERROR</div>"
        
    def on_stop_clicked(b):
        nonlocal stop_plotting, stop_metrics
        
        try:
            with output_area:
                print("🛑 Initiating system shutdown...")
            
            monitoring_state["is_running"] = False
            
            # Stop all monitoring threads
            stop_plotting = True
            stop_metrics = True
            
            if plot_update_thread and plot_update_thread.is_alive():
                plot_update_thread.join(timeout=2.0)
            if metrics_update_thread and metrics_update_thread.is_alive():
                metrics_update_thread.join(timeout=2.0)
                
            inspector.stop()
            
            system_status.value = "<div style='font-size: 18px; font-weight: bold; color: #dc2626;'>🔴 SYSTEM OFFLINE</div>"
            start_button.disabled = False
            stop_button.disabled = True
            emergency_button.disabled = True
            
            prediction_display.value = "<div style='font-size: 16px; font-weight: bold; color: #64748b;'>Current Prediction: <span style='color: #475569;'>STANDBY</span></div>"
            confidence_display.value = "<div style='font-size: 16px; font-weight: bold; color: #64748b;'>Confidence Level: <span style='color: #475569;'>N/A</span></div>"
            alert_display.value = ""
            
            update_display()  # Final update
            
            with output_area:
                print("✅ System shutdown completed successfully")
                print(f"📊 Session Summary:")
                print(f"   - Total Predictions: {monitoring_state['total_predictions']}")
                print(f"   - Low Confidence Alerts: {monitoring_state['low_confidence_count']}")
                if monitoring_state["start_time"]:
                    session_time = int(time.time() - monitoring_state["start_time"])
                    print(f"   - Session Duration: {session_time//60}m {session_time%60}s")
                
        except Exception as e:
            with output_area:
                print(f"⚠️ Error during shutdown: {e}")
    
    def on_emergency_stop_clicked(b):
        """Emergency stop handler."""
        nonlocal stop_plotting, stop_metrics
        
        with output_area:
            print("🚨 EMERGENCY STOP ACTIVATED 🚨")
        
        # Immediate shutdown
        monitoring_state["is_running"] = False
        stop_plotting = True
        stop_metrics = True
        
        try:
            inspector.stop()
        except:
            pass
        
        system_status.value = "<div style='font-size: 18px; font-weight: bold; color: #dc2626;'>🚨 EMERGENCY STOP</div>"
        alert_display.value = """
        <div style='background-color: #fef2f2; border: 2px solid #dc2626; border-radius: 8px; padding: 15px; margin: 10px 0;'>
            <div style='color: #dc2626; font-weight: bold; font-size: 16px; text-align: center;'>
                🚨 EMERGENCY STOP ACTIVATED 🚨<br/>
                <span style='font-size: 14px;'>System halted for safety</span>
            </div>
        </div>
        """
        
        start_button.disabled = False
        stop_button.disabled = True
        emergency_button.disabled = True
    
    def on_clear_clicked(b):
        """Enhanced clear function."""
        inspector.clear_history()
        monitoring_state["current_prediction"] = {"value": "STANDBY", "confidence": 0.0}
        monitoring_state["total_predictions"] = 0
        monitoring_state["low_confidence_count"] = 0
        
        prediction_display.value = "<div style='font-size: 16px; font-weight: bold; color: #64748b;'>Current Prediction: <span style='color: #475569;'>STANDBY</span></div>"
        confidence_display.value = "<div style='font-size: 16px; font-weight: bold; color: #64748b;'>Confidence Level: <span style='color: #475569;'>N/A</span></div>"
        alert_display.value = ""
        
        with plot_output:
            clear_output()
        
        with output_area:
            print("🗑️ Data cleared - System ready for new monitoring session")
            
    # Connect event handlers
    threshold_slider.observe(on_threshold_change, names='value')
    start_button.on_click(on_start_clicked)
    stop_button.on_click(on_stop_clicked)
    emergency_button.on_click(on_emergency_stop_clicked)
    clear_button.on_click(on_clear_clicked)
    
    # Initial states
    stop_button.disabled = True
    emergency_button.disabled = True
    update_threshold_display()
    
    # Enhanced layout with industrial design
    header = widgets.HTML(
        value="""
        <div style='background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 100%); 
                    color: white; padding: 20px; border-radius: 12px; 
                    box-shadow: 0 8px 25px rgba(0,0,0,0.15); text-align: center; 
                    font-family: Arial, sans-serif; margin-bottom: 20px;'>
            <h2 style='margin: 0; font-size: 24px; font-weight: bold;'>
                🏭 INDUSTRIAL PREDICTIVE MAINTENANCE MONITOR
            </h2>
            <p style='margin: 5px 0 0 0; font-size: 14px; opacity: 0.9;'>
                Advanced Machine Learning-Based Fault Detection System
            </p>
        </div>
        """
    )
    
    # Control panel with professional styling
    control_panel = widgets.VBox([
        widgets.HTML("<div style='font-size: 16px; font-weight: bold; color: #1e3a8a; margin-bottom: 10px;'>🎛️ SYSTEM CONTROLS</div>"),
        widgets.HBox([start_button, stop_button, emergency_button, clear_button], 
                     layout=widgets.Layout(justify_content='space-around'))
    ], layout=widgets.Layout(
        border='2px solid #e2e8f0', border_radius='8px', 
        padding='15px', margin='10px 0', background_color='#f8fafc'
    ))
    
    # Threshold configuration panel
    threshold_panel = widgets.VBox([
        widgets.HTML("<div style='font-size: 16px; font-weight: bold; color: #1e3a8a; margin-bottom: 10px;'>⚙️ THRESHOLD CONFIGURATION</div>"),
        threshold_slider,
        threshold_display
    ], layout=widgets.Layout(
        border='2px solid #e2e8f0', border_radius='8px', 
        padding='15px', margin='10px 0', background_color='#f8fafc'
    ))
    
    # Status monitoring panel
    status_panel = widgets.VBox([
        widgets.HTML("<div style='font-size: 16px; font-weight: bold; color: #1e3a8a; margin-bottom: 10px;'>📊 SYSTEM STATUS</div>"),
        system_status,
        prediction_display,
        confidence_display,
        metrics_display
    ], layout=widgets.Layout(
        border='2px solid #e2e8f0', border_radius='8px', 
        padding='15px', margin='10px 0', background_color='#f8fafc'
    ))
    
    # Alert panel
    alert_panel = widgets.VBox([
        widgets.HTML("<div style='font-size: 16px; font-weight: bold; color: #dc2626; margin-bottom: 10px;'>🚨 ALERT SYSTEM</div>"),
        alert_display
    ], layout=widgets.Layout(
        border='2px solid #e2e8f0', border_radius='8px', 
        padding='15px', margin='10px 0', background_color='#f8fafc'
    ))
    
    # Enhanced instructions
    instructions = widgets.HTML("""
    <div style='background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); 
                border: 2px solid #0ea5e9; border-radius: 8px; padding: 15px; margin: 10px 0;'>
        <div style='font-size: 16px; font-weight: bold; color: #0c4a6e; margin-bottom: 10px;'>
            📋 OPERATING INSTRUCTIONS
        </div>
        <div style='font-size: 14px; color: #0e7490; line-height: 1.6;'>
            <b>🚀 System Operation:</b><br/>
            • Set confidence threshold using the slider (recommended: 0.70)<br/>
            • Click <b>START MONITORING</b> to begin real-time analysis<br/>
            • Monitor threshold line on plots for safety compliance<br/>
            • Watch for low confidence alerts and system recommendations<br/>
            • Use <b>EMERGENCY STOP</b> for immediate shutdown if needed<br/><br/>
            
            <b>🎯 Alert System:</b><br/>
            • <span style='color: #16a34a;'>🟢 Green Zone:</span> Confidence ≥ Threshold (Normal Operation)<br/>
            • <span style='color: #dc2626;'>🔴 Red Zone:</span> Confidence &lt; Threshold (Alert Condition)<br/>
            • System will generate alerts for low confidence predictions<br/>
            • Metrics tracking provides session statistics and performance data
        </div>
    </div>
    """)
    
    # Main UI layout
    ui = widgets.VBox([
        header,
        instructions,
        widgets.HBox([
            widgets.VBox([control_panel, threshold_panel], layout=widgets.Layout(width='50%')),
            widgets.VBox([status_panel, alert_panel], layout=widgets.Layout(width='50%'))
        ]),
        widgets.HTML("<div style='font-size: 16px; font-weight: bold; color: #1e3a8a; margin: 20px 0 10px 0; text-align: center;'>📈 REAL-TIME MONITORING DASHBOARD</div>"),
        plot_output,
        widgets.HTML("<div style='font-size: 14px; font-weight: bold; color: #64748b; margin: 10px 0; text-align: center;'>📋 SYSTEM LOG</div>"),
        output_area
    ], layout=widgets.Layout(width='100%'))
    
    return ui


def simulate_streaming_from_file(file_path: Path, 
                                inspector: LiveAudioInspector,
                                playback_speed: float = 1.0,
                                chunk_duration: float = 0.1) -> pd.DataFrame:
    """
    Simulate real-time streaming by reading from an audio file.
    
    Args:
        file_path: Path to audio file to simulate streaming from
        inspector: Live inspector instance (should be configured but not started)
        playback_speed: Speed multiplier for simulation (1.0 = real-time)
        chunk_duration: Duration of each chunk to stream (seconds)
        
    Returns:
        DataFrame with prediction results
    """
    import warnings
    
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("soundfile is required for file simulation")
    
    # Suppress warnings during simulation
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning, module='librosa')
        warnings.filterwarnings('ignore', category=FutureWarning, module='librosa')
        warnings.filterwarnings('ignore', message='.*PySoundFile failed.*')
        
        print(f"Simulating streaming from: {file_path}")
        
        # Load audio file
        audio_data, original_sr = sf.read(file_path, dtype=np.float32)
    
        # Resample if needed
        if original_sr != TARGET_SAMPLE_RATE:
            from scipy import signal
            audio_data = signal.resample(audio_data, int(len(audio_data) * TARGET_SAMPLE_RATE / original_sr))
            
        # Convert to mono if needed
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
            
        # Streaming parameters
        chunk_samples = int(chunk_duration * TARGET_SAMPLE_RATE)
        chunk_delay = chunk_duration / playback_speed
        
        # Set up callback to collect results
        results = []
        def collect_results(result):
            results.append(result)
            print(f"Time: {result['timestamp']:.1f}s | Prediction: {result['prediction']} | Confidence: {result['confidence']:.3f}")
            
        inspector.set_callback(collect_results)
        
        # Start inspector (but don't start audio stream)
        inspector.is_running = True
        inspector.analysis_thread = threading.Thread(target=inspector._analysis_worker)
        inspector.analysis_thread.start()
        
        try:
            # Stream audio data in chunks
            total_chunks = len(audio_data) // chunk_samples
            print(f"Streaming {total_chunks} chunks of {chunk_duration}s each...")
            
            for i in range(total_chunks):
                start_idx = i * chunk_samples
                end_idx = start_idx + chunk_samples
                chunk = audio_data[start_idx:end_idx]
                
                # Add chunk to buffer
                inspector.audio_buffer.extend(chunk)
                
                # Wait for real-time simulation
                time.sleep(chunk_delay)
                
                # Progress indicator
                if i % 10 == 0:
                    progress = (i / total_chunks) * 100
                    print(f"Progress: {progress:.1f}%")
                    
        finally:
            # Stop inspector
            inspector.stop()
            
        print(f"Simulation complete. Generated {len(results)} predictions.")
        return pd.DataFrame(results) if results else pd.DataFrame()