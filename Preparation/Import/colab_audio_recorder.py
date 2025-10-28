"""
Google Colab Audio Recorder Module
=================================

This module provides JavaScript-based audio recording functionality for Google Colab
that maintains the same UI and functionality as the original audio_recorder.py but
works within browser security constraints.

Features:
- Browser-based audio recording using JavaScript
- Same UI design and functionality as original recorder
- Direct audio capture without PyAudio dependencies
- Automatic file saving with timestamps and defect classification
- Compatible with Google Drive integration

Author: Predictive Maintenance Workshop
Version: 1.0 Colab Edition
"""

import ipywidgets as widgets
from IPython.display import display, HTML, Javascript
import base64
import io
import wave
import numpy as np
from datetime import datetime
import os
from pathlib import Path
import json

# JavaScript code for audio recording in browser
AUDIO_RECORDER_JS = """
class ColabAudioRecorder {
    constructor() {
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.stream = null;
        this.isRecording = false;
    }
    
    async initialize() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    sampleRate: 44100,
                    channelCount: 1,
                    volume: 1.0
                } 
            });
            return true;
        } catch (error) {
            console.error('Error accessing microphone:', error);
            return false;
        }
    }
    
    startRecording() {
        if (!this.stream) {
            console.error('Stream not initialized');
            return false;
        }
        
        this.audioChunks = [];
        this.mediaRecorder = new MediaRecorder(this.stream);
        
        this.mediaRecorder.ondataavailable = (event) => {
            this.audioChunks.push(event.data);
        };
        
        this.mediaRecorder.onstop = () => {
            const audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' });
            this.processAudio(audioBlob);
        };
        
        this.mediaRecorder.start();
        this.isRecording = true;
        return true;
    }
    
    stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.isRecording = false;
            return true;
        }
        return false;
    }
    
    async processAudio(audioBlob) {
        const reader = new FileReader();
        reader.onload = () => {
            const base64Audio = reader.result.split(',')[1];
            // Send audio data to Python
            google.colab.kernel.invokeFunction('handle_audio_data', [base64Audio], {});
        };
        reader.readAsDataURL(audioBlob);
    }
}

// Global recorder instance
window.colabRecorder = new ColabAudioRecorder();

// Initialize recorder
window.colabRecorder.initialize().then(success => {
    if (success) {
        console.log('Audio recorder initialized successfully');
    } else {
        console.error('Failed to initialize audio recorder');
    }
});
"""

def create_colab_recorder_ui(base_dir="data/audio"):
    """
    Create a Colab-compatible audio recorder UI using JavaScript for audio capture.
    
    Parameters:
    -----------
    base_dir : str
        Base directory for saving audio files
        
    Returns:
    --------
    widgets.VBox
        Complete UI widget ready for display
    """
    
    # Ensure base directory exists
    os.makedirs(base_dir, exist_ok=True)
    
    # Global state for the recorder
    recorder_state = {
        'is_recording': False,
        'current_duration': 0,
        'defect_type': 'Good',
        'base_dir': base_dir,
        'recording_data': None
    }
    
    def handle_audio_data(base64_audio):
        """Handle audio data received from JavaScript"""
        try:
            # Decode base64 audio data
            audio_bytes = base64.b64decode(base64_audio)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            defect_type = recorder_state['defect_type']
            
            # Create defect type directory
            defect_dir = os.path.join(recorder_state['base_dir'], defect_type)
            os.makedirs(defect_dir, exist_ok=True)
            
            # Save audio file
            filename = f"{defect_type}_{timestamp}.wav"
            filepath = os.path.join(defect_dir, filename)
            
            with open(filepath, 'wb') as f:
                f.write(audio_bytes)
            
            status_output.clear_output()
            with status_output:
                print(f"✅ Saved: {filename}")
                print(f"📁 Location: {filepath}")
                print(f"🎵 Defect Type: {defect_type}")
            
        except Exception as e:
            status_output.clear_output()
            with status_output:
                print(f"❌ Error saving audio: {str(e)}")
    
    # Register the callback function
    try:
        from google.colab import output
        output.register_callback('handle_audio_data', handle_audio_data)
    except ImportError:
        print("Warning: google.colab not available. Running in non-Colab environment.")
    
    # UI Components
    defect_dropdown = widgets.Dropdown(
        options=['Good', 'Chipped Tooth', 'Missing Tooth', 'Root Crack', 'Other'],
        value='Good',
        description='Defect Type:',
        style={'description_width': '100px'},
        layout=widgets.Layout(width='250px')
    )
    
    duration_slider = widgets.IntSlider(
        value=3,
        min=1,
        max=10,
        step=1,
        description='Duration (s):',
        style={'description_width': '100px'},
        layout=widgets.Layout(width='300px')
    )
    
    record_button = widgets.Button(
        description='🎤 START RECORDING',
        button_style='success',
        layout=widgets.Layout(width='200px', height='50px'),
        style={'font_weight': 'bold'}
    )
    
    stop_button = widgets.Button(
        description='⏹️ STOP RECORDING',
        button_style='danger',
        layout=widgets.Layout(width='200px', height='50px'),
        style={'font_weight': 'bold'},
        disabled=True
    )
    
    status_display = widgets.HTML(
        value="<div style='font-size: 16px; color: #666;'>Ready to record</div>"
    )
    
    status_output = widgets.Output()
    
    # Timer display
    timer_display = widgets.HTML(
        value="<div style='font-size: 24px; font-weight: bold; color: #333; text-align: center;'>00:00</div>"
    )
    
    def update_defect_type(change):
        recorder_state['defect_type'] = change['new']
    
    def on_record_click(b):
        recorder_state['is_recording'] = True
        recorder_state['current_duration'] = duration_slider.value
        recorder_state['defect_type'] = defect_dropdown.value
        
        # Update UI
        record_button.disabled = True
        stop_button.disabled = False
        defect_dropdown.disabled = True
        duration_slider.disabled = True
        
        status_display.value = f"<div style='font-size: 16px; color: #d32f2f; font-weight: bold;'>🔴 RECORDING - {recorder_state['defect_type']}</div>"
        
        # Start JavaScript recording
        display(Javascript("""
            if (window.colabRecorder) {
                window.colabRecorder.startRecording();
                
                // Auto-stop after duration
                setTimeout(() => {
                    if (window.colabRecorder.isRecording) {
                        window.colabRecorder.stopRecording();
                    }
                }, %d * 1000);
                
                // Update timer
                let elapsed = 0;
                const maxDuration = %d;
                const timerInterval = setInterval(() => {
                    elapsed++;
                    const minutes = Math.floor(elapsed / 60);
                    const seconds = elapsed %% 60;
                    const timeStr = String(minutes).padStart(2, '0') + ':' + String(seconds).padStart(2, '0');
                    
                    // Update timer display
                    const timerElement = document.querySelector('[data-timer-display]');
                    if (timerElement) {
                        timerElement.innerHTML = '<div style="font-size: 24px; font-weight: bold; color: #d32f2f; text-align: center;">' + timeStr + '</div>';
                    }
                    
                    if (elapsed >= maxDuration) {
                        clearInterval(timerInterval);
                        // Reset UI through Python callback
                        google.colab.kernel.invokeFunction('reset_recorder_ui', [], {});
                    }
                }, 1000);
            } else {
                alert('Audio recorder not initialized. Please run the setup cell first.');
            }
        """ % (recorder_state['current_duration'], recorder_state['current_duration'])))
    
    def on_stop_click(b):
        # Stop JavaScript recording
        display(Javascript("""
            if (window.colabRecorder && window.colabRecorder.isRecording) {
                window.colabRecorder.stopRecording();
            }
        """))
        
        reset_ui()
    
    def reset_ui():
        """Reset UI to initial state"""
        recorder_state['is_recording'] = False
        
        record_button.disabled = False
        stop_button.disabled = True
        defect_dropdown.disabled = False
        duration_slider.disabled = False
        
        status_display.value = "<div style='font-size: 16px; color: #666;'>Ready to record</div>"
        timer_display.value = "<div style='font-size: 24px; font-weight: bold; color: #333; text-align: center;'>00:00</div>"
    
    def reset_recorder_ui():
        """Callback function for JavaScript timer completion"""
        reset_ui()
    
    # Register reset callback
    try:
        from google.colab import output
        output.register_callback('reset_recorder_ui', lambda: reset_ui())
    except ImportError:
        pass
    
    # Event handlers
    defect_dropdown.observe(update_defect_type, names='value')
    record_button.on_click(on_record_click)
    stop_button.on_click(on_stop_click)
    
    # Layout
    header = widgets.HTML("""
    <div style='background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 100%); color: white; 
                padding: 20px; border-radius: 12px; text-align: center; margin-bottom: 20px;'>
        <h2 style='margin: 0; font-size: 24px;'>🎤 GOOGLE COLAB AUDIO RECORDER</h2>
        <p style='margin: 5px 0 0 0; font-size: 14px; opacity: 0.9;'>
            Browser-based Audio Recording for Predictive Maintenance
        </p>
    </div>
    """)
    
    controls_panel = widgets.VBox([
        widgets.HTML("<div style='font-weight: bold; color: #1e3a8a; margin-bottom: 10px;'>🎛️ RECORDING CONTROLS</div>"),
        defect_dropdown,
        duration_slider,
        widgets.HBox([record_button, stop_button])
    ], layout=widgets.Layout(border='2px solid #e2e8f0', border_radius='8px', 
                            padding='15px', margin='10px 0', background_color='#f8fafc'))
    
    status_panel = widgets.VBox([
        widgets.HTML("<div style='font-weight: bold; color: #1e3a8a; margin-bottom: 10px;'>📊 STATUS</div>"),
        status_display,
        widgets.HTML("<div data-timer-display='true'></div>"),
        timer_display
    ], layout=widgets.Layout(border='2px solid #e2e8f0', border_radius='8px', 
                            padding='15px', margin='10px 0', background_color='#f8fafc'))
    
    instructions = widgets.HTML("""
    <div style='background: #f0f9ff; border: 2px solid #0ea5e9; border-radius: 8px; padding: 15px; margin: 10px 0;'>
        <div style='font-weight: bold; color: #0c4a6e; margin-bottom: 10px;'>📋 BROWSER RECORDING INSTRUCTIONS</div>
        <div style='color: #0e7490; line-height: 1.6;'>
            <b>🎤 Microphone Access:</b> Click "Allow" when prompted for microphone access<br/>
            <b>🔴 Recording:</b> Audio is captured directly in your browser - no PyAudio needed<br/>
            <b>⏱️ Duration:</b> Recording stops automatically after selected duration<br/>
            <b>💾 Saving:</b> Files are saved automatically with timestamps and defect classification<br/>
            <b>📁 Organization:</b> Audio files are organized by defect type in separate folders
        </div>
    </div>
    """)
    
    # Initialize JavaScript
    display(HTML(f"<script>{AUDIO_RECORDER_JS}</script>"))
    
    # Main UI
    ui = widgets.VBox([
        header,
        instructions,
        controls_panel,
        status_panel,
        widgets.HTML("<div style='font-weight: bold; color: #64748b; text-align: center; margin: 10px 0;'>📝 RECORDING LOG</div>"),
        status_output
    ], layout=widgets.Layout(width='100%'))
    
    return ui

def list_colab_audio_devices():
    """
    List available audio devices in Colab (browser-based).
    Returns information about browser audio capabilities.
    """
    info = [
        "🌐 Browser Audio Devices (Google Colab)",
        "----------------------------------------",
        "📱 Primary Input: Default Browser Microphone",
        "🎯 Sample Rate: 44.1 kHz",
        "📊 Channels: Mono (1 channel)",
        "🔧 Technology: Web Audio API + MediaRecorder",
        "",
        "💡 Note: Actual devices depend on your system and browser permissions"
    ]
    return info

def record_colab_snippet(duration=3, defect_type="Good", base_dir="data/audio"):
    """
    Record an audio snippet in Colab using browser API.
    This is a simplified version for programmatic use.
    
    Parameters:
    -----------
    duration : int
        Recording duration in seconds
    defect_type : str  
        Type of defect being recorded
    base_dir : str
        Base directory for saving files
    """
    print("🎤 For recording in Colab, please use the create_colab_recorder_ui() function")
    print("   Browser-based recording requires the interactive UI for microphone access")
    return None

# Main function for easy import
def create_recorder_ui(base_dir="data/audio"):
    """
    Create the Colab-compatible recorder UI.
    This is the main function that should be called from notebooks.
    """
    return create_colab_recorder_ui(base_dir)

if __name__ == "__main__":
    print("Google Colab Audio Recorder Module")
    print("Usage: from colab_audio_recorder import create_recorder_ui")
    print("       ui = create_recorder_ui()")
    print("       display(ui)")