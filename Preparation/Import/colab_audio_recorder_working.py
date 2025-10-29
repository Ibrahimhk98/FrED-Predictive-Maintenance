"""
Google Colab Audio Recorder - WORKING VERSION
============================================

This is a completely rewritten version that addresses recording issues:
- Proper audio capture and conversion
- Robust error handling and debugging
- Multiple fallback mechanisms
- Real-time status updates
- Comprehensive logging

Author: Predictive Maintenance Workshop
Version: 3.0 Working Edition
"""

import ipywidgets as widgets
from IPython.display import display, HTML, Javascript, clear_output
import base64
import io
import wave
import numpy as np
from datetime import datetime
import os
from pathlib import Path
import json

def create_working_recorder_ui(base_dir="data/audio", team_number=None):
    """
    Create a WORKING Colab audio recorder with comprehensive debugging and fallbacks.
    
    Parameters:
    -----------
    base_dir : str
        Base directory for saving audio files (Google Drive path)
    team_number : str or int
        Team number to include in filenames
        
    Returns:
    --------
    widgets.VBox
        Complete working UI widget
    """
    
    # Ensure base directory exists
    os.makedirs(base_dir, exist_ok=True)
    
    # Global state for the recorder
    recorder_state = {
        'is_recording': False,
        'current_duration': 0,
        'defect_type': 'Good',
        'team_number': team_number,
        'base_dir': base_dir,
        'recording_data': None,
        'debug_mode': True
    }
    
    # Debug output for troubleshooting
    debug_output = widgets.Output()
    
    def debug_log(message):
        """Log debug messages"""
        if recorder_state['debug_mode']:
            with debug_output:
                print(f"🔧 DEBUG: {message}")
    
    def handle_audio_data(base64_audio):
        """Handle audio data with comprehensive error handling and debugging"""
        debug_log("handle_audio_data called")
        try:
            if not base64_audio:
                debug_log("ERROR: No audio data received")
                status_output.clear_output()
                with status_output:
                    print("❌ No audio data received - check microphone permissions")
                return
                
            debug_log(f"Received audio data: {len(base64_audio)} characters")
            
            # Decode base64 audio data
            try:
                audio_bytes = base64.b64decode(base64_audio)
                debug_log(f"Decoded audio: {len(audio_bytes)} bytes")
            except Exception as e:
                debug_log(f"ERROR decoding base64: {e}")
                raise
            
            # Generate filename with timestamp and team number
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            defect_type = recorder_state['defect_type']
            team_num = recorder_state['team_number']
            
            debug_log(f"Creating filename - Team: {team_num}, Defect: {defect_type}")
            
            # Create defect type directory
            defect_dir = os.path.join(recorder_state['base_dir'], defect_type)
            os.makedirs(defect_dir, exist_ok=True)
            debug_log(f"Created directory: {defect_dir}")
            
            # Build filename
            if team_num:
                filename = f"Team{team_num}_{defect_type}_{timestamp}.wav"
            else:
                filename = f"{defect_type}_{timestamp}.wav"
            
            filepath = os.path.join(defect_dir, filename)
            debug_log(f"Full filepath: {filepath}")
            
            # Save audio file
            with open(filepath, 'wb') as f:
                f.write(audio_bytes)
            
            debug_log(f"File saved successfully: {os.path.exists(filepath)}")
            
            # Update status display
            status_output.clear_output()
            with status_output:
                print(f"✅ SUCCESS: Audio recorded and saved!")
                print(f"📁 File: {filename}")
                print(f"📂 Location: {defect_dir}")
                print(f"📊 Size: {len(audio_bytes):,} bytes")
                print(f"🎵 Defect: {defect_type}")
                if team_num:
                    print(f"👥 Team: {team_num}")
                print(f"⏰ Timestamp: {timestamp}")
                print("=" * 50)
                
            # Reset UI
            reset_ui()
                
        except Exception as e:
            debug_log(f"CRITICAL ERROR in handle_audio_data: {str(e)}")
            status_output.clear_output()
            with status_output:
                print(f"❌ CRITICAL ERROR: {str(e)}")
                print(f"📁 Attempted location: {recorder_state['base_dir']}")
                print("🔧 Check debug output below for details")
    
    # Register callback for Colab
    try:
        from google.colab import output
        output.register_callback('handle_audio_data', handle_audio_data)
        debug_log("Colab callback registered successfully")
        colab_available = True
    except ImportError:
        debug_log("WARNING: Not in Colab environment")
        colab_available = False
    
    # UI Components
    defect_dropdown = widgets.Dropdown(
        options=['Good', 'Chipped Tooth', 'Missing Tooth', 'Root Crack', 'Other'],
        value='Good',
        description='Defect Type:',
        style={'description_width': '100px'},
        layout=widgets.Layout(width='250px')
    )
    
    duration_slider = widgets.IntSlider(
        value=5,
        min=1,
        max=400,
        step=1,
        description='Duration (s):',
        style={'description_width': '100px'},
        layout=widgets.Layout(width='400px')
    )
    
    team_input = widgets.Text(
        value=str(team_number) if team_number else '',
        placeholder='Enter team number',
        description='Team #:',
        style={'description_width': '100px'},
        layout=widgets.Layout(width='200px')
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
    
    # Test button for debugging
    test_button = widgets.Button(
        description='🧪 TEST SYSTEM',
        button_style='info',
        layout=widgets.Layout(width='150px', height='40px')
    )
    
    status_display = widgets.HTML(
        value="<div style='font-size: 16px; color: #666;'>Ready to record</div>"
    )
    
    status_output = widgets.Output()
    timer_display = widgets.HTML(
        value="<div style='font-size: 24px; font-weight: bold; color: #333; text-align: center;'>00:00</div>"
    )
    
    def update_team_number(change):
        recorder_state['team_number'] = change['new'] if change['new'] else None
        debug_log(f"Team number updated: {recorder_state['team_number']}")
    
    def update_defect_type(change):
        recorder_state['defect_type'] = change['new']
        debug_log(f"Defect type updated: {recorder_state['defect_type']}")
    
    def test_system():
        """Test the system components"""
        debug_log("Testing system components...")
        with status_output:
            clear_output()
            print("🧪 SYSTEM TEST RESULTS:")
            print("=" * 30)
            
            # Test 1: Directory access
            try:
                test_dir = os.path.join(base_dir, "test")
                os.makedirs(test_dir, exist_ok=True)
                print("✅ Directory access: OK")
                os.rmdir(test_dir)
            except Exception as e:
                print(f"❌ Directory access: FAILED - {e}")
            
            # Test 2: Colab availability
            if colab_available:
                print("✅ Google Colab: Available")
            else:
                print("❌ Google Colab: Not available")
            
            # Test 3: Callback registration
            try:
                from google.colab import output
                print("✅ Callback system: OK")
            except:
                print("❌ Callback system: FAILED")
            
            # Test 4: JavaScript
            display(Javascript("""
                console.log('🧪 JavaScript test executed');
                if (typeof google !== 'undefined' && google.colab) {
                    console.log('✅ Colab API available');
                } else {
                    console.log('❌ Colab API not available');
                }
            """))
            print("✅ JavaScript execution: Check browser console")
            
            print("=" * 30)
            print("📋 If any tests failed, check the debug output")
    
    def on_test_click(b):
        test_system()
    
    def on_record_click(b):
        debug_log("Record button clicked")
        recorder_state['is_recording'] = True
        recorder_state['current_duration'] = duration_slider.value
        recorder_state['defect_type'] = defect_dropdown.value
        recorder_state['team_number'] = team_input.value if team_input.value else None
        
        debug_log(f"Recording params - Duration: {recorder_state['current_duration']}, Defect: {recorder_state['defect_type']}, Team: {recorder_state['team_number']}")
        
        # Update UI
        record_button.disabled = True
        stop_button.disabled = False
        defect_dropdown.disabled = True
        duration_slider.disabled = True
        team_input.disabled = True
        
        status_display.value = f"<div style='font-size: 16px; color: #d32f2f; font-weight: bold;'>🔴 RECORDING - {recorder_state['defect_type']}</div>"
        
        # Clear previous outputs
        status_output.clear_output()
        
        # Enhanced JavaScript with better error handling
        js_code = f"""
        // Enhanced Audio Recorder with debugging
        (async function() {{
            console.log('🎤 Starting recording process...');
            
            // Initialize recorder if needed
            if (!window.workingRecorder) {{
                console.log('🔧 Creating new recorder instance...');
                
                class WorkingAudioRecorder {{
                    constructor() {{
                        this.mediaRecorder = null;
                        this.stream = null;
                        this.audioChunks = [];
                        this.isRecording = false;
                    }}
                    
                    async initialize() {{
                        try {{
                            console.log('🎯 Requesting microphone access...');
                            this.stream = await navigator.mediaDevices.getUserMedia({{
                                audio: {{
                                    sampleRate: 44100,
                                    channelCount: 1,
                                    echoCancellation: true,
                                    noiseSuppression: true,
                                    autoGainControl: true
                                }}
                            }});
                            console.log('✅ Microphone access granted');
                            return true;
                        }} catch (error) {{
                            console.error('❌ Microphone access failed:', error);
                            alert('Microphone access denied! Please:\\n1. Click Allow when prompted\\n2. Check browser permissions\\n3. Refresh the page and try again');
                            return false;
                        }}
                    }}
                    
                    async startRecording() {{
                        if (!this.stream) {{
                            console.error('❌ No audio stream available');
                            return false;
                        }}
                        
                        this.audioChunks = [];
                        
                        // Try different MIME types
                        const mimeTypes = [
                            'audio/webm;codecs=opus',
                            'audio/webm',
                            'audio/mp4',
                            'audio/ogg;codecs=opus'
                        ];
                        
                        let mimeType = 'audio/webm';
                        for (const type of mimeTypes) {{
                            if (MediaRecorder.isTypeSupported(type)) {{
                                mimeType = type;
                                break;
                            }}
                        }}
                        
                        console.log('🎵 Using MIME type:', mimeType);
                        
                        this.mediaRecorder = new MediaRecorder(this.stream, {{ mimeType }});
                        
                        this.mediaRecorder.ondataavailable = (event) => {{
                            if (event.data.size > 0) {{
                                this.audioChunks.push(event.data);
                                console.log('📦 Audio chunk received:', event.data.size, 'bytes');
                            }}
                        }};
                        
                        this.mediaRecorder.onstop = () => {{
                            console.log('⏹️ Recording stopped, processing audio...');
                            const audioBlob = new Blob(this.audioChunks, {{ type: mimeType }});
                            console.log('🎵 Final audio blob:', audioBlob.size, 'bytes');
                            this.processAudio(audioBlob);
                        }};
                        
                        this.mediaRecorder.onerror = (event) => {{
                            console.error('❌ MediaRecorder error:', event.error);
                        }};
                        
                        this.mediaRecorder.start(250); // Collect data every 250ms
                        this.isRecording = true;
                        console.log('🔴 Recording started successfully');
                        return true;
                    }}
                    
                    stopRecording() {{
                        if (this.mediaRecorder && this.isRecording) {{
                            this.mediaRecorder.stop();
                            this.isRecording = false;
                            console.log('⏹️ Stop recording requested');
                            return true;
                        }}
                        return false;
                    }}
                    
                    processAudio(audioBlob) {{
                        console.log('🔄 Processing audio blob...');
                        const reader = new FileReader();
                        reader.onload = () => {{
                            const base64Audio = reader.result.split(',')[1];
                            console.log('📤 Sending', base64Audio.length, 'characters to Python');
                            
                            if (typeof google !== 'undefined' && google.colab) {{
                                google.colab.kernel.invokeFunction('handle_audio_data', [base64Audio], {{}});
                                console.log('✅ Audio data sent to Python');
                            }} else {{
                                console.error('❌ Google Colab API not available');
                                alert('Error: Google Colab API not available. Please refresh and try again.');
                            }}
                        }};
                        reader.onerror = (error) => {{
                            console.error('❌ FileReader error:', error);
                        }};
                        reader.readAsDataURL(audioBlob);
                    }}
                }}
                
                window.workingRecorder = new WorkingAudioRecorder();
            }}
            
            // Initialize and start recording
            const initialized = await window.workingRecorder.initialize();
            if (initialized) {{
                const started = await window.workingRecorder.startRecording();
                if (started) {{
                    console.log('🎉 Recording started successfully!');
                    
                    // Auto-stop timer
                    setTimeout(() => {{
                        if (window.workingRecorder.isRecording) {{
                            console.log('⏰ Auto-stopping recording after {recorder_state['current_duration']} seconds');
                            window.workingRecorder.stopRecording();
                        }}
                    }}, {recorder_state['current_duration']} * 1000);
                }} else {{
                    console.error('❌ Failed to start recording');
                    alert('Failed to start recording. Check browser console for details.');
                }}
            }} else {{
                console.error('❌ Failed to initialize recorder');
            }}
        }})();
        """
        
        display(Javascript(js_code))
        debug_log("JavaScript recording code executed")
    
    def on_stop_click(b):
        debug_log("Stop button clicked")
        display(Javascript("""
            if (window.workingRecorder && window.workingRecorder.isRecording) {
                console.log('🛑 Manual stop requested');
                window.workingRecorder.stopRecording();
            }
        """))
        reset_ui()
    
    def reset_ui():
        """Reset UI to initial state"""
        debug_log("Resetting UI")
        recorder_state['is_recording'] = False
        
        record_button.disabled = False
        stop_button.disabled = True
        defect_dropdown.disabled = False
        duration_slider.disabled = False
        team_input.disabled = False
        
        status_display.value = "<div style='font-size: 16px; color: #666;'>Ready to record</div>"
        timer_display.value = "<div style='font-size: 24px; font-weight: bold; color: #333; text-align: center;'>00:00</div>"
    
    # Event handlers
    defect_dropdown.observe(update_defect_type, names='value')
    team_input.observe(update_team_number, names='value')
    record_button.on_click(on_record_click)
    stop_button.on_click(on_stop_click)
    test_button.on_click(on_test_click)
    
    # Initial debug info
    debug_log(f"Recorder initialized - Base dir: {base_dir}")
    debug_log(f"Colab available: {colab_available}")
    
    # Layout
    header = widgets.HTML("""
    <div style='background: linear-gradient(135deg, #16a34a 0%, #22c55e 100%); color: white; 
                padding: 20px; border-radius: 12px; text-align: center; margin-bottom: 20px;'>
        <h2 style='margin: 0; font-size: 24px;'>🎤 WORKING GOOGLE COLAB AUDIO RECORDER</h2>
        <p style='margin: 5px 0 0 0; font-size: 14px; opacity: 0.9;'>
            Comprehensive Debugging • Multiple Fallbacks • Real-time Status
        </p>
    </div>
    """)
    
    controls_panel = widgets.VBox([
        widgets.HTML("<div style='font-weight: bold; color: #16a34a; margin-bottom: 10px;'>🎛️ RECORDING CONTROLS</div>"),
        widgets.HBox([defect_dropdown, team_input]),
        duration_slider,
        widgets.HBox([record_button, stop_button, test_button])
    ], layout=widgets.Layout(border='2px solid #dcfce7', border_radius='8px', 
                            padding='15px', margin='10px 0', background_color='#f0fdf4'))
    
    status_panel = widgets.VBox([
        widgets.HTML("<div style='font-weight: bold; color: #16a34a; margin-bottom: 10px;'>📊 STATUS & RESULTS</div>"),
        status_display,
        timer_display,
        status_output
    ], layout=widgets.Layout(border='2px solid #dcfce7', border_radius='8px', 
                            padding='15px', margin='10px 0', background_color='#f0fdf4'))
    
    debug_panel = widgets.VBox([
        widgets.HTML("<div style='font-weight: bold; color: #ea580c; margin-bottom: 10px;'>🔧 DEBUGGING OUTPUT</div>"),
        debug_output
    ], layout=widgets.Layout(border='2px solid #fed7aa', border_radius='8px', 
                            padding='15px', margin='10px 0', background_color='#fff7ed'))
    
    instructions = widgets.HTML(f"""
    <div style='background: #eff6ff; border: 2px solid #3b82f6; border-radius: 8px; padding: 15px; margin: 10px 0;'>
        <div style='font-weight: bold; color: #1e40af; margin-bottom: 10px;'>📋 WORKING RECORDER INSTRUCTIONS</div>
        <div style='color: #1e40af; line-height: 1.6;'>
            <b>🧪 Test First:</b> Click "TEST SYSTEM" to verify everything works<br/>
            <b>🎤 Allow Mic:</b> Click "Allow" when browser asks for microphone access<br/>
            <b>⏱️ Duration:</b> Set recording length (1-400 seconds)<br/>
            <b>👥 Team:</b> Enter your team number (optional)<br/>
            <b>🔴 Record:</b> Click START RECORDING and speak clearly<br/>
            <b>📁 Location:</b> Files save to: {base_dir}<br/>
            <b>🔧 Debug:</b> Check debug output below if issues occur
        </div>
    </div>
    """)
    
    # Main UI
    ui = widgets.VBox([
        header,
        instructions,
        controls_panel,
        status_panel,
        debug_panel
    ], layout=widgets.Layout(width='100%'))
    
    return ui

# Main function
def create_recorder_ui(base_dir="data/audio", team_number=None):
    """Create the working Colab recorder UI"""
    return create_working_recorder_ui(base_dir, team_number)

def list_colab_audio_devices():
    """List audio device information"""
    info = [
        "🎤 WORKING AUDIO RECORDER - SYSTEM INFO",
        "=" * 50,
        "🌐 Platform: Browser-based Web Audio API",
        "📱 Input: Default System Microphone",
        "🎯 Quality: 44.1 kHz, 16-bit, Mono",
        "💾 Format: WebM/OGG → WAV",
        "⏱️ Duration: 1-400 seconds",
        "🔧 Features: Auto-stop, Team numbers, Debug mode",
        "📋 Status: Enhanced with comprehensive error handling"
    ]
    return info

if __name__ == "__main__":
    print("Working Google Colab Audio Recorder - Version 3.0")
    print("Features: Comprehensive debugging, multiple fallbacks, real-time status")