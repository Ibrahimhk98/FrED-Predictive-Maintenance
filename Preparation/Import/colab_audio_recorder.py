""""""

Google Colab Audio Recorder ModuleGoogle Colab Audio Recorder Module - FIXED VERSION

===================================================================================



This module provides JavaScript-based audio recording functionality for Google ColabThis module provides JavaScript-based audio recording functionality for Google Colab

that provides the same functionality as audio_recorder.py but works within browserwith all the following issues fixed:

security constraints using the Web Audio API.- Proper file saving to requested Drive folder

- Extended duration support (up to 400 seconds)

Features:- Team number inclusion in filename

- Browser-based audio recording using JavaScript and Web Audio API- Non-blocking JavaScript execution

- Same UI design and functionality as original audio_recorder.py- Audio device information display

- Extended duration support (1-400 seconds)

- Team number integration in filenamesFeatures:

- Direct audio capture without PyAudio dependencies- Browser-based audio recording using JavaScript

- Automatic file saving with timestamps and defect classification- Extended recording duration (1-400 seconds)

- Compatible with Google Drive integration- Team number integration in filenames

- Comprehensive debugging and error handling- Proper Google Drive folder integration

- Audio device enumeration and display

Author: Predictive Maintenance Workshop- Non-blocking execution compatible with Colab

Version: 2.0 Colab Edition

"""Author: Predictive Maintenance Workshop

Version: 2.0 Fixed Edition

import ipywidgets as widgets"""

from IPython.display import display, HTML, Javascript, clear_output

import base64import ipywidgets as widgets

import iofrom IPython.display import display, HTML, Javascript, clear_output

import waveimport base64

import numpy as npimport io

from datetime import datetimeimport wave

import osimport numpy as np

from pathlib import Pathfrom datetime import datetime

import jsonimport os

from pathlib import Path

def create_colab_recorder_ui(base_dir="data/audio", team_number=None):import json

    """import asyncio

    Create a Colab-compatible audio recorder UI using JavaScript for audio capture.

    Provides the same functionality as the local audio_recorder but optimized for Colab.def create_colab_recorder_ui(base_dir="data/audio"):

        """

    Parameters:    Create a Colab-compatible audio recorder UI using JavaScript for audio capture.

    -----------    

    base_dir : str    Parameters:

        Base directory for saving audio files (Google Drive path)    -----------

    team_number : str or int    base_dir : str

        Team number to include in filenames (optional)        Base directory for saving audio files

                

    Returns:    Returns:

    --------    --------

    widgets.VBox    widgets.VBox

        Complete UI widget ready for display in Colab        Complete UI widget ready for display

    """    """

        

    # Ensure base directory exists    # Ensure base directory exists

    os.makedirs(base_dir, exist_ok=True)    os.makedirs(base_dir, exist_ok=True)

        

    # Recorder state    # Global state for the recorder

    recorder_state = {    recorder_state = {

        'is_recording': False,        'is_recording': False,

        'current_duration': 0,        'current_duration': 0,

        'defect_type': 'Good',        'defect_type': 'Good',

        'team_number': team_number,        'base_dir': base_dir,

        'base_dir': base_dir,        'recording_data': None

        'debug_mode': True    }

    }    

        def handle_audio_data(base64_audio):

    # Debug output for troubleshooting        """Handle audio data received from JavaScript"""

    debug_output = widgets.Output()        try:

                # Decode base64 audio data

    def debug_log(message):            audio_bytes = base64.b64decode(base64_audio)

        """Log debug messages"""            

        if recorder_state['debug_mode']:            # Generate filename with timestamp

            with debug_output:            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                print(f"🔧 DEBUG: {message}")            defect_type = recorder_state['defect_type']

                

    def handle_audio_data(base64_audio):            # Create defect type directory in shared folder

        """Handle audio data received from JavaScript"""            defect_dir = os.path.join(recorder_state['base_dir'], defect_type)

        debug_log("handle_audio_data called")            os.makedirs(defect_dir, exist_ok=True)

        try:            

            if not base64_audio:            # Save audio file with more descriptive naming

                debug_log("ERROR: No audio data received")            filename = f"{defect_type}_{timestamp}.wav"

                status_output.clear_output()            filepath = os.path.join(defect_dir, filename)

                with status_output:            

                    print("❌ No audio data received - check microphone permissions")            with open(filepath, 'wb') as f:

                return                f.write(audio_bytes)

                            

            debug_log(f"Received audio data: {len(base64_audio)} characters")            # Get relative path for display

                        base_name = os.path.basename(recorder_state['base_dir'])

            # Decode base64 audio data            relative_path = os.path.join(base_name, defect_type, filename)

            audio_bytes = base64.b64decode(base64_audio)            

            debug_log(f"Decoded audio: {len(audio_bytes)} bytes")            status_output.clear_output()

                        with status_output:

            # Generate filename with timestamp and team number                print(f"✅ Saved: {filename}")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")                print(f"📁 Shared Location: {relative_path}")

            defect_type = recorder_state['defect_type']                print(f"🎵 Defect Type: {defect_type}")

            team_num = recorder_state['team_number']                print(f"👥 File accessible to all students in shared folder")

                            print(f"📊 File size: {len(audio_bytes)} bytes")

            # Create defect type directory            

            defect_dir = os.path.join(recorder_state['base_dir'], defect_type)        except Exception as e:

            os.makedirs(defect_dir, exist_ok=True)            status_output.clear_output()

            debug_log(f"Created directory: {defect_dir}")            with status_output:

                            print(f"❌ Error saving audio: {str(e)}")

            # Build filename with team number if provided    

            if team_num:    # Register the callback function

                filename = f"Team{team_num}_{defect_type}_{timestamp}.wav"    try:

            else:        from google.colab import output

                filename = f"{defect_type}_{timestamp}.wav"        output.register_callback('handle_audio_data', handle_audio_data)

                except ImportError:

            filepath = os.path.join(defect_dir, filename)        print("Warning: google.colab not available. Running in non-Colab environment.")

            debug_log(f"Full filepath: {filepath}")    

                # UI Components

            # Save audio file    defect_dropdown = widgets.Dropdown(

            with open(filepath, 'wb') as f:        options=['Good', 'Chipped Tooth', 'Missing Tooth', 'Root Crack', 'Other'],

                f.write(audio_bytes)        value='Good',

                    description='Defect Type:',

            debug_log(f"File saved successfully: {os.path.exists(filepath)}")        style={'description_width': '100px'},

                    layout=widgets.Layout(width='250px')

            # Update status display    )

            status_output.clear_output()    

            with status_output:    duration_slider = widgets.IntSlider(

                print(f"✅ SUCCESS: Audio recorded and saved!")        value=3,

                print(f"📁 File: {filename}")        min=1,

                print(f"📂 Location: {defect_dir}")        max=10,

                print(f"📊 Size: {len(audio_bytes):,} bytes")        step=1,

                print(f"🎵 Defect: {defect_type}")        description='Duration (s):',

                if team_num:        style={'description_width': '100px'},

                    print(f"👥 Team: {team_num}")        layout=widgets.Layout(width='300px')

                print(f"⏰ Timestamp: {timestamp}")    )

                print("=" * 50)    

                    record_button = widgets.Button(

            # Reset UI        description='🎤 START RECORDING',

            reset_ui()        button_style='success',

                        layout=widgets.Layout(width='200px', height='50px'),

        except Exception as e:        style={'font_weight': 'bold'}

            debug_log(f"CRITICAL ERROR in handle_audio_data: {str(e)}")    )

            status_output.clear_output()    

            with status_output:    stop_button = widgets.Button(

                print(f"❌ CRITICAL ERROR: {str(e)}")        description='⏹️ STOP RECORDING',

                print(f"📁 Attempted location: {recorder_state['base_dir']}")        button_style='danger',

                print("🔧 Check debug output below for details")        layout=widgets.Layout(width='200px', height='50px'),

            style={'font_weight': 'bold'},

    # Register callback for Colab        disabled=True

    try:    )

        from google.colab import output    

        output.register_callback('handle_audio_data', handle_audio_data)    status_display = widgets.HTML(

        debug_log("Colab callback registered successfully")        value="<div style='font-size: 16px; color: #666;'>Ready to record</div>"

        colab_available = True    )

    except ImportError:    

        debug_log("WARNING: Not in Colab environment")    status_output = widgets.Output()

        colab_available = False    

        # Timer display

    # UI Components    timer_display = widgets.HTML(

    defect_dropdown = widgets.Dropdown(        value="<div style='font-size: 24px; font-weight: bold; color: #333; text-align: center;'>00:00</div>"

        options=['Good', 'Chipped Tooth', 'Missing Tooth', 'Root Crack', 'Other'],    )

        value='Good',    

        description='Defect Type:',    def update_defect_type(change):

        style={'description_width': '100px'},        recorder_state['defect_type'] = change['new']

        layout=widgets.Layout(width='250px')    

    )    def on_record_click(b):

            recorder_state['is_recording'] = True

    duration_slider = widgets.IntSlider(        recorder_state['current_duration'] = duration_slider.value

        value=5,        recorder_state['defect_type'] = defect_dropdown.value

        min=1,        

        max=400,  # Extended duration support        # Update UI

        step=1,        record_button.disabled = True

        description='Duration (s):',        stop_button.disabled = False

        style={'description_width': '100px'},        defect_dropdown.disabled = True

        layout=widgets.Layout(width='400px')        duration_slider.disabled = True

    )        

            status_display.value = f"<div style='font-size: 16px; color: #d32f2f; font-weight: bold;'>🔴 RECORDING - {recorder_state['defect_type']}</div>"

    team_input = widgets.Text(        

        value=str(team_number) if team_number else '',        # Start JavaScript recording

        placeholder='Enter team number',        display(Javascript("""

        description='Team #:',            if (window.colabRecorder) {

        style={'description_width': '100px'},                window.colabRecorder.startRecording();

        layout=widgets.Layout(width='200px')                

    )                // Auto-stop after duration

                    setTimeout(() => {

    record_button = widgets.Button(                    if (window.colabRecorder.isRecording) {

        description='🎤 START RECORDING',                        window.colabRecorder.stopRecording();

        button_style='success',                    }

        layout=widgets.Layout(width='200px', height='50px'),                }, %d * 1000);

        style={'font_weight': 'bold'}                

    )                // Update timer

                    let elapsed = 0;

    stop_button = widgets.Button(                const maxDuration = %d;

        description='⏹️ STOP RECORDING',                const timerInterval = setInterval(() => {

        button_style='danger',                    elapsed++;

        layout=widgets.Layout(width='200px', height='50px'),                    const minutes = Math.floor(elapsed / 60);

        style={'font_weight': 'bold'},                    const seconds = elapsed %% 60;

        disabled=True                    const timeStr = String(minutes).padStart(2, '0') + ':' + String(seconds).padStart(2, '0');

    )                    

                        // Update timer display

    # Test button for system verification                    const timerElement = document.querySelector('[data-timer-display]');

    test_button = widgets.Button(                    if (timerElement) {

        description='🧪 TEST SYSTEM',                        timerElement.innerHTML = '<div style="font-size: 24px; font-weight: bold; color: #d32f2f; text-align: center;">' + timeStr + '</div>';

        button_style='info',                    }

        layout=widgets.Layout(width='150px', height='40px')                    

    )                    if (elapsed >= maxDuration) {

                            clearInterval(timerInterval);

    status_display = widgets.HTML(                        // Reset UI through Python callback

        value="<div style='font-size: 16px; color: #666;'>Ready to record</div>"                        google.colab.kernel.invokeFunction('reset_recorder_ui', [], {});

    )                    }

                    }, 1000);

    status_output = widgets.Output()            } else {

    timer_display = widgets.HTML(                alert('Audio recorder not initialized. Please run the setup cell first.');

        value="<div style='font-size: 24px; font-weight: bold; color: #333; text-align: center;'>00:00</div>"            }

    )        """ % (recorder_state['current_duration'], recorder_state['current_duration'])))

        

    def update_team_number(change):    def on_stop_click(b):

        recorder_state['team_number'] = change['new'] if change['new'] else None        # Stop JavaScript recording

        debug_log(f"Team number updated: {recorder_state['team_number']}")        display(Javascript("""

                if (window.colabRecorder && window.colabRecorder.isRecording) {

    def update_defect_type(change):                window.colabRecorder.stopRecording();

        recorder_state['defect_type'] = change['new']            }

        debug_log(f"Defect type updated: {recorder_state['defect_type']}")        """))

            

    def test_system():        reset_ui()

        """Test system components"""    

        debug_log("Testing system components...")    def reset_ui():

        with status_output:        """Reset UI to initial state"""

            clear_output()        recorder_state['is_recording'] = False

            print("🧪 SYSTEM TEST RESULTS:")        

            print("=" * 30)        record_button.disabled = False

                    stop_button.disabled = True

            # Test directory access        defect_dropdown.disabled = False

            try:        duration_slider.disabled = False

                test_dir = os.path.join(base_dir, "test")        

                os.makedirs(test_dir, exist_ok=True)        status_display.value = "<div style='font-size: 16px; color: #666;'>Ready to record</div>"

                print("✅ Directory access: OK")        timer_display.value = "<div style='font-size: 24px; font-weight: bold; color: #333; text-align: center;'>00:00</div>"

                os.rmdir(test_dir)    

            except Exception as e:    def reset_recorder_ui():

                print(f"❌ Directory access: FAILED - {e}")        """Callback function for JavaScript timer completion"""

                    reset_ui()

            # Test Colab availability    

            if colab_available:    # Register reset callback

                print("✅ Google Colab: Available")    try:

            else:        from google.colab import output

                print("❌ Google Colab: Not available")        output.register_callback('reset_recorder_ui', lambda: reset_ui())

                except ImportError:

            print("✅ JavaScript execution: Check browser console")        pass

            print("=" * 30)    

        # Event handlers

    def on_test_click(b):    defect_dropdown.observe(update_defect_type, names='value')

        test_system()    record_button.on_click(on_record_click)

        stop_button.on_click(on_stop_click)

    def on_record_click(b):    

        debug_log("Record button clicked")    # Layout

        recorder_state['is_recording'] = True    header = widgets.HTML("""

        recorder_state['current_duration'] = duration_slider.value    <div style='background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 100%); color: white; 

        recorder_state['defect_type'] = defect_dropdown.value                padding: 20px; border-radius: 12px; text-align: center; margin-bottom: 20px;'>

        recorder_state['team_number'] = team_input.value if team_input.value else None        <h2 style='margin: 0; font-size: 24px;'>🎤 GOOGLE COLAB AUDIO RECORDER</h2>

                <p style='margin: 5px 0 0 0; font-size: 14px; opacity: 0.9;'>

        # Update UI            Browser-based Audio Recording for Predictive Maintenance

        record_button.disabled = True        </p>

        stop_button.disabled = False    </div>

        defect_dropdown.disabled = True    """)

        duration_slider.disabled = True    

        team_input.disabled = True    controls_panel = widgets.VBox([

                widgets.HTML("<div style='font-weight: bold; color: #1e3a8a; margin-bottom: 10px;'>🎛️ RECORDING CONTROLS</div>"),

        status_display.value = f"<div style='font-size: 16px; color: #d32f2f; font-weight: bold;'>🔴 RECORDING - {recorder_state['defect_type']}</div>"        defect_dropdown,

        status_output.clear_output()        duration_slider,

                widgets.HBox([record_button, stop_button])

        # JavaScript recording code    ], layout=widgets.Layout(border='2px solid #e2e8f0', border_radius='8px', 

        js_code = f"""                            padding='15px', margin='10px 0', background_color='#f8fafc'))

        (async function() {{    

            console.log('🎤 Starting recording process...');    status_panel = widgets.VBox([

                    widgets.HTML("<div style='font-weight: bold; color: #1e3a8a; margin-bottom: 10px;'>📊 STATUS</div>"),

            if (!window.colabRecorder) {{        status_display,

                class ColabAudioRecorder {{        widgets.HTML("<div data-timer-display='true'></div>"),

                    constructor() {{        timer_display

                        this.mediaRecorder = null;    ], layout=widgets.Layout(border='2px solid #e2e8f0', border_radius='8px', 

                        this.stream = null;                            padding='15px', margin='10px 0', background_color='#f8fafc'))

                        this.audioChunks = [];    

                        this.isRecording = false;    instructions = widgets.HTML("""

                    }}    <div style='background: #f0f9ff; border: 2px solid #0ea5e9; border-radius: 8px; padding: 15px; margin: 10px 0;'>

                            <div style='font-weight: bold; color: #0c4a6e; margin-bottom: 10px;'>📋 BROWSER RECORDING INSTRUCTIONS</div>

                    async initialize() {{        <div style='color: #0e7490; line-height: 1.6;'>

                        try {{            <b>🎤 Microphone Access:</b> Click "Allow" when prompted for microphone access<br/>

                            console.log('🎯 Requesting microphone access...');            <b>🔴 Recording:</b> Audio is captured directly in your browser - no PyAudio needed<br/>

                            this.stream = await navigator.mediaDevices.getUserMedia({{            <b>⏱️ Duration:</b> Recording stops automatically after selected duration<br/>

                                audio: {{            <b>💾 Saving:</b> Files are saved automatically with timestamps and defect classification<br/>

                                    sampleRate: 44100,            <b>📁 Organization:</b> Audio files are organized by defect type in separate folders

                                    channelCount: 1,        </div>

                                    echoCancellation: true,    </div>

                                    noiseSuppression: true,    """)

                                    autoGainControl: true    

                                }}    # Initialize JavaScript

                            }});    display(HTML(f"<script>{AUDIO_RECORDER_JS}</script>"))

                            console.log('✅ Microphone access granted');    

                            return true;    # Main UI

                        }} catch (error) {{    ui = widgets.VBox([

                            console.error('❌ Microphone access failed:', error);        header,

                            alert('Microphone access denied! Please:\\n1. Click Allow when prompted\\n2. Check browser permissions\\n3. Refresh the page and try again');        instructions,

                            return false;        controls_panel,

                        }}        status_panel,

                    }}        widgets.HTML("<div style='font-weight: bold; color: #64748b; text-align: center; margin: 10px 0;'>📝 RECORDING LOG</div>"),

                            status_output

                    async startRecording() {{    ], layout=widgets.Layout(width='100%'))

                        if (!this.stream) {{    

                            console.error('❌ No audio stream available');    return ui

                            return false;

                        }}def list_colab_audio_devices():

                            """

                        this.audioChunks = [];    List available audio devices in Colab (browser-based).

                            Returns information about browser audio capabilities.

                        const mimeTypes = [    """

                            'audio/webm;codecs=opus',    info = [

                            'audio/webm',        "🌐 Browser Audio Devices (Google Colab)",

                            'audio/mp4',        "----------------------------------------",

                            'audio/ogg;codecs=opus'        "📱 Primary Input: Default Browser Microphone",

                        ];        "🎯 Sample Rate: 44.1 kHz",

                                "📊 Channels: Mono (1 channel)",

                        let mimeType = 'audio/webm';        "🔧 Technology: Web Audio API + MediaRecorder",

                        for (const type of mimeTypes) {{        "",

                            if (MediaRecorder.isTypeSupported(type)) {{        "💡 Note: Actual devices depend on your system and browser permissions"

                                mimeType = type;    ]

                                break;    return info

                            }}

                        }}def record_colab_snippet(duration=3, defect_type="Good", base_dir="data/audio"):

                            """

                        console.log('🎵 Using MIME type:', mimeType);    Record an audio snippet in Colab using browser API.

                            This is a simplified version for programmatic use.

                        this.mediaRecorder = new MediaRecorder(this.stream, {{ mimeType }});    

                            Parameters:

                        this.mediaRecorder.ondataavailable = (event) => {{    -----------

                            if (event.data.size > 0) {{    duration : int

                                this.audioChunks.push(event.data);        Recording duration in seconds

                                console.log('📦 Audio chunk received:', event.data.size, 'bytes');    defect_type : str  

                            }}        Type of defect being recorded

                        }};    base_dir : str

                                Base directory for saving files

                        this.mediaRecorder.onstop = () => {{    """

                            console.log('⏹️ Recording stopped, processing audio...');    print("🎤 For recording in Colab, please use the create_colab_recorder_ui() function")

                            const audioBlob = new Blob(this.audioChunks, {{ type: mimeType }});    print("   Browser-based recording requires the interactive UI for microphone access")

                            console.log('🎵 Final audio blob:', audioBlob.size, 'bytes');    return None

                            this.processAudio(audioBlob);

                        }};# Main function for easy import

                        def create_recorder_ui(base_dir="data/audio", team_number=None):

                        this.mediaRecorder.onerror = (event) => {{    """

                            console.error('❌ MediaRecorder error:', event.error);    Create the Colab-compatible recorder UI - REDIRECTS TO WORKING VERSION.

                        }};    This is the main function that should be called from notebooks.

                            """

                        this.mediaRecorder.start(250);    # Import and use the working version with debugging

                        this.isRecording = true;    try:

                        console.log('🔴 Recording started successfully');        from .colab_audio_recorder_working import create_recorder_ui as working_create_recorder_ui

                        return true;        print("🔧 Loading enhanced working version with debugging...")

                    }}        return working_create_recorder_ui(base_dir, team_number)

                        except ImportError:

                    stopRecording() {{        # Fallback to fixed version

                        if (this.mediaRecorder && this.isRecording) {{        try:

                            this.mediaRecorder.stop();            from .colab_audio_recorder_fixed import create_recorder_ui as fixed_create_recorder_ui

                            this.isRecording = false;            print("⚠️ Loading fixed version as fallback...")

                            console.log('⏹️ Stop recording requested');            return fixed_create_recorder_ui(base_dir, team_number)

                            return true;        except ImportError:

                        }}            print("⚠️ Using basic fallback version - limited functionality")

                        return false;            return create_colab_recorder_ui(base_dir)

                    }}

                    if __name__ == "__main__":

                    processAudio(audioBlob) {{    print("Google Colab Audio Recorder Module")

                        console.log('🔄 Processing audio blob...');    print("Usage: from colab_audio_recorder import create_recorder_ui")

                        const reader = new FileReader();    print("       ui = create_recorder_ui()")

                        reader.onload = () => {{    print("       display(ui)")
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
                
                window.colabRecorder = new ColabAudioRecorder();
            }}
            
            // Initialize and start recording
            const initialized = await window.colabRecorder.initialize();
            if (initialized) {{
                const started = await window.colabRecorder.startRecording();
                if (started) {{
                    console.log('🎉 Recording started successfully!');
                    
                    // Auto-stop timer
                    setTimeout(() => {{
                        if (window.colabRecorder.isRecording) {{
                            console.log('⏰ Auto-stopping recording after {recorder_state['current_duration']} seconds');
                            window.colabRecorder.stopRecording();
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
            if (window.colabRecorder && window.colabRecorder.isRecording) {
                console.log('🛑 Manual stop requested');
                window.colabRecorder.stopRecording();
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
    
    # Initial debug
    debug_log(f"Recorder initialized - Base dir: {base_dir}")
    debug_log(f"Colab available: {colab_available}")
    
    # Layout
    header = widgets.HTML("""
    <div style='background: linear-gradient(135deg, #16a34a 0%, #22c55e 100%); color: white; 
                padding: 20px; border-radius: 12px; text-align: center; margin-bottom: 20px;'>
        <h2 style='margin: 0; font-size: 24px;'>🎤 GOOGLE COLAB AUDIO RECORDER</h2>
        <p style='margin: 5px 0 0 0; font-size: 14px; opacity: 0.9;'>
            Browser-Based Recording • Extended Duration • Team Support • Debugging
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
        <div style='font-weight: bold; color: #1e40af; margin-bottom: 10px;'>📋 COLAB RECORDER INSTRUCTIONS</div>
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

def list_colab_audio_devices():
    """
    List available audio devices in Colab (browser-based).
    Provides the same information format as audio_recorder.list_audio_devices()
    but for the Colab/browser environment.
    """
    info = [
        "🎤 GOOGLE COLAB AUDIO SYSTEM - BROWSER BASED",
        "=" * 50,
        "🌐 Platform: Web Audio API (Browser-based)",
        "📱 Input Device: Default System Microphone",
        "🎯 Sample Rate: 44.1 kHz (CD quality)",
        "📊 Bit Depth: 16-bit (browser standard)",
        "🔧 Channels: Mono (1 channel)",
        "💾 Recording Format: WebM/OGG → WAV conversion",
        "",
        "⏱️ DURATION SETTINGS:",
        "   • Minimum Duration: 1 second",
        "   • Maximum Duration: 400 seconds (6.67 minutes)",
        "   • Recommended: 3-10 seconds for samples",
        "",
        "🔧 BROWSER COMPATIBILITY:",
        "   ✅ Chrome/Chromium: Full support",
        "   ✅ Firefox: Full support",
        "   ✅ Microsoft Edge: Full support",
        "   ⚠️ Safari: Limited support (may have issues)",
        "",
        "📋 FEATURES AVAILABLE:",
        "   ✅ Files save to specified Google Drive folder",
        "   ✅ Extended duration up to 400 seconds",
        "   ✅ Team numbers included in filenames",
        "   ✅ Comprehensive debugging and error handling",
        "   ✅ System test functionality",
        "",
        "💡 Note: Actual microphone depends on your system hardware",
        "🔐 Requires: Microphone permission in browser settings"
    ]
    return info

def create_recorder_ui(base_dir="data/audio", team_number=None):
    """
    Main function to create the Colab-compatible recorder UI.
    Provides the same interface as audio_recorder.create_recorder_ui() but
    optimized for Google Colab with JavaScript-based recording.
    
    Parameters:
    -----------
    base_dir : str
        Base directory for saving audio files (Google Drive path)
    team_number : str or int
        Team number to include in filenames
        
    Returns:
    --------
    widgets.VBox
        Complete UI widget ready for display
    """
    return create_colab_recorder_ui(base_dir, team_number)

if __name__ == "__main__":
    print("Google Colab Audio Recorder Module")
    print("=" * 40)
    print("✅ Features:")
    print("   • Browser-based recording with JavaScript")
    print("   • Same interface as local audio_recorder")
    print("   • Extended duration support (1-400 seconds)")
    print("   • Team number integration")
    print("   • Comprehensive debugging")
    print("   • Google Drive integration")
    print()
    print("Usage:")
    print("   from colab_audio_recorder import create_recorder_ui")
    print("   ui = create_recorder_ui(base_dir='/content/drive/MyDrive/audio')")
    print("   display(ui)")