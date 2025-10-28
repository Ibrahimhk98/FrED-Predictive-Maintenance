"""
Enhanced Live Inspector UI Module
================================

This module provides an advanced user interface for the live audio inspector
with continuous updates, CSV logging, and machine identification capabilities.

Features:
- Real-time graph updates every 2 seconds during monitoring
- Automatic CSV data logging with configurable intervals
- Machine ID input for multi-machine monitoring
- Professional industrial-style UI design
- Comprehensive alert system with threshold monitoring
- Background threading for continuous operation

Author: Predictive Maintenance Workshop
Version: 2.0 Enhanced
"""

import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import threading
import time
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
import csv
from pathlib import Path


def create_enhanced_live_inspector_ui(inspector):
    """
    Create enhanced UI with continuous updates, CSV logging, and machine number input.
    
    Parameters:
    -----------
    inspector : LiveAudioInspector
        The configured live audio inspector instance
    
    Returns:
    --------
    widgets.VBox
        Complete UI widget ready for display
    
    Features:
    ---------
    - Continuous real-time plot updates (every 2 seconds)
    - Automatic CSV logging with machine metadata
    - Machine ID configuration for multi-machine setups
    - Professional threshold monitoring with visual alerts
    - Background threading for non-blocking operation
    - Comprehensive data export capabilities
    """
    
    # Machine number input
    machine_number_input = widgets.Text(
        value='Machine_001',
        placeholder='Enter machine ID (e.g., Machine_001)',
        description='Machine ID:',
        style={'description_width': '100px'},
        layout=widgets.Layout(width='250px')
    )
    
    # Enhanced control buttons
    start_button = widgets.Button(
        description='▶️ START MONITORING',
        button_style='success',
        layout=widgets.Layout(width='180px', height='45px'),
        style={'font_weight': 'bold'}
    )
    
    stop_button = widgets.Button(
        description='⏹️ STOP MONITORING',
        button_style='danger',
        layout=widgets.Layout(width='180px', height='45px'),
        style={'font_weight': 'bold'}
    )
    
    save_button = widgets.Button(
        description='💾 SAVE DATA',
        button_style='info',
        layout=widgets.Layout(width='150px', height='45px'),
        style={'font_weight': 'bold'}
    )
    
    clear_button = widgets.Button(
        description='🗑️ CLEAR DATA',
        button_style='warning',
        layout=widgets.Layout(width='150px', height='45px'),
        style={'font_weight': 'bold'}
    )
    
    # Auto-save controls
    auto_save_checkbox = widgets.Checkbox(
        value=True,
        description='Auto-save every:',
        style={'description_width': '120px'}
    )
    
    save_interval_input = widgets.IntSlider(
        value=30,
        min=10,
        max=300,
        step=10,
        description='seconds',
        style={'description_width': '70px'},
        layout=widgets.Layout(width='200px')
    )
    
    # Threshold slider
    threshold_slider = widgets.FloatSlider(
        value=0.70,
        min=0.0,
        max=1.0,
        step=0.01,
        description='Confidence Threshold:',
        style={'description_width': '150px'},
        layout=widgets.Layout(width='400px'),
        readout_format='.3f'
    )
    
    # Status displays
    system_status = widgets.HTML(
        value="<div style='font-size: 18px; font-weight: bold; color: #dc2626;'>🔴 SYSTEM OFFLINE</div>"
    )
    
    prediction_display = widgets.HTML(
        value="<div style='font-size: 16px; font-weight: bold; color: #64748b;'>Current Prediction: STANDBY</div>"
    )
    
    confidence_display = widgets.HTML(
        value="<div style='font-size: 16px; font-weight: bold; color: #64748b;'>Confidence Level: N/A</div>"
    )
    
    save_status_display = widgets.HTML(
        value="<div style='font-size: 14px; color: #64748b;'>No data saved yet</div>"
    )
    
    alert_display = widgets.HTML(value="")
    
    # Output areas
    output_area = widgets.Output()
    plot_output = widgets.Output()
    
    # State tracking
    monitoring_state = {
        "is_running": False,
        "start_time": None,
        "total_predictions": 0,
        "low_confidence_count": 0,
        "last_save_time": 0,
        "csv_file_path": None
    }
    
    # Thread control
    plot_update_thread = None
    auto_save_thread = None
    stop_plotting = False
    stop_auto_save = False
    
    def get_machine_id():
        """Get current machine ID."""
        machine_id = machine_number_input.value.strip()
        return machine_id if machine_id else "Unknown_Machine"
    
    def save_data_to_csv(filename_suffix=""):
        """Save monitoring data to CSV."""
        try:
            machine_id = get_machine_id()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            filename = f"machine_monitoring_{machine_id}_{timestamp}"
            if filename_suffix:
                filename += f"_{filename_suffix}"
            filename += ".csv"
            
            filepath = Path(filename)
            
            # Get results dataframe
            df = inspector.get_results_dataframe()
            
            if df.empty:
                save_status_display.value = "<div style='color: #dc2626;'>No data to save</div>"
                return None
            
            # Enhance dataframe with metadata
            df_enhanced = df.copy()
            df_enhanced['machine_id'] = machine_id
            df_enhanced['model_used'] = str(type(inspector.model).__name__)
            df_enhanced['feature_level'] = inspector.feature_level
            df_enhanced['threshold_used'] = threshold_slider.value
            df_enhanced['session_start'] = datetime.fromtimestamp(monitoring_state["start_time"]) if monitoring_state["start_time"] else None
            
            # Reorder columns
            column_order = ['machine_id', 'timestamp', 'prediction', 'confidence', 'threshold_used', 
                          'model_used', 'feature_level', 'session_start']
            df_enhanced = df_enhanced.reindex(columns=column_order + [col for col in df_enhanced.columns if col not in column_order])
            
            # Save to CSV
            df_enhanced.to_csv(filepath, index=False, quoting=csv.QUOTE_NONNUMERIC)
            
            monitoring_state["last_save_time"] = time.time()
            monitoring_state["csv_file_path"] = filepath
            
            save_status_display.value = f"""
            <div style='color: #16a34a;'>
                ✅ Saved: {filename}<br/>
                📊 Records: {len(df_enhanced)} | {datetime.now().strftime("%H:%M:%S")}
            </div>
            """
            
            with output_area:
                print(f"💾 Data saved: {filename} ({len(df_enhanced)} records)")
            
            return filepath
            
        except Exception as e:
            save_status_display.value = f"<div style='color: #dc2626;'>❌ Save failed: {str(e)}</div>"
            return None
    
    def auto_save_worker():
        """Background worker for automatic saving."""
        while not stop_auto_save and monitoring_state["is_running"]:
            if auto_save_checkbox.value and len(inspector.results_history) > 0:
                interval = save_interval_input.value
                current_time = time.time()
                
                if current_time - monitoring_state["last_save_time"] >= interval:
                    save_data_to_csv("auto")
            
            time.sleep(5)
    
    def update_display():
        """Continuously update plots during monitoring."""
        with plot_output:
            clear_output(wait=True)
            
            df = inspector.get_results_dataframe()
            if not df.empty and len(df) > 1:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
                    fig.patch.set_facecolor('#f8fafc')
                    
                    # Convert to relative time
                    df['relative_time'] = df['timestamp'] - df['timestamp'].iloc[0]
                    threshold_val = threshold_slider.value
                    machine_id = get_machine_id()
                    
                    # Plot 1: Predictions over time
                    unique_preds = df['prediction'].unique()
                    colors = {'good': '#16a34a', 'normal': '#16a34a', 'healthy': '#16a34a'}
                    fault_colors = ['#dc2626', '#ea580c', '#7c2d12']
                    fault_idx = 0
                    
                    for pred in unique_preds:
                        if pred.lower() in colors:
                            color = colors[pred.lower()]
                        else:
                            color = fault_colors[fault_idx % len(fault_colors)]
                            fault_idx += 1
                        
                        mask = df['prediction'] == pred
                        confidence_vals = df[mask]['confidence']
                        time_vals = df[mask]['relative_time']
                        
                        # Separate above/below threshold
                        above = confidence_vals >= threshold_val
                        below = confidence_vals < threshold_val
                        
                        if above.any():
                            ax1.scatter(time_vals[above], confidence_vals[above], 
                                      c=color, label=f'{pred} (Normal)', s=40, alpha=0.8)
                        if below.any():
                            ax1.scatter(time_vals[below], confidence_vals[below], 
                                      c=color, label=f'{pred} (Alert)', s=40, alpha=0.8, marker='x')
                    
                    # Add threshold line and zones
                    ax1.axhline(y=threshold_val, color='#dc2626', linestyle='--', linewidth=2, 
                               label=f'Threshold ({threshold_val:.3f})')
                    ax1.fill_between(df['relative_time'], 0, threshold_val, alpha=0.1, color='red', label='Alert Zone')
                    ax1.fill_between(df['relative_time'], threshold_val, 1, alpha=0.1, color='green', label='Safe Zone')
                    
                    ax1.set_xlabel('Time (seconds)')
                    ax1.set_ylabel('Confidence Level')
                    ax1.set_title(f'🔍 Real-Time Monitoring - {machine_id}')
                    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    ax1.grid(True, alpha=0.3)
                    ax1.set_ylim(-0.05, 1.05)
                    
                    # Plot 2: Confidence trend
                    conf_vals = df['confidence'].values
                    time_vals = df['relative_time'].values
                    
                    # Moving average
                    window = min(5, len(conf_vals))
                    if len(conf_vals) >= window:
                        moving_avg = np.convolve(conf_vals, np.ones(window)/window, mode='valid')
                        moving_time = time_vals[window-1:]
                        ax2.plot(moving_time, moving_avg, 'b-', linewidth=2, label='Trend')
                    
                    ax2.plot(time_vals, conf_vals, 'lightblue', alpha=0.7, label='Raw Confidence')
                    ax2.axhline(y=threshold_val, color='#dc2626', linestyle='--', linewidth=2)
                    
                    # Highlight low confidence periods
                    below_threshold = conf_vals < threshold_val
                    if np.any(below_threshold):
                        ax2.fill_between(time_vals, 0, conf_vals, where=below_threshold, 
                                       alpha=0.3, color='red', label='Alert Periods')
                    
                    ax2.set_xlabel('Time (seconds)')
                    ax2.set_ylabel('Confidence Level')
                    ax2.set_title(f'📊 Confidence Analysis - {machine_id}')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    ax2.set_ylim(-0.05, 1.05)
                    
                    # Latest value annotation
                    if not df.empty:
                        latest = df.iloc[-1]
                        status_color = '#16a34a' if latest["confidence"] >= threshold_val else '#dc2626'
                        status_text = 'NORMAL' if latest["confidence"] >= threshold_val else 'ALERT'
                        
                        ax1.annotate(f'{latest["prediction"]}\\nConf: {latest["confidence"]:.3f}\\n{status_text}',
                                   xy=(latest['relative_time'], latest['confidence']),
                                   xytext=(20, 20), textcoords='offset points',
                                   bbox=dict(boxstyle='round', facecolor=status_color, alpha=0.8),
                                   arrowprops=dict(arrowstyle='->', color=status_color),
                                   fontsize=10, color='white', weight='bold')
                    
                    plt.tight_layout()
                    plt.subplots_adjust(right=0.85)
                    plt.show()
    
    def continuous_plot_updater():
        """Continuously update plots while monitoring."""
        while not stop_plotting and inspector.is_running:
            if len(inspector.results_history) > 0:
                update_display()
            time.sleep(2.0)  # Update every 2 seconds
    
    def prediction_callback(result):
        """Handle new predictions."""
        monitoring_state["total_predictions"] += 1
        threshold_val = threshold_slider.value
        confidence = result['confidence']
        prediction = result['prediction']
        
        # Update displays
        if confidence >= threshold_val:
            pred_color = '#16a34a'
            status_icon = '🟢'
            status_text = 'NORMAL'
        else:
            pred_color = '#dc2626'
            status_icon = '🔴'
            status_text = 'ALERT'
            monitoring_state["low_confidence_count"] += 1
            
            # Alert message
            alert_display.value = f"""
            <div style='background: #fef2f2; border: 2px solid #dc2626; border-radius: 8px; padding: 15px; margin: 10px 0;'>
                <div style='color: #dc2626; font-weight: bold; text-align: center;'>
                    🚨 LOW CONFIDENCE ALERT 🚨<br/>
                    Confidence: {confidence:.3f} &lt; Threshold: {threshold_val:.3f}
                </div>
            </div>
            """
        
        prediction_display.value = f"""
        <div style='font-size: 16px; font-weight: bold;'>
            Current Prediction: <span style='color: {pred_color};'>{status_icon} {prediction}</span>
        </div>
        """
        
        confidence_display.value = f"""
        <div style='font-size: 16px; font-weight: bold;'>
            Confidence: <span style='color: {pred_color};'>{confidence:.3f}</span> ({status_text})
        </div>
        """
    
    # Event handlers
    def on_start_clicked(b):
        nonlocal plot_update_thread, auto_save_thread, stop_plotting, stop_auto_save
        
        try:
            with output_area:
                print(f"🚀 Starting monitoring for {get_machine_id()}...")
            
            monitoring_state["is_running"] = True
            monitoring_state["start_time"] = time.time()
            monitoring_state["total_predictions"] = 0
            monitoring_state["low_confidence_count"] = 0
            monitoring_state["last_save_time"] = time.time()
            
            inspector.set_callback(prediction_callback)
            inspector.start()
            
            system_status.value = "<div style='font-size: 18px; font-weight: bold; color: #16a34a;'>🟢 SYSTEM ONLINE</div>"
            start_button.disabled = True
            stop_button.disabled = False
            save_button.disabled = False
            
            # Start background threads
            stop_plotting = False
            stop_auto_save = False
            
            plot_update_thread = threading.Thread(target=continuous_plot_updater)
            plot_update_thread.daemon = True
            plot_update_thread.start()
            
            auto_save_thread = threading.Thread(target=auto_save_worker)
            auto_save_thread.daemon = True
            auto_save_thread.start()
            
            with output_area:
                print("✅ System online! Graphs updating continuously...")
                
        except Exception as e:
            with output_area:
                print(f"❌ Startup failed: {e}")
            system_status.value = "<div style='color: #dc2626;'>🔴 ERROR</div>"
    
    def on_stop_clicked(b):
        nonlocal stop_plotting, stop_auto_save
        
        try:
            with output_area:
                print("🛑 Stopping monitoring...")
            
            monitoring_state["is_running"] = False
            stop_plotting = True
            stop_auto_save = True
            
            # Wait for threads
            if plot_update_thread and plot_update_thread.is_alive():
                plot_update_thread.join(timeout=2.0)
            if auto_save_thread and auto_save_thread.is_alive():
                auto_save_thread.join(timeout=2.0)
            
            inspector.stop()
            
            # Final save
            if auto_save_checkbox.value and len(inspector.results_history) > 0:
                save_data_to_csv("final")
            
            system_status.value = "<div style='color: #dc2626;'>🔴 OFFLINE</div>"
            start_button.disabled = False
            stop_button.disabled = True
            save_button.disabled = True
            
            prediction_display.value = "<div style='color: #64748b;'>Current Prediction: STANDBY</div>"
            confidence_display.value = "<div style='color: #64748b;'>Confidence Level: N/A</div>"
            alert_display.value = ""
            
            # Final plot update
            update_display()
            
            with output_area:
                print("✅ Monitoring stopped")
                print(f"📊 Session summary: {monitoring_state['total_predictions']} predictions, {monitoring_state['low_confidence_count']} alerts")
                
        except Exception as e:
            with output_area:
                print(f"❌ Stop error: {e}")
    
    def on_save_clicked(b):
        """Manual save handler."""
        save_data_to_csv("manual")
    
    def on_clear_clicked(b):
        """Clear data handler."""
        inspector.clear_history()
        monitoring_state["total_predictions"] = 0
        monitoring_state["low_confidence_count"] = 0
        
        prediction_display.value = "<div style='color: #64748b;'>Current Prediction: STANDBY</div>"
        confidence_display.value = "<div style='color: #64748b;'>Confidence Level: N/A</div>"
        alert_display.value = ""
        save_status_display.value = "<div style='color: #64748b;'>Data cleared</div>"
        
        with plot_output:
            clear_output()
        
        with output_area:
            print("🗑️ Data cleared")
    
    # Connect event handlers
    start_button.on_click(on_start_clicked)
    stop_button.on_click(on_stop_clicked)
    save_button.on_click(on_save_clicked)
    clear_button.on_click(on_clear_clicked)
    
    # Initial states
    stop_button.disabled = True
    save_button.disabled = True
    
    # Create UI layout
    header = widgets.HTML("""
    <div style='background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 100%); color: white; 
                padding: 20px; border-radius: 12px; text-align: center; margin-bottom: 20px;'>
        <h2 style='margin: 0; font-size: 24px;'>🏭 ENHANCED PREDICTIVE MAINTENANCE MONITOR</h2>
        <p style='margin: 5px 0 0 0; font-size: 14px; opacity: 0.9;'>
            Real-time Monitoring with Continuous Updates, CSV Logging & Machine Identification
        </p>
    </div>
    """)
    
    machine_panel = widgets.VBox([
        widgets.HTML("<div style='font-weight: bold; color: #1e3a8a; margin-bottom: 10px;'>🏭 MACHINE CONFIGURATION</div>"),
        machine_number_input
    ], layout=widgets.Layout(border='2px solid #e2e8f0', border_radius='8px', 
                            padding='15px', margin='10px 0', background_color='#f8fafc'))
    
    control_panel = widgets.VBox([
        widgets.HTML("<div style='font-weight: bold; color: #1e3a8a; margin-bottom: 10px;'>🎛️ SYSTEM CONTROLS</div>"),
        widgets.HBox([start_button, stop_button, save_button, clear_button])
    ], layout=widgets.Layout(border='2px solid #e2e8f0', border_radius='8px', 
                            padding='15px', margin='10px 0', background_color='#f8fafc'))
    
    logging_panel = widgets.VBox([
        widgets.HTML("<div style='font-weight: bold; color: #1e3a8a; margin-bottom: 10px;'>💾 DATA LOGGING</div>"),
        widgets.HBox([auto_save_checkbox, save_interval_input]),
        save_status_display
    ], layout=widgets.Layout(border='2px solid #e2e8f0', border_radius='8px', 
                            padding='15px', margin='10px 0', background_color='#f8fafc'))
    
    threshold_panel = widgets.VBox([
        widgets.HTML("<div style='font-weight: bold; color: #1e3a8a; margin-bottom: 10px;'>⚙️ THRESHOLD CONFIG</div>"),
        threshold_slider
    ], layout=widgets.Layout(border='2px solid #e2e8f0', border_radius='8px', 
                            padding='15px', margin='10px 0', background_color='#f8fafc'))
    
    status_panel = widgets.VBox([
        widgets.HTML("<div style='font-weight: bold; color: #1e3a8a; margin-bottom: 10px;'>📊 SYSTEM STATUS</div>"),
        system_status,
        prediction_display,
        confidence_display
    ], layout=widgets.Layout(border='2px solid #e2e8f0', border_radius='8px', 
                            padding='15px', margin='10px 0', background_color='#f8fafc'))
    
    alert_panel = widgets.VBox([
        widgets.HTML("<div style='font-weight: bold; color: #dc2626; margin-bottom: 10px;'>🚨 ALERTS</div>"),
        alert_display
    ], layout=widgets.Layout(border='2px solid #e2e8f0', border_radius='8px', 
                            padding='15px', margin='10px 0', background_color='#f8fafc'))
    
    instructions = widgets.HTML("""
    <div style='background: #f0f9ff; border: 2px solid #0ea5e9; border-radius: 8px; padding: 15px; margin: 10px 0;'>
        <div style='font-weight: bold; color: #0c4a6e; margin-bottom: 10px;'>📋 ENHANCED FEATURES</div>
        <div style='color: #0e7490; line-height: 1.6;'>
            <b>🆕 New Features:</b><br/>
            • <b>Continuous Graph Updates:</b> Plots refresh automatically every 2 seconds during monitoring<br/>
            • <b>CSV Data Logging:</b> Auto-save or manual save with machine ID, timestamps, and metadata<br/>
            • <b>Machine Identification:</b> Enter unique machine number for multi-machine monitoring<br/><br/>
            
            <b>📊 CSV Data Includes:</b> Machine ID, Timestamp, Prediction, Confidence, Threshold, Model Used, Feature Level<br/>
            <b>⚙️ Auto-save:</b> Configurable intervals (10-300 seconds) + final save on stop<br/>
            <b>🎯 Real-time:</b> Live plots update continuously while monitoring is active
        </div>
    </div>
    """)
    
    # Main UI
    ui = widgets.VBox([
        header,
        instructions,
        machine_panel,
        widgets.HBox([
            widgets.VBox([control_panel, logging_panel], layout=widgets.Layout(width='50%')),
            widgets.VBox([threshold_panel, status_panel], layout=widgets.Layout(width='50%'))
        ]),
        alert_panel,
        widgets.HTML("<div style='font-weight: bold; color: #1e3a8a; text-align: center; margin: 20px 0 10px 0;'>📈 REAL-TIME DASHBOARD (Updates Every 2 Seconds)</div>"),
        plot_output,
        widgets.HTML("<div style='font-weight: bold; color: #64748b; text-align: center; margin: 10px 0;'>📋 SYSTEM LOG</div>"),
        output_area
    ], layout=widgets.Layout(width='100%'))
    
    return ui


# Convenience function for quick setup
def display_enhanced_ui(inspector):
    """
    Quick function to create and display the enhanced UI.
    
    Parameters:
    -----------
    inspector : LiveAudioInspector
        The configured live audio inspector instance
    
    Usage:
    ------
    from enhanced_live_inspector_ui import display_enhanced_ui
    display_enhanced_ui(inspector)
    """
    ui = create_enhanced_live_inspector_ui(inspector)
    display(ui)
    return ui