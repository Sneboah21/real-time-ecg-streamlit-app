# ============================================================
# REAL-TIME HEALTH MONITORING SYSTEM - STREAMLIT DEPLOYMENT
# Module 5: Deployment & Real-Time Implementation
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import time
from tensorflow import keras
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import wfdb
from scipy import signal
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Real-Time Health Monitoring System",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================
# 1. LOAD MODEL AND PREPROCESSING FUNCTIONS
# ============================================================

@st.cache_resource
def load_model():
    """Load the trained ECG classification model"""
    try:
        model = keras.models.load_model('models/ecg_arrhythmia_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_resource
def load_label_encoder():
    """Load the label encoder"""
    try:
        with open('models/label_encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
        return encoder
    except Exception as e:
        st.error(f"Error loading label encoder: {e}")
        return None


def bandpass_filter(ecg_signal, lowcut=0.5, highcut=40, fs=360, order=4):
    """Apply Butterworth bandpass filter"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    filtered_signal = signal.filtfilt(b, a, ecg_signal)
    return filtered_signal


def notch_filter(ecg_signal, freq=60, fs=360, quality=30):
    """Apply notch filter to remove powerline interference"""
    b, a = signal.iirnotch(freq, quality, fs)
    filtered_signal = signal.filtfilt(b, a, ecg_signal)
    return filtered_signal


def normalize_signal(ecg_signal):
    """Z-score normalization"""
    mean = np.mean(ecg_signal)
    std = np.std(ecg_signal)
    normalized = (ecg_signal - mean) / (std + 1e-8)
    return normalized


def preprocess_ecg(ecg_signal):
    """Complete preprocessing pipeline"""
    filtered = bandpass_filter(ecg_signal)
    filtered = notch_filter(filtered)
    normalized = normalize_signal(filtered)
    return normalized


def detect_r_peaks(ecg_signal, fs=360):
    """Simple R-peak detection using scipy"""
    peaks, _ = signal.find_peaks(ecg_signal, distance=fs * 0.6, height=0.5)
    return peaks


def calculate_heart_rate(peaks, fs=360):
    """Calculate heart rate from R-peaks"""
    if len(peaks) < 2:
        return 75
    rr_intervals = np.diff(peaks) / fs
    heart_rate = 60 / np.mean(rr_intervals)
    return heart_rate


def analyze_full_record(ecg_signal, model, label_encoder, fs=360):
    """
    Analyze ALL beats in the ECG signal and return comprehensive statistics
    """
    # Preprocess
    preprocessed = preprocess_ecg(ecg_signal)

    # Detect all R-peaks
    peaks = detect_r_peaks(preprocessed)

    # Store predictions for each beat
    beat_predictions = []
    beat_confidences = []

    # Analyze each beat
    for peak in peaks:
        start = peak - 90
        end = peak + 90

        # Check bounds
        if start >= 0 and end < len(preprocessed):
            segment = preprocessed[start:end]

            if len(segment) == 180:
                # Predict
                input_data = segment.reshape(1, 180, 1)
                prediction_probs = model.predict(input_data, verbose=0)
                prediction_class = np.argmax(prediction_probs, axis=1)[0]
                confidence = prediction_probs[0][prediction_class] * 100

                predicted_label = label_encoder.classes_[prediction_class]
                beat_predictions.append(predicted_label)
                beat_confidences.append(confidence)

    # Calculate statistics
    total_beats = len(beat_predictions)
    normal_count = beat_predictions.count('Normal')
    atrial_count = beat_predictions.count('Atrial')
    ventricular_count = beat_predictions.count('Ventricular')

    # Calculate percentages (burden)
    normal_pct = (normal_count / total_beats * 100) if total_beats > 0 else 0
    atrial_pct = (atrial_count / total_beats * 100) if total_beats > 0 else 0
    ventricular_pct = (ventricular_count / total_beats * 100) if total_beats > 0 else 0

    # Determine overall diagnosis based on burden
    if ventricular_pct > 10:
        diagnosis = "Ventricular Arrhythmia"
        severity = "High Risk"
        color = "error"
    elif ventricular_pct > 1:
        diagnosis = "Occasional Ventricular Beats"
        severity = "Monitor"
        color = "warning"
    elif atrial_pct > 10:
        diagnosis = "Atrial Arrhythmia"
        severity = "Moderate Risk"
        color = "warning"
    elif atrial_pct > 1:
        diagnosis = "Occasional Atrial Beats"
        severity = "Monitor"
        color = "info"
    else:
        diagnosis = "Normal Sinus Rhythm"
        severity = "Healthy"
        color = "success"

    return {
        'total_beats': total_beats,
        'normal_count': normal_count,
        'atrial_count': atrial_count,
        'ventricular_count': ventricular_count,
        'normal_pct': normal_pct,
        'atrial_pct': atrial_pct,
        'ventricular_pct': ventricular_pct,
        'diagnosis': diagnosis,
        'severity': severity,
        'color': color,
        'mean_confidence': np.mean(beat_confidences) if beat_confidences else 0,
        'heart_rate': calculate_heart_rate(peaks, fs),
        'preprocessed': preprocessed,
        'peaks': peaks
    }


# ============================================================
# 2. STREAMLIT APP INTERFACE
# ============================================================

st.title("🫀 Real-Time Health Monitoring System")
st.markdown("### ECG Arrhythmia Detection & Analysis")
st.markdown("---")

# Load model
model = load_model()
label_encoder = load_label_encoder()

if model is None or label_encoder is None:
    st.stop()

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")

    input_method = st.radio(
        "Select Input Method:",
        ["Upload ECG File", "Load PhysioNet Record", "Simulate Real-Time Stream"]
    )

    st.markdown("---")
    st.header("📊 Model Information")
    st.info(f"""
    **Classes:** {len(label_encoder.classes_)}

    **Types:**
    - Normal
    - Atrial
    - Ventricular

    **Accuracy:** 99.28%
    """)

    st.markdown("---")
    st.header("🎯 Alert Thresholds")
    st.caption("Triggers alerts when heart rate goes outside these ranges")
    hr_min = st.slider("Min Heart Rate (bpm)", 40, 60, 50)
    hr_max = st.slider("Max Heart Rate (bpm)", 100, 140, 120)

# ============================================================
# 3. INPUT METHOD 1: UPLOAD ECG FILE (ENHANCED WITH FULL ANALYSIS)
# ============================================================

if input_method == "Upload ECG File":
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📁 Upload ECG Data")
        uploaded_file = st.file_uploader(
            "Choose a CSV or TXT file containing ECG signal",
            type=['csv', 'txt', 'dat']
        )

    with col2:
        st.header("📈 Live Metrics")

    if uploaded_file is not None:
        try:
            ecg_data = np.loadtxt(uploaded_file)

            if len(ecg_data) < 180:
                with col1:
                    st.error("ECG signal too short. Need at least 180 samples")
                st.stop()

            # Analyze full record
            with col1:
                with st.spinner("Analyzing ECG signal..."):
                    results = analyze_full_record(ecg_data, model, label_encoder)

            # Visualize
            with col1:
                fig = go.Figure()
                time_axis = np.arange(len(results['preprocessed'])) / 360

                fig.add_trace(go.Scatter(
                    x=time_axis, y=results['preprocessed'],
                    mode='lines',
                    name='ECG Signal',
                    line=dict(color='green', width=1)
                ))

                fig.add_trace(go.Scatter(
                    x=results['peaks'] / 360, y=results['preprocessed'][results['peaks']],
                    mode='markers',
                    name='R-peaks',
                    marker=dict(color='red', size=8)
                ))

                fig.update_layout(
                    title=f"Uploaded ECG File Analysis - {uploaded_file.name}",
                    xaxis_title="Time (seconds)",
                    yaxis_title="Normalized Amplitude",
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)

            # Display comprehensive results in col2
            with col2:
                st.metric(
                    label="❤️ Heart Rate",
                    value=f"{results['heart_rate']:.0f} bpm"
                )

                st.metric(
                    label="📊 Total Beats",
                    value=f"{results['total_beats']}"
                )

                st.markdown("---")
                st.subheader("Beat Distribution")

                # Show beat counts and percentages
                st.metric("✅ Normal", f"{results['normal_count']}",
                          delta=f"{results['normal_pct']:.1f}%")

                st.metric("⚠️ Atrial", f"{results['atrial_count']}",
                          delta=f"{results['atrial_pct']:.1f}%")

                st.metric("🚨 Ventricular", f"{results['ventricular_count']}",
                          delta=f"{results['ventricular_pct']:.1f}%")

                st.markdown("---")

                # Overall diagnosis
                st.subheader("📋 Diagnosis")

                if results['color'] == 'success':
                    st.success(f"**{results['diagnosis']}**")
                elif results['color'] == 'warning':
                    st.warning(f"**{results['diagnosis']}**")
                elif results['color'] == 'error':
                    st.error(f"**{results['diagnosis']}**")
                else:
                    st.info(f"**{results['diagnosis']}**")

                st.info(f"**Severity:** {results['severity']}")
                st.caption(f"Avg Confidence: {results['mean_confidence']:.1f}%")

                st.markdown("---")

                # Clinical recommendations
                st.subheader("💡 Recommendations")

                if results['ventricular_pct'] > 10:
                    st.error("🚨 **URGENT**")
                    st.warning("Ventricular burden > 10%")
                    st.info("Immediate cardiology consult")
                elif results['ventricular_pct'] > 1:
                    st.warning("⚠️ Monitor PVCs")
                    st.info("Follow-up recommended")
                elif results['atrial_pct'] > 10:
                    st.warning("⚠️ Significant PACs")
                    st.info("Consider Holter monitor")
                else:
                    st.success("✅ Normal rhythm")
                    st.info("Routine follow-up")

                # Heart rate alerts
                st.markdown("---")
                if results['heart_rate'] < hr_min:
                    st.error(f"⚠️ **BRADYCARDIA**\nHR: {results['heart_rate']:.0f} bpm")
                elif results['heart_rate'] > hr_max:
                    st.error(f"⚠️ **TACHYCARDIA**\nHR: {results['heart_rate']:.0f} bpm")
                else:
                    st.success("✅ HR within normal range")

            # Show detailed breakdown in main column
            with col1:
                st.markdown("---")
                st.subheader("📊 Detailed Beat Analysis")

                # Create dataframe
                beat_data = pd.DataFrame({
                    'Beat Type': ['Normal', 'Atrial', 'Ventricular'],
                    'Count': [results['normal_count'], results['atrial_count'], results['ventricular_count']],
                    'Percentage': [f"{results['normal_pct']:.1f}%", f"{results['atrial_pct']:.1f}%",
                                   f"{results['ventricular_pct']:.1f}%"],
                    'Clinical Significance': [
                        '✅ Expected',
                        '⚠️ Monitor if > 1%' if results['atrial_pct'] > 1 else '✅ Benign',
                        '🚨 Concerning if > 10%' if results['ventricular_pct'] > 10 else (
                            '⚠️ Monitor' if results['ventricular_pct'] > 1 else '✅ Benign')
                    ]
                })

                st.dataframe(beat_data, use_container_width=True, hide_index=True)

                # Visualization
                fig_pie = go.Figure(data=[go.Pie(
                    labels=['Normal', 'Atrial', 'Ventricular'],
                    values=[results['normal_count'], results['atrial_count'], results['ventricular_count']],
                    marker=dict(colors=['#00cc66', '#ffaa00', '#ff4444']),
                    hole=0.3
                )])
                fig_pie.update_layout(
                    title="Beat Type Distribution",
                    height=350,
                    showlegend=True
                )
                st.plotly_chart(fig_pie, use_container_width=True)

                # Clinical burden thresholds info
                st.info("""
                **Clinical Burden Thresholds:**
                - < 1% abnormal beats: Normal variation (healthy)
                - 1-10% abnormal beats: Monitor, usually benign
                - > 10% ventricular beats: Clinically significant
                - > 30% burden: High risk, requires treatment
                """)

        except Exception as e:
            with col1:
                st.error(f"Error: {e}")

# ============================================================
# 4. INPUT METHOD 2: LOAD PHYSIONET RECORD
# ============================================================

elif input_method == "Load PhysioNet Record":
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("🏥 PhysioNet MIT-BIH Database")
        record_options = ['100', '101', '103', '105', '106', '109', '111',
                          '115', '119', '200', '201', '208', '210', '213']
        selected_record = st.selectbox("Select Patient Record:", record_options)

        # Add analysis mode selection
        analysis_mode = st.radio(
            "Analysis Mode:",
            ["Single Beat Analysis", "Full Record Analysis (All Beats)"],
            help="Single beat analyzes one beat. Full record analyzes ALL beats for comprehensive diagnosis."
        )

        load_button = st.button("Load & Analyze Record", type="primary")

    with col2:
        st.header("📈 Live Metrics")

    if load_button:
        try:
            with col1:
                with st.spinner(f"Loading record {selected_record}..."):
                    record = wfdb.rdrecord(selected_record, pn_dir='mitdb')
                    annotation = wfdb.rdann(selected_record, 'atr', pn_dir='mitdb')
                    ecg_signal = record.p_signal[:3600, 0]
                    preprocessed = preprocess_ecg(ecg_signal)

            # Visualize
            with col1:
                fig = go.Figure()
                time_axis = np.arange(len(preprocessed)) / 360

                fig.add_trace(go.Scatter(
                    x=time_axis, y=preprocessed,
                    mode='lines',
                    name='ECG Signal',
                    line=dict(color='green', width=1)
                ))

                peaks = detect_r_peaks(preprocessed)
                fig.add_trace(go.Scatter(
                    x=peaks / 360, y=preprocessed[peaks],
                    mode='markers',
                    name='R-peaks',
                    marker=dict(color='red', size=8)
                ))

                fig.update_layout(
                    title=f"Patient Record {selected_record} - Real-Time ECG",
                    xaxis_title="Time (seconds)",
                    yaxis_title="Normalized Amplitude",
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)

            # Choose analysis mode
            if analysis_mode == "Single Beat Analysis":
                # Original single-beat analysis
                first_peak = peaks[0] if len(peaks) > 0 else 90
                segment = preprocessed[first_peak - 90:first_peak + 90]

                if len(segment) == 180:
                    input_data = segment.reshape(1, 180, 1)
                    prediction_probs = model.predict(input_data, verbose=0)
                    prediction_class = np.argmax(prediction_probs, axis=1)[0]
                    confidence = prediction_probs[0][prediction_class] * 100
                    predicted_label = label_encoder.classes_[prediction_class]
                    heart_rate = calculate_heart_rate(peaks)

                    with col2:
                        st.metric(
                            label="❤️ Heart Rate",
                            value=f"{heart_rate:.0f} bpm",
                            delta=f"{heart_rate - 75:.0f}"
                        )

                        if predicted_label == "Normal":
                            st.success(f"✅ **Status:** {predicted_label}")
                        elif predicted_label == "Atrial":
                            st.warning(f"⚠️ **Status:** {predicted_label} Arrhythmia")
                        else:
                            st.error(f"🚨 **Status:** {predicted_label} Arrhythmia")

                        st.info(f"🎯 **Confidence:** {confidence:.2f}%")
                        st.caption(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
                        st.caption("📌 **Note:** Analyzing ONE beat only")

                        st.markdown("---")

                        if heart_rate < hr_min:
                            st.error(f"⚠️ **BRADYCARDIA**\nHR: {heart_rate:.0f} bpm")
                        elif heart_rate > hr_max:
                            st.error(f"⚠️ **TACHYCARDIA**\nHR: {heart_rate:.0f} bpm")
                        else:
                            st.success("✅ HR Normal")

                        if predicted_label != "Normal":
                            st.warning(f"⚠️ **{predicted_label.upper()}**")

            else:  # Full Record Analysis
                with col1:
                    with st.spinner("Analyzing all beats in the record..."):
                        results = analyze_full_record(ecg_signal, model, label_encoder)

                with col2:
                    # Display comprehensive results
                    st.metric(
                        label="❤️ Heart Rate",
                        value=f"{results['heart_rate']:.0f} bpm"
                    )

                    st.metric(
                        label="📊 Total Beats",
                        value=f"{results['total_beats']}"
                    )

                    st.markdown("---")
                    st.subheader("Beat Distribution")

                    # Show beat counts and percentages
                    st.metric("✅ Normal", f"{results['normal_count']}",
                              delta=f"{results['normal_pct']:.1f}%")

                    st.metric("⚠️ Atrial", f"{results['atrial_count']}",
                              delta=f"{results['atrial_pct']:.1f}%")

                    st.metric("🚨 Ventricular", f"{results['ventricular_count']}",
                              delta=f"{results['ventricular_pct']:.1f}%")

                    st.markdown("---")

                    # Overall diagnosis
                    st.subheader("📋 Diagnosis")

                    if results['color'] == 'success':
                        st.success(f"**{results['diagnosis']}**")
                    elif results['color'] == 'warning':
                        st.warning(f"**{results['diagnosis']}**")
                    elif results['color'] == 'error':
                        st.error(f"**{results['diagnosis']}**")
                    else:
                        st.info(f"**{results['diagnosis']}**")

                    st.info(f"**Severity:** {results['severity']}")
                    st.caption(f"Avg Confidence: {results['mean_confidence']:.1f}%")

                    st.markdown("---")

                    # Clinical recommendations
                    st.subheader("💡 Recommendations")

                    if results['ventricular_pct'] > 10:
                        st.error("🚨 **URGENT**")
                        st.warning("Ventricular burden > 10%")
                        st.info("Immediate cardiology consult")
                    elif results['ventricular_pct'] > 1:
                        st.warning("⚠️ Monitor PVCs")
                        st.info("Follow-up recommended")
                    elif results['atrial_pct'] > 10:
                        st.warning("⚠️ Significant PACs")
                        st.info("Consider Holter monitor")
                    else:
                        st.success("✅ Normal rhythm")
                        st.info("Routine follow-up")

                # Show detailed breakdown in main column
                with col1:
                    st.markdown("---")
                    st.subheader("📊 Detailed Beat Analysis")

                    # Create dataframe
                    beat_data = pd.DataFrame({
                        'Beat Type': ['Normal', 'Atrial', 'Ventricular'],
                        'Count': [results['normal_count'], results['atrial_count'], results['ventricular_count']],
                        'Percentage': [f"{results['normal_pct']:.1f}%", f"{results['atrial_pct']:.1f}%",
                                       f"{results['ventricular_pct']:.1f}%"],
                        'Clinical Significance': [
                            '✅ Expected',
                            '⚠️ Monitor if > 1%' if results['atrial_pct'] > 1 else '✅ Benign',
                            '🚨 Concerning if > 10%' if results['ventricular_pct'] > 10 else (
                                '⚠️ Monitor' if results['ventricular_pct'] > 1 else '✅ Benign')
                        ]
                    })

                    st.dataframe(beat_data, use_container_width=True, hide_index=True)

                    # Visualization
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=['Normal', 'Atrial', 'Ventricular'],
                        values=[results['normal_count'], results['atrial_count'], results['ventricular_count']],
                        marker=dict(colors=['#00cc66', '#ffaa00', '#ff4444']),
                        hole=0.3
                    )])
                    fig_pie.update_layout(
                        title="Beat Type Distribution",
                        height=350,
                        showlegend=True
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                    # Clinical burden thresholds info
                    st.info("""
                    **Clinical Burden Thresholds:**
                    - < 1% abnormal beats: Normal variation (healthy)
                    - 1-10% abnormal beats: Monitor, usually benign
                    - > 10% ventricular beats: Clinically significant
                    - > 30% burden: High risk, requires treatment
                    """)

        except Exception as e:
            with col1:
                st.error(f"Error: {e}")

# ============================================================
# 5. INPUT METHOD 3: SIMULATE REAL-TIME STREAM (FIXED)
# ============================================================

elif input_method == "Simulate Real-Time Stream":
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📡 Real-Time ECG Streaming Simulation")

        signal_type = st.selectbox(
            "Select Signal Type:",
            [
                "100 - Normal",
                "106 - Ventricular Arrhythmia",
                "119 - Ventricular Arrhythmia",
                "201 - Atrial Arrhythmia",
                "208 - Mixed Arrhythmias"
            ]
        )

        record_num = signal_type.split(' ')[0]
        duration = st.slider("Duration (seconds)", 5, 30, 10)
        start_button = st.button("▶️ Start Real-Time Monitoring", type="primary")

    with col2:
        st.header("📈 Live Metrics")
        hr_placeholder = st.empty()
        status_placeholder = st.empty()
        confidence_placeholder = st.empty()
        time_placeholder = st.empty()

        st.markdown("---")
        st.subheader("📊 Cumulative Statistics")
        total_beats_placeholder = st.empty()
        normal_count_placeholder = st.empty()
        atrial_count_placeholder = st.empty()
        ventricular_count_placeholder = st.empty()

        st.markdown("---")
        alert_placeholder = st.empty()

    if start_button:
        try:
            record = wfdb.rdrecord(record_num, pn_dir='mitdb')
            full_signal = record.p_signal[:, 0]

            with col1:
                chart_placeholder = st.empty()
                progress_placeholder = st.empty()

            total_samples = duration * 360
            buffer = []

            # Track beat predictions (only unique beats)
            beat_predictions_list = []
            last_analysis_index = -360  # Track when we last analyzed

            for i in range(0, total_samples, 36):
                chunk = full_signal[i:i + 36]
                buffer.extend(chunk)

                if len(buffer) > 1800:
                    buffer = buffer[-1800:]

                # Update chart
                time_axis = np.arange(len(buffer)) / 360

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=time_axis, y=buffer,
                    mode='lines',
                    line=dict(color='green', width=1.5),
                    fill='tozeroy',
                    fillcolor='rgba(0,255,0,0.1)'
                ))

                fig.update_layout(
                    title=f"Live Stream - Record {record_num}",
                    xaxis_title="Time (seconds)",
                    yaxis_title="Amplitude (mV)",
                    height=400,
                    xaxis_range=[max(0, time_axis[-1] - 5), time_axis[-1] if len(time_axis) > 0 else 5]
                )

                chart_placeholder.plotly_chart(fig, use_container_width=True)

                # Analyze only every ~1 second (every heartbeat, not every 0.1s)
                # This prevents counting the same beat multiple times
                if len(buffer) >= 180 and (i - last_analysis_index) >= 270:  # ~0.75 seconds apart
                    segment = np.array(buffer[-180:])
                    preprocessed = preprocess_ecg(segment)

                    input_data = preprocessed.reshape(1, 180, 1)
                    prediction_probs = model.predict(input_data, verbose=0)
                    prediction_class = np.argmax(prediction_probs, axis=1)[0]
                    confidence = prediction_probs[0][prediction_class] * 100
                    predicted_label = label_encoder.classes_[prediction_class]

                    peaks = detect_r_peaks(preprocessed)
                    heart_rate = calculate_heart_rate(peaks)

                    # Add to predictions list (unique beats only)
                    beat_predictions_list.append(predicted_label)
                    last_analysis_index = i

                    # Calculate cumulative statistics
                    total_beats_analyzed = len(beat_predictions_list)
                    normal_beats = beat_predictions_list.count('Normal')
                    atrial_beats = beat_predictions_list.count('Atrial')
                    ventricular_beats = beat_predictions_list.count('Ventricular')

                    normal_pct = (normal_beats / total_beats_analyzed * 100) if total_beats_analyzed > 0 else 0
                    atrial_pct = (atrial_beats / total_beats_analyzed * 100) if total_beats_analyzed > 0 else 0
                    ventricular_pct = (
                                ventricular_beats / total_beats_analyzed * 100) if total_beats_analyzed > 0 else 0

                    # Update live metrics
                    with hr_placeholder:
                        st.metric("❤️ Heart Rate", f"{heart_rate:.0f} bpm")

                    with status_placeholder:
                        if predicted_label == "Normal":
                            st.success(f"✅ **Current: {predicted_label}**")
                        elif predicted_label == "Atrial":
                            st.warning(f"⚠️ **Current: {predicted_label}**")
                        else:
                            st.error(f"🚨 **Current: {predicted_label}**")

                    with confidence_placeholder:
                        st.info(f"🎯 **Confidence:** {confidence:.1f}%")

                    with time_placeholder:
                        st.caption(f"⏰ {datetime.now().strftime('%H:%M:%S')}")

                    # Update cumulative statistics
                    with total_beats_placeholder:
                        st.metric("📊 Total Beats", f"{total_beats_analyzed}")

                    with normal_count_placeholder:
                        st.metric("✅ Normal", f"{normal_beats}", delta=f"{normal_pct:.1f}%")

                    with atrial_count_placeholder:
                        st.metric("⚠️ Atrial", f"{atrial_beats}", delta=f"{atrial_pct:.1f}%")

                    with ventricular_count_placeholder:
                        st.metric("🚨 Ventricular", f"{ventricular_beats}", delta=f"{ventricular_pct:.1f}%")

                    # Update alerts
                    with alert_placeholder:
                        alert_placeholder.empty()
                        if heart_rate < hr_min or heart_rate > hr_max:
                            st.error(f"⚠️ HR: {heart_rate:.0f} bpm - Alert!")

                        if predicted_label != "Normal":
                            st.warning(f"⚠️ {predicted_label} detected!")

                progress_placeholder.progress(min(i / total_samples, 1.0))
                time.sleep(0.1)

            progress_placeholder.empty()

            # Final summary
            if len(beat_predictions_list) > 0:
                total_beats_analyzed = len(beat_predictions_list)
                normal_beats = beat_predictions_list.count('Normal')
                atrial_beats = beat_predictions_list.count('Atrial')
                ventricular_beats = beat_predictions_list.count('Ventricular')

                normal_pct = (normal_beats / total_beats_analyzed * 100)
                atrial_pct = (atrial_beats / total_beats_analyzed * 100)
                ventricular_pct = (ventricular_beats / total_beats_analyzed * 100)

                with col1:
                    st.success("✅ Monitoring session completed!")
                    st.markdown("---")
                    st.subheader("📋 Session Summary")

                    # Determine final diagnosis
                    if ventricular_pct > 10:
                        final_diagnosis = "Ventricular Arrhythmia - High Risk"
                        diagnosis_color = "error"
                    elif ventricular_pct > 1:
                        final_diagnosis = "Occasional Ventricular Beats - Monitor"
                        diagnosis_color = "warning"
                    elif atrial_pct > 10:
                        final_diagnosis = "Atrial Arrhythmia - Moderate Risk"
                        diagnosis_color = "warning"
                    elif atrial_pct > 1:
                        final_diagnosis = "Occasional Atrial Beats - Monitor"
                        diagnosis_color = "info"
                    else:
                        final_diagnosis = "Normal Sinus Rhythm - Healthy"
                        diagnosis_color = "success"

                    # Display final diagnosis
                    if diagnosis_color == "success":
                        st.success(f"**Final Diagnosis:** {final_diagnosis}")
                    elif diagnosis_color == "warning":
                        st.warning(f"**Final Diagnosis:** {final_diagnosis}")
                    elif diagnosis_color == "error":
                        st.error(f"**Final Diagnosis:** {final_diagnosis}")
                    else:
                        st.info(f"**Final Diagnosis:** {final_diagnosis}")

                    # Create summary table
                    summary_data = pd.DataFrame({
                        'Beat Type': ['Normal', 'Atrial', 'Ventricular', 'TOTAL'],
                        'Count': [normal_beats, atrial_beats, ventricular_beats, total_beats_analyzed],
                        'Percentage': [
                            f"{normal_pct:.1f}%",
                            f"{atrial_pct:.1f}%",
                            f"{ventricular_pct:.1f}%",
                            "100.0%"
                        ],
                        'Status': [
                            '✅ Expected',
                            '⚠️ Monitor' if atrial_pct > 1 else '✅ Benign',
                            '🚨 Urgent' if ventricular_pct > 10 else (
                                '⚠️ Monitor' if ventricular_pct > 1 else '✅ Benign'),
                            '-'
                        ]
                    })

                    st.dataframe(summary_data, use_container_width=True, hide_index=True)

                    # Pie chart
                    if total_beats_analyzed > 0:
                        fig_summary = go.Figure(data=[go.Pie(
                            labels=['Normal', 'Atrial', 'Ventricular'],
                            values=[normal_beats, atrial_beats, ventricular_beats],
                            marker=dict(colors=['#00cc66', '#ffaa00', '#ff4444']),
                            hole=0.4
                        )])
                        fig_summary.update_layout(
                            title=f"Final Beat Distribution ({duration}s monitoring)",
                            height=350
                        )
                        st.plotly_chart(fig_summary, use_container_width=True)

        except Exception as e:
            with col1:
                st.error(f"Error: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p><strong>Real-Time Health Monitoring System</strong></p>
    <p>Built with Streamlit | Powered by 1D CNN | Dataset: MIT-BIH Arrhythmia Database</p>
    <p>⚠️ For Educational Purposes Only - Not for Clinical Use</p>
</div>
""", unsafe_allow_html=True)

