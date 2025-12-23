"""
Real-time PyQt5 dashboard for EEG motor imagery classification.
Shows live EEG data, classification probabilities, and radar chart.
"""
import sys
import numpy as np
from datetime import datetime
from collections import deque
from typing import Optional, Dict, List
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QGridLayout
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont
import torch


class EEGPlotWidget(QWidget):
    """Widget for plotting real-time EEG signals."""
    def __init__(self, n_channels: int = 64, buffer_size: int = 1000):
        super().__init__()
        self.n_channels = n_channels
        self.buffer_size = buffer_size
        self.data_buffer = deque(maxlen=buffer_size)
        self.time_buffer = deque(maxlen=buffer_size)
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
    def update_plot(self, data: np.ndarray, timestamps: np.ndarray):
        """Update EEG plot with new data."""
        # data: (n_channels, n_samples) or (n_samples,)
        if len(data.shape) == 1:
            data = data[np.newaxis, :]
        
        # Add to buffer
        for i in range(data.shape[1]):
            self.data_buffer.append(data[:, i])
            self.time_buffer.append(timestamps[i])
        
        if len(self.data_buffer) < 2:
            return
        
        # Plot
        self.ax.clear()
        data_array = np.array(self.data_buffer).T
        time_array = np.array(self.time_buffer)
        
        # Plot selected channels (limit to avoid clutter)
        channels_to_plot = min(10, self.n_channels)
        channel_indices = np.linspace(0, self.n_channels - 1, channels_to_plot, dtype=int)
        
        for i, ch_idx in enumerate(channel_indices):
            offset = i * 50  # Vertical offset between channels
            self.ax.plot(time_array, data_array[ch_idx] + offset,
                        linewidth=0.5, alpha=0.7, label=f"Ch{ch_idx}")
        
        self.ax.set_xlabel("Time (s)", fontsize=10)
        self.ax.set_ylabel("Channel (with offset)", fontsize=10)
        self.ax.set_title(
            f"Real-time EEG Signals (showing {channels_to_plot}/{self.n_channels} channels)",
            fontsize=12, fontweight="bold"
        )
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc="upper right", fontsize=8, ncol=2)
        
        self.canvas.draw()


class ProbabilityPlotWidget(QWidget):
    """Widget for plotting classification probabilities."""
    def __init__(self, class_names: List[str]):
        super().__init__()
        self.class_names = class_names
        self.prob_history = {name: deque(maxlen=50) for name in class_names}
        self.time_history = deque(maxlen=50)
        
        self.figure = Figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
    def update_probabilities(self, probabilities: Dict[str, float], timestamp: float):
        """Update probability plot with new classification."""
        self.time_history.append(timestamp)
        
        for class_name, prob in probabilities.items():
            if class_name in self.prob_history:
                self.prob_history[class_name].append(prob)
        
        if len(self.time_history) < 2:
            return
        
        self.ax.clear()
        time_array = np.array(self.time_history)
        
        colors = {
            "Rest": "gray",
            "Left Hand": "blue",
            "Right Hand": "green",
            "Both Fists": "orange",
            "Both Feet": "red"
        }
        
        for class_name in self.class_names:
            if len(self.prob_history[class_name]) > 0:
                probs = np.array(self.prob_history[class_name])
                color = colors.get(class_name, "black")
                self.ax.plot(
                    time_array[-len(probs):], probs,
                    label=class_name, color=color, linewidth=2, marker="o", markersize=4
                )
        
        self.ax.set_xlabel("Time (s)", fontsize=10)
        self.ax.set_ylabel("Probability", fontsize=10)
        self.ax.set_title("Classification Probabilities Over Time", fontsize=12, fontweight="bold")
        self.ax.set_ylim(0, 1)
        self.ax.legend(loc="upper left", fontsize=9)
        self.ax.grid(True, alpha=0.3)
        
        self.canvas.draw()


class RadarChartWidget(QWidget):
    """Widget for radar chart (spider chart) of class probabilities."""
    def __init__(self, class_names: List[str]):
        super().__init__()
        self.class_names = class_names
        self.n_classes = len(class_names)
        
        self.figure = Figure(figsize=(6, 6))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111, projection="polar")
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
    def update_radar(self, probabilities: Dict[str, float]):
        """Update radar chart with current probabilities."""
        self.ax.clear()
        
        # Prepare data
        angles = np.linspace(0, 2 * np.pi, self.n_classes, endpoint=False).tolist()
        values = [probabilities.get(name, 0.0) for name in self.class_names]
        
        # Complete the circle
        angles += angles[:1]
        values += values[:1]
        
        # Plot
        self.ax.plot(angles, values, "o-", linewidth=2, markersize=8)
        self.ax.fill(angles, values, alpha=0.25)
        
        # Set labels
        self.ax.set_xticks(angles[:-1])
        self.ax.set_xticklabels(self.class_names, fontsize=10)
        self.ax.set_ylim(0, 1)
        self.ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        self.ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title(
            "Classification Probabilities (Radar Chart)",
            fontsize=12, fontweight="bold", pad=20
        )
        
        self.canvas.draw()


class RealTimeDashboard(QMainWindow):
    """Main dashboard window for real-time EEG classification."""
    def __init__(
        self, model, preprocess_fn, class_names: List[str],
        sfreq: int = 250, chunk_size: int = 125
    ):  # 0.5s chunks at 250Hz
        super().__init__()
        self.model = model
        self.preprocess_fn = preprocess_fn
        self.class_names = class_names
        self.sfreq = sfreq
        self.chunk_size = chunk_size  # Samples per chunk
        
        # Data buffer (30 seconds)
        buffer_samples = 30 * sfreq
        self.data_buffer = deque(maxlen=buffer_samples)
        self.time_buffer = deque(maxlen=buffer_samples)
        self.start_time = datetime.now()
        
        # Statistics
        self.prediction_count = 0
        self.current_prediction = "None"
        self.current_confidence = 0.0
        self.current_probabilities = {name: 0.0 for name in class_names}
        
        self.init_ui()
        self.setup_timer()
        
    def init_ui(self):
        """Initialize user interface."""
        self.setWindowTitle("Real-time EEG Motor Imagery Classification Dashboard")
        self.setGeometry(100, 100, 1600, 1000)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Left panel: EEG plot
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        eeg_plot = EEGPlotWidget(n_channels=64)
        left_layout.addWidget(eeg_plot)
        left_panel.setLayout(left_layout)
        self.eeg_plot = eeg_plot
        
        # Right panel: Statistics and plots
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        
        # Statistics panel
        stats_widget = QWidget()
        stats_layout = QGridLayout()
        
        # Labels for statistics
        self.time_label = QLabel("Time: 00:00:00")
        self.time_label.setFont(QFont("Arial", 12, QFont.Bold))
        stats_layout.addWidget(QLabel("Current Time:"), 0, 0)
        stats_layout.addWidget(self.time_label, 0, 1)
        
        self.prediction_label = QLabel("Prediction: None")
        self.prediction_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.prediction_label.setStyleSheet("color: blue;")
        stats_layout.addWidget(QLabel("Prediction:"), 1, 0)
        stats_layout.addWidget(self.prediction_label, 1, 1)
        
        self.confidence_label = QLabel("Confidence: 0.00%")
        self.confidence_label.setFont(QFont("Arial", 12))
        stats_layout.addWidget(QLabel("Confidence:"), 2, 0)
        stats_layout.addWidget(self.confidence_label, 2, 1)
        
        self.count_label = QLabel("Classifications: 0")
        stats_layout.addWidget(QLabel("Total Classifications:"), 3, 0)
        stats_layout.addWidget(self.count_label, 3, 1)
        
        stats_widget.setLayout(stats_layout)
        stats_widget.setMaximumHeight(150)
        right_layout.addWidget(stats_widget)
        
        # Probability plot
        prob_plot = ProbabilityPlotWidget(self.class_names)
        right_layout.addWidget(prob_plot)
        self.prob_plot = prob_plot
        
        # Radar chart
        radar_chart = RadarChartWidget(self.class_names)
        right_layout.addWidget(radar_chart)
        self.radar_chart = radar_chart
        
        right_panel.setLayout(right_layout)
        
        # Add panels to main layout
        main_layout.addWidget(left_panel, 2)  # 2:1 ratio
        main_layout.addWidget(right_panel, 1)
        
    def setup_timer(self):
        """Setup timer for periodic updates."""
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_dashboard)
        self.update_timer.start(100)  # Update every 100ms
        
    def add_data_chunk(self, data: np.ndarray):
        """
        Add a chunk of EEG data and classify if buffer has 30 seconds.
        
        Args:
            data: EEG data (n_channels, n_samples)
        """
        # Generate timestamps
        current_time = (datetime.now() - self.start_time).total_seconds()
        n_samples = data.shape[1]
        timestamps = np.linspace(
            current_time - n_samples/self.sfreq,
            current_time, n_samples
        )
        
        # Add to buffer
        for i in range(n_samples):
            self.data_buffer.append(data[:, i])
            self.time_buffer.append(timestamps[i])
        
        # Check if we have enough data (30 seconds)
        if len(self.data_buffer) >= 30 * self.sfreq:
            # Process chunk and classify
            self.classify_current_buffer()
        
        # Update plots
        self.update_dashboard()
    
    def classify_current_buffer(self):
        """Classify current buffer of data."""
        if len(self.data_buffer) < 30 * self.sfreq:
            return
        
        # Get latest chunk (0.5 seconds)
        recent_data = np.array(list(self.data_buffer)[-self.chunk_size:]).T
        recent_data = recent_data[np.newaxis, ...]  # Add batch dimension
        
        # Preprocess
        processed_data = self.preprocess_fn(recent_data)
        
        # Predict
        device = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.no_grad():
            if isinstance(processed_data, np.ndarray):
                processed_data = torch.FloatTensor(processed_data).to(device)
            outputs = self.model(processed_data)
            if hasattr(outputs, "cpu"):
                outputs = outputs.cpu().numpy()
            probabilities = torch.softmax(torch.FloatTensor(outputs), dim=1).numpy()[0]
        
        # Update predictions
        prob_dict = {name: float(prob) for name, prob in zip(self.class_names, probabilities)}
        predicted_idx = np.argmax(probabilities)
        self.current_prediction = self.class_names[predicted_idx]
        self.current_confidence = float(probabilities[predicted_idx])
        self.current_probabilities = prob_dict
        self.prediction_count += 1
        
        # Update probability plot
        current_time = (datetime.now() - self.start_time).total_seconds()
        self.prob_plot.update_probabilities(prob_dict, current_time)
        
        # Update radar chart
        self.radar_chart.update_radar(prob_dict)
    
    def update_dashboard(self):
        """Update all dashboard components."""
        # Update time
        elapsed = datetime.now() - self.start_time
        time_str = str(elapsed).split(".")[0]
        self.time_label.setText(f"Time: {time_str}")
        
        # Update prediction
        self.prediction_label.setText(f"Prediction: {self.current_prediction}")
        
        # Update confidence
        self.confidence_label.setText(f"Confidence: {self.current_confidence:.2%}")
        
        # Update count
        self.count_label.setText(f"Classifications: {self.prediction_count}")
        
        # Update EEG plot
        if len(self.data_buffer) > 0:
            data_array = np.array(list(self.data_buffer)).T
            time_array = np.array(list(self.time_buffer))
            self.eeg_plot.update_plot(data_array, time_array)


def run_dashboard(model, preprocess_fn, class_names: List[str], data_source=None):
    """
    Run the real-time dashboard.
    
    Args:
        model: Trained model for classification
        preprocess_fn: Preprocessing function
        class_names: List of class names
        data_source: Optional data source (generator or callback)
    """
    app = QApplication(sys.argv)
    dashboard = RealTimeDashboard(model, preprocess_fn, class_names)
    dashboard.show()
    
    # If data source is provided, connect it
    if data_source is not None:
        def add_data():
            try:
                chunk = next(data_source)
                dashboard.add_data_chunk(chunk)
            except StopIteration:
                pass
        
        timer = QTimer()
        timer.timeout.connect(add_data)
        timer.start(500)  # Every 0.5 seconds
    
    sys.exit(app.exec_())
