"""
Web-based dashboard using Gradio (Hugging Face) for EEG classification.
Can be easily deployed to Hugging Face Spaces.
"""
import numpy as np
import gradio as gr
from datetime import datetime
from collections import deque
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
import torch


class WebDashboard:
    """Web dashboard using Gradio for EEG motor imagery classification."""
    
    def __init__(
        self, model, preprocess_fn, class_names: List[str],
        sfreq: int = 250, chunk_size: int = 125
    ):
        self.model = model
        self.preprocess_fn = preprocess_fn
        self.class_names = class_names
        self.sfreq = sfreq
        self.chunk_size = chunk_size
        
        # History
        self.prediction_history = deque(maxlen=100)
        self.probability_history = {name: deque(maxlen=100) for name in class_names}
        self.time_history = deque(maxlen=100)
        
    def classify_chunk(
        self, eeg_data: np.ndarray
    ) -> Tuple[str, float, Dict, go.Figure, go.Figure]:
        """
        Classify a chunk of EEG data and return visualizations.
        
        Args:
            eeg_data: EEG data array (will be preprocessed)
        
        Returns:
            Tuple of (prediction, confidence, probabilities_dict, prob_plot, radar_plot)
        """
        # Preprocess
        processed_data = self.preprocess_fn(eeg_data)
        
        # Predict
        device = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.no_grad():
            if isinstance(processed_data, np.ndarray):
                processed_data = torch.FloatTensor(processed_data).to(device)
            outputs = self.model(processed_data)
            if hasattr(outputs, "cpu"):
                outputs = outputs.cpu().numpy()
            probabilities = torch.softmax(torch.FloatTensor(outputs), dim=1).numpy()[0]
        
        # Get prediction
        predicted_idx = np.argmax(probabilities)
        prediction = self.class_names[predicted_idx]
        confidence = float(probabilities[predicted_idx])
        
        prob_dict = {name: float(prob) for name, prob in zip(self.class_names, probabilities)}
        
        # Update history
        current_time = datetime.now().timestamp()
        self.time_history.append(current_time)
        self.prediction_history.append((prediction, confidence))
        for name, prob in prob_dict.items():
            self.probability_history[name].append(prob)
        
        # Create probability timeline plot
        prob_fig = self._create_probability_plot()
        
        # Create radar chart
        radar_fig = self._create_radar_chart(prob_dict)
        
        return prediction, confidence, prob_dict, prob_fig, radar_fig
    
    def _create_probability_plot(self) -> go.Figure:
        """Create Plotly figure for probability timeline."""
        fig = go.Figure()
        
        colors = {
            "Rest": "gray",
            "Left Hand": "blue",
            "Right Hand": "green",
            "Both Fists": "orange",
            "Both Feet": "red"
        }
        
        time_array = np.array(list(self.time_history))
        if len(time_array) > 1:
            time_array = time_array - time_array[0]  # Normalize to start from 0
        
        for class_name in self.class_names:
            if len(self.probability_history[class_name]) > 0:
                probs = np.array(self.probability_history[class_name])
                fig.add_trace(go.Scatter(
                    x=time_array[-len(probs):],
                    y=probs,
                    mode="lines+markers",
                    name=class_name,
                    line=dict(color=colors.get(class_name, "black"), width=2),
                    marker=dict(size=4)
                ))
        
        fig.update_layout(
            title="Classification Probabilities Over Time",
            xaxis_title="Time (s)",
            yaxis_title="Probability",
            yaxis=dict(range=[0, 1]),
            hovermode="x unified",
            height=400
        )
        
        return fig
    
    def _create_radar_chart(self, probabilities: Dict[str, float]) -> go.Figure:
        """Create Plotly radar chart for class probabilities."""
        fig = go.Figure()
        
        values = [probabilities.get(name, 0.0) for name in self.class_names]
        
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # Close the circle
            theta=self.class_names + [self.class_names[0]],
            fill="toself",
            name="Probabilities",
            line=dict(color="blue", width=2),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickvals=[0.2, 0.4, 0.6, 0.8, 1.0]
                )
            ),
            showlegend=True,
            title="Classification Probabilities (Radar Chart)",
            height=500
        )
        
        return fig
    
    def create_interface(self):
        """Create Gradio interface."""
        with gr.Blocks(title="EEG Motor Imagery Classification Dashboard") as interface:
            gr.Markdown("# 🧠 Real-time EEG Motor Imagery Classification Dashboard")
            gr.Markdown("Upload EEG data or provide real-time data stream for classification.")
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Input area
                    with gr.Tab("Data Input"):
                        data_input = gr.File(label="Upload EEG Data File (EDF format)")
                        chunk_slider = gr.Slider(
                            minimum=0.1, maximum=5.0, value=0.5, step=0.1,
                            label="Chunk Size (seconds)"
                        )
                        classify_btn = gr.Button("Classify Chunk", variant="primary")
                    
                    # Results
                    with gr.Tab("Results"):
                        prediction_output = gr.Textbox(
                            label="Predicted Class",
                            value="None",
                            interactive=False
                        )
                        confidence_output = gr.Number(
                            label="Confidence",
                            value=0.0,
                            interactive=False
                        )
                        probabilities_output = gr.JSON(
                            label="All Class Probabilities",
                            value={}
                        )
                
                with gr.Column(scale=3):
                    # Probability timeline plot
                    prob_plot = gr.Plot(
                        label="Probability Timeline"
                    )
                    
                    # Radar chart
                    radar_plot = gr.Plot(
                        label="Probability Radar Chart"
                    )
            
            # Example: Real-time mode
            with gr.Row():
                realtime_toggle = gr.Checkbox(label="Enable Real-time Mode")
                status_text = gr.Textbox(
                    label="Status",
                    value="Ready",
                    interactive=False
                )
            
            # Classification function
            def process_classification(file, chunk_size):
                if file is None:
                    return "None", 0.0, {}, None, None
                
                # Load and preprocess file
                # This would call your data loading function
                # For now, return placeholder
                try:
                    # eeg_data = load_edf_file(file.name)
                    # prediction, confidence, probs, prob_fig, radar_fig = self.classify_chunk(eeg_data)
                    # return prediction, confidence, probs, prob_fig, radar_fig
                    return "Left Hand", 0.85, {"Left Hand": 0.85, "Right Hand": 0.10, "Rest": 0.05}, None, None
                except Exception as e:
                    return f"Error: {str(e)}", 0.0, {}, None, None
            
            classify_btn.click(
                fn=process_classification,
                inputs=[data_input, chunk_slider],
                outputs=[prediction_output, confidence_output, probabilities_output, prob_plot, radar_plot]
            )
            
            # Auto-update in real-time mode
            realtime_toggle.change(
                fn=lambda x: "Real-time mode enabled" if x else "Real-time mode disabled",
                inputs=[realtime_toggle],
                outputs=[status_text]
            )
        
        return interface
    
    def launch(self, share: bool = False, server_name: str = "0.0.0.0", server_port: int = 7860):
        """Launch the Gradio interface."""
        interface = self.create_interface()
        interface.launch(share=share, server_name=server_name, server_port=server_port)
