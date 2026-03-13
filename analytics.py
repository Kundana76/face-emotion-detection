# src/analytics.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import os

class EmotionAnalytics:
    """Analytics and visualization for emotion data"""
    
    def __init__(self, data_source):
        """
        Initialize analytics
        
        Args:
            data_source: DataFrame or path to CSV file
        """
        if isinstance(data_source, str):
            self.df = pd.read_csv(data_source)
        else:
            self.df = data_source
        
        # Convert timestamp
        if 'timestamp' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.colors = px.colors.qualitative.Set3
    
    def plot_emotion_distribution(self, save_path=None):
        """Plot emotion distribution pie chart"""
        emotion_counts = self.df['emotion'].value_counts()
        
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'pie'}, {'type': 'bar'}]],
            subplot_titles=('Emotion Distribution', 'Emotion Counts')
        )
        
        # Pie chart
        fig.add_trace(
            go.Pie(
                labels=emotion_counts.index,
                values=emotion_counts.values,
                marker=dict(colors=self.colors),
                textinfo='label+percent',
                hole=0.3
            ),
            row=1, col=1
        )
        
        # Bar chart
        fig.add_trace(
            go.Bar(
                x=emotion_counts.index,
                y=emotion_counts.values,
                marker_color=self.colors,
                text=emotion_counts.values,
                textposition='auto',
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text="Emotion Distribution Analysis",
            showlegend=False,
            height=500
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_confidence_trends(self, save_path=None):
        """Plot confidence trends over time"""
        # Group by time and emotion
        self.df['hour'] = self.df['timestamp'].dt.hour
        self.df['date'] = self.df['timestamp'].dt.date
        
        # Average confidence by emotion
        avg_confidence = self.df.groupby('emotion')['confidence'].mean().reset_index()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Confidence by Emotion',
                'Confidence Over Time',
                'Emotion Timeline',
                'Confidence Distribution'
            )
        )
        
        # Bar chart of average confidence
        fig.add_trace(
            go.Bar(
                x=avg_confidence['emotion'],
                y=avg_confidence['confidence'],
                marker_color=self.colors,
                text=avg_confidence['confidence'].round(3),
                textposition='auto',
                name='Avg Confidence'
            ),
            row=1, col=1
        )
        
        # Confidence over time
        time_conf = self.df.groupby('timestamp')['confidence'].mean().reset_index()
        fig.add_trace(
            go.Scatter(
                x=time_conf['timestamp'],
                y=time_conf['confidence'],
                mode='lines+markers',
                name='Confidence Trend',
                line=dict(color='blue', width=2)
            ),
            row=1, col=2
        )
        
        # Emotion timeline
        for emotion in self.emotions:
            emotion_data = self.df[self.df['emotion'] == emotion]
            if not emotion_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=emotion_data['timestamp'],
                        y=[emotion] * len(emotion_data),
                        mode='markers',
                        name=emotion,
                        marker=dict(size=8),
                        showlegend=False
                    ),
                    row=2, col=1
                )
        
        # Confidence distribution
        for emotion in self.emotions:
            emotion_data = self.df[self.df['emotion'] == emotion]
            if not emotion_data.empty:
                fig.add_trace(
                    go.Violin(
                        y=emotion_data['confidence'],
                        name=emotion,
                        box_visible=True,
                        meanline_visible=True,
                        showlegend=False
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            title_text="Confidence Analysis Dashboard",
            height=800,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_heatmap(self, save_path=None):
        """Plot emotion transition heatmap"""
        # Create emotion transition matrix
        emotions_seq = self.df['emotion'].tolist()
        unique_emotions = list(set(emotions_seq))
        
        # Count transitions
        transition_matrix = np.zeros((len(unique_emotions), len(unique_emotions)))
        
        for i in range(len(emotions_seq) - 1):
            current_idx = unique_emotions.index(emotions_seq[i])
            next_idx = unique_emotions.index(emotions_seq[i + 1])
            transition_matrix[current_idx][next_idx] += 1
        
        # Normalize
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_matrix = np.divide(transition_matrix, row_sums, where=row_sums != 0)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=transition_matrix,
            x=unique_emotions,
            y=unique_emotions,
            colorscale='Viridis',
            text=np.round(transition_matrix, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Emotion Transition Heatmap",
            xaxis_title="Next Emotion",
            yaxis_title="Current Emotion",
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def generate_report(self, output_dir='data/reports'):
        """Generate comprehensive analytics report"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Emotion Analytics Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                .stats {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 20px 0; }}
                .stat-card {{ background: #f5f5f5; padding: 20px; border-radius: 10px; }}
                .stat-value {{ font-size: 24px; font-weight: bold; color: #0066cc; }}
                .stat-label {{ color: #666; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                img {{ max-width: 100%; height: auto; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>Emotion Detection Analytics Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-value">{len(self.df)}</div>
                    <div class="stat-label">Total Detections</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{self.df['emotion'].nunique()}</div>
                    <div class="stat-label">Unique Emotions</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{self.df['confidence'].mean():.3f}</div>
                    <div class="stat-label">Avg Confidence</div>
                </div>
            </div>
            
            <h2>Emotion Distribution</h2>
            <table>
                <tr>
                    <th>Emotion</th>
                    <th>Count</th>
                    <th>Percentage</th>
                    <th>Avg Confidence</th>
                </tr>
        """
        
        # Add emotion statistics
        emotion_stats = self.df.groupby('emotion').agg({
            'emotion': 'count',
            'confidence': 'mean'
        }).rename(columns={'emotion': 'count', 'confidence': 'avg_confidence'})
        
        for emotion, row in emotion_stats.iterrows():
            percentage = (row['count'] / len(self.df)) * 100
            html_content += f"""
                <tr>
                    <td>{emotion}</td>
                    <td>{row['count']}</td>
                    <td>{percentage:.1f}%</td>
                    <td>{row['avg_confidence']:.3f}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <h2>Time Series Analysis</h2>
        """
        
        # Add plots
        fig1 = self.plot_emotion_distribution()
        fig2 = self.plot_confidence_trends()
        fig3 = self.plot_heatmap()
        
        plot_paths = []
        for i, fig in enumerate([fig1, fig2, fig3]):
            plot_path = os.path.join(output_dir, f'plot_{timestamp}_{i}.html')
            fig.write_html(plot_path)
            plot_paths.append(plot_path)
            html_content += f'<iframe src="{os.path.basename(plot_path)}" width="100%" height="600"></iframe>'
        
        html_content += """
        </body>
        </html>
        """
        
        # Save report
        report_path = os.path.join(output_dir, f'report_{timestamp}.html')
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        return report_path