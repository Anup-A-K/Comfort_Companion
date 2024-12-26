from collections import deque
from statistics import mean
import time
import json
from datetime import datetime

class MetricsTracker:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.response_times = deque(maxlen=window_size)
        self.user_ratings = deque(maxlen=window_size)
        self.similarity_scores = deque(maxlen=window_size)
        self.metrics_history = []
        self.metrics_file = "metrics_log.txt"
        
        # Initialize metrics file
        with open(self.metrics_file, 'w') as f:
            f.write(f"Metrics Log Started: {datetime.now().isoformat()}\n")
            f.write("-" * 50 + "\n\n")

    def calculate_similarity(self, reference: str, hypothesis: str) -> float:
        """Calculate simple word overlap similarity"""
        try:
            ref_words = set(reference.lower().split())
            hyp_words = set(hypothesis.lower().split())
            
            if not ref_words or not hyp_words:
                return 0.0
                
            # Calculate overlap
            overlap = len(ref_words.intersection(hyp_words))
            precision = overlap / len(hyp_words) if hyp_words else 0
            recall = overlap / len(ref_words) if ref_words else 0
            
            # F1 score
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0
                
            return f1
            
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0

    def evaluate_response(self, reference: str, hypothesis: str):
        """Evaluate response using similarity metric"""
        similarity = self.calculate_similarity(reference, hypothesis)
        self.similarity_scores.append(similarity)
        self._log_metric(f"Similarity score: {similarity:.3f}")

    def track_latency(self, start_time):
        """Track response time"""
        latency = time.time() - start_time
        self.response_times.append(latency)
        self._log_metric(f"Latency: {latency:.3f} seconds")
        return latency

    def add_user_rating(self, rating: float):
        """Add user feedback rating"""
        self.user_ratings.append(rating)
        self._log_metric(f"User Rating: {rating}")

    def get_metrics(self):
        """Get current metrics summary"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "latency": {
                "current": self.response_times[-1] if self.response_times else None,
                "average": mean(self.response_times) if self.response_times else None,
                "max": max(self.response_times) if self.response_times else None
            },
            "user_feedback": {
                "average_rating": mean(self.user_ratings) if self.user_ratings else None,
                "total_ratings": len(self.user_ratings)
            },
            "similarity": {
                "current": self.similarity_scores[-1] if self.similarity_scores else None,
                "average": mean(self.similarity_scores) if self.similarity_scores else None
            }
        }
        self.metrics_history.append(metrics)
        self._log_metrics_summary(metrics)
        return metrics

    def _log_metric(self, message):
        """Log individual metric to file"""
        try:
            with open(self.metrics_file, 'a') as f:
                f.write(f"[{datetime.now().isoformat()}] {message}\n")
        except Exception as e:
            print(f"Error logging metric: {e}")

    def _log_metrics_summary(self, metrics):
        """Log metrics summary to file"""
        try:
            with open(self.metrics_file, 'a') as f:
                f.write("\n=== Metrics Summary ===\n")
                f.write(f"Timestamp: {metrics['timestamp']}\n")
                
                # Latency metrics
                f.write("\nLatency:\n")
                f.write(f"  Current: {metrics['latency']['current']:.3f} seconds\n")
                f.write(f"  Average: {metrics['latency']['average']:.3f} seconds\n")
                f.write(f"  Max: {metrics['latency']['max']:.3f} seconds\n")
                
                # Similarity scores
                f.write("\nSimilarity Scores:\n")
                f.write(f"  Current: {metrics['similarity']['current']:.3f}\n")
                f.write(f"  Average: {metrics['similarity']['average']:.3f}\n")
                
                # User feedback metrics
                f.write("\nUser Feedback:\n")
                f.write(f"  Average Rating: {metrics['user_feedback']['average_rating']:.2f}\n")
                f.write(f"  Total Ratings: {metrics['user_feedback']['total_ratings']}\n")
                f.write("-" * 50 + "\n\n")
        except Exception as e:
            print(f"Error logging metrics summary: {e}")

    def save_metrics(self, filename='metrics_history.json'):
        """Save metrics history to JSON file"""
        try:
            with open(filename, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
        except Exception as e:
            print(f"Error saving metrics: {e}") 