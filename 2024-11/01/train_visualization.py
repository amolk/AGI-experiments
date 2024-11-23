import matplotlib.pyplot as plt
import numpy as np


class ConfidenceVisualizer:
    def __init__(self, confidence_tracker):
        self.tracker = confidence_tracker

    def plot_confidence_evolution(self):
        """Plot the evolution of confidence for each level over training"""
        plt.figure(figsize=(12, 6))

        for level, confidences in enumerate(self.tracker.confidence_history):
            steps = np.arange(len(confidences))
            plt.plot(steps, confidences, label=f"Level {level}")

            # Add smoothed trend line
            window_size = min(100, len(confidences))
            if window_size > 0:
                smoothed = np.convolve(
                    confidences, np.ones(window_size) / window_size, mode="valid"
                )
                plt.plot(steps[window_size - 1 :], smoothed, "--", alpha=0.5)

        plt.xlabel("Training Steps")
        plt.ylabel("Confidence")
        plt.title("Evolution of Level Confidences During Training")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add confidence regime annotations
        plt.axhspan(0.8, 1.0, alpha=0.2, color="g", label="High Confidence")
        plt.axhspan(0.4, 0.8, alpha=0.2, color="y", label="Medium Confidence")
        plt.axhspan(0.0, 0.4, alpha=0.2, color="r", label="Low Confidence")

        plt.show()

    def plot_confidence_distribution(self, recent_steps=1000):
        """Plot the distribution of confidences for each level"""
        num_levels = len(self.tracker.confidence_history)
        fig, axes = plt.subplots(1, num_levels, figsize=(15, 5))

        for level, confidences in enumerate(self.tracker.confidence_history):
            recent_conf = confidences[-recent_steps:]
            if len(recent_conf) > 0:
                axes[level].hist(recent_conf, bins=50, alpha=0.7, density=True)
                axes[level].set_title(f"Level {level} Confidence")
                axes[level].set_xlabel("Confidence")
                axes[level].set_ylabel("Density")

        plt.tight_layout()
        plt.show()

    def plot_confidence_correlation(self):
        """Plot correlation between adjacent level confidences"""
        num_levels = len(self.tracker.confidence_history)

        if num_levels < 2:
            return

        fig, axes = plt.subplots(1, num_levels - 1, figsize=(15, 5))

        for i in range(num_levels - 1):
            conf_lower = self.tracker.confidence_history[i]
            conf_higher = self.tracker.confidence_history[i + 1]

            # Take min length in case of different lengths
            min_len = min(len(conf_lower), len(conf_higher))

            if min_len > 0:
                axes[i].scatter(
                    conf_lower[-min_len:], conf_higher[-min_len:], alpha=0.1
                )
                axes[i].set_title(f"Level {i} vs {i+1}")
                axes[i].set_xlabel(f"Level {i} Confidence")
                axes[i].set_ylabel(f"Level {i+1} Confidence")

        plt.tight_layout()
        plt.show()


# Extend ConfidenceTracker with visualization
class ConfidenceTracker:
    def __init__(self, model):
        self.model = model
        self.confidence_history = [[] for _ in range(model.num_levels)]
        self.visualizer = ConfidenceVisualizer(self)

    def update(self, outputs):
        for level, output in enumerate(outputs):
            self.confidence_history[level].append(output["confidence"].mean().item())

    def plot_confidence_evolution(self):
        self.visualizer.plot_confidence_evolution()

    def plot_confidence_distribution(self):
        self.visualizer.plot_confidence_distribution()

    def plot_confidence_correlation(self):
        self.visualizer.plot_confidence_correlation()


# Example usage in training loop:
"""
# In training loop
if batch_idx % plot_interval == 0:
    model.confidence_tracker.plot_confidence_evolution()
    model.confidence_tracker.plot_confidence_distribution()
    model.confidence_tracker.plot_confidence_correlation()
"""
