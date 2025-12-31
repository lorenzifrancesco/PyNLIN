""" 
When run in background, continously
update a plot of the final channel on off gain. 
"""
import numpy as np
import matplotlib.pyplot as plt
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import os
from modules import cfg
from pynlin.utils import nu2lambda

class SignalPlotter(FileSystemEventHandler):
    def __init__(self, file_path, n_modes=2):
        self.file_path = file_path
        self.n_modes = n_modes
        self.cf = cfg.load_toml_to_struct("./input/config.toml")
        self.fig, self.ax = plt.subplots(figsize=(4, 3))
        self.lines = [self.ax.plot([], [], label=f"Mode {i+1}")[0] for i in range(self.n_modes+1)]
        # print the target gain on a constant line on the plot 
        self.x_axis = 1e6*nu2lambda(np.linspace(self.cf.center_frequency - self.cf.channel_spacing * self.cf.n_channels / 2, self.cf.center_frequency + self.cf.channel_spacing * self.cf.n_channels / 2, self.cf.n_channels))
        self.ax.set_xlabel(r"Channel wavelength [$\mu m$]")
        self.ax.set_ylabel(r"On Off Gain [dB]")
        self.ref = - self.cf.launch_power + 0.2e-3 * self.cf.fiber_length
        plt.tight_layout()
        # self.ax.legend()
        self.last_save_time = time.time()

    def update_plot(self):
        try:
            signals = np.load(self.file_path)
            for i, line in enumerate(self.lines):
                if i == self.n_modes:
                  line.set_data(self.x_axis, np.ones(signals.shape[0]) * np.mean(signals, axis=(0, 1)) + self.ref)
                else:
                  line.set_data(self.x_axis, signals[:, i] + self.ref)
                
            data_min = np.min(signals) + self.ref
            data_max = np.max(signals) + self.ref
            center = np.mean(signals, axis=(0, 1)) + self.ref
            # center = (data_max + data_min) / 2
            half_range = 8 / 2
            self.ax.set_ylim(center - half_range, center + half_range)
            
            self.ax.relim()
            self.ax.autoscale_view()
            if time.time() - self.last_save_time > 0.5:
                self.fig.savefig("media/signal_plot.png")
                self.last_save_time = time.time()

        except Exception as e:
            print(f"Error updating plot: {e}")

    def on_modified(self, event):
        if event.src_path == self.file_path:
            self.update_plot()

def monitor_file(file_path, n_modes=2):
    event_handler = SignalPlotter(file_path, n_modes=n_modes)
    observer = Observer()
    observer.schedule(event_handler, path=os.path.dirname(file_path) or ".", recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    file_path = "results/gain_walker.npy"
    monitor_file(file_path, n_modes=4)