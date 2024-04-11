"""
Timing utilities, by Robert-Jan Bruintjes; modified by yclin.
"""
import time
import numpy as np
import datetime


def start_timing():
    start_time = time.time()
    return start_time


def end_timing(timing_data):
    """Returns time in seconds."""
    start_time = timing_data
    end_time = time.time()
    elapsed_time = end_time - start_time
    return elapsed_time


class MyTimer(object):
    def __init__(self, name=None, keep_n=10):
        self.name = name
        self.times = []
        self.anomaly = False
        self.keep = np.zeros((keep_n), dtype="float32")
        self.keep_n = keep_n
        self.keep_idx = 0

    def write(self, time):
        self.keep[self.keep_idx] = time
        self.keep_idx = (self.keep_idx + 1) % self.keep_n

    def __enter__(self):
        self.timing = start_timing()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if not self.anomaly:
            time = end_timing(self.timing)
            self.write(time)
        else:
            self.anomaly = False

    def start(self):
        self.__enter__()

    def stop(self):
        self.__exit__(None, None, None)

    def manual_entry(self, start_time, end_time):
        if not self.anomaly:
            elapsed_time = end_time - start_time
            self.write(elapsed_time)
        else:
            self.anomaly = False

    def mark_anomaly(self):
        self.anomaly = True

    def last(self):
        return self.keep[(self.keep_idx - 1) % self.keep_n]

    def mean(self):
        return np.mean(self.keep)

    def __str__(self):
        name_display = f"{self.name}" if self.name is not None else "Time"
        return f"{name_display} (s): {self.mean():.4f}"
