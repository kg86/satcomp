import time


class Timer:
    def __init__(self):
        self.t = time.time()
        self.times = dict()

    def __call__(self):
        x = time.time() - self.t
        self.t = time.time()
        return x

    def record(self, name):
        self.times[name] = time.time() - self.t
        self.t = time.time()
