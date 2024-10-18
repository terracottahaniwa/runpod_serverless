import threading
import time


class CounterTimer():
    def __init__(self):
        self.intervall = 1
        self.counter = 0
        self.step = 1
        self.is_ruuning = False
        self.hook = None
        self.thread = threading.Thread(
            target = self.run
        )

    def __enter__(self):
        self.is_running = True
        self.thread.start()
        return self

    def run(self):
        while self.is_running:
            if self.hook:
                self.hook(self)
            time.sleep(self.intervall)
            self.counter += self.step
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.is_running = False
        self.thread.join()


class ReturnableThread(threading.Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._return = None

    def run(self):
        self._return = self._target(
            *self._args,
            **self._kwargs
        )

    def join(self):
        super().join()
        return self._return

