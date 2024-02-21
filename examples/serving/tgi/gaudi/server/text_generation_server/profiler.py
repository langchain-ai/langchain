import os
import threading
import queue

from contextlib import contextmanager
import time
import json

class FileWriter(threading.Thread):
    def __init__(self, filename, write_queue):
        super().__init__()
        self.filename = filename
        self.write_queue = write_queue
        self.daemon = True
        self.timer_event = threading.Event()

    def _drain_write_queue(self):
        content = ""
        while True:
            try:
                element = self.write_queue.get_nowait()
                content += element
            except queue.Empty:
                break
        return content

    def run(self):
        # don't check the queue too often
        while not self.timer_event.wait(1):
            # Block and wait for the next item in the queue
            content = self.write_queue.get()
            # Collect any other items in the queue
            content += self._drain_write_queue()

            with open(self.filename, "a") as outfile:
                outfile.write(content)

class Profiler():
    profiling_trace_events = queue.Queue()
    event_tid = {"counter": 1, "external": 2, "internal": 3, "own": 4}
    filename = "server_events.json"

    def __init__(self):
        self.enabled = os.getenv("TGI_PROFILER_ENABLED", "false").lower() == "true" and int(os.getenv("RANK", "0")) == 0
        if self.enabled:
            # initialize the trace file
            with open(self.filename, "w") as outfile:
                outfile.write('{"traceEvents": ')
            file_writer = FileWriter(self.filename, self.profiling_trace_events)
            file_writer.start()

    @contextmanager
    def record_event(self, type, name, args=None, util=None):
        if self.enabled:
            start = time.time() * 1000000.0
            if util is not None:
                self.profiling_trace_events.put(json.dumps([{
                "pid": 1,
                "tid": self.event_tid["counter"],
                "ph": "C",
                "name": "util",
                "ts": start,
                "args": {
                    "util": util["util"],
                }}]))

            event = {
                "pid": 1,
                "tid": self.event_tid[type],
                "ph": "X",
                "name": name,
                "ts": start,
                "dur": None,
                "args": args
            }
            yield

            end = time.time() * 1000000.0
            event["dur"] = end - start

            self.profiling_trace_events.put(json.dumps([event]))
        else:
            yield