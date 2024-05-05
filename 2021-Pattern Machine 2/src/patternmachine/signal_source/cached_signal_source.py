from patternmachine.signal_source.signal_source import SignalSource


class CachedSignalSource(SignalSource):
    def __init__(self) -> None:
        self.frames = self.load_frames()
        self.next_frame_id = 0
        super().__init__()

    def load_frames(self) -> None:
        raise NotImplementedError()

    def seek(self, frame_index=0):
        self.next_frame_id = frame_index

    @property
    def item_count(self):
        return len(self.frames)

    def item(self):
        while True:
            yield self.frames[self.next_frame_id % len(self.frames)]
            self.next_frame_id += 1
