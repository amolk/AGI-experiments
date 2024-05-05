class Clock:
    def __init__(self, tau):
        self.tau = tau
        self.listeners = []
        self.reset()

    def register(self, iterator):
        assert iterator is not None
        assert hasattr(iterator, "next"), f"{iterator} does not have next()"
        assert hasattr(iterator, "current"), f"{iterator} does not have current property"

        self.listeners.append(iterator)

    def tick(self):
        self.t += 1
        for listener in self.listeners:
            listener.next(self)

    def reset(self):
        self.t = -1
