import timeit


class _TimerDecorator:
    def __init__(self, callback, with_args=False):
        self.callback = callback
        self.with_args = with_args

    def __call__(self, func):
        if self.with_args:

            def _decorated_func(*args, **kwargs):
                with Timer() as t:
                    _res = func(*args, **kwargs)
                self.callback(t, *args, **kwargs)
                return _res
        else:

            def _decorated_func(*args, **kwargs):
                with Timer() as t:
                    _res = func(*args, **kwargs)
                self.callback(t)
                return _res

        return _decorated_func


class Timer:
    """Main function class."""
    def __enter__(self):
        self.start = timeit.default_timer()
        return self

    def __exit__(self, *args):
        self.stop = timeit.default_timer()
        self.time = self.stop - self.start

    def __str__(self):
        return f'Duration: {self.time}, Start: {self.start}, Stop: {self.stop}'

    @classmethod
    def set(cls, *args, **kwargs):
        """Utility to decorate functions."""
        return _TimerDecorator(*args, **kwargs)
