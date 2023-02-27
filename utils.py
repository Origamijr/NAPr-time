import librosa
import inspect

def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')


def show_spectrogram(s):
    from config import data_config
    librosa.display.specshow(s,sr=data_config()['sr'])


class MovingAverage:
    def __init__(self, alpha=0):
        self.alpha = alpha
        self.value = 0
        self.n = 0

    def add(self, x, count=1):
        self.n += count
        if self.alpha == 0:
            self.value += count * (x - self.value) / self.n
        else:
            self.value *= 1 - self.alpha
            self.value += self.alpha * x
        return self.value
    
    def reset(self):
        self.value = 0
        self.n = 0


def filter_kwargs(kwargs, exclude=None, adapt_f=None):
    if adapt_f is not None:
        sig = inspect.signature(adapt_f)
        filter_keys = [param.name for param in sig.parameters.values() if param.kind == param.POSITIONAL_OR_KEYWORD]
        kwargs = {filter_key: kwargs[filter_key] for filter_key in filter_keys if filter_key in kwargs}
    if exclude is not None:
        kwargs = {key: kwargs[key] for key in kwargs if key not in exclude}
    return kwargs

if __name__ == "__main__":
    print(is_interactive())