from retina.option import Option
from retina.profiler import Profiler

if __name__ == '__main__':
    runtime_config = Option()
    profiler = Profiler(runtime_config)
    profiler.profile()
