from retina.serving import Serving
from retina.option import Option, parse_options
from retina.profiler import Profiler

if __name__ == '__main__':
    runtime_config = parse_options()

    if runtime_config.task == "serve":
        client = Serving(runtime_config)
        start_time, end_time = client.querying()
        
    
