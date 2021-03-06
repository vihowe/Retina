import os
from retina import profiler
from retina.serving import Serving
from retina.option import Option, parse_options, ServingConfig
from retina.profiler import Profiler

if __name__ == '__main__':
    runtime_config = parse_options()

    if runtime_config.task == "serve":
        serve_config = ServingConfig()
        client = Serving(runtime_config, serve_config)
        queries = client.querying()
        
        latency_dir= os.path.join(runtime_config.data_path, 'latency')
        # latency_file = os.path.join(latency_dir, runtime_config.models_name[0]+'_'+str(runtime_config.infer_bs)+'_'+str(runtime_config.total_queries)+'.csv')
        latency_file = os.path.join(latency_dir, runtime_config.models_name[0]+'.csv')
        os.makedirs(latency_dir, exist_ok=True)
        client.record_latency(queries, latency_file)
        
    else:
        profiler = Profiler(runtime_config)
        profiler.profile()
        
    
