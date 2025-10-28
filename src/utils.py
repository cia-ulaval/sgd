def create_run_name(config):
    return 'dataset=' + config['dataset'] + \
           '_noiseSTD=' + str(config['noise_std']) + \
           '_noiseTYPE=' + config['noise_type'] + \
           '_lr=' + str(config['lr']) + \
           '_seed=' + str(config['seed'])