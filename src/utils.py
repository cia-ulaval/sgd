import time


def create_run_name(config):
    return 'dataset=' + config['dataset'] + \
           '_noiseSTD=' + str(config['noise_std']) + \
           '_covarianceMODE=' + str(config['covariance_mode']) + \
           '_numSAMPLES=' + str(config['num_noise_samples']) + \
           '_lr=' + str(config['lr']) + \
           '_seed=' + str(config['seed']) + \
           f'_t={time.time_ns()}'
