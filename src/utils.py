import time


def create_run_name(config):
    return 'dataset=' + config['dataset'] + \
           '_noiseSTD=' + str(config['noise_std']) + \
           '_covarianceMODE=' + str(config['covariance_mode']) + \
           '_noiseSCHEDULER=' + str(config.get('noise_scheduler')) + \
           '_a=' + str(config.get('noise_scheduler_trial', {}).get("a", None)) + \
           '_b=' + str(config.get('noise_scheduler_trial', {}).get("b", None)) + \
           '_c=' + str(config.get('noise_scheduler_trial', {}).get("c", None)) + \
           '_numSAMPLES=' + str(config['num_noise_samples_batch'] * config['num_noise_samples_accumulation']) + \
           '_lr=' + str(config['lr']) + \
           '_seed=' + str(config['seed']) + \
           f'_t={time.time_ns()}'
