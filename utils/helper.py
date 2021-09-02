import logging
import logging.config
import os
import json
import sys

def set_gpu(gpus):
    os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

def get_logger(file_name, log_dir, config_dir):
	config_dict = json.load(open( os.path.join(config_dir, 'log_config.json')))
	config_dict['handlers']['file_handler']['filename'] = os.path.join(log_dir,  file_name.replace('/', '-'))
	logging.config.dictConfig(config_dict)
	logger = logging.getLogger(file_name)

	std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
	consoleHandler = logging.StreamHandler(sys.stdout)
	consoleHandler.setFormatter(logging.Formatter(std_out_format))
	logger.addHandler(consoleHandler)

	return logger

def get_combined_results(left_results, right_results):
	results = {}
	for setting_name in ['raw', 'filtered']:
		count = float(left_results['{}_count'.format(setting_name)])

		results['left_{}_mr'.format(setting_name)] = (left_results['{}_mr'.format(setting_name)]/count)*1
		results['left_{}_mrr'.format(setting_name)] = (left_results['{}_mrr'.format(setting_name)]/count)*100

		results['right_{}_mr'.format(setting_name)] = (right_results['{}_mr'.format(setting_name)]/count)*1
		results['right_{}_mrr'.format(setting_name)] = (right_results['{}_mrr'.format(setting_name)]/count)*100

		results['{}_mr'.format(setting_name)] = ((left_results['{}_mr'.format(setting_name)] + right_results['{}_mr'.format(setting_name)])/(2*count))*1
		results['{}_mrr'.format(setting_name)] = ((left_results['{}_mrr'.format(setting_name)] + right_results['{}_mrr'.format(setting_name)])/(2*count))*100

		for k in range(10):
			results['left_{}_hits@{}'.format(setting_name, k+1)] = (left_results['{}_hits@{}'.format(setting_name, k+1)]/count)*100
			results['right_{}_hits@{}'.format(setting_name, k+1)] = (right_results['{}_hits@{}'.format(setting_name, k+1)]/count)*100
			results['{}_hits@{}'.format(setting_name, k+1)]= ((left_results['{}_hits@{}'.format(setting_name, k+1)]+right_results['{}_hits@{}'.format(setting_name, k+1)])/(2*count))*100
	return results

def log_results(epoch, results, type, logger):
	for setting_name in ['raw', 'filtered']:
		logger.info('[Epoch {} {}]: {} MR:  Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(epoch, type, setting_name.title(), results['left_{}_mr'.format(setting_name)], results['right_{}_mr'.format(setting_name)], results['{}_mr'.format(setting_name)]))
		logger.info('[Epoch {} {}]: {} MRR: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(epoch, type, setting_name.title(), results['left_{}_mrr'.format(setting_name)], results['right_{}_mrr'.format(setting_name)], results['{}_mrr'.format(setting_name)]))
		for k in range(10):
			str = '\n' if k==9 else ''
			logger.info('[Epoch {} {}]: {} Hits@{}: Tail : {:.5}, Head : {:.5}, Avg : {:.5}{}'.format(epoch, type, setting_name.title(), k+1, results['left_{}_hits@{}'.format(setting_name, k+1)], results['right_{}_hits@{}'.format(setting_name, k+1)], results['{}_hits@{}'.format(setting_name, k+1)], str))
