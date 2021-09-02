import torch
import numpy as np

import sys, os, time, uuid, argparse
from pprint import pprint

from utils.helper import *
from utils.lr import LR
import utils.analyser as A

from data_pipeline.dataset import *
# from data_pipeline.datapipeline import *

from model.models import *

from torch.utils.data import DataLoader

class Main(object):

    def __init__(self, params):
        self.params = params
        self.logger = get_logger(self.params.name, self.params.log_dir, self.params.config_dir)

        '''Logging all Parametrs'''
        self.logger.info('PID: {}'.format(os.getpid()))
        self.logger.info(vars(self.params))

        '''Set Device as CPU/GPU'''
        if self.params.gpu != '-1' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.cuda.set_rng_state(torch.cuda.get_rng_state())
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device('cpu')

        '''Getting Datapipeline, Model and Optimizer'''
        self.data_loader                           = LoadData(params.data_path, params.dataset_name, params.data_pipeline_strategy_train, params.reverse)
        self.datapipeline                          = {}
        self.datapipeline['train']                 = self.add_datapipeline_train(self.params.data_pipeline_strategy_train, 'train')
        self.datapipeline['valid_tail_prediction'] = self.add_datapipeline_test(self.params.data_pipeline_strategy_test, 'valid', 'tail_prediction')
        self.datapipeline['valid_head_prediction'] = self.add_datapipeline_test(self.params.data_pipeline_strategy_test, 'valid', 'head_prediction')
        self.datapipeline['test_tail_prediction']  = self.add_datapipeline_test(self.params.data_pipeline_strategy_test, 'test', 'tail_prediction')
        self.datapipeline['test_head_prediction']  = self.add_datapipeline_test(self.params.data_pipeline_strategy_test, 'test', 'head_prediction')

        self.model        = self.add_model(self.params.model_name)
        self.optimizer    = self.add_optimizer(self.model.parameters())
        self.lr_scheduler = LR(self.optimizer,self.params)


    def add_datapipeline_train(self, data_pipeline_strategy, type):
        if data_pipeline_strategy   == 'one_to_n':
            return DataLoader(DataPipeline_Train_Tail_1toN(self.params, type, self.data_loader), batch_size=self.params.batch_size, shuffle=True, num_workers=self.params.num_data_workers, collate_fn=DataPipeline.collate_fn, pin_memory=True, drop_last=False)
        elif data_pipeline_strategy   == 'one_to_x':
            if self.params.reverse:
                return DataLoader(DataPipeline_Train_Tail_1toX(self.params, type, self.data_loader), batch_size=self.params.batch_size, shuffle=True, num_workers=self.params.num_data_workers, collate_fn=DataPipeline.collate_fn, pin_memory=True, drop_last=False)
            else:
                tail_batch = DataLoader(DataPipeline_Train_Tail_1toX(self.params, type, self.data_loader), batch_size=self.params.batch_size, shuffle=True, num_workers=self.params.num_data_workers, collate_fn=DataPipeline.collate_fn, pin_memory=True, drop_last=False)
                head_batch = DataLoader(DataPipeline_Train_Head_1toX(self.params, type, self.data_loader), batch_size=self.params.batch_size, shuffle=True, num_workers=self.params.num_data_workers, collate_fn=DataPipeline.collate_fn, pin_memory=True, drop_last=False)
                return BidirectionalOneShotIterator(tail_batch, head_batch)
        elif data_pipeline_strategy == 'one_to_nx':
            return DataLoader(DataPipeline_Train_Tail_1toNX(self.params, type, self.data_loader), batch_size=self.params.batch_size, shuffle=True, num_workers=self.params.num_data_workers, collate_fn=DataPipeline.collate_fn_nx, pin_memory=True, drop_last=False)
        else: raise NotImplementedError

    def add_datapipeline_test(self, data_pipeline_strategy, type, mode):
        if data_pipeline_strategy == 'one_to_n':
            if mode == 'tail_prediction':
                return DataLoader(DataPipeline_Test_Tail_1toN(self.params, type, self.data_loader), batch_size=self.params.eval_batch_size, shuffle=False, num_workers=self.params.num_data_workers, collate_fn=DataPipeline.collate_fn, pin_memory=True, drop_last=False)
            else:
                return DataLoader(DataPipeline_Test_Head_1toN(self.params, type, self.data_loader), batch_size=self.params.eval_batch_size, shuffle=False, num_workers=self.params.num_data_workers, collate_fn=DataPipeline.collate_fn, pin_memory=True, drop_last=False)
        else: raise NotImplementedError

    def add_model(self, model_name):
        if model_name.lower() == 'fcconve':
            model = FCConvE(self.params)
        elif model_name.lower() == 'fce':
            model = FCE(self.params)
        else:
            raise NotImplementedError

        model.to(self.device)
        return model

    def add_optimizer(self, parameters):
        if self.params.opt == 'adam'  : return torch.optim.Adam(parameters, lr=self.params.lr, weight_decay=self.params.l2)
        else                          : return torch.optim.SGD(parameters,  lr=self.params.lr, weight_decay=self.params.l2)

    def predict(self, type='valid',mode='tail_prediction', save_ranks=False):
        self.model.eval()

        results = {}

        epoch_start_time = time.time()
        step_start_time  = time.time()
        all_ranks = []

        # for step, batch in enumerate(self.datapipeline['{}_{}'.format(type, mode)].get_batch(False)):
        for step, batch in enumerate(self.datapipeline['{}_{}'.format(type, mode)]):

            input, output                                              = batch
            h, r, t, negative_entities, subsampling_weight, input_mode = input

            h, r, t, negative_entities, subsampling_weight, input_mode = h.to(self.device), r.to(self.device), t.to(self.device), negative_entities.to(self.device), subsampling_weight.to(self.device), input_mode
            y                                                          = output.to(self.device)

            entity_embedding                                           = None

            pred                                                       = self.model.forward(h, r, t, entity_embedding, negative_entities, input_mode, strategy=self.params.data_pipeline_strategy_test)

            if step % 100 == 0:
                self.logger.info('[{},{} Step{}]: Time:{}'.format(type.title(), mode.title(), step, time.strftime('%H:%M:%S:%f',time.gmtime(time.time()-step_start_time))))
                step_start_time = time.time()

            '''Calculate MR, MRR, and Hits@K'''
            range                     = torch.arange(pred.size()[0], device=self.device)
            target_entity             = t if input_mode == 'tail_prediction' else h
            raw_ranks                 = 1 + torch.argsort(torch.argsort(pred, dim =1, descending=True), dim =1, descending=False)[range,target_entity]

            target_pred               = pred[range,target_entity]
            pred[y]                   = -np.inf # -1.0
            pred[range,target_entity] = target_pred
            filtered_ranks            = 1 + torch.argsort(torch.argsort(pred, dim =1, descending=True), dim =1, descending=False)[range,target_entity]
            all_ranks.append(torch.stack([h, r, t, filtered_ranks], dim=1))

            for setting_name, setting_ranks in [('raw', raw_ranks.float()), ('filtered', filtered_ranks.float())]:
                results['{}_count'.format(setting_name)]            = torch.numel(setting_ranks) + results.get('{}_count'.format(setting_name), 0)
                results['{}_mr'.format(setting_name)]               = torch.sum(setting_ranks).item() + results.get('{}_mr'.format(setting_name), 0)
                results['{}_mrr'.format(setting_name)]              = torch.sum(1.0/setting_ranks).item() + results.get('{}_mrr'.format(setting_name), 0)
                for k in torch.arange(10):
                    k = k.item()
                    results['{}_hits@{}'.format(setting_name, k+1)] = torch.numel(setting_ranks[setting_ranks <= (k+1)]) + results.get('{}_hits@{}'.format(setting_name, k+1), 0)

        self.logger.info('[{},{}]: Total Time:{}'.format(type.title(), mode.title(), time.strftime('%H:%M:%S:%f',time.gmtime(time.time()-epoch_start_time))))
        all_ranks = torch.cat(all_ranks, dim=0)
        if save_ranks:
            self.dump_ranks(type, mode, all_ranks)

        return results

    def run_epoch(self, epoch, val_mrr = 0):
        self.model.train()
        losses = []

        epoch_start_time                          = time.time()
        step_start_time                           = time.time()

        # for step, batch in enumerate(self.datapipeline['train'].get_batch()):
        for step, batch in enumerate(self.datapipeline['train']):
            self.optimizer.zero_grad()

            input, output                                        = batch
            h, r, t, negative_entities, subsampling_weight, mode = input

            h, r, t, negative_entities, subsampling_weight, mode = h.to(self.device), r.to(self.device), t.to(self.device), negative_entities.to(self.device), subsampling_weight.to(self.device), mode
            output                                               = output.to(self.device)

            entity_embedding                                     = None

            pred                                  = self.model.forward(h, r, t, entity_embedding, negative_entities, mode, strategy=self.params.data_pipeline_strategy_train)
            loss                                  = self.model.loss(pred, output, subsampling_weight)

            loss.backward()

            self.optimizer.step()
            if self.params.lr_type == 'clr': self.lr_scheduler.step(epoch)

            losses.append(loss.cpu().item())

            if step % 100 == 0:
                self.logger.info('[Epoch {}, Step{}]: Train Loss:{:.5},  Val MRR:{:.5},  Test MRR:{:.5},  Time:{}'.format(epoch, step, np.mean(losses), self.best_val_mrr, self.best_test_mrr, time.strftime('%H:%M:%S',time.gmtime(time.time()-step_start_time))))
                step_start_time = time.time()

        if self.params.lr_type != 'clr': self.lr_scheduler.step(epoch, val_mrr)

        loss     = np.mean(losses)
        self.logger.info('[Epoch:{}]:  Training Loss:{:.4} Total Time:{}\n'.format(epoch, loss, time.strftime('%H:%M:%S:%f',time.gmtime(time.time()-epoch_start_time))))
        return loss

    def dump_ranks(self, typ, mode, ranks):
        save_path = os.path.join(self.params.ranks_dir, "{name}.{typ}.{mode}".format(name=self.params.name, typ=typ, mode=mode))
        ranks = ranks.cpu().numpy()
        np.savetxt(save_path, ranks, fmt="%d")

    def save_scores(self, save_path):
        pass

    def save_model(self, save_path):
        state = {
            'state_dict':       self.model.state_dict(),
            'best_val'  :       self.best_val,
            'best_test' :       self.best_test,
            'best_epoch':       self.best_epoch,
            'optimizer' :       self.optimizer.state_dict(),
            'args'      :       vars(self.params)
            }
        torch.save(state, save_path)

    def load_model(self, load_path):
        state              = torch.load(load_path)
        self.best_val      = state['best_val']
        self.best_test     = state['best_test']
        self.best_epoch    = state['best_epoch']
        self.best_val_mrr  = self.best_val['filtered_mrr']
        self.best_test_mrr = self.best_test['filtered_mrr']
        self.model.load_state_dict(state['state_dict'])
        self.optimizer.load_state_dict(state['optimizer'])

    def analyse(self):
        self.best_val_mrr, self.best_val, self.best_test, self.best_test_mrr, self.best_epoch = 0., {}, {}, 0., 0.
        val_mrr = 0
        save_path = os.path.join(self.params.model_dir,self.params.name)
        self.load_model(save_path)
        self.logger.info('Successfully Loaded previous model')
        weight = self.model.fc1.weight.detach().cpu().numpy()
        save = 'False'
        A.analyse(weight, self.params.model_name, self.params.dataset_name)
        if save:
            outfile = os.path.join(os.pardir, "interactions", self.params.model_name.lower(), self.params.dataset_name.lower())
            if not os.path.exists(outfile):
                os.makedirs(outfile)
            outfile = os.path.join(outfile, 'weight')
            np.save(outfile, weight)
            print("saved the model:%s" % outfile)


    def fit(self):
        self.best_val_mrr, self.best_val, self.best_test, self.best_test_mrr, self.best_epoch = 0., {}, {}, 0., 0.
        val_mrr = 0
        save_path = os.path.join(self.params.model_dir,self.params.name)

        if self.params.restore:
            self.load_model(save_path)
            self.logger.info('Successfully Loaded previous model')

            test_left_results  = self.predict(type ='test',mode ='tail_prediction', save_ranks=True)
            test_right_results = self.predict(type ='test',mode ='head_prediction', save_ranks=True)
            test_results       = get_combined_results(test_left_results, test_right_results)
            log_results("-1", test_results, 'Test', self.logger)

            valid_left_results  = self.predict(type ='valid',mode ='tail_prediction', save_ranks=True)
            valid_right_results = self.predict(type ='valid',mode ='head_prediction', save_ranks=True)
            valid_results       = get_combined_results(valid_left_results, valid_right_results)
            log_results("-1", valid_results, 'Valid', self.logger)

            return
        # with torch.no_grad():
        #     val_left_results       = self.predict(type ='valid',mode ='tail_prediction', save_ranks=True)
        #     val_right_results      = self.predict(type ='valid',mode ='head_prediction', save_ranks=True)
        # val_results                = get_combined_results(val_left_results, val_right_results)

        for epoch in range(self.params.nepochs):

            train_loss                 = self.run_epoch(epoch, val_mrr)

            if epoch >= (self.params.val_epoch_start-1):
                with torch.no_grad():
                    val_left_results       = self.predict(type ='valid',mode ='tail_prediction')
                    val_right_results      = self.predict(type ='valid',mode ='head_prediction')
                val_results                = get_combined_results(val_left_results, val_right_results)
                log_results(epoch, val_results, 'Validation', self.logger)

                val_mrr                = val_results['filtered_mrr']

                if val_results['filtered_mrr'] > self.best_val_mrr:
                    with torch.no_grad():
                        test_left_results  = self.predict(type ='test',mode ='tail_prediction', save_ranks=True)
                        test_right_results = self.predict(type ='test',mode ='head_prediction', save_ranks=True)
                    test_results           = get_combined_results(test_left_results, test_right_results)
                    log_results(epoch, test_results, 'Test', self.logger)

                    self.best_val          = val_results
                    self.best_val_mrr      = val_results['filtered_mrr']
                    self.best_test         = test_results
                    self.best_test_mrr     = self.best_test['filtered_mrr']
                    self.best_epoch        = epoch
                    self.save_model(save_path)

            self.logger.info('[Epoch {}]:  Training Loss: {:.5},  Valid MRR: {:.5},  Test MRR: {:.5}\n\n\n'.format(epoch, train_loss, self.best_val_mrr, self.best_test_mrr))


def getParser():
    parser = argparse.ArgumentParser(description="Parser For Arguments", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #paths
    parser.add_argument("--data_path",                      type=str,               default=os.path.join("data"),               help="Required, Base path for data folder")

    #dataset
    parser.add_argument("--dataset_name",                   type=str,               default="FB15K-237",                help="Required, Dataset Name")
    parser.add_argument("--file_numbers",                   type=int,nargs="+",     default=[-1,1,2,3,4,5,6,7],          help="File Number Partitions reqd for Training, for Serial specify all partitons")
    parser.add_argument("--prefetch_buffer_size",           type=int,               default=0,                          help="Preftch Buffer Size used for training, O: means optimal buffer size")

    #training
    parser.add_argument("--num_data_workers",               type=int,               default=3,                          help="No. of data workers for data loading")
    parser.add_argument("--seed",                           type=int,               default=42,                         help="Seed Number for Random Generator")
    parser.add_argument("--no_reverse",                     action='store_true',    default=False,                      help="Whether to do (h,r,?) and(t,r_rev,?) training, or (h,r,?) and (?,r,t)")
    parser.add_argument("--reverse",                        type=str,               default='True',                      help="Whether to do (h,r,?) and(t,r_rev,?) training, or (h,r,?) and (?,r,t)")
    parser.add_argument("--is_label_smoothing",             action='store_true',    default=True,                      help="Label Smoothing enable or disable, reqd True")
    parser.add_argument("--label_smoothing_epsilon",        type=float,             default=0.1,                        help="Label smoothing value")
    parser.add_argument("--batch_size",                     type=int,               default=128,                        help="Batch Size")
    parser.add_argument("--nepochs",                        type=int,               default=100,                          help="Number of epochs")
    parser.add_argument("--embedding_dim",                  type=int,               default=200,                        help="Embedding Dimension")
    parser.add_argument("--rel_embedding_dim",              type=int,               default=200,                        help="Relation Embedding Dimension")
    parser.add_argument("--ent_x_size",                     type=int,               default=500,                       help="Scored against X entities, reqd for ONE_TO_X")
    parser.add_argument("--data_pipeline_strategy_train",   type=str,               default='one_to_n',                 help="ONE_TO_N='one_to_n', ONE_TO_X='one_to_x', ONE_TO_NX='one_to_nx'")
    parser.add_argument("--model_name",                     type=str,               default='fcconve',                    help="Model Name: fcconve, fce,")
    parser.add_argument("--num_gpus_per_worker",            type=int,               default=1,                          help="For CPU:0, For GPU Serial:1, For GPU PS and COLLECTIVE_ALL_REDUCE: 1+")

    parser.add_argument("--ent_proj_dim",                  type=int,               default=200,                        help="Ent ProjFCCOnvE  Dimension")
    parser.add_argument("--rel_proj_dim",                  type=int,               default=200,                        help="Rel ProjFCConvE Dimension")

    #embedding
    parser.add_argument("--embedding_strategy",             type=str,               default='standard',                 help="Standard:1, Hash:2, SVD:3, Sketch:4, MCB:5, Kronecker:6")

    #testing
    parser.add_argument("--eval_prefetch_buffer_size",      type=int,               default=0,                          help="Preftch Buffer Size used for evaluation, O: means optimal buffer size")
    parser.add_argument("--eval_batch_size",                type=int,               default=128,                        help="Eval Batch Size")
    parser.add_argument("--data_pipeline_strategy_test",    type=str,               default='one_to_n',                 help="ONE_TO_N='one_to_n', ONE_TO_X='one_to_x'")
    parser.add_argument("--val_epoch_start",                type=int,               default=0,                          help="Required, Dataset Name")


    #Model Parameters
    parser.add_argument("--k_h",                            type=int,               default=10,                         help="Convolution Height. For Eg embedding_dim =200 == 10x20")
    parser.add_argument("--k_w",                            type=int,               default=20,                         help="Convolution Width. For Eg embedding_dim =200 == 10x20")
    parser.add_argument("--k_in_channels",                  type=int,               default=1,                          help="Convolution Channels. Valid Value are 1 and 2")
    parser.add_argument('--k_out_channels_1',               type=int,               default=32,                         help='Number of filters in convolution')
    parser.add_argument('--k_out_channels_2',               type=int,               default=16,                         help='final capsule size in CapsConvE')
    parser.add_argument('--ker_sz',                         type=int,nargs="+",     default=[3,3],                      help='Filter Kernel Size')
    parser.add_argument('--bias',                           action='store_false',    default=True,                       help='Use bias or not')
    parser.add_argument('--concat_form',                    type=str,               default='alternate',                help='Input concat form')
    parser.add_argument('--permute_dim',                    action='store_true',    default=False,                      help='Restore from the previously saved model')
    parser.add_argument('--group_convolution',              type=str,               default='pointwise',                help='depthwise/pointwise')

    #FC
    #parser.add_argument("--n_hidden",                       type=int,               default=1000,                       help="number of hidden units for Dark model")
    parser.add_argument('--fc1',                            type=int,               default=400,                        help='size of the firsct FC layer in Dark')

    #Dropout
    parser.add_argument('--dropout1',                       type=float,             default=0.2,                        help='Dropout for input map')
    parser.add_argument('--dropout2',                       type=float,             default=0.3,                        help='Dropout for feature map')
    parser.add_argument('--dropout3',                       type=float,             default=0.4,                        help='Dropout for full connected layer')

    #Learning Rate and Optimizer Parameters
    parser.add_argument("--lr_type",                        type=str,               default='reduce',                   help="For Standard/Adaptive: 'standard', CLR : 'clr', Decay : 'decay', ReduceLROnPlateau : 'reduce'")
    parser.add_argument("--lr",                             type=float,             default=0.0005,                     help="For Standard/Adaptive: Starting Learning Rate, CLR : Minimum LR")
    parser.add_argument("--lr_exp_decay_rate",              type=float,             default=0.995,                      help="For Standard/Adaptive: Starting Learning Rate, CLR : Minimum LR")
    parser.add_argument("--lr_mul_decay_rate",              type=float,             default=0.9,                        help="For Standard/Adaptive: Starting Learning Rate, CLR : Minimum LR")
    parser.add_argument("--reduce_patiene_epoch",           type=int,               default=3,                          help="No. of epochs for reaching min lr to max lr, i.e. half of cycle")
    parser.add_argument("--steps",                          type=int,nargs="+",     default=[30,60],                    help="File Number Partitions reqd for Training, for Serial specify all partitons")
    parser.add_argument("--clr_max_lr",                     type=float,             default=0.006,                      help="For CLR: Maximum Learning Rate ")
    parser.add_argument("--clr_step_epochs",                type=int,               default=3,                          help="No. of epochs for reaching min lr to max lr, i.e. half of cycle")
    parser.add_argument("--clr_max_cycle_epochs",           type=int,               default=15,                         help="Max no. of epochs to perform CLR: It should be twice nultiple of lr_step_epochs")
    parser.add_argument("--clr_gamma",                      type=float,             default=0.99994,                    help="For CLR exp_range: decaying paramter")
    parser.add_argument("--clr_mode",                       type=str,               default='triangular',               help="For CLR: triangular / triangular2 / exp_range ")
    parser.add_argument("--lr_type_after_max_clr_epochs",   type=str,               default='standard',                 help="For Standard/Adaptive: 'standard', Decay : 'decay', ReduceLROnPlateau : 'reduce'")
    parser.add_argument("--l2",                             type=float,             default=0.0,                         help="L2 Regularization for Optimizer")
    parser.add_argument("--opt",                            type=str,               default='adam',                     help="Optimizer to Use Adam/SGD ")

    #Loss
    parser.add_argument("--loss",                           type=str,               default='bce',                      help="Loss adversial/logsigmoid/bce/soft_margin/hard_margin")
    parser.add_argument("--adversarial_temperature",        type=int,               default=1,                          help="Adversial Temperature for adversial loss")
    parser.add_argument("--loss_uniform_weight",            type=str,               default='False',                    help="Uniform weight to positive and negative sample loss or based on subsampling weight, for adversial and logsigmoid")
    parser.add_argument("--gamma",                          type=float,             default=9.0,                        help="Margin")

    #GPU Ids
    parser.add_argument("--gpu",                            type=str,               default='0',                        help="Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0, For Multiple GPU = 0,1")

    #Model Directory
    parser.add_argument("--base_dir",                       type=str,               default=os.path.abspath(os.curdir), help="base directory. All other directories will be take relative to this one.")
    parser.add_argument("--ranks_dir",                      type=str,               default=os.path.join('logs', 'ranks1'),         help="Set Base Directory for saving ranks of test/valid triples.")
    parser.add_argument("--model_dir",                      type=str,               default=os.path.join('logs', 'saved_models'),  help="Set Base Model Directory for saving or restoring model.")
    parser.add_argument("--log_dir",                        type=str,               default=os.path.join('logs', 'log'),           help="Set Base Log Directory for logging.")
    parser.add_argument("--config_dir",                     type=str,               default=os.path.join('config'),                help="Set config directory for loading json logging config.")
    parser.add_argument("--name",                           type=str,               default='testrun_'+str(uuid.uuid4())[:8],                  help="Set filename for saving or restoring models")

    #Debug
    parser.add_argument("--restore",                        action='store_true',    default=False,                      help="Wheteher to restore model for Evaluation")
    parser.add_argument("--run",                            type=str,             default='manual',                   help="manual or gridsearch run")
    parser.add_argument("--parameters",                     type=str,             default='manual',                   help="manual or gridsearch run")
    parser.add_argument("--kg_analysis",                    type=str,             default='False',                   help="KG Analysis model or Default Model")
    parser.add_argument("--analyse",                        action='store_true',    default=False,                      help="True for running the model analysis")
    return parser

def set_params(params):
    params.ranks_dir  = os.path.join(params.base_dir, params.ranks_dir)
    params.model_dir  = os.path.join(params.base_dir, params.model_dir)
    params.log_dir    = os.path.join(params.base_dir, params.log_dir)

    if not os.path.exists(params.ranks_dir): os.makedirs(params.ranks_dir)
    if not os.path.exists(params.model_dir): os.makedirs(params.model_dir)
    if not os.path.exists(params.log_dir)  : os.makedirs(params.log_dir)

    params.config_dir = os.path.join(params.base_dir, params.config_dir)
    params.data_path  = os.path.join(params.base_dir, params.data_path)
    params.pmi_file   = os.path.join(params.data_path, params.pmi_file)

    params.reverse    = True if params.reverse == 'True' else False
    params.reverse    = True if params.data_pipeline_strategy_train in ['one_to_n', 'one_to_nx'] else params.reverse
    if params.no_reverse:    params.reverse = False
    if params.analyse:
        params.restore = True
    return params

def main():
    sys.path.append(os.path.dirname(os.getcwd()))
    parser = getParser()
    try:
        args = parser.parse_args()
    except:
        sys.exit(1)

    if not (args.restore or args.analyse):
        args.name = args.name + '_' + time.strftime("%d_%m_%Y") + '_' + time.strftime("%H:%M:%S")

    '''Seeting Seed'''
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    '''Setting GPU Devices to Run'''
    if args.num_gpus_per_worker > 0:
        set_gpu(args.gpu)

    args = set_params(args)

    model = Main(args)
    if args.analyse:
        model.analyse()
    else:
        model.fit()

if __name__ == "__main__":
        main()
