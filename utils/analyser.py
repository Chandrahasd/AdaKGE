import os
import numpy as np
import matplotlib
matplotlib.use('qt4agg')
from matplotlib import pyplot as plt
import scipy
import scipy.stats as stats
from utils.plot_utils import plotDistribution, plotone, plotBoth

def analyse_one(weights):
    x = np.sum(np.abs(weight), 1)
    plt.hist(x, bins=100)

def analyse(weights, model_name, dataset_name, normalize=False):
    x,y = weights.shape
    weights = np.abs(weights)
    max_weight = weights.max()
    # if normalize:
        # weights = weights/max_weight
    ent = weights[:,:y//2]
    rel = weights[:,y//2:]
    nbins = 100
    ent = np.sum(ent, 1)
    rel = np.sum(rel, 1)
    max_weight = ent.max() if ent.max() > rel.max() else rel.max()
    if normalize:
        ent = ent/max_weight
        rel = rel/max_weight
    # legends = ['%s:%s'%(model_name, dataset_name)]
    # plotDistribution(ent.reshape(ent.shape[0], 1), legends=legends)
    # fig, (axe, axr) = plt.subplots(nrows=2, ncols=1)
    # axe.hist(ent, bins=nbins)
    # axr.hist(rel, bins=nbins)
    # axe.set_xlim([0, 300])
    # axr.set_xlim([0, 300])
    # fig.suptitle('%s:%s'%(model_name, dataset_name))
    # plt.show()
    plotBoth(ent, rel, model_name, dataset_name)

def main():
    datasets = ['wn18', 'wn18rr', 'fb15k', 'fb15k-237']
    modelnames = ['fcconve']
    basedir = os.path.join(os.pardir, "interactions")
    for model in modelnames:
        for dataset in datasets:
            weight_file = os.path.join(basedir, model, dataset, 'weight.npy')
            weights = np.load(weight_file)
            analyse(weights, model, dataset, normalize=True)
    # plt.show()

if __name__ == "__main__":
    main()
