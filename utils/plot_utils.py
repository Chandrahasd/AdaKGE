import os
import numpy as np
import matplotlib
matplotlib.use('qt4agg')
from matplotlib import pyplot as plt
import scipy
import scipy.stats as stats
from itertools import product

name_map = {'hole':"HolE", "complex":"ComplEx", "distmult":"DistMult", "transe":"TransE", "transr":"TransR", 'stranse':"STransE", "add":"Additive", "mult":"Multiplicative", 'fcconve':'FCConvE', 'fb15k':'FB15K', 'fb15k237':'FB15K-237', 'fb15k-237':'FB15K-237', 'wn18':'WN18', 'wn18rr':'WN18RR'}
name_defs = {"atm":"Alignment to Mean (ATM)", "length":"Avg Vector Length", "conicity":"Conicity", "Average conicity":"Average Conicity", "Average length":"Average Vector length", 'maw':'Mean Absolute Weight'}
datasets = ["fb15k", "wn18", 'fb15k237', 'wn18rr']
num_relations = {'fb15k':1345, 'fb15k-237':237, 'fb15k237':237, 'wn18':18, 'wn18rr':11}

cone_perf_offsets = {"complex":{1: (-0.02, 0.9), 50: (0.01, 0.1)},
                     "hole"   :{1: (0.01, 0.1), 50: (-0.18,-0.9), 100:(-0.20,-0.1)},
                     "distmult":{1:(-0.02, -2.5), 50:(0.01, -1.1)},
                     "transe" :{1:(0.01,0.1)},
                     "transr": {1:(0.01,-1.1), 50:(0.01, -0.9)},
                     "stranse":{},
                     }
len_perf_offsets = { "complex" :{1:(-1.8, 1.0),  50: (0.1, -0.5), 100:(0.1, -0.6)},
                     "hole"    :{1:(-0.5, 1.0),   50: (0.15,-1), 100:(0.15, 0.0)},
                     "distmult":{1:(-2.4, -0.5), 50:(0.1, -0.5), 100:(0.1, -0.5)},
                     "transe"  :{1:(0.1, -0.5)},
                     "transr"  :{1:(0.1,-1.0),  50:(0.1, -1.2), 100:(0.1, -0.5)},
                     "stranse" :{1:(0.1, -0.5)},
                     }

niw_relation_offsets = {
        'fb15k':(0.01, 0.5),
        'fb15k-237':(-0.15, 0.0),
        'wn18':(0.01, 0.0),
        'wn18rr':(0.01, 0.0),
    }

niw_entity_offsets = {
        'fb15k':(0.01, 0.0),
        'fb15k-237':(0.01, 0.0),
        'wn18':(0.01, 0.0),
        'wn18rr':(0.01, 0.0),
    }

perf_offsets = {"conicity":cone_perf_offsets, "length":len_perf_offsets}
default_offset = (0.01, -0.5)
borderwidth = 5

#safe_colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
#safe_colors = ['#c7c7c7', '#737373', '#bdbdbd', '#525252', '#969696', '#252525', '#f0f0f0', '#f781bf', '#984ea3', '#999999', '#e41a1c', '#dede00']
#safe_colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33", "#a65628", "#f781bf"]
#safe_colors = ["#302c21", "#b1beac", "#324340", "#d8d0c2", "#696356", "#a7bdb7", "#ab928a", "#8eacb4"]
#safe_colors = ["#456a74", "#002a33", "#2178a3", "#0d353f", "#417792", "#274c56", "#2c647e", "#13526c"]
#safe_colors = ["#89a2ec", "#2580fe", "#2b5a9b", "#4372de", "#4795e0", "#3e6cba", "#4f8eed", "#6e88d0"]
#safe_colors = ['#c7c787', "#b1beac", "#324340", "#d8d0c2", "#696356", "#a7bdb7", "#ab928a", "#8eacb4"]
safe_colors = ["#a7bdb7","#a7bdb7","#a7bdb7","#a7bdb7","#a7bdb7","#a7bdb7","#a7bdb7","#d8d0c2","#d8d0c2","#d8d0c2","#d8d0c2","#d8d0c2","#d8d0c2", "#696356", "#a7bdb7", "#ab928a", "#8eacb4"]

cfg = {
    'linewidth':6,
    'color': 'k',
    'marker': '+',
    'linestyle': '--',
    'fontsize': 60,
    'borderwidth': 5,
    'npointsx': 5,
    'npointsy': 4,
    'weight': 'regular'
}

def plotone(weights, ax, **kwargs): #color='k', marker='+', linestyle='--'):
    # parse the arguments
    name = kwargs.get('name', 'Entity')
    dataset_name = kwargs.get('dataset_name', 'FB15K-237')
    color = kwargs.get('color', cfg.get('color', 'k'))
    marker = kwargs.get('marker', cfg.get('marker', '+'))
    linestyle = kwargs.get('linestyle', cfg.get('linestyle', '--'))

    linewidth = cfg.get('linewidth', 2)
    fontsize = cfg.get('fontsize', 12)
    borderwidth = cfg.get('borderwidth', 5)
    npointsx = cfg.get('npointsx', 5)
    npointsy = cfg.get('npointsy', 4)
    weight = cfg.get('weight', 'normal')

    # plot the density
    density = stats.gaussian_kde(weights)
    nSamples = weights.shape[0]
    markevery = nSamples//15
    x,y = np.histogram(weights, nSamples)
    if name not in ['Entity']:
        ax.axvline(x=weights.mean(), ymax=1, color=color, linestyle=linestyle, linewidth=linewidth*1.7)
        img = ax.plot(y, density(y), c=color, label=name, marker=marker, markevery=markevery, markersize=1.5*linewidth, markeredgewidth=linewidth, linewidth=linewidth)
    else:
        ax.axvline(x=weights.mean(), ymax=1, color=color, linestyle=linestyle, linewidth=linewidth*1.0)
        img = ax.plot(y, density(y), c=color, label=name, marker=marker, markevery=markevery, markersize=6*linewidth, markeredgewidth=linewidth, linewidth=linewidth)

    # set legend
    ax.set_label(name)
    ax.legend(loc='upper left', prop={'weight':weight, 'size':1.0*fontsize})
    # annotate the name
    if name.lower() == 'entity':
        x_offset = niw_entity_offsets[dataset_name.lower()][0]
        y_offset = niw_entity_offsets[dataset_name.lower()][1]
        wy = 11.0
    else:
        x_offset = niw_relation_offsets[dataset_name.lower()][0]
        y_offset = niw_relation_offsets[dataset_name.lower()][1]
        wy = 9.0
    # x_offset = -0.35
    # y_offset = 0
    wx = weights.mean()
    # wy = density(wx)
    # wy = 11.0 if name.lower() == 'entity' else 9.0
    ax.annotate("%0.3f" %(wx), (wx+x_offset, wy+y_offset), fontsize=fontsize*0.8, weight=weight)
    # ax.annotate("mean=%0.3f" %(wx), (wx+x_offset, wy+y_offset), fontsize=fontsize*0.8, weight=weight)

    # set axis properties
    for i in ax.spines.values():
        i.set_linewidth(borderwidth)
    minx = 0.0
    maxx = 1.0
    maxy = 12.0
    plt.xlim(minx, maxx)
    xlabel = "NAIW"
    ylabel = "Density"
    ax.set_xlabel(xlabel, fontsize=fontsize, weight=weight)
    if dataset_name.lower() in ['fb15k', 'wn18']:
        ax.set_ylabel(ylabel, fontsize=fontsize, weight=weight)
    plt.xticks(np.arange(minx, maxx+0.1, (maxx-minx)/npointsx), fontsize=fontsize*0.9, weight=weight)
    plt.yticks(np.arange(0,maxy+0.1, maxy/npointsy)[1:], fontsize=fontsize*0.9, weight=weight)
    # plt.ylim(miny,maxy)
    # if show:
        # plt.show()
    return img

def plotBoth(ent, rel, model_name, dataset_name):
    colors = ["k", "#a7bdb7","#d8d0c2","#d8d0c2","#d8d0c2","#d8d0c2","#d8d0c2", "#696356", "#a7bdb7", "#ab928a", "#8eacb4"]
    markers = "|s.+x3ov^<>p"
    # legends = ['%s:%s'%(model_name, dataset_name)]
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ent_linestyle = '--'
    rel_linestyle = ':'
    weight = cfg.get('weight', 'normal')
    imgs = []
    imgs.append(plotone(ent, ax, color=colors[0], marker=markers[0], linestyle=ent_linestyle, dataset_name=dataset_name, name='Entity'))
    imgs.append(plotone(rel, ax, color=colors[0], marker=markers[1], linestyle=rel_linestyle, dataset_name=dataset_name, name='Relation'))
    # plt.legend(imgs, loc='upper left', fontsize=cfg['fontsize'])
    ax.set_title('%s (%d relations)'%(name_map[dataset_name], num_relations[dataset_name]), fontsize=cfg['fontsize']*1.0, weight=weight, fontdict={"verticalalignment":"bottom"})
    # ax.set_title('%s:%s'%(name_map[model_name], name_map[dataset_name]), fontsize=cfg['fontsize']*1.3, weight=weight, fontdict={"verticalalignment":"bottom"})
    fig = plt.gcf()
    fig.set_size_inches(20, 16)
    outdir = 'plots'
    outfile = os.path.join(outdir, '%s'%(dataset_name))
    plt.savefig(outfile+".png", dpi=100, bbox_inches='tight')

def plotDistribution(gp, xlabel="maw", ylabel="Density", legends=None, modelName="hole", outfile=None, show=True, combined=True):
    markers = "+.x3ov^<>p"
    fontsize = 60
    linewidth = 10
    fig, ax = plt.subplots()
    figs = []
    npoints = 5
    npointsx = 4
    miny = 0
    maxy = 0
    minx = 0
    maxx = 0
    m,n = gp.shape
    conicity_x = 0
    conicity_y = 0
    conx_offset = 0
    cony_offset = 0
    if combined:
        gp = gp.reshape(1, m*n)
    for i in np.arange(gp.shape[0]):
        gpi = gp[i,:]
        nSamples = gpi.shape[0]
        density = stats.gaussian_kde(gpi)
        x,y = np.histogram(gpi, nSamples)
        maxy = max(np.ceil(density(y).max()), maxy)
        maxx = max(np.ceil(y.max()), maxx)
        minx = min(np.floor(y.min()), minx)
        figs.append(ax.plot(y, density(y), c="k", label=legends[i], marker=markers[i], markevery=100, markeredgewidth=10, markersize=30, linewidth=linewidth))
        #figs.append(ax.plot(y, density(y), c="k", label=legends[i], marker=markers[i], markevery=gpi.shape[0]/10, markeredgewidth=10, markersize=30, linewidth=linewidth))
        #figs.append(ax.plot(y, density(y), c=colors[i], label=legends[i], marker=markers[i], markevery=10, markeredgewidth=10, markersize=30, linewidth=linewidth))
    plt.axvline(x=gpi.mean(), ymax=1, color='k', linestyle='--', linewidth=linewidth*0.7)
    #plt.axvline(x=gpi.mean(), ymax=density(gpi.mean())/maxy, color='k', linestyle='--', linewidth=linewidth*0.7)
    conicity_x = gpi.mean()
    conicity_y = density(gpi.mean())

    if conicity_x > 0.2:
        conicity_x = -conicity_x
    #ax.annotate("Avg Length = %0.2f" %(gpi.mean()), (conicity_x+conx_offset, conicity_y+cony_offset), fontsize=fontsize*0.9, weight='bold')
    ax.annotate("%s = %0.2f" %(name_defs['conicity'], gpi.mean()), (conicity_x+conx_offset, conicity_y+cony_offset), fontsize=fontsize*0.9, weight='bold')

    #plt.legend(figs,  legendLabels, loc='upper right')
    #ax.legend(bbox_to_anchor=(0.001, 1.06, .998, .1), loc=3, ncol=gp.shape[0], mode='expand', borderaxespad=0., fontsize=fontsize*0.55)
    #ax.legend(loc='upper left')
    #plt.legend(figs, legendLabels, loc='upper right')
    for i in ax.spines.values():
        i.set_linewidth(borderwidth)
    if xlabel in ['conicity', 'atm']:
        plt.xlim(-1.0,1.0)
        minx = -1.0
        maxx = 1.0
    else:
        plt.xlim(minx, maxx+1)
        maxx = maxx+1
    xlabel = name_defs[xlabel]
    ax.set_title(name_map[modelName], fontsize=fontsize*1.3, weight='bold', fontdict={"verticalalignment":"bottom"})
    if modelName in ['transe', 'distmult']:
        ax.set_ylabel(ylabel, fontsize=fontsize, weight='bold')
    #if modelName in ['distmult', 'hole', 'complex']:
    ax.set_xlabel(xlabel, fontsize=fontsize, weight='bold')
    """
    if modelName in ['stranse', "transe", "transr"]:
        maxy = 4.0
        npoints = 4
    if modelName in ['distmult', "complex", "hole"]:
        maxy = 8
        npoints = 4
    """
    plt.xticks(np.arange(minx, maxx+0.1, (maxx-minx)/npointsx), fontsize=fontsize*0.9, weight='bold')
    #plt.xticks(list(plt.xticks()[0]) + [gp.mean()])
    plt.yticks(np.arange(0,maxy+0.1, maxy/npoints)[1:], fontsize=fontsize*0.9, weight='bold')
    plt.ylim(miny,maxy)
    fig = plt.gcf()
    # fig.set_size_inches(20, 16)
    # plt.savefig(outfile+".png", dpi=100)
    if show:
        plt.show()


