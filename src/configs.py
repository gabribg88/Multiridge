####
fig_width = 6
fig_height = 6
fig_size =  [fig_width,fig_height]
fig_params = {'lines.linewidth': 1,
              'axes.grid': True,
              'axes.xmargin': 0,
              'axes.ymargin': 0,
              'grid.linestyle': '--',
              'grid.linewidth': 0.5,
              'axes.labelsize': 13,
              'axes.titlesize': 14,
              'font.size': 11,
              'legend.fontsize': 12,
              'xtick.labelsize': 12,
              'ytick.labelsize': 12,
              'figure.figsize': fig_size,
              'savefig.dpi': 600,
              'axes.linewidth': 1.0,
              ##
              'xtick.direction': 'in',
              'xtick.major.size': 3,
              'xtick.major.width': 0.5,
              'xtick.minor.size': 1.5,
              'xtick.minor.width': 0.5,
              'xtick.minor.visible': True,
              'xtick.top': True,
              ##
              'ytick.direction': 'in',
              'ytick.major.size': 3,
              'ytick.major.width': 0.5,
              'ytick.minor.size': 1.5,
              'ytick.minor.width': 0.5,
              'ytick.minor.visible': True,
              'ytick.right': True,
              ##
              'font.family': 'serif',
              'mathtext.fontset': 'dejavuserif',
              }

###################
FORMAT = {'exp_id': 'str',
          'seed': 'int64',
          'torch_dtype': 'int64',
          'samples_number': 'int64',
          'features_number': 'int64',
          'snr_db': 'int64',
          'informative_frac': 'float64',
          'folds_number': 'int64',
          'initialization': 'float64',
          'epochs_number': 'int64',
          'learning_rate': 'float64'}