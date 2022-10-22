'''hyrank###########'''
config_hyrank = {}
config_hyrank['dataset'] = 'hyrank'
config_hyrank['verbose'] = True
config_hyrank['save_every_epoch'] = 200
config_hyrank['print_every'] = 100

config_hyrank['lr'] = 1e-5
config_hyrank['lr_schedule'] = 'manual_smooth'  # manual, plateau, or a number, or manual_smooth
config_hyrank['batch_size'] = 64
config_hyrank['epoch_num'] = 200
config_hyrank['init_std'] = 0.0099999
config_hyrank['init_bias'] = 0.0
config_hyrank['batch_norm'] = True
config_hyrank['batch_norm_eps'] = 1e-05
config_hyrank['batch_norm_decay'] = 0.9
config_hyrank['conv_filters_dim'] = 3

config_hyrank['e_pretrain'] = True
# config_hyrank['e_pretrain_sample_size'] = 800 #fix training
config_hyrank['e_pretrain_sample_size'] = 300 #1% training

config_hyrank['e_num_filters'] = 64
config_hyrank['e_num_layers'] = 3

config_hyrank['g_num_filters'] = 64
config_hyrank['g_num_layers'] = 3

config_hyrank['zdim'] = 64
config_hyrank['cost'] = 'pdis'  # l2, l2sq, l1, pdis
config_hyrank['lambda'] = 0.01
config_hyrank['n_classes'] = 14

config_hyrank['mlp_classifier'] = False
config_hyrank['eval_strategy'] = 1
config_hyrank['sampling_size'] = 10
config_hyrank['augment_z'] = True
config_hyrank['augment_x'] = False
config_hyrank['aug_rate'] = 0.4
config_hyrank['LVO'] = True

config_hyrank['window_size'] = 13
config_hyrank['num_pcs'] = 50
config_hyrank['train_ratio'] = 0.05