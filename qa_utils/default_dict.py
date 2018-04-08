class DefaultDict():

    def __init__(self):
        self.train_params = {
            'learning_rate': 1e-3,
            'lr_decay_n_iters': 3000,
            'lr_decay_rate': 0.8,
            'max_epoches': 1000,
            'early_stopping': 5,
            'cache_dir': '../results',
            'display_batch_interval': 20,
            'summary_interval': 10,
            'evaluate_interval': 5,
            'saving_interval': 1000, # useless now
            'epoch_reshuffle': True
        }
        self.model_params = {
            'use_q_len': False,
            'model_name': 'unnamed_model',
            'lstm_dim': 1024, 
            'lstm_step': 20, # equals max_n_frames. 60 for 'xhy'
            'ques_embed_dim': 100, # useful only for custom embedding
            'attention_dim': 512,
            'regularization_beta': 1e-7,
            'use_notifier': False, # whether 'is_training' is needed
            'dropout_prob': 0.6,
            'boundary_dim': 128 # for boundary-aware lstm
        }
        self.data_params = {
            'dataset': 'yqf', # 'yqf' or 'xhy'
            'batch_size': 100,
            'n_classes': 300, # 495 for 'xhy'
            'n_types': 4, # 5 for 'xhy'
            'use_frame': True,
            'input_video_dim': 4096,
            'max_n_frames': 20, # 60 for 'xhy'
            'use_motion': False,
            'input_motion_dim': 4096,
            'max_n_motions': 5, # 45 for 'xhy'
            'max_n_q_words': 26,
            'use_qvec': False, # for pretrained word vector
            'input_ques_dim': 300,
            'ques_vecs_file': '/home/fenixlin/data/video_qa/yqf/word2vec_300d.mat', # path differs for 'xhy'
            'redis_key_file': '/home/fenixlin/data/video_qa/yqf/tgif_key_dict.pkl', # only for 'yqf'
            'train_file': '/home/fenixlin/data/video_qa/yqf/train_gif_qa.pkl',
            'valid_file': '/home/fenixlin/data/video_qa/yqf/val_gif_qa.pkl',
            'test_file': '/home/fenixlin/data/video_qa/yqf/test_gif_qa.pkl'
        }

    @property
    def train_params(self):
        return self.train_params

    @property
    def model_params(self):
        return self.model_params

    @property
    def data_params(self):
        return self.data_params
