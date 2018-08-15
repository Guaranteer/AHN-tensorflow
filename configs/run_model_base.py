import sys
sys.path.append('..')
from models.model_base import Model
from trainers.trainer_base import Trainer

train_params = {
    'learning_rate': 1e-4,
    'lr_decay_n_iters': 3000,
    'lr_decay_rate': 0.8,
    'max_epoches': 1000,
    'early_stopping': 10,
    'cache_dir': '../results/',
    'summary_dir': '../summaries/',
    'display_batch_interval': 20,
    'summary_interval': 10,
    'evaluate_interval': 5,
    'saving_interval': 1000,
    'epoch_reshuffle': True
}

model_params = {
    'lstm_dim': 384,
    'second_lstm_dim': 384,
    'attention_dim': 256,
    'regularization_beta': 1e-7,
    'dropout_prob': 0.6
}

data_params = {
    'batch_size': 100,
    'n_classes': 5032,
    'n_types': 5,
    'input_video_dim': 202,
    'max_n_frames': 240,
    'max_n_q_words': 47,
    'input_ques_dim': 300,
    'ref_dim':300,

    'word2vec': '../../data/word2vec/word2vec.bin',
    'database':'../data/actNet200-V1-3.pkl',
    'question':'../data/question.json',
    'answer':'../data/answer.json',
    'real_train_proposals':'../data/train.json',
    'real_val_proposals':'../data/val_1.json',
    'real_test_proposals': '../data/val_2.json',
    'feature_path':'../../data/tsn_score/'
}

if __name__ == '__main__':
    model = Model(data_params, model_params)
    trainer = Trainer(train_params, model_params, data_params, model)
    trainer.train()
