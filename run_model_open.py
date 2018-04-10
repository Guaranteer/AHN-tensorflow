import sys
sys.path.append('./qa_utils/')
from model_open import Model
from trainer_open import Trainer

train_params = {
    'learning_rate': 1e-3,
    'lr_decay_n_iters': 1200,
    'lr_decay_rate': 0.8,
    'max_epoches': 1000,
    'early_stopping': 5,
    'cache_dir': './model/base/',
    'display_batch_interval': 20,
    'summary_interval': 10,
    'evaluate_interval': 5,
    'saving_interval': 1000,
    'epoch_reshuffle': True
}
model_params = {
    'lstm_dim': 384,
    'second_lstm_dim':384,
    'ref_dim':300,
    'attention_dim': 256,
    'regularization_beta': 1e-7,
    'dropout_prob': 0.6,
    'decode_dim': 256
}

data_params = {
    # general
    'batch_size': 64,
    'n_types': 5,
    'n_words': 9028,

    # feature
    'input_video_dim': 202,
    'max_n_frames': 240,

    # question answer
    'max_n_q_words': 20,
    'max_n_a_words': 10,
    'input_ques_dim': 300,

    # path
    'word_embedding': './qg/embedding.pkl',
    'word2index': './qg/word2index.pkl',
    'index2word': './qg/index2word.pkl',
<<<<<<< HEAD
    'train_json': './qg/train_clean1.json',
    'val_json': './qg/val_clean1.json',
    'test_json': './qg/test_clean1.json',
=======
    'train_json': './qg/train_clean.json',
    'val_json': './qg/val_clean.json',
    'test_json': './qg/test_clean.json',
>>>>>>> 15a0c9bc304f527f1c9f6d601121752501c2fecf
    'feature_path':'./tsn_score/'
}

if __name__ == '__main__':
    model = Model(data_params, model_params)
    trainer = Trainer(train_params, model_params, data_params, model)
    trainer.train()
