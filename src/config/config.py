data_dir = 'src/utils/data/'
model_dir = 'src/utils/model/'
model_day = {
    "optimizer": "adam",
    "seq_len": 14,
    "horizon": 1,
    "input_dim": 2,
    "output_dim": 1,
    "rnn_units": 200,
    "rnn_layers": 2,
    "dropout": 0
}
model_week = {
    "optimizer": "adam",
    "seq_len": 63,
    "horizon": 7,
    "input_dim": 2,
    "output_dim": 1,
    "rnn_units": 200,
    "rnn_layers": 2,
    "dropout": 0
}
    
model_month = {
    "optimizer": "adam",
    "seq_len": 63,
    "horizon": 30,
    "input_dim": 2,
    "output_dim": 1,
    "rnn_units": 200,
    "rnn_layers": 2,
    "dropout": 0
}