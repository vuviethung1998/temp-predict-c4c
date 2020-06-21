# data_dir_temp = '../utils/data/temp/'
data_dir_temp = 'src/utils/data/temp/'
# data_dir_power = '../utils/data/power/'
data_dir_power = 'src/utils/data/power/'
# model_dir_temp = '../utils/model/temp/'
model_dir_temp = 'src/utils/model/temp/'
# model_dir_power = '../utils/model/power/'
model_dir_power = 'src/utils/model/power/'
model_temp_day = {
    "optimizer": "adam",
    "seq_len": 14,
    "horizon": 1,
    "input_dim": 2,
    "output_dim": 1,
    "rnn_units": 200,
    "rnn_layers": 2,
    "dropout": 0
}
model_temp_week = {
    "optimizer": "adam",
    "seq_len": 63,
    "horizon": 7,
    "input_dim": 2,
    "output_dim": 1,
    "rnn_units": 200,
    "rnn_layers": 2,
    "dropout": 0
}
model_temp_month = {
    "optimizer": "adam",
    "seq_len": 63,
    "horizon": 30,
    "input_dim": 2,
    "output_dim": 1,
    "rnn_units": 200,
    "rnn_layers": 2,
    "dropout": 0
}
model_power_day = {
    "optimizer": "adam",
    "seq_len": 14,
    "horizon": 1,
    "input_dim": 4,
    "output_dim": 1,
    "rnn_units": 200,
    "rnn_layers": 2,
    "dropout": 0
}
model_power_week = {
    "optimizer": "adam",
    "seq_len": 63,
    "horizon": 7,
    "input_dim": 4,
    "output_dim": 1,
    "rnn_units": 200,
    "rnn_layers": 2,
    "dropout": 0
}
model_power_month = {
    "optimizer": "adam",
    "seq_len": 63,
    "horizon": 30,
    "input_dim": 4,
    "output_dim": 1,
    "rnn_units": 200,
    "rnn_layers": 2,
    "dropout": 0
}