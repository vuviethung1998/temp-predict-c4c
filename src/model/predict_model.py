import pandas as pd
import numpy  as np
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, Dense, Input
from keras.models import Model
from src.config.config import *
from keras import backend as K
import tensorflow as tf

#--------------------------
#get config
def getConfig(type, data_type):
    if type=='day':
        if data_type == 'power':
            cfg = model_power_day
        else:
            cfg = model_temp_day
    if type=='week':
        if data_type == 'power':
            cfg = model_power_week
        else:
            cfg = model_temp_week
    if type=='month':
        if data_type == 'power':
            cfg = model_power_month
        else:
            cfg = model_temp_month
    return cfg

# ---------------------------------------------------------------
#  Model Construction
def lstm_enc(input, rnn_unit, rnn_depth, rnn_dropout):
    """
    The intermediate LSTM layers return sequences, while the last returns a single element.
    The input is also a sequence. In order to match the shape of input and output of the LSTM
    to sum them we can do it only for all layers but the last.
    """

    x = input
    states = []
    for i in range(rnn_depth):
        lstm_layer = LSTM(rnn_unit, return_sequences=True,
                          return_state=True, name='LSTM_enc_{}'.format(i+1))
        x_rnn, state_h, state_c = lstm_layer(x)
        states += [state_h, state_c]
        x = x_rnn
    return x, states

def lstm_dec(input, rnn_unit, rnn_depth, rnn_dropout, init_states):
    """
    The intermediate LSTM layers return sequences, while the last returns a single element.
    The input is also a sequence. In order to match the shape of input and output of the LSTM
    to sum them we can do it only for all layers but the last.
    """
    layers_lstm = []
    x = input
    states = []
    for i in range(rnn_depth):
        lstm_layer = LSTM(rnn_unit, return_sequences=True,
                          return_state=True, name='LSTM_dec_{}'.format(i+1))
        layers_lstm.append(lstm_layer)
        x_rnn, state_h, state_c = lstm_layer(x, initial_state=init_states[2*i:2*(i+1)])
        states += [state_h, state_c]
        x = x_rnn
    return layers_lstm, x, states



def model_layer(model_dir ,model_type, optimizer  , input_dim, output_dim, rnn_units, rnn_layers,drop_out=0):
    # Model
    # encoder_inputs = Input(shape=(None, input_dim))
    # _, encoder_states = lstm_enc(encoder_inputs, rnn_unit=rnn_units,
    #                                 rnn_depth=rnn_layers,
    #                                 rnn_dropout=drop_out)
    # 
    # decoder_inputs = Input(shape=(None, output_dim))
    # layers, decoder_outputs, _ = lstm_dec(decoder_inputs, rnn_unit=rnn_units,
    #                                         rnn_depth=rnn_layers,
    #                                         rnn_dropout=drop_out,
    #                                         init_states=encoder_states)
    # 
    # decoder_dense = Dense(output_dim, activation='relu')
    # decoder_outputs = decoder_dense(decoder_outputs)
    # model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    # 
    # 
    # model.load_weights(model_dir + 'best_weight_{}.hdf5'.format(model_type))
    # model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
    # 
    # # Inference encoder_model
    # encoder_model = Model(encoder_inputs, encoder_states)
    # 
    # # Inference decoder_model
    # decoder_states_inputs = []
    # decoder_states = []
    # decoder_outputs = decoder_inputs
    # for i in range(rnn_layers):
    #     decoder_state_input_h = Input(shape=(rnn_units,))
    #     decoder_state_input_c = Input(shape=(rnn_units,))
    #     decoder_states_inputs += [decoder_state_input_h, decoder_state_input_c]
    #     d_o, state_h, state_c = layers[i](decoder_outputs, initial_state=decoder_states_inputs[2*i:2*(i+1)])
    #     decoder_outputs = d_o
    #     decoder_states += [state_h, state_c]
    # decoder_outputs = decoder_dense(decoder_outputs)
    # decoder_model = Model([decoder_inputs] + decoder_states_inputs,[decoder_outputs] + decoder_states)
    # 
    # return model, encoder_model, decoder_model
    encoder_inputs = Input(shape=(None, input_dim))
    _, encoder_states = lstm_enc( encoder_inputs, rnn_unit=rnn_units,
                                  rnn_depth=rnn_layers,
                                  rnn_dropout=drop_out)
    decoder_inputs = Input(shape=(None, output_dim))
    
    layers, decoder_outputs, _ = lstm_dec(decoder_inputs, rnn_unit=rnn_units,
                                          rnn_depth=rnn_layers,
                                          rnn_dropout=drop_out,
                                          init_states= encoder_states)
    decoder_dense = Dense(output_dim, activation='relu')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    
    
    model.load_weights(model_dir + 'best_weight_{}.hdf5'.format(model_type))
    model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
    
    # Inference encoder_model
    encoder_model = Model(encoder_inputs, encoder_states)
    
    # Inference decoder_model
    decoder_states_inputs = []
    decoder_states = []
    decoder_outputs = decoder_inputs
    for i in range(rnn_layers):
        decoder_state_input_h = Input(shape=(rnn_units,))
        decoder_state_input_c = Input(shape=(rnn_units,))
        decoder_states_inputs += [decoder_state_input_h, decoder_state_input_c]
        d_o, state_h, state_c = layers[i](decoder_outputs, initial_state=decoder_states_inputs[2*i:2*(i+1)])
        decoder_outputs = d_o
        decoder_states += [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs,[decoder_outputs] + decoder_states)
    return model, encoder_model, decoder_model

# Data create
# -----------------------------------------------------------
def create_data(data, seq_len, horizon, input_dim, output_dim): # seq len: lay bao nhieu data de hoc, horizon: lay bao nhieu data de du doan
    _data = data.copy()
    T = _data.shape[0]
    en_x = np.zeros(shape=((T - seq_len - horizon), seq_len, input_dim)) # (T - seqlen - horizon) : so bo = T / (seqlen + horizon) - 1
    de_x = np.zeros(shape=((T - seq_len - horizon), horizon, output_dim))
    de_y = np.zeros(shape=((T - seq_len - horizon), horizon, output_dim))

    # lấy điện năng,  
    load = _data[:, -1].copy()
    load = load.reshape(load.shape[0], 1)
    
    for i in range(T - seq_len - horizon):
        for j in range(input_dim):
            en_x[i, :, j] = _data[ i: i + seq_len, j]

        de_x[i, :, :] = load[i + seq_len - 1:i + seq_len + horizon - 1]
        de_x[i, 0, :] = 0
        de_y[i, :, :] = load[i + seq_len:i + seq_len + horizon]

    return en_x, de_x, de_y


def _predict(source, encoder_model, decoder_model, output_dim, horizon ):
    states_value = encoder_model.predict(source)
    target_seq = np.zeros((1, 1, output_dim))
    preds = np.zeros(shape=(horizon, output_dim), dtype='float32')
    for i in range(horizon):
        output = decoder_model.predict([target_seq] + states_value)
        output_tokens = output[0]
        # output_tokens = output_tokens[0, -1, 0]
        preds[i] = output_tokens
        target_seq = output_tokens

        # Update states
        states_value = output[1:]
    return preds


def predict(test_data_norm, scaler, seq_len, horizon, input_dim, output_dim, encoder_model, decoder_model):

    scaler = scaler
    data_test = test_data_norm.copy()
    l = seq_len
    h = horizon

    data = np.zeros(shape=(l, input_dim), dtype='float32')
    data[:l, :] = data_test[:l, :]

    _data = np.zeros(shape=(l + h, input_dim), dtype='float32')
    _data[:l, :] = data_test[:l, :]

    input = np.zeros(shape=(1, l, input_dim))
    input[0, :, :] = data.copy()
    yhats = _predict(input, encoder_model, decoder_model, output_dim, horizon)

    _data[l: l + h] = yhats
    predicted_data = scaler.inverse_transform(_data[l: l + h])[:,-1]

    # clear session to load again the model
    K.clear_session()
    return  predicted_data


def get_data_by_date_temp(day, month, year, type):
    data = pd.read_csv(data_dir_temp + 'month_day_year_temp.csv')

    # get idx respective to day
    idx_lst = data.index[ (data['day'] == day) & (data['month'] == month) & (data['year'] == year)].tolist()

    if len(idx_lst) == 0:
        return -1

    idx = idx_lst[0]

    # get data respective to type of data
    # if data is day: get 7 data points
    # elif data is month: get 63 data points
    # elif data is year: get 63 data points
    if type == 'day':
        return_data = data[idx-14:idx]
    elif type == 'week':
        return_data = data[idx-63:idx]
    elif type == 'month':
        return_data = data[idx-63:idx]

    test_data2d = return_data[['month', 'temp']].copy()
    data_ret = {}
    scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
    scaler.fit(test_data2d)

    test_data2d_norm = scaler.transform(test_data2d)

    data_ret['test_data_norm'] = test_data2d_norm.copy()

    data_ret['scaler'] = scaler

    return data_ret

def get_data_by_date_power(day, month, year, type):
    data = pd.read_csv(data_dir_power+'month_day_year_temp_increase_power.csv')

    # get idx respective to day
    idx_lst = data.index[ (data['day'] == day) & (data['month'] == month) & (data['year'] == year)].tolist()

    if len(idx_lst) == 0:
        return -1

    idx = idx_lst[0]

    # get data respective to type of data
    # if data is day: get 7 data points
    # elif data is month: get 63 data points
    # elif data is year: get 63 data points
    if type == 'day':
        return_data = data[idx-14:idx]
    elif type == 'week':
        return_data = data[idx-63:idx]
    elif type == 'month':
        return_data = data[idx-63:idx]

    test_data2d = return_data[['increase', 'holiday', 'month', 'power']].copy()
    data_ret = {}
    scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
    scaler.fit(test_data2d)

    test_data2d_norm = scaler.transform(test_data2d)

    data_ret['test_data_norm'] = test_data2d_norm.copy()

    data_ret['scaler'] = scaler

    return data_ret


def get_temp(day, month,year, type='day'):
    config = getConfig(type, data_type='temp')
    optimizer = config['optimizer']
    input_dim = config['input_dim']
    output_dim = config['output_dim']
    rnn_units =  config['rnn_units']
    rnn_layers = config['rnn_layers']
    dropout = config['dropout']
    seq_len = config['seq_len']
    horizon = config['horizon']
    model_type = type

    # get previous date based on type of model
    data = get_data_by_date_temp(day, month, year, type)
    if data == -1:
        return "Previous date not found in data store!!!"

    # model_type, optimizer  , input_dim, output_dim, rnn_units, rnn_layers,drop_out=0
    loaded_model, encoder_model, decoder_model = model_layer(model_dir= model_dir_temp,model_type=model_type, optimizer=optimizer ,\
                                                             input_dim=input_dim, output_dim=output_dim, \
                                                             rnn_units=rnn_units,rnn_layers=rnn_layers, \
                                                             drop_out= dropout)

    predicted_data = predict(test_data_norm=data['test_data_norm'], \
                                                    scaler=data['scaler'],seq_len=seq_len, horizon=horizon, \
                                                    input_dim=input_dim, output_dim=output_dim, \
                                                    encoder_model=encoder_model, decoder_model=decoder_model)
    return predicted_data


def get_power(day, month,year, type='day'):
    config = getConfig(type, data_type='power')
    optimizer = config['optimizer']
    input_dim = config['input_dim']
    output_dim = config['output_dim']
    rnn_units =  config['rnn_units']
    rnn_layers = config['rnn_layers']
    dropout = config['dropout']
    seq_len = config['seq_len']
    horizon = config['horizon']
    model_type = type

    # get previous date based on type of model
    data = get_data_by_date_power(day, month, year, type)
    if data == -1:
        return "Previous date not found in data store!!!"

    # model_type, optimizer  , input_dim, output_dim, rnn_units, rnn_layers,drop_out=0
    loaded_model, encoder_model, decoder_model = model_layer(model_dir= model_dir_power,model_type=model_type, optimizer=optimizer , \
                                                             input_dim=input_dim, output_dim=output_dim, \
                                                             rnn_units=rnn_units,rnn_layers=rnn_layers, \
                                                             drop_out= dropout)

    predicted_data = predict(test_data_norm=data['test_data_norm'], \
                             scaler=data['scaler'],seq_len=seq_len, horizon=horizon, \
                             input_dim=input_dim, output_dim=output_dim, \
                             encoder_model=encoder_model, decoder_model=decoder_model)
    return predicted_data

if __name__=="__main__":
    predicted_data = get_power(5, 6, 2017, 'day')
    print(predicted_data)