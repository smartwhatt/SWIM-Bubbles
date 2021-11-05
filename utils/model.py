from tensorflow import keras
from keras.layers import Input, LSTM, Dense
from keras.models import Model


def create_model(num_encoder_tokens, num_decoder_tokens):
    #Dimensionality
    dimensionality = 256

    #Encoder
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder_lstm = LSTM(dimensionality, return_state=True)
    encoder_outputs, state_hidden, state_cell = encoder_lstm(encoder_inputs)
    encoder_states = [state_hidden, state_cell]
    #Decoder
    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    decoder_lstm = LSTM(dimensionality, return_sequences=True, return_state=True)
    decoder_outputs, decoder_state_hidden, decoder_state_cell = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    #Model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    #Compiling
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'], sample_weight_mode='temporal')

    return model