import re
import numpy as np


def clean_text(txt):
    txt = txt.lower()
    txt = re.sub(r"i'm", "i am", txt)
    txt = re.sub(r"he's", "he is", txt)
    txt = re.sub(r"she's", "she is", txt)
    txt = re.sub(r"that's", "that is", txt)
    txt = re.sub(r"what's", "what is", txt)
    txt = re.sub(r"where's", "where is", txt)
    txt = re.sub(r"\'ll", " will", txt)
    txt = re.sub(r"\'ve", " have", txt)
    txt = re.sub(r"\'re", " are", txt)
    txt = re.sub(r"\'d", " would", txt)
    txt = re.sub(r"won't", "will not", txt)
    txt = re.sub(r"can't", "can not", txt)
    txt = re.sub(r"[^\w\s]", "", txt)
    txt = re.sub(r"\[\w+\]", 'hi', txt)
    txt = " ".join(re.findall(r"\w+", txt))
    return txt


def import_movie_data(linesfile, convfile):
    with open(linesfile, encoding="utf-8", errors="ignore") as file:
        lines = file.read().split("\n")

    with open(convfile, encoding="utf-8", errors="ignore") as file:
        convs = file.read().split("\n")

    # import exchange impormation
    exchn = []
    for conv in convs:
        exchn.append(conv.split(" +++$+++ ")
                     [-1][1:-1].replace("'", "").replace(",", "").split())

    # import dialog
    diag = {}
    for line in lines:
        index = line.split(" +++$+++ ")
        diag[index[0]] = index[-1]

    question = []
    answer = []

    for conv in exchn:
        for i in range(len(conv) - 1):
            question.append(diag[conv[i]])
            answer.append(diag[conv[i+1]])
    return (question, answer)


def add_token(pairs, pairsnum=-1):
    input_docs = []
    target_docs = []
    input_tokens = set()
    target_tokens = set()
    for line in pairs[:pairsnum]:
        input_doc, target_doc = line[0], line[1]
        # Appending each input sentence to input_docs
        input_docs.append(input_doc)
        # Splitting words from punctuation
        target_doc = " ".join(re.findall(r"[\w']+|[^\s\w]", target_doc))
        # Redefine target_doc below and append it to target_docs
        target_doc = '<START> ' + target_doc + ' <END>'
        target_docs.append(target_doc)

        # Now we split up each sentence into words and add each unique word to our vocabulary set
        for token in re.findall(r"[\w']+|[^\s\w]", input_doc):
            if token not in input_tokens:
                input_tokens.add(token)
        for token in target_doc.split():
            if token not in target_tokens:
                target_tokens.add(token)
    input_tokens = sorted(list(input_tokens))
    target_tokens = sorted(list(target_tokens))
    num_encoder_tokens = len(input_tokens)
    num_decoder_tokens = len(target_tokens)

    return [(input_docs, target_docs), (input_tokens, target_tokens), (num_encoder_tokens, num_decoder_tokens)]


def create_feature_dict(input_tokens, target_tokens):
    input_features_dict = dict(
        [(token, i) for i, token in enumerate(input_tokens)])

    target_features_dict = dict(
        [(token, i) for i, token in enumerate(target_tokens)])

    reverse_input_features_dict = dict(
        (i, token) for token, i in input_features_dict.items())

    reverse_target_features_dict = dict(
        (i, token) for token, i in target_features_dict.items())

    return [(input_features_dict, target_features_dict), (reverse_input_features_dict, reverse_target_features_dict)]


def create_training_data(input_docs, target_docs, num_encoder_tokens, num_decoder_tokens, input_features_dict, target_features_dict):
    # Maximum length of sentences in input and target documents
    max_encoder_seq_length = max(
        [len(re.findall(r"[\w']+|[^\s\w]", input_doc)) for input_doc in input_docs])
    max_decoder_seq_length = max(
        [len(re.findall(r"[\w']+|[^\s\w]", target_doc)) for target_doc in target_docs])
    encoder_input_data = np.zeros(
        (len(input_docs), max_encoder_seq_length, num_encoder_tokens),
        dtype='float32')

    decoder_input_data = np.zeros(
        (len(input_docs), max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')

    decoder_target_data = np.zeros(
        (len(input_docs), max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')

    for line, (input_doc, target_doc) in enumerate(zip(input_docs, target_docs)):
        for timestep, token in enumerate(re.findall(r"[\w']+|[^\s\w]", input_doc)):
            # Assign 1. for the current line, timestep, & word in encoder_input_data
            encoder_input_data[line, timestep, input_features_dict[token]] = 1.

        for timestep, token in enumerate(target_doc.split()):
            decoder_input_data[line, timestep,
                               target_features_dict[token]] = 1.
            if timestep > 0:
                decoder_target_data[line, timestep - 1,
                                    target_features_dict[token]] = 1.

    return [encoder_input_data, (decoder_input_data, decoder_target_data), (max_encoder_seq_length, max_decoder_seq_length)]
