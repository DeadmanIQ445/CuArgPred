import numpy as np
from scipy.linalg import toeplitz
from copy import copy

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Conv1D, Dense, Conv2D, Flatten, Input, Embedding, concatenate
from tensorflow.keras import Model

tf.get_logger().setLevel('WARNING')


class DefaultEmbedding(Model):
    """
    Creates an embedder that provides the default value for the index -1. The default value is a zero-vector
    """

    def __init__(self, init_vectors=None, shape=None, trainable=True):
        super(DefaultEmbedding, self).__init__()

        if init_vectors is not None:
            self.embs = tf.Variable(init_vectors, dtype=tf.float32,
                                    trainable=trainable, name="default_embedder_var")
            shape = init_vectors.shape
        else:
            # TODO
            # the default value is no longer constant. need to replace this with a standard embedder
            self.embs = tf.Variable(tf.random.uniform(shape=(shape[0], shape[1]), dtype=tf.float32),
                                    name="default_embedder_pad")
        # self.pad = tf.zeros(shape=(1, init_vectors.shape[1]), name="default_embedder_pad")
        # self.pad = tf.random.uniform(shape=(1, init_vectors.shape[1]), name="default_embedder_pad")
        self.pad = tf.Variable(tf.random.uniform(shape=(1, shape[1]), dtype=tf.float32),
                               name="default_embedder_pad")

    def __call__(self, ids):
        emb_matr = tf.concat([self.embs, self.pad], axis=0)
        return tf.nn.embedding_lookup(params=emb_matr, ids=ids)
        # return tf.expand_dims(tf.nn.embedding_lookup(params=self.emb_matr, ids=ids), axis=3)


class PositionalEncoding(Model):
    def __init__(self, seq_len, pos_emb_size):
        """
        Create positional embedding matrix for tokens. Currently not using because it results in N^2 computational
        complexity. Should move this functionality to batch preparation.
        :param seq_len: maximum sequence length
        :param pos_emb_size: the dimensionality of positional embeddings
        """
        super(PositionalEncoding, self).__init__()

        positions = list(range(seq_len * 2))
        position_splt = positions[:seq_len]
        position_splt.reverse()
        self.position_encoding = tf.constant(toeplitz(position_splt, positions[seq_len:]),
                                             dtype=tf.int32,
                                             name="position_encoding")
        self.position_embedding = tf.Variable(tf.random.uniform(shape=(seq_len * 2, pos_emb_size), dtype=tf.float32),
                                              name="position_embedding")

    def __call__(self):
        return tf.nn.embedding_lookup(self.position_embedding, self.position_encoding, name="position_lookup")


class TextCnnLayer(Model):
    """

    """

    def __init__(self, out_dim, kernel_shape, activation=None):
        super(TextCnnLayer, self).__init__()

        self.kernel_shape = kernel_shape
        self.out_dim = out_dim
        self.textConv = Conv1D(filters=out_dim, kernel_size=3,
                               activation=activation, data_format='channels_last', padding='same')
        # self.textConv = Conv1D(filters=out_dim, kernel_size=3,
        #                           activation=activation, data_format='channels_last')

        padding_size = (self.kernel_shape[0] - 1) // 2
        assert padding_size * 2 + 1 == self.kernel_shape[0]
        self.pad_constant = tf.constant([[0, 0], [padding_size, padding_size], [0, 0], [0, 0]])

    def __call__(self, x):
        # padded = tf.pad(x, self.pad_constant)
        convolve = self.textConv(x)
        return tf.squeeze(convolve, axis=0)


class TextCnn(Model):
    """
    TextCnn model for classifying tokens in a sequence. The model uses following pipeline:

    token_embeddings (provided from outside) ->
    several convolutional layers, get representations for all tokens ->
    pass representation for all tokens through a dense network ->
    classify each token
    """

    def __init__(self, input_size, h_sizes, seq_len,
                 pos_emb_size, cnn_win_size, dense_size, num_classes,
                 activation=None, dense_activation=None, drop_rate=0.2):
        """

        :param input_size: dimensionality of input embeddings
        :param h_sizes: sizes of hidden CNN layers, internal dimensionality of token embeddings
        :param seq_len: maximum sequence length
        :param pos_emb_size: dimensionality of positional embeddings
        :param cnn_win_size: width of cnn window
        :param dense_size: number of unius in dense network
        :param num_classes: number of output units
        :param activation: activation for cnn
        :param dense_activation: activation for dense layers
        :param drop_rate: dropout rate for dense network
        """
        super(TextCnn, self).__init__()

        self.seq_len = seq_len
        self.h_sizes = h_sizes
        self.dense_size = dense_size
        self.num_classes = num_classes

        def infer_kernel_sizes(h_sizes):
            """
            Compute kernel sizes from the desired dimensionality of hidden layers
            :param h_sizes:
            :return:
            """
            kernel_sizes = copy(h_sizes)
            kernel_sizes.pop(-1)  # pop last because it is the output of the last CNN layer
            kernel_sizes.insert(0, input_size)  # the first kernel size should be (cnn_win_size, input_size)
            kernel_sizes = [(cnn_win_size, ks) for ks in kernel_sizes]
            return kernel_sizes

        kernel_sizes = infer_kernel_sizes(h_sizes)
        # self.layers_tok = [ TextCnnLayer(out_dim=h_size, kernel_shape=kernel_size, activation=activation)
        # for h_size, kernel_size in zip(h_sizes, kernel_sizes)]

        self.layers_tok = []
        # self.layers_pos = [TextCnnLayer(out_dim=h_size, kernel_shape=(cnn_win_size, pos_emb_size), activation=activation)
        #                for h_size, _ in zip(h_sizes, kernel_sizes)]

        # self.positional = PositionalEncoding(seq_len=seq_len, pos_emb_size=pos_emb_size)

        if dense_activation is None:
            dense_activation = activation

        self.attention = tfa.layers.MultiHeadAttention(head_size=1024, num_heads=1)

        self.dense_1 = Dense(dense_size, activation=dense_activation)
        self.dropout_1 = tf.keras.layers.Dropout(rate=drop_rate)
        self.dense_2 = Dense(num_classes, activation=None)  # logits
        self.dropout_2 = tf.keras.layers.Dropout(rate=drop_rate)

    def __call__(self, embs, training=True):

        temp_cnn_emb = embs  # shape (?, seq_len, input_size)
        # pass embeddings through several CNN layers
        for l in self.layers_tok:
            temp_cnn_emb = l(temp_cnn_emb)  # shape (?, seq_len, h_size)
        # reshape before passing through a dense network
        # token_features = tf.reshape(temp_cnn_emb, shape=(-1, self.h_sizes[-1])) # shape (? * seq_len, h_size[-1])
        # token_features = Flatten()(temp_cnn_emb)
        # local_h2 = self.dense_1(embs) # shape (? * seq_len, dense_size)
        # local_h2 = self.dropout_1(local_h2)
        tag_logits = self.dense_2(embs)  # shape (? * seq_len, num_classes)
        return tag_logits
        # return tf.reshape(tag_logits, (-1, self.seq_len, self.num_classes)) # reshape back, shape (?, seq_len, num_classes)
        # return tf.reshape(tag_logits, (-1, self.num_classes)) # reshape back, shape (?, seq_len, num_classes)


class TypePredictor(Model):
    """
    TypePredictor model predicts types for Python functions using the following inputs
    Tokens: FastText embeddings for tokens trained on a large collection of texts
    Graph: Graph embeddings pretrained with GNN model
    Prefix: Embeddings for the first n characters of a token
    Suffix: Embeddings for the last n characters of a token
    """

    def __init__(self, tok_embedder, num_classes, train_embeddings=False,
                 h_sizes=None, dense_size=8192,
                 seq_len=512, pos_emb_size=10, cnn_win_size=3,
                 crf_transitions=None, suffix_prefix_dims=50, suffix_prefix_buckets=1000):
        """
        Initialize TypePredictor. Model initializes embedding layers and then passes embeddings to TextCnn model
        :param tok_embedder: Embedder for tokens, TODO мб Сюда подавать енкодер Берта
        :param graph_embedder: Embedder for graph nodes
        :param train_embeddings: whether to finetune embeddings
        :param h_sizes: hiddenlayer sizes
        :param dense_size: sizes of dense layers
        :param num_classes: number of output classes
        :param seq_len: maximum length of sentences
        :param pos_emb_size: dimensionality of positional embeddings
        :param cnn_win_size: width of cnn window
        :param crf_transitions: CRF transition probabilities
        :param suffix_prefix_dims: dimensionality of suffix and prefix embeddings
        :param suffix_prefix_buckets: number of suffix and prefix embeddings
        """
        super(TypePredictor, self).__init__()

        if h_sizes is None:
            h_sizes = [1024]

        assert num_classes is not None, "set num_classes"

        self.seq_len = seq_len
        self.transition_params = crf_transitions

        # initialize embeddings
        with tf.device('/CPU:0'):
            self.tok_emb = tok_embedder

        # compute final embedding size after concatenation
        input_dim = 1024

        self.text_cnn = TextCnn(input_size=input_dim, h_sizes=h_sizes,
                                seq_len=seq_len, pos_emb_size=pos_emb_size,
                                cnn_win_size=cnn_win_size, dense_size=dense_size,
                                num_classes=num_classes, activation=tf.nn.relu,
                                dense_activation=tf.nn.tanh)

        self.crf_transition_params = None


    @tf.function(input_signature=[
        { "input_word_ids": tf.TensorSpec(shape=[None,512], dtype=tf.int32),
           "input_type_ids": tf.TensorSpec(shape=[None,512], dtype=tf.int32),
           "input_mask": tf.TensorSpec(shape=[None,512], dtype=tf.int32)}])
    def call(self, token_ids):
        """
        Inference
        :param token_ids: ids for tokens, shape (?, seq_len)
        :param prefix_ids: ids for prefixes, shape (?, seq_len)
        :param suffix_ids: ids for suffixes, shape (?, seq_len)
        :param graph_ids: ids for graph nodes, shape (?, seq_len)
        :param training: whether to finetune embeddings
        :return: logits for token classes, shape (?, seq_len, num_classes)
        """
        tok_emb = self.tok_emb(token_ids)[0]
        logits = self.text_cnn(tok_emb, training=False)
        return logits


    def __call__(self, token_ids, training=True):
        """
        Inference
        :param token_ids: ids for tokens, shape (?, seq_len)
        :param prefix_ids: ids for prefixes, shape (?, seq_len)
        :param suffix_ids: ids for suffixes, shape (?, seq_len)
        :param graph_ids: ids for graph nodes, shape (?, seq_len)
        :param training: whether to finetune embeddings
        :return: logits for token classes, shape (?, seq_len, num_classes)
        """
        tok_emb = self.tok_emb(token_ids)[0]
        logits = self.text_cnn(tok_emb, training=training)
        return logits


    # @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
    # def call(self, token_ids):
    #     tok_emb = self.tok_emb(token_ids)[0]
    #     logits = self.text_cnn(tok_emb, training=training)
    #     return logits

    def loss(self, logits, labels, lengths, class_weights=None, extra_mask=None):
        """
        Compute cross-entropy loss for each meaningful tokens. Mask padded tokens.
        :param logits: shape (?, seq_len, num_classes)
        :param labels: ids of labels, shape (?, seq_len)
        :param lengths: actual sequence lenghts, shape (?,)
        :param class_weights: optionally provide weights for each token, shape (?, seq_len)
        :param extra_mask: mask for hiding some of the token labels, not counting them towards the loss, shape (?, seq_len)
        :return: average cross-entropy loss
        """

        a = tf.one_hot(labels, depth=logits.shape[-1]) > 0
        losses = tf.nn.softmax_cross_entropy_with_logits(a, logits, axis=-1)
        seq_mask = tf.sequence_mask(lengths, self.seq_len)
        if extra_mask is not None:
            seq_mask = tf.math.logical_and(seq_mask, extra_mask)
        if class_weights is None:
            loss = tf.reduce_mean(tf.boolean_mask(losses, seq_mask))
        else:
            loss = tf.reduce_mean(tf.boolean_mask(losses * class_weights, seq_mask))

        return loss

    
    def score(self, logits, labels, lengths, scorer=None, extra_mask=None):
        """
        Compute precision, recall and f1 scores using the provided scorer function
        :param logits: shape (?, seq_len, num_classes)
        :param labels: ids of token labels, shape (?, seq_len)
        :param lengths: tensor of actual sentence lengths, shape (?,)
        :param scorer: scorer function, takes `pred_labels` and `true_labels` as aguments
        :param extra_mask: mask for hiding some of the token labels, not counting them towards the score, shape (?, seq_len)
        :return:
        """
        mask = tf.sequence_mask(lengths, self.seq_len)
        if extra_mask is not None:
            mask = tf.math.logical_and(mask, extra_mask)
        true_labels = tf.boolean_mask(labels, mask)
        argmax = tf.math.argmax(logits, axis=-1)
        masked_pred = tf.boolean_mask(logits, mask)
        estimated_labels = tf.cast(tf.boolean_mask(argmax, mask), tf.int32)
        score = scorer(estimated_labels.numpy(), true_labels.numpy(), average='micro')
        top_k = tf.nn.top_k(masked_pred, k=5).indices
        t_k_score = []
        for i in range(top_k.shape[0]):
            t_k_score.append(1 if true_labels[i] in top_k[i] else 0)
        p, r, f1 = score[0], score[1], score[2]
        return p, r, f1, sum(t_k_score) / len(t_k_score) if len(t_k_score) > 0 else 0.0


def train_step_finetune(model, optimizer, token_ids, labels, lengths,
                        extra_mask=None, class_weights=None, scorer=None, finetune=False):
    """
    Make a train step
    :param model: TypePrediction model instance
    :param optimizer: tf optimizer
    :param token_ids: ids for tokens, shape (?, seq_len)
    :param prefix: ids for prefixes, shape (?, seq_len)
    :param suffix: ids for suffixes, shape (?, seq_len)
    :param graph_ids: ids for graph nodes, shape (?, seq_len)
    :param labels: ids for labels, shape (?, )
    :param lengths: actual sequence lengths, shape (?, )
    :param extra_mask: additional mask to hide tokens that should be labeled, but are not labeled, shape (?, seq_len)
    :param class_weights: weight of each token, shape (?, seq_len)
    :param scorer: scorer function, takes `pred_labels` and `true_labels` as aguments
    :param finetune: whether to train embeddings
    :return: values for loss, precision, recall and f1-score
    """
    with tf.GradientTape() as tape:
        logits = model(token_ids, training=True)  # TODO
        loss = model.loss(logits, labels, lengths, class_weights=class_weights, extra_mask=extra_mask)
        p, r, f1, t_k_score = model.score(logits, labels, lengths, scorer=scorer, extra_mask=extra_mask)
        gradients = tape.gradient(loss, model.trainable_variables)
        if not finetune:
            # do not update embeddings
            # pop embeddings related to embedding matrices
            optimizer.apply_gradients((g, v) for g, v in zip(gradients, model.trainable_variables) if
                                      not v.name.startswith("default_embedder"))
        else:
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, p, r, f1, t_k_score


# @tf.function
def test_step(model, token_ids, labels, lengths, extra_mask=None, class_weights=None, scorer=None):
    """

    :param model: TypePrediction model instance
    :param token_ids: ids for tokens, shape (?, seq_len)
    :param prefix: ids for prefixes, shape (?, seq_len)
    :param suffix: ids for suffixes, shape (?, seq_len)
    :param graph_ids: ids for graph nodes, shape (?, seq_len)
    :param labels: ids for labels, shape (?, )
    :param lengths: actual sequence lengths, shape (?, )
    :param extra_mask: additional mask to hide tokens that should be labeled, but are not labeled, shape (?, seq_len)
    :param class_weights: weight of each token, shape (?, seq_len)
    :param scorer: scorer function, takes `pred_labels` and `true_labels` as aguments
    :return: values for loss, precision, recall and f1-score
    """
    logits = model(token_ids, training=False)
    loss = model.loss(logits, labels, lengths, class_weights=class_weights, extra_mask=extra_mask)
    p, r, f1, t_k_score = model.score(logits, labels, lengths, scorer=scorer, extra_mask=extra_mask)
    t_k_score = model.score(logits, labels, lengths, scorer=scorer, extra_mask=extra_mask)
    return loss, p, r, f1, t_k_score


def train(model, train_batches, test_batches, epochs, report_every=10, scorer=None, learning_rate=0.001,
          learning_rate_decay=1., finetune=False, decrease_every=None, lower_bound=1e-5, MODEL_SAVE_PATH="./model"):
    lr = tf.Variable(learning_rate, trainable=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    train_losses = []
    test_losses = []
    train_f1s = []
    test_f1s = []
    def pr_av(x): return sum(x) / len(x) if len(x)!=0 else 0
    try:
        for e in range(epochs):
            losses = []
            ps = []
            rs = []
            f1s = []
            top_ks = []
            for ind, batch in enumerate(train_batches):
                loss, p, r, f1, t_k_score = train_step_finetune(model=model, optimizer=optimizer, token_ids=batch[0],
                                                                labels=batch[1],
                                                                lengths=batch[2],
                                                                extra_mask=batch[0]['input_mask'] > 0,
                                                                scorer=scorer,
                                                                finetune=finetune and e / epochs > 0.6)


                if not np.isnan(loss):
                    losses.append(loss.numpy())
                    ps.append(p)
                    rs.append(r)
                    f1s.append(f1)
                    top_ks.append(t_k_score)

                if ind % report_every == 0 and ind > 0:
                    print(f'loss = {pr_av(losses)}, acc = {pr_av(ps)}, top 5 = {pr_av(top_ks)}, batch={ind}')

                if not decrease_every is None and ind % decrease_every == 0 and ind > 0 and lr>lower_bound:
                    lr.assign(lr * learning_rate_decay)

                if ind % 10 == 0: 
                    tf.saved_model.save(model, export_dir=MODEL_SAVE_PATH)
            for ind, batch in enumerate(test_batches):
                # token_ids, graph_ids, labels, class_weights, lengths = b
                test_loss, test_p, test_r, test_f1, test_top_ks = test_step(model=model, token_ids=batch[0],
                                                                            labels=batch[1],
                                                                            lengths=batch[2],
                                                                            extra_mask=batch[0]['input_mask'] > 0,
                                                                            # class_weights=batch['class_weights'],
                                                                            scorer=scorer)

            print(
                f"Epoch: {e}, Train Loss: {pr_av(losses)}, Train P: {pr_av(ps)}, Train R: {pr_av(rs)}, Train F1: {pr_av(f1s)}, "
                f"Test loss: {test_loss}, Test P: {test_p}, Test R: {test_r}, Test F1: {test_f1}, Test Top 5: {test_top_ks}")

            train_losses.append(float(sum(losses) / len(losses)))
            train_f1s.append(float(sum(f1s) / len(f1s)))
            test_losses.append(float(test_loss))
            test_f1s.append(float(test_f1))

            lr.assign(lr * learning_rate_decay)

            tf.saved_model.save(model, export_dir=MODEL_SAVE_PATH)

    except KeyboardInterrupt:
        pass

    return train_losses, train_f1s, test_losses, test_f1s
