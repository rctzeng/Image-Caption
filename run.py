import os
import _pickle as cPickle
import numpy as np
import pandas as pd
import multiprocessing
import optparse
import time
from sklearn.model_selection import train_test_split
import tensorflow as tf

# configurations
bsize, lr, grad_clip, mx_epoch, mx_num_batch = 256, 1e-4, 1e-3, 20, 401 # training
pn, proc_bsize = 12, 256 # parallel preparing training batches
MAX_LEN = 15 # model txt window size
val_ratio = 0.2

def generate_ids(size=32, min_len=10, max_len=15):
    """
    implements curricular learning: 
        examples with shorter captions are selected earlier
    -------------------------------
    return ids of examples
    """
    global cur_epoch
    while True:
        st = time.time()
        for C_LEN in range(min_len, max_len+1):
            print('#### len == {} ####'.format(C_LEN))
            ids = np.where(len_cap==C_LEN)[0]
            order = np.random.permutation(len(ids))
            for i in range(0, len(ids), size):
                yield [ids[x] for x in order[i:i+size]]
        ed = time.time()
        cur_epoch += 1
        print('#### {} epoch takes {}s ####'.format(cur_epoch, ed-st))
def pad(x, ed, max_len=15):
    """
    preprocess captions
    -------------------------------
    x: caption in string format
    ed: voc_id of <ED> token
    max_len: padding length
    """
    x = eval(x)
    while len(x)<max_len:
        x.append(ed)
    x = x[:max_len-1]
    x[-1] = ed
    return x
def generate_batch(img_map, df_cap, ids, vocab_size=2187):
    """
    generates training instances
    -------------------------------
    img_map: processed image features
    df_cap: processed captions
    ids: id of examples to be generated into training instances
    """
    imgs, curs, nxts = [], [], []
    for idx in ids:
        row = df_cap.iloc[idx]
        cap = row['caption']
        if row['img_id'] not in img_map.keys():
            continue
        img = img_map[row['img_id']]
        cur = [ed for i in range(MAX_LEN-1)]
        for i in range(1, len(cap)):
            nxt = np.zeros((vocab_size))
            nxt[cap[i]] = 1
            cur[i-1] = cap[i-1]
            imgs.append(img.ravel())
            curs.append(cur[:])
            nxts.append(nxt.ravel())
    return np.array(imgs), np.array(curs), np.array(nxts)

def image_caption_model(embedding_matrix=None, vocab_size=2187, lang_dim=100, img_dim=256, lang_len=14, rnn_unit=512, bsize=32, reuse=False):
    def xavier_init(size):
        """implements weight init from Xavier Glorot(AISTATS'10)"""
        return tf.random_normal(shape=size, stddev=1./tf.sqrt(size[0]/2.))
    # define input/output
    img_input = tf.placeholder(tf.float32, [bsize, img_dim])
    lang_input = tf.placeholder(tf.int32, [bsize, lang_len])
    nxt_word = tf.placeholder(tf.float32, [bsize, vocab_size])
    # define params
    rnn_in = [tf.Variable(xavier_init([lang_dim*lang_len+img_dim, rnn_unit])), tf.Variable(tf.zeros(shape=[rnn_unit]))]
    rnn_out = [tf.Variable(xavier_init([rnn_unit, vocab_size])), tf.Variable(tf.zeros(shape=[vocab_size]))]
    a_txt = [tf.Variable(xavier_init([lang_len*lang_dim, lang_len])), tf.Variable(xavier_init([lang_len]))]
    a_img = [tf.Variable(xavier_init([img_dim, img_dim])), tf.Variable(xavier_init([img_dim]))]
    # embed txt input
    embedding_matrix = tf.Variable(xavier_init((vocab_size, lang_dim)), dtype=tf.float32) if embedding_matrix is None else tf.Variable(embedding_matrix, dtype=tf.float32, trainable=False)
    lang_embed = tf.nn.embedding_lookup(embedding_matrix, lang_input)
    # add attention on txt_input
    lang_embed = tf.reshape(lang_embed, [bsize, lang_dim*lang_len])
    a = tf.sigmoid(tf.matmul(lang_embed, a_txt[0]) + a_txt[1])
    a_lang = tf.multiply(lang_embed, tf.concat([a for i in range(lang_dim)], axis=1))
    # add attention on img_input
    a_image = tf.multiply(tf.sigmoid(tf.matmul(img_input, a_img[0]) + a_img[1]), img_input)
    # merge txt_input and image_input
    x = tf.concat([a_lang, a_image], axis=1)
    # add dropout
    x = tf.nn.dropout(x, 0.9)
    # feed merged inputs into GRU
    with tf.variable_scope('cell_def'):
        cell = tf.nn.rnn_cell.GRUCell(rnn_unit, reuse=reuse)
    # input for GRU must in 3-dimensional: (batch_size, num_steps, num_rnn_units)
    x = tf.reshape(tf.matmul(x, rnn_in[0]) + rnn_in[1], [bsize, 1, rnn_unit])
    with tf.variable_scope('rnn_def'):
        init_state = cell.zero_state(bsize, dtype=tf.float32)
        x, _ = tf.nn.dynamic_rnn(cell, x, initial_state=init_state)
    x = tf.reshape(x, [bsize, rnn_unit])
    # predict next word
    out_logit = tf.matmul(x, rnn_out[0]) + rnn_out[1]
    # define loss function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_logit, labels=nxt_word))
    nxt_pred = tf.sigmoid(out_logit)
    params =  cell.weights + rnn_in + rnn_out + a_txt + a_img
    return [img_input, lang_input], nxt_word, loss, nxt_pred, params

def fit_model(train_data, val_data=None, epochs=200, bsize=32, lr=1e-4, param_vals=None):
    """
    trains the model for a fixed number of steps
    note: [loss, solver, params] of model must be defined in global graph
    -------------------------------
    """
    def mb_generator(X, Y, bsize=32, shuffle=True):
        """mini-batch iterator"""
        N = len(Y)
        i_epoch = 0
        while True:
            order = np.random.permutation(N) if shuffle else list(range(N))
            for i in range(0, N-bsize, bsize):
                bimgs = [X[0][order[idx]] for idx in order[i:i+bsize]]
                btxt = [X[1][order[idx]] for idx in order[i:i+bsize]]
                bnxt = [Y[order[idx]] for idx in order[i:i+bsize]]
                yield [bimgs, btxt], bnxt, i_epoch
            i_epoch += 1
    
    train_loss, val_loss, tloss, epch = [], [], [], 0
    train_gen = mb_generator(*train_data, bsize=bsize, shuffle=False)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        # load weights to model
        if param_vals is not None:
            if not assign_init:
                global var_placeholders, assign_ops
                var_placeholders = [tf.placeholder(tf.float32, shape=p.shape) for p in param_vals]
                assign_ops = [v.assign(p) for (v, p) in zip(params, var_placeholders)]
            sess.run([assign_ops], feed_dict={v:p for v,p in zip(var_placeholders, param_vals)})
        # train model
        while epch < epochs:
            bX, bY, tepch = next(train_gen)
            _, cur_loss = sess.run([solver, loss], feed_dict={inputs[0]:bX[0], inputs[1]:bX[1], output:bY})
            if tepch > epch:
                epch = tepch
                train_loss.append(np.mean(tloss))
                if val_data is not None:
                    vloss = []
                    val_gen = mb_generator(*val_data, bsize=bsize, shuffle=False)
                    bvi = 0
                    while bvi < 1:
                        bvX, bvY, bvi = next(val_gen)
                        vloss.append(sess.run([loss], feed_dict={inputs[0]:bvX[0], inputs[1]:bvX[1], output:bvY})[0])
                    val_loss.append(np.mean(vloss))
                tloss = [cur_loss]
            else:
                tloss.append(cur_loss)
        param_vals = sess.run(params)
    return {'loss':train_loss, 'val_loss':val_loss}, param_vals
def get_minibatch(ids):
    """functions for multiprocessing"""
    return generate_batch(img_train, df_train, ids)
def train(hist, param_vals, lr=1e-4, num_process=12, ckpt_path=None):
    """
    speedup preparation of training instances by multi-process
    trains and checkpoints the model
    -------------------------------
    hist: training history
    param_vals: model parameter values
    num_process: # of process to speedup training set preparation
    """
    # parallelly prepares training instance
    pool=multiprocessing.Pool(processes=num_process)
    ids = [next(idx_gen) for i in range(num_process)]
    results=pool.map(get_minibatch, ids)
    pool.close()
    pool.join()
    imgs = np.vstack([rst[0] for rst in results])
    curs = np.vstack([rst[1] for rst in results])
    nxts = np.vstack([rst[2] for rst in results])
    train_img, train_cur, train_nxt = imgs, curs, nxts
    # train model
    h, param_vals = fit_model(train_data=[[train_img, train_cur], train_nxt], val_data=[[val_img, val_cur], val_nxt], bsize=bsize, epochs=1, lr=lr, param_vals=param_vals)
    print('{}th train-loss: {}, val-loss: {}'.format(cur_epoch, np.mean(h['loss']), np.mean(h['val_loss'])))
    # record training history and model parameters
    hist['loss'] += h['loss']
    hist['val_loss'] += h['val_loss']
    if ckpt_path is not None:
        cPickle.dump(hist, open('{}.pkl'.format(ckpt_path), 'wb'))
        cPickle.dump(param_vals, open('{}_params.pkl'.format(ckpt_path), 'wb'))
    return hist, param_vals

def parse_arg():
    """
    parsing arguments from commands
    -------------------------------
    -g: run on which GPU
    -n: name of the model to do checkpoint
    -r: set if you want to start a new training;
        otherwise, program will continue from last
        training checkpoint specified by -n 
    """
    parser = optparse.OptionParser()
    parser.add_option('-r', action='store_true', dest='reset', default=False)
    parser.add_option('-g', dest='gpu_id')
    parser.add_option('-n', dest='name')
    (options, args) = parser.parse_args()
    return options

# parse argument
opts = parse_arg()
os.environ["CUDA_VISIBLE_DEVICES"]='{}'.format(opts.gpu_id)
name = opts.name
ckpt_path = 'model_ckpt/{}'.format(name)

# load data
enc_map, dec_map = cPickle.load(open('dataset/text/enc_map.pkl', 'rb')), cPickle.load(open('dataset/text/dec_map.pkl', 'rb'))
df_train, img_train = pd.read_csv('dataset/text/train_enc_cap.csv'), cPickle.load(open('dataset/train_img256.pkl', 'rb'))
df_train, df_val = train_test_split(df_train, test_size=0.025, random_state=1337)
len_cap, vlen_cap = np.array([len(eval(x)) for x in df_train['caption']]), np.array([len(eval(x)) for x in df_val['caption']])
idx_gen = generate_ids(min_len=min(len_cap), max_len=15, size=proc_bsize)
ed = enc_map['<ED>']
# process dataset
df_train['caption'] = df_train['caption'].apply(lambda x: pad(x, ed))
df_val['caption'] = df_val['caption'].apply(lambda x: pad(x, ed))
val_img, val_cur, val_nxt = generate_batch(img_train, df_val, range(df_val.shape[0]))

def gradient_clip(loss, params, opt, clip_th=1e-4):
    grads_and_vars = opt.compute_gradients(loss, params)
    clipped_grads_and_vars = [(tf.clip_by_norm(grad, clip_th), var) for grad, var in grads_and_vars]
    solver = opt.apply_gradients(clipped_grads_and_vars)
    return solver

# define model
model = image_caption_model(embedding_matrix=cPickle.load(open('dataset/text/embedding_matrix.pkl', 'rb')), bsize=bsize)
opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)
inputs, output, loss, nxt_pred, params = model
solver = gradient_clip(loss, params, opt, clip_th=grad_clip)
# load weights
assign_init = False
param_vals = None if opts.reset else cPickle.load(open('{}_params.pkl'.format(ckpt_path), 'rb'))

# training
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
hist = {'loss':[], 'val_loss':[]} if opts.reset else cPickle.load(open('{}.pkl'.format(ckpt_path), 'rb'))
cur_epoch, prev_epoch, prev_vloss, es_cnt = 0, 0, 100, 0

for i in range(mx_num_batch):
    hist, param_vals = train(hist, param_vals, lr=lr, num_process=pn, ckpt_path=ckpt_path)
    if hist['val_loss'][-1] > prev_vloss:
        es_cnt += 1
    else:
        prev_vloss, es_cnt = hist['val_loss'][-1], 0
    if es_cnt > 10: # simple lr decay
        lr = max(lr/2, 1e-5)
        print('scaling down lr={}'.format(lr))
        prev_vloss, es_cnt = hist['val_loss'][-1], 0
    if prev_epoch < cur_epoch:
        lr = 1e-4
        prev_vloss, es_cnt = 100, 0
        prev_epoch = cur_epoch
    if cur_epoch > mx_epoch:
        print('lr={}, epch={}'.format(lr, cur_epoch))
        break