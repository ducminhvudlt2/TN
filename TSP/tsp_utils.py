import collections
import os
import warnings

import numpy as np
import tensorflow as tf


def create_TSP_dataset(
        n_problems,
        n_nodes,
        data_dir,
        seed=None,
        data_type='train'):

    # set random number generator
    if seed == None:
        rnd = np.random
    else:
        rnd = np.random.RandomState(seed)

    # build task name and datafiles
    task_name = 'tsp-size-{}-len-{}-{}.txt'.format(n_problems, n_nodes,data_type)
    fname = os.path.join(data_dir, task_name)

    # cteate/load data
    if os.path.exists(fname):
        print('Loading dataset for {}...'.format(task_name))
        data = np.loadtxt(fname)
        data = data.reshape(-1, n_nodes,2)
    else:
        print('Creating dataset for {}...'.format(task_name))
        # Generate a training set of size n_problems
        data= rnd.uniform(0,1,size=(n_problems,n_nodes,2))
        np.savetxt(fname, data.reshape(-1, n_nodes*2))

    return data

class DataGenerator(object):
    def __init__(self,
                 args):
        self.args = args
        self.rnd = np.random.RandomState(seed= args['random_seed'])
        print('Created train iterator.')

        # create test data
        self.n_problems = args['test_size']
        self.test_data = create_TSP_dataset(self.n_problems,args['n_nodes'],'./data',
            seed = args['random_seed']+1,data_type='test')

        self.reset()

    def reset(self):
        self.count = 0

    def get_train_next(self):
        input_data = self.rnd.uniform(0,1,
            size=[self.args['batch_size'],self.args['n_nodes'],2])

        return input_data

    def get_test_next(self):

        if self.count<self.args['test_size']:
            input_data = self.test_data[self.count:self.count+1]
            self.count +=1
        else:
            warnings.warn("The test iterator reset.")
            self.count = 0
            input_data = self.test_data[self.count:self.count+1]
            self.count +=1

        return input_data

    def get_test_all(self):
        return self.test_data

class State(collections.namedtuple("State",
                                        ("mask"))):
    pass
class Env(object):
    def __init__(self,
                 args):

        self.n_nodes = args['n_nodes']
        self.input_dim = args['input_dim']
        self.input_data = tf.placeholder(tf.float32,\
            shape=[None,self.n_nodes,args['input_dim']])
        self.input_pnt = self.input_data
        self.batch_size = tf.shape(self.input_data)[0]

    def reset(self,beam_width=1):
        self.beam_width = beam_width

        self.input_pnt = self.input_data
        self.mask = tf.zeros([self.batch_size*beam_width,self.n_nodes],dtype=tf.float32)

        state = State(mask = self.mask )

        return state

    def step(self,
             idx,
             beam_parent=None):

        # if the environment is used in beam search decoder
        if beam_parent is not None:
            # BatchBeamSeq: [batch_size*beam_width x 1]
            # [0,1,2,3,...,127,0,1,...],
            batchBeamSeq = tf.expand_dims(tf.tile(tf.cast(tf.range(self.batch_size), tf.int64),
                                                 [self.beam_width]),1)
            # batchedBeamIdx:[batch_size*beam_width]
            batchedBeamIdx= batchBeamSeq + tf.cast(self.batch_size,tf.int64)*beam_parent

            #MASK:[batch_size*beam_width x sourceL]
            self.mask = tf.gather_nd(self.mask,batchedBeamIdx)

        self.mask = self.mask + tf.one_hot(tf.squeeze(idx,1),self.n_nodes)

        state = State(mask = self.mask )

        return state


def reward_func(sample_solution=None):

    # make sample_solution of shape [sourceL x batch_size x input_dim]
    sample_solution = tf.stack(sample_solution,0)

    sample_solution_tilted = tf.concat((tf.expand_dims(sample_solution[-1],0),
         sample_solution[:-1]),0)
    # get the reward based on the route lengths


    route_lens_decoded = tf.reduce_sum(tf.pow(tf.reduce_sum(tf.pow(\
        (sample_solution_tilted - sample_solution) ,2), 2) , .5), 0)
    return route_lens_decoded
