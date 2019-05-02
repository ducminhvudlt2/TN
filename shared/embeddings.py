import tensorflow as tf


class Embedding(object):
    def __init__(self,emb_type, embedding_dim):
        self.emb_type = emb_type
        self.embedding_dim = embedding_dim

    def __call__(self,input_pnt):
        # returns the embeded tensor. Should be implemented in child classes
        pass

class LinearEmbedding(Embedding):
    def __init__(self,embedding_dim,_scope=''):
        super(LinearEmbedding,self).__init__('linear',embedding_dim)
        self.project_emb = tf.layers.Conv1D(embedding_dim,1,
            _scope=_scope+'Embedding/conv1d')

    def __call__(self,input_pnt):
        # emb_inp_pnt: [batch_size, max_time, embedding_dim]
        emb_inp_pnt = self.project_emb(input_pnt)
        # emb_inp_pnt = tf.Print(emb_inp_pnt,[emb_inp_pnt])
        return emb_inp_pnt



if __name__ == "__main__":
    sess = tf.InteractiveSession()
    input_pnt = tf.random_uniform([2,10,2])
    Embedding = LinearEmbedding(128)
    emb_inp_pnt = Embedding(input_pnt)
    sess.run(tf.global_variables_initializer())
    print(sess.run([emb_inp_pnt,tf.shape(emb_inp_pnt)]))
