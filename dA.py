import os
import sys
import timeit

import numpy as np
import cPickle
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


from make_data_sim_exp import Load_data
from scipy.special import expit
from lasagne.updates import apply_momentum, adagrad



class dA(object):

    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        input=None,
        desired=None,
        n_visible=15,
        n_hidden=24,
        W=None,
        bhid=None,
        bvis=None):

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
            
        self.numpy_rng = numpy_rng

        # note : W' was written as `W_prime` and b' as `b_prime`
        if not W:
            initial_W = np.asarray(
                numpy_rng.uniform(
                    low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * np.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(
                value=np.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        if not bhid:
            bhid = theano.shared(
                value=np.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )

        self.W = W
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        # if no input is given, generate a variable representing the input
        if input is None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.desired=desired

        self.params = [self.W, self.b, self.b_prime]

    def get_corrupted_input(self, input, corruption_level, noise):
        
        corruption = self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX)
        corrupted = corruption * input
        replacement=0.5
        
        # replace corrupted features with 0.5
        corrupted = corrupted + T.switch(T.eq(corruption, 0.), replacement, 0.)
        if noise <= 0:
            return corrupted
        else:
            noise = self.theano_rng.normal(size=input.shape, std = noise,
                                       dtype=theano.config.floatX)
            return corrupted + noise

    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer """
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed input given the values of the
        hidden layer

        """
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self, corruption_level, learning_rate, noise = 0.0, momentum=0):
        """ This function computes the cost and the updates for one trainng
        step of the dA """

        tilde_x = self.get_corrupted_input(self.x, corruption_level, noise)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        L = - T.sum(self.desired * T.log(z) + (1 - self.desired) * T.log(1 - z), axis=1)
        cost = T.mean(L)

        # adagrad with momentum on cost
        updates_ada = adagrad(cost, self.params, learning_rate=learning_rate)
        updates = apply_momentum(updates_ada, self.params, momentum=momentum)

        return (cost, updates)
        
    def get_cost(self, corruption_level, noise = 0.0):
        
        tilde_x = self.get_corrupted_input(self.x, corruption_level, noise)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        L = - T.sum(self.desired * T.log(z) + (1 - self.desired) * T.log(1 - z), axis=1)
        cost = T.mean(L)

        return cost

def test_dA(learning_rate=0.1, training_epochs=15,
            batch_size=40, output_folder='dA_plots',cut=np.inf, m=0):

    datasets = Load_data(cut)
    train_set_x, train_set_y = datasets[0]
    
    # calculating mini batch sizes from arbitrary input sizes
    batch_size = min(batch_size,train_set_x.get_value(borrow=True).shape[0])

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  
    des = T.matrix('des')

    #####################################
    # BUILDING THE MODEL CORRUPTION 30% #
    #####################################

    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = dA(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        desired=des,
        n_visible=15,
        n_hidden=24
    )

    cost, updates = da.get_cost_updates(
        corruption_level=0.3, noise = 0.01,
        learning_rate=learning_rate,
        momentum=m
    )

    train_da = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            des: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    start_time = timeit.default_timer()

    ############
    # TRAINING #
    ############

    # go through training epochs
    for epoch in xrange(training_epochs):
        # go through trainng set
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(train_da(batch_index))

        print 'Training epoch %d, cost ' % epoch, np.mean(c)

    end_time = timeit.default_timer()

    training_time = (end_time - start_time)

    print >> sys.stderr, ('The 30% corruption code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % (training_time / 60.))

    with open('best_model.pkl', 'w') as f:
        cPickle.dump(da, f)
        
def test():
    classifier = cPickle.load(open('best_model.pkl'))

    datasets = Load_data()
    test_set_x, test_set_y= datasets[1]
    g=np.dot(np.asarray(test_set_x.get_value()[:,:]), classifier.W.get_value()) +\
                    classifier.b.get_value()
    h=expit(g)
    i=np.dot(h, classifier.W.get_value().T) + classifier.b_prime.get_value()
    predictions=expit(i)
    labels=np.asarray(test_set_y.get_value()[:,:])
    inputs=np.asarray(test_set_x.get_value()[:,:])
    return inputs, labels, predictions

if __name__ == '__main__':
    test_dA(training_epochs=6000, batch_size=200, learning_rate=0.2, m=0.8)
