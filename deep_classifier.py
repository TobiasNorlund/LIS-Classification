# Load classification problem
import classification_base
import numpy as np
import sklearn.metrics as skmet
import sklearn.cross_validation
import theano

(X, Y, X_val, X_test) = classification_base.load(load_val=True, load_test=False)

Y = Y[:,0]

# Split X to train and test sets
X_train_train, X_train_test, Y_train_train, Y_train_test= sklearn.cross_validation.train_test_split(X, Y,  test_size=0.33, random_state=42)

X_val = theano.shared(np.asarray(X_val, dtype=theano.config.floatX), borrow=True)
X_train_train = theano.shared(np.asarray(X_train_train, dtype=theano.config.floatX), borrow=True)
X_train_test = theano.shared(np.asarray(X_train_test, dtype=theano.config.floatX), borrow=True)
Y_train_train = theano.shared(np.asarray(Y_train_train, dtype=theano.config.floatX), borrow=True)
Y_train_test = theano.shared(np.asarray(Y_train_test, dtype=theano.config.floatX), borrow=True)

datasets = [(X_train_train, Y_train_train), (X_train_test, Y_train_test), (0, 0)]

# --------------------------

from DBN import DBN
import time

finetune_lr=0.1
pretraining_epochs=100
pretrain_lr=0.01
k=1
training_epochs=1000
batch_size=10

train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]
test_set_x, test_set_y = datasets[2]

# compute number of minibatches for training, validation and testing
n_train_batches = X_val.get_value(borrow=True).shape[0] / batch_size

# numpy random generator
numpy_rng = np.random.RandomState(123)
print('... building the model')
# construct the Deep Belief Network
dbn = DBN(numpy_rng=numpy_rng, n_ins=53,
            hidden_layers_sizes=[100, 100, 100],
            n_outs=7)


# start-snippet-2
#########################
# PRETRAINING THE MODEL #
#########################
print('... getting the pretraining functions')
pretraining_fns = dbn.pretraining_functions(train_set_x=X_val,
                                            batch_size=batch_size,
                                            k=k)

print('... pre-training the model')
start_time = time.clock()
## Pre-train layer-wise
for i in xrange(dbn.n_layers):
    # go through pretraining epochs
    for epoch in xrange(pretraining_epochs):
        # go through the training set
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(pretraining_fns[i](index=batch_index,
                                        lr=pretrain_lr))
        print('Pre-training layer %i, epoch %d, cost ' % (i, epoch),)
        print(numpy.mean(c))

end_time = time.clock()
# end-snippet-2
print >> sys.stderr, ('The pretraining code for file ' +
                        os.path.split(__file__)[1] +
                        ' ran for %.2fm' % ((end_time - start_time) / 60.))
########################
# FINETUNING THE MODEL #
########################

# get the training, validation and testing function for the model
print('... getting the finetuning functions')
train_fn, validate_model, test_model = dbn.build_finetune_functions(
    datasets=datasets,
    batch_size=batch_size,
    learning_rate=finetune_lr
)


print('... finetuning the model')
# compute number of minibatches for training, validation and testing
n_train_batches = X_train_train.get_value(borrow=True).shape[0] / batch_size
# early-stopping parameters
patience = 4 * n_train_batches  # look as this many examples regardless
patience_increase = 2.    # wait this much longer when a new best is
                            # found
improvement_threshold = 0.995  # a relative improvement of this much is
                                # considered significant
validation_frequency = min(n_train_batches, patience / 2)
                                # go through this many
                                # minibatches before checking the network
                                # on the validation set; in this case we
                                # check every epoch

best_validation_loss = numpy.inf
test_score = 0.
start_time = time.clock()

done_looping = False
epoch = 0

while (epoch < training_epochs) and (not done_looping):
    epoch = epoch + 1
    for minibatch_index in xrange(n_train_batches):

        minibatch_avg_cost = train_fn(minibatch_index)
        iter = (epoch - 1) * n_train_batches + minibatch_index

        if (iter + 1) % validation_frequency == 0:

            validation_losses = validate_model()
            this_validation_loss = numpy.mean(validation_losses)
            print(
                'epoch %i, minibatch %i/%i, validation error %f %%'
                % (
                    epoch,
                    minibatch_index + 1,
                    n_train_batches,
                    this_validation_loss * 100.
                )
            )

            # if we got the best validation score until now
            if this_validation_loss < best_validation_loss:

                #improve patience if loss improvement is good enough
                if (
                    this_validation_loss < best_validation_loss *
                    improvement_threshold
                ):
                    patience = max(patience, iter * patience_increase)

                # save best validation score and iteration number
                best_validation_loss = this_validation_loss
                best_iter = iter

                # test it on the test set
                #test_losses = test_model()
                #test_score = numpy.mean(test_losses)
                #print(('     epoch %i, minibatch %i/%i, test error of '
                #        'best model %f %%') %
                #        (epoch, minibatch_index + 1, n_train_batches,
                #        test_score * 100.))

        if patience <= iter:
            done_looping = True
            break

end_time = time.clock()
print(
    (
        'Optimization complete with best validation score of %f %%, '
        'obtained at iteration %i, '
        'with test performance %f %%'
    ) % (best_validation_loss * 100., best_iter + 1, test_score * 100.)
)
print >> sys.stderr, ('The fine tuning code for file ' +
                        os.path.split(__file__)[1] +
                        ' ran for %.2fm' % ((end_time - start_time)
                                            / 60.))
