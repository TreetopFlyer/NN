# NN
Neural Network

## NN.TrainingSet

The NN.TrainingSet members are used to prepare labeled training data to be fed into a neural network

`NN.TrainingSet.Create()`
Static Constructor:
Instantiate a new TrainingSet.

`NN.TrainingSet.AddPoint(inTrainingSet, inLabelVector, inDataVector)`
Static Method:
Append the vector inDataVector to inTrainingSet under the label inLabelVector

`NN.TrainingSet.AddCloud(inTrainingSet, inLabelVector, inDataVectors)`
Static Method:
Append all of the vectors in inDataVectors to inTrainingSet, where each vector will be labeled with inLabelVector

`NN.TrainingSet.Randomize(inTrainingSet)`
Static Method:
Randomize the order of the TrainingSet inTrainingSet

### Example useage

    // to add points one-at-a-time:
    var ts = NN.TrainingSet.Create();
    NN.TrainingSet.AddPoint(ts, [0, 1], [0.1, 0.05]);
    NN.TrainingSet.AddPoint(ts, [0, 1], [0.0, -0.06]);
    NN.TrainingSet.AddPoint(ts, [1, 0], [0.99, 0.85]);
    NN.TrainingSet.AddPoint(ts, [1, 0], [1.2, 1.05]);
    
    // the same thing, but in "batch" format:
    var ts = NN.TrainingSet.Create();
    NN.TrainingSet.AddCloud(ts, [0, 1], [[0.1, 0.05], [0.0, -0.06]]);
    NN.TrainingSet.AddCloud(ts, [1, 0], [[0.99, 0.85], [1.2, 1.05]]);



## NN.Network

The NN.Network members are used to create and train multi-layer neural networks.

`NN.Network.Create(inInputDimensions, inHiddenLayer[n]Dimensions, inOutputDimensions)`
Static Constructor: This constructor takes a variable number of integer arguments.
Each argument passed in represents a new layer in the network.
The integer value of the argument represents the number of neurons in that layer.

    //create a single-layer network with two input units and 1 output neuron:
    var nn = NN.Network.Create(2, 1);
    
    //create a multi-layer network with 2 input units, 10 hidden neurons, and 3 output neurons:
    var nn = NN.Network.Create(2, 10, 3);
    
    //create a multi-layer network with 5 input units, 50 hidden neurons, followed by another layer of 20 hidden neurons, and 3 output neurons:
    var nn = NN.Network.Create(5, 50, 20, 3);
    
    /*
    The FIRST argument should be the number of dimensions in your data vectors.
    The LAST argument should the number of dimensions in your label vectors.
    The numbers in between represent the hidden layers (if any) and their size
    */

`NN.Network.Batch(inNetwork, inTrainingSet, inIterations)`
Static Method: Train Network inNetwork with TrainingSet inTrainingSet for iterations inIterations.
This method does "batch" style training where the entire data set is presented to the network each iteration.

`NN.Network.Stochastic(inNetwork, inTrainingSet, inIterations)`
Static Method: Train Network inNetwork with TrainingSet inTrainingSet for iterations inIterations.
This method does "stochastic" style training where random data vectors are selected from the training set, and the network is trained on them one-at-a-time.
The network will have examined each point randomly in the training set in a single iteration.

`NN.Network.Observe(inNetwork, inData)`
Static Method: Present the Network inNetwork with the un-labeled data inData. Returns a set of labels for the data.

### Example useage

    /*
      create some points around [0, 0] and label them with [0]
      create some points around [1, 1] and label them with [1]
    */
    var ts = NN.TrainingSet.Create();
    NN.TrainingSet.AddCloud(ts, [0], [[0.1, 0.05], [0.0, -0.06]]);
    NN.TrainingSet.AddCloud(ts, [1], [[0.99, 0.85], [1.2, 1.05]]);
    
    /*
      create a network that will accept our 2d training data and 1d label vectors, and throw in 10 hidden units.
      train the network for 1000 iterations.
      see what it labels the points at [0, 0] and [1, 1]
      (this network is serious overkill for such basic training data)
    */
    var nn = NN.Network.Create(2, 10, 1);
    NN.Network.Batch(nn, ts, 1000);
    NN.Observe(nn1, [[0, 0]]); // should output ~0
    NN.Observe(nn1, [[1, 1]]); // should output ~1
