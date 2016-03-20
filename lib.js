var NN = {};

NN.TrainingSet = {};
NN.TrainingSet.Instances = [];
NN.TrainingSet.Create = function()
{
    var obj = {};

    obj.Input = [];
    obj.Output = [];
    obj.Order = [];
    
    NN.TrainingSet.Instances.push(obj);
    return obj;
};
NN.TrainingSet.AddPoint = function(inTrainingSet, inType, inData)
{
    inTrainingSet.Input.push(inData);
    inTrainingSet.Output.push(inType);
    inTrainingSet.Order.push(inTrainingSet.Order.length);
};
NN.TrainingSet.AddCloud = function(inTrainingSet, inLabel, inCloud)
{
    var i;
    for(i=0; i<inCloud.length; i++)
    {
        NN.TrainingSet.AddPoint(inTrainingSet, inLabel, inCloud[i]);
    }
};
NN.TrainingSet.Randomize = function(inTrainingSet)
{
      var newOrder = [];
      var selection;
      while(inTrainingSet.Order.length != 0)
      {
          selection = Math.floor(inTrainingSet.Order.length * Math.random());
          inTrainingSet.Order.splice(selection, 1);
          newOrder.push(selection);
      }
      inTrainingSet.Order = newOrder;
};


NN.Layer = {};
NN.Layer.Create = function(sizeIn, sizeOut)
{
    var i;
    var min = [];
    var max = [];
    var obj = {};
    
    sizeIn++;
    
    obj.Forward = {};
    for(i=0; i<sizeIn; i++)
    {
        min.push(-1);
        max.push(1);
    }
    obj.Forward.Matrix = M.Box([min, max], sizeOut);
    obj.Forward.StageInput = [];
    obj.Forward.StageAffine = [];
    obj.Forward.StageSigmoid = [];
    obj.Forward.StageDerivative = [];
    
    obj.Backward = {};
    obj.Backward.Matrix = M.Transpose(obj.Forward.Matrix);
    obj.Backward.StageInput = [];
    obj.Backward.StageDerivative = [];
    obj.Backward.StageAffine = [];
    
    return obj;
};
NN.Layer.Forward = function(inLayer, inInput)
{
    inLayer.Forward.StageInput = M.Pad(inInput); // Pad the input
    inLayer.Forward.StageAffine = M.Transform(inLayer.Forward.Matrix, inLayer.Forward.StageInput);
    inLayer.Forward.StageSigmoid = M.Sigmoid(inLayer.Forward.StageAffine);
    
    return inLayer.Forward.StageSigmoid;
};
NN.Layer.Error = function(inLayer, inTarget)
{
    return M.Subtract(inLayer.Forward.StageSigmoid, inTarget);
};
NN.Layer.Backward = function(inLayer, inInput)
{
    /* We need the derivative of the forward pass, but only during the backward pass.
    That's why-- even though it "belongs" to the forward pass-- it is being calculated here. */
    inLayer.Forward.StageDerivative = M.Derivative(inLayer.Forward.StageSigmoid);
    
    /* This transpose matrix is for sending the error back to a previous layer.
    And again, even though it is derived directly from the forward matrix, it is only needed during the backward pass so we calculate it here.*/
    inLayer.Backward.Matrix = M.Transpose(inLayer.Forward.Matrix);
    
    /* When the error vector arrives at a layer, it always needs to be multiplied (read 'supressed') by the derivative of
    what the layer output earlier during the forward pass.
    So despite its name, Backward.StageDerivative contains the result of this *multiplication* and not some new derivative calculation.*/
    inLayer.Backward.StageInput = inInput;
    inLayer.Backward.StageDerivative = M.Multiply(inLayer.Backward.StageInput, inLayer.Forward.StageDerivative);
    inLayer.Backward.StageAffine = M.Transform(inLayer.Backward.Matrix, inLayer.Backward.StageDerivative);
    
    return M.Unpad(inLayer.Backward.StageAffine);// Unpad the output
};
NN.Layer.Adjust = function(inLayer, inLearningRate)
{
    var deltas;
    var vector;
    var scalar;
    var i, j;
    
    for(i=0; i<inLayer.Forward.StageInput.length; i++)
    {
        deltas = M.Outer(inLayer.Forward.StageInput[i], inLayer.Backward.StageDerivative[i]);
        deltas = M.Scale(deltas, inLearningRate);
        
        inLayer.Forward.Matrix = M.Subtract(inLayer.Forward.Matrix, deltas);
    }
};
NN.Layer.Stochastic = function(inLayer, inTrainingSet, inIterations)
{
    /* this method is ONLY for testing individual layers, and does not translate to network-level training */
    var i, j;
    var current;
    var error;
    for(i=0; i<inIterations; i++)
    {
        NN.TrainingSet.Randomize(inTrainingSet);
        for(j=0; j<inTrainingSet.Order.length; j++)
        {
            current = inTrainingSet.Order[j];
            NN.Layer.Forward(inLayer, [inTrainingSet.Input[current]]);
            error = M.Subtract(inLayer.Forward.StageSigmoid, [inTrainingSet.Output[current]]);
            NN.Layer.Backward(inLayer, error);
            NN.Layer.Adjust(inLayer, 0.1);
        }
    }
};

NN.Network = {};
NN.Network.Instances = [];
NN.Network.Create = function()
{
    var obj = {};
    var i;    
    
    obj.Layers = [];
    obj.LearningRate = 0.8;
    obj.Error = [];
    
    for(i=0; i<arguments.length-1; i++)
    {
        obj.Layers.push(NN.Layer.Create(arguments[i], arguments[i+1]));
    }
    
    NN.Network.Instances.push(obj);
    return obj;
};
NN.Network.Observe = function(inNetwork, inBatch)
{
      var input = M.Clone(inBatch);
      var i;
      for(i=0; i<inNetwork.Layers.length; i++)
      {
          input = NN.Layer.Forward(inNetwork.Layers[i], input);
      }
      return inNetwork.Layers[inNetwork.Layers.length-1].Forward.StageSigmoid;
};
NN.Network.Error = function(inNetwork, inTraining)
{
      return M.Subtract(inNetwork.Layers[inNetwork.Layers.length-1].Forward.StageSigmoid, inTraining);
};
NN.Network.Learn = function(inNetwork, inError)
{
      var input = inError;
      var i;
      for(i=inNetwork.Layers.length-1; i>=0; i--)
      {
          input = NN.Layer.Backward(inNetwork.Layers[i], input);
          NN.Layer.Adjust(inNetwork.Layers[i], inNetwork.LearningRate);
      }
};


NN.Network.Batch = function(inNetwork, inTrainingSet, inIterations)
{
    var i;
    for(i=0; i<inIterations; i++)
    {
        NN.Network.Observe(inNetwork, inTrainingSet.Input);
        inNetwork.Error = NN.Network.Error(inNetwork, inTrainingSet.Output)
        NN.Network.Learn(inNetwork, inNetwork.Error);
    }
};
NN.Network.Stochastic = function(inNetwork, inTrainingSet, inIterations)
{
    var i, j;
    var current;
    
    for(i=0; i<inIterations; i++)
    {
        NN.TrainingSet.Randomize(inTrainingSet);
        for(j=0; j<inTrainingSet.Order.length; j++)
        {
            current = inTrainingSet.Order[j];
            NN.Network.Observe(inNetwork, [inTrainingSet.Input[current]]);
            inNetwork.Error = NN.Network.Error(inNetwork, [inTrainingSet.Output[current]]);
            NN.Network.Learn(inNetwork, inNetwork.Error);
        }
    }
};