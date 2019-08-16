"use strict";

const tf = require("@tensorflow/tfjs-node");

// Default values
let model = tf.sequential();
let optimizer = tf.train.adam();
let loss = tf.losses.meanSquaredError;

const models = {
  lstm_hidden_cells: (input_shape, output_shape) => {
    // Config
    loss = tf.losses.huberLoss;
    optimizer = tf.train.adam();

    model.add(
      tf.layers.inputLayer({
        inputShape: input_shape
      })
    );

    const cells = [
      tf.layers.lstmCell({ units: 128 }),
      tf.layers.lstmCell({ units: 128 }),
      tf.layers.lstmCell({ units: 64 }),
      tf.layers.lstmCell({ units: 48 }),
      tf.layers.lstmCell({ units: 32 })
    ];

    model.add(
      tf.layers.rnn({
        cell: cells,
        returnSequences: false
      })
    );

    model.add(
      tf.layers.dense({
        units: output_shape,
        activation: "softmax"
      })
    );

    model.compile({
      optimizer,
      loss
    });

    return model;
  },

  lstm_hibrid: (input_shape, output_shape) => {
    // Config
    loss = tf.losses.huberLoss;
    optimizer = tf.train.adam();

    const cells = [
      tf.layers.lstmCell({ units: 64 }),
      tf.layers.lstmCell({ units: 128 }),
      tf.layers.lstmCell({ units: 128 })
    ];

    model.add(
      tf.layers.rnn({
        cell: cells,
        inputShape: input_shape,
        returnSequences: true
      })
    );

    model.add(
      tf.layers.lstm({
        units: 128,
        returnSequences: false
      })
    );

    model.add(
      tf.layers.dense({
        units: 128,
        activation: "relu"
      })
    );

    model.add(
      tf.layers.dense({
        units: 32,
        activation: "relu"
      })
    );

    model.add(
      tf.layers.dense({
        units: output_shape,
        activation: "softmax"
      })
    );

    model.compile({
      optimizer,
      loss
    });

    return model;
  },

  lstm: (input_shape, output_shape) => {
    // Config
    loss = tf.losses.huberLoss;
    optimizer = tf.train.adam();

    model.add(
      tf.layers.lstm({
        units: 128,
        inputShape: input_shape,
        returnSequences: true
      })
    );

    model.add(
      tf.layers.dropout({
        rate: 0.1
      })
    );
    model.add(tf.layers.batchNormalization());

    model.add(
      tf.layers.lstm({
        units: 128,
        returnSequences: true
      })
    );

    model.add(
      tf.layers.dropout({
        rate: 0.1
      })
    );
    model.add(tf.layers.batchNormalization());

    model.add(
      tf.layers.lstm({
        units: 64,
        activation: "tanh"
      })
    );

    model.add(
      tf.layers.dropout({
        rate: 0.1
      })
    );
    model.add(tf.layers.batchNormalization());

    model.add(
      tf.layers.dense({
        units: output_shape,
        activation: "softmax"
      })
    );

    model.compile({
      optimizer,
      loss
    });

    return model;
  },

  create_model_qlearn: (input_size, action_count) => {
    // Huber Loss designed for Q learning
    loss = tf.losses.huberLoss;

    model.add(
      tf.layers.dense({
        units: 24,
        inputShape: [input_size],
        activation: "relu",
        kernelInitializer: "varianceScaling"
      })
    );

    model.add(
      tf.layers.dense({
        units: 64,
        activation: "relu",
        kernelInitializer: "varianceScaling"
      })
    );

    model.add(
      tf.layers.dense({
        units: 64,
        activation: "relu",
        kernelInitializer: "varianceScaling"
      })
    );

    model.add(
      tf.layers.dense({
        units: 24,
        activation: "relu",
        kernelInitializer: "varianceScaling"
      })
    );

    model.add(
      tf.layers.dense({
        units: action_count,
        activation: "relu",
        kernelInitializer: "varianceScaling"
      })
    );

    model.compile({
      optimizer,
      loss
    });

    return model;
  },

  create_model_lstm: () => {
    // Config
    loss = tf.losses.meanSquaredError;
    optimizer = tf.train.adamax();
    const input_layer_shape = 8;
    const input_layer_neurons = 120;
    const rnn_input_layer_features = 10;
    const rnn_output_neurons = 10;
    const output_layer_neurons = 1;

    const rnn_input_layer_timesteps =
      input_layer_neurons / rnn_input_layer_features;

    const rnn_input_shape = [
      rnn_input_layer_features,
      rnn_input_layer_timesteps
    ];

    console.log("RNN Input shape: ", rnn_input_shape);

    model.add(
      tf.layers.dense({
        units: input_layer_neurons,
        inputShape: [input_layer_shape],
        activation: "elu",
        useBias: false
      })
    );
    model.add(tf.layers.reshape({ targetShape: rnn_input_shape }));

    const cells = [
      tf.layers.lstmCell({ units: rnn_input_layer_features }),
      tf.layers.lstmCell({ units: rnn_input_layer_features })
    ];

    model.add(
      tf.layers.rnn({
        cell: cells,
        inputShape: rnn_input_shape,
        returnSequences: false
      })
    );

    model.add(
      tf.layers.dense({
        units: output_layer_neurons,
        inputShape: [rnn_output_neurons],
        activation: "elu",
        useBias: true
      })
    );

    model.compile({
      optimizer,
      loss
    });

    return model;
  },

  create_model_rnn: (input_shape, output_shape) => {
    optimizer = tf.train.adam(0.001, 0.00001);

    model.add(
      tf.layers.dense({
        units: 128,
        inputShape: [input_shape],
        activation: "relu",
        useBias: true
      })
    );

    model.add(
      tf.layers.dropout({
        rate: 0.2
      })
    );

    model.add(
      tf.layers.dense({
        units: 128,
        activation: "relu",
        useBias: true
      })
    );

    model.add(
      tf.layers.dropout({
        rate: 0.2
      })
    );

    model.add(
      tf.layers.dense({
        units: 25,
        activation: "relu",
        useBias: true
      })
    );

    model.add(
      tf.layers.dropout({
        rate: 0.1
      })
    );

    model.add(
      tf.layers.dense({
        units: output_shape,
        activation: "relu",
        useBias: true
      })
    );

    model.compile({
      optimizer,
      loss
    });

    return model;
  },

  create_modell_smma: () => {
    model.add(
      tf.layers.dense({
        units: 2,
        inputShape: [1],
        activation: "tanh",
        useBias: true
      })
    );

    model.add(
      tf.layers.dense({
        units: 50,
        activation: "tanh",
        useBias: true
      })
    );

    model.add(
      tf.layers.dense({
        units: 1,
        activation: "tanh",
        useBias: true
      })
    );

    model.compile({
      optimizer,
      loss
    });

    return model;
  },

  create_modell_volumed_smma: input_shape => {
    model.add(
      tf.layers.dense({
        units: input_shape * 5,
        inputShape: [input_shape],
        activation: "elu",
        useBias: true
      })
    );

    model.add(
      tf.layers.dense({
        units: 50,
        activation: "elu",
        useBias: true
      })
    );

    model.add(
      tf.layers.dense({
        units: 1,
        activation: "elu",
        useBias: true
      })
    );

    model.compile({
      optimizer,
      loss
    });

    return model;
  },

  create_model_cnn: () => {
    // Define input layer
    model.add(
      tf.layers.inputLayer({
        inputShape: [48, 1]
      })
    );

    // Add the first convolutional layer
    model.add(
      tf.layers.conv1d({
        kernelSize: 2,
        filters: 64,
        strides: 1,
        use_bias: true,
        activation: "relu",
        kernelInitializer: "VarianceScaling"
      })
    );

    // Add the Average Pooling layer
    model.add(
      tf.layers.averagePooling1d({
        poolSize: [2],
        strides: [1]
      })
    );

    // Add the second convolutional layer
    model.add(
      tf.layers.conv1d({
        kernelSize: 2,
        filters: 64,
        strides: 1,
        use_bias: true,
        activation: "relu",
        kernelInitializer: "VarianceScaling"
      })
    );

    // Add the Average Pooling layer
    model.add(
      tf.layers.averagePooling1d({
        poolSize: [2],
        strides: [1]
      })
    );

    // Add Flatten layer, reshape input to (number of samples, number of features)
    model.add(tf.layers.flatten({}));

    // Add Dense layer,
    model.add(
      tf.layers.dense({
        units: 1,
        kernelInitializer: "VarianceScaling",
        activation: "linear"
      })
    );

    this.model.compile({
      optimizer,
      loss
    });

    return model;
  }
};

module.exports = models;
