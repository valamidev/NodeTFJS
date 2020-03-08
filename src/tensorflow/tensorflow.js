"use strict";

const _ = require("lodash");
const logger = require("../logger");
const tf = require("@tensorflow/tfjs-node-gpu");
const tf_model = require("../tf_models/tf_models");

class Tensorflow {
  constructor() {
    this.train = {};
    this.tensor_shapes = {};
    this.input = [];
    this.output = [];

    this.status = "Loaded";
  }

  async load_model(name) {
    try {
      this.model = await tf.loadLayersModel(`file://./${name}/model.json`);
    } catch (e) {
      logger.error("Tensorflow load_model ", e);
    }
  }

  load_train_tensor(tensor_data) {
    tensor_data.map(elem => {
      this.input.push(elem.input);
      this.output.push(elem.output);
    });

    this.create_train_tensor();
  }

  create_train_tensor() {
    this.train.input = tf.tensor(this.input);
    this.train.output = tf.tensor(this.output);

    console.log("Input shape:", this.train.input.shape);
    console.log("Output shape:", this.train.output.shape);

    this.tensor_shapes.input = _.slice(this.train.input.shape, 1);
    this.tensor_shapes.output = _.slice(this.train.output.shape, 1);
  }

  async get_predict(input) {
    try {
      let outputs = await this.model.predict(input);

      let output_value = await outputs.dataSync();

      return output_value;
    } catch (e) {
      logger.error("Tensorflow predict error ", e);
    }
  }

  async api_get_predict(input) {
    try {
      if (this.status != "Ready") {
        return [0, 0];
      }

      let predict_tensor = tf.tensor([input]);

      let outputs = await this.model.predict(predict_tensor);

      let result = await outputs.dataSync();

      return result;
    } catch (e) {
      logger.error("Tensorflow api_predict error ", e);
    }
  }

  async test_model(tensor_data) {
    try {
      let input_tensors = [];
      let output_tensors = [];

      tensor_data.map(elem => {
        input_tensors.push(elem.input);
        output_tensors.push(elem.output);
      });

      for (let i = 0; i < tensor_data.length; i++) {
        let predict_tensor = tf.tensor([input_tensors[i]]);

        let predict = await this.get_predict(predict_tensor);

        let control = tf.tensor([output_tensors[i]]);

        let pred_res = Number(predict);
        let control_res = Number(control.dataSync());

        console.log(`Pred: ${predict} Real: ${control} `);
      }
    } catch (e) {
      logger.error("Tensorflow train error ", e);
    }
  }

  async train_modell(
    settings = {
      model: "rnn",
      name: "",
      loop: 5,
      epochs: 10
    }
  ) {
    try {
      // Base config
      let config = {
        verbose: 1,
        epochs: settings.epochs,
        batchSize: 8192,
        validationSplit: 0.1
      };

      switch (settings.model) {
        case "rnn":
          this.model = tf_model.create_model_rnn(
            this.tensor_shapes.input,
            this.tensor_shapes.output[0]
          );
          break;
        case "lstm":
          this.model = tf_model.lstm(
            this.tensor_shapes.input,
            this.tensor_shapes.output[0]
          );
          break;
        case "lstm_hidden_cells":
          this.model = tf_model.lstm_hidden_cells(
            this.tensor_shapes.input,
            this.tensor_shapes.output[0]
          );
          break;
        case "lstm_hibrid":
          this.model = tf_model.lstm_hibrid(
            this.tensor_shapes.input,
            this.tensor_shapes.output[0]
          );
          break;
        default:
          this.model = tf_model.create_model_rnn(
            this.tensor_shapes.input,
            this.tensor_shapes.output[0]
          );
          break;
      }

      let response = {};

      // Where actual learning happen
      for (let i = 0; i < settings.loop; i++) {
        response = await this.model.fit(
          this.train.input,
          this.train.output,
          config
        );

        this.status = `Training`;
      }

      // Save model if name set
      if (settings.name != "") {
        await this.model.save(`file://./${settings.name}`);
      }

      this.status = "Ready";
      // Return last loss this could verify if the train were success
      return _.last(response.history.loss);
    } catch (e) {
      logger.error("Tensorflow train error ", e);
    }
  }

  async re_train(
    settings = {
      loop: 5,
      epochs: 10
    }
  ) {
    try {
      let config = {
        verbose: 1,
        shuffle: true,
        epochs: settings.epochs,
        batchSize: 4096,
        validationSplit: 0.03,
        stepsPerEpoch: 2,
        validationSteps: 2
      };

      let response = {};

      // Where actual learning happen
      for (let i = 0; i < settings.loop; i++) {
        response = await this.model.fit(
          this.train.input,
          this.train.output,
          config
        );

        this.status = `Training`;
      }

      // Return last loss this could verify if the train were success
      this.status = "Ready";
      return _.last(response.history.loss);
    } catch (e) {
      logger.error("Tensorflow re-train error ", e);
    }
  }
}

module.exports = new Tensorflow();
