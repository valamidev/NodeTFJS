"use strict";

const _ = require("lodash");
const fs = require("fs");
const util = require("./utils/utils");
const tensorflow = require("./tensorflow/tensorflow");
const httpAPI = require("./httpAPI");

// Set HTTP api port!
const API = new httpAPI(3333);

async function main() {
  try {
    // HTTP API
    API.add_tf_control_routes("lstm", tensorflow);
    // HTTP API

    let rawdata = await fs.readFileSync(
      "./sample/trade_history_ao_mome_trix_rsi"
    );
    let trade_signals = JSON.parse(rawdata);

    let tensor_data = util.trade_singal_extractor(trade_signals, 4);

    console.log("Train sample data: ", tensor_data.train[0]);
    console.log("Test sample data: ", tensor_data.test[0]);

    await tensorflow.load_train_tensor(tensor_data.train);

    await tensorflow.train_modell({
      model: "lstm_hidden_cells",
      name: "",
      loop: 5,
      epochs: 50
    });

    await tensorflow.test_model(tensor_data.test);
  } catch (e) {
    console.log(e);
  }
}

main();
