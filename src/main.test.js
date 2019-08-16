"use strict";

const _ = require("lodash");
const fs = require("fs");
const util = require("./utils/utils");
const tensorflow = require("./tensorflow/tensorflow");

function create_test_data(count) {
  let trade_signals = [];

  for (let i = 0; i < count; i++) {
    let rand = Math.random();
    let buy_in = [];

    let hardening = 43124 * Math.random();

    if (rand > 0.5) {
      // Ascending
      for (let k = 1; k <= 11; k++) {
        buy_in.push(hardening + k * Math.random() * hardening);
      }
    } else {
      // Descending
      for (let k = 11; k >= 1; k--) {
        buy_in.push(hardening - k * Math.random() * hardening);
      }
    }

    trade_signals.push({ buy_price: 500, sell_price: 1000 * rand, buy_in });
  }

  return [trade_signals];
}

async function test() {
  try {
    let trade_signals = create_test_data(1000);

    console.log(trade_signals);

    let tensor_data = util.trade_singal_extractor(trade_signals, 1);

    console.log("Train sample data: ", tensor_data.train[0]);
    console.log("Test sample data: ", tensor_data.test[0]);

    await tensorflow.load_train_tensor(tensor_data.train);

    await tensorflow.train_modell({
      model: "lstm_hidden_cells",
      name: "",
      loop: 1,
      epochs: 50
    });

    await tensorflow.test_model(tensor_data.test);
  } catch (e) {
    console.log(e);
  }
}

test();
