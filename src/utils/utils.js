"use strict";
const fs = require("fs");
const _ = require("lodash");

const utils = {
  trade_singal_extractor: (trade_signals, input_indicator_counts) => {
    let result = {
      train: [],
      test: []
    };
    let tensor_good = [];
    let tensor_bad = [];
    let tensor_frame = {};

    trade_signals.map(elem => {
      elem.map(trade => {
        // Set Buy status tensor
        tensor_frame = { input: [], output: [], profit: [] };

        tensor_frame.input = utils.trade_singal_input(
          trade.buy_in,
          input_indicator_counts
        );

        let buy_price = trade.buy_price;
        // Trade history last element is the selling price
        let sell_price = trade.sell_price;

        tensor_frame.profit = ((sell_price - buy_price) / sell_price) * 100;

        if (sell_price >= buy_price * 1.009) {
          tensor_frame.output = [1, 0];
          tensor_good.push(tensor_frame);
        }

        if (sell_price <= buy_price * 0.991) {
          tensor_frame.output = [0, 1];
          tensor_bad.push(tensor_frame);
        }
      });
    });

    // Make even dataset!
    let even_count = _.min([tensor_good.length, tensor_bad.length]);

    //Mix datas
    tensor_good = _.shuffle(tensor_good);
    tensor_bad = _.shuffle(tensor_bad);

    // Create Train and Test dataset

    result.train = _.take(tensor_good, even_count).concat(
      _.take(tensor_bad, even_count)
    );

    // Shuffle Good and Bad results close to evenly
    result.train = _.shuffle(result.train);

    // Let 100 data for train
    result.train = _.slice(
      result.train,
      0,
      parseInt(result.train.length - 100)
    );

    // Last 100 data for train
    result.test = _.slice(
      result.train,
      parseInt(result.train.length - 100),
      result.train.length
    );

    // console.log(normalize([1, 2, 23, 23, 123]));

    return result;
  },

  trade_singal_input: (input_array, input_indicator_counts) => {
    let shape_y = ~~(input_array.length / input_indicator_counts);
    /*
        [
        y [x,x,x],
        y [x,x,x]
        ]
        */

    let result = [];

    for (let i = 0; i < shape_y; i++) {
      let row = [];
      for (let k = 0; k < input_indicator_counts; k++) {
        // Add every indicator to a single row
        row.push(input_array[i * input_indicator_counts + k]);
      }
      result.push(row);
    }

    result = utils.min_max_scale_2d(result);

    return result;
  },

  min_max_scale_2d: array => {
    let orig_array = array;

    let min = [];
    let max = [];

    // Get min/max values
    for (let i = 0; i < orig_array.length; i++) {
      if (typeof orig_array[i] === "object" && orig_array[i] !== null) {
        for (let k = 0; k < orig_array[i].length; k++) {
          // Init values
          if (typeof min[k] == "undefined" || typeof max[k] == "undefined") {
            min[k] = orig_array[i][k];
            max[k] = orig_array[i][k];
          }

          if (orig_array[i][k] < min[k]) {
            min[k] = orig_array[i][k];
          }

          if (orig_array[i][k] > max[k]) {
            max[k] = orig_array[i][k];
          }
        }
      }
    }

    // Execute normalization!
    /*
    z= x −min(x) /max(x)−min(x)
    */
    for (let i = 0; i < orig_array.length; i++) {
      if (typeof orig_array[i] === "object" && orig_array[i] !== null) {
        for (let k = 0; k < orig_array[i].length; k++) {
          orig_array[i][k] = (orig_array[i][k] - min[k]) / (max[k] - min[k]);
        }
      }
    }

    return orig_array;
  }
};

module.exports = utils;
