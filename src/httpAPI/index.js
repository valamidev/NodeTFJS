"use strict";
const Koa = require("koa");
const Router = require("koa-router");
const parse = require("co-body");

const utils = require("../utils/utils");

const app = new Koa();

class HttpAPI {
  constructor(port) {
    app.listen(port);
  }

  add_tf_control_routes(name, tensorflow_object) {
    let router = new Router({
      prefix: `/${name}`
    });
    // Add Tensorflow control routes
    // Input shape /* GET */
    // Status /* GET */
    // Predict /* POST */

    router.get("/shapes", (ctx, next) => {
      ctx.body = tensorflow_object.tensor_shapes;
    });

    router.get("/status", (ctx, next) => {
      ctx.body = tensorflow_object.status;
    });

    router.post("/predict", async (ctx, next) => {
      try {
        let post_data = await parse(ctx);

        let input_data = utils.trade_singal_input(post_data.input, 4);

        ctx.body = await tensorflow_object.api_get_predict(input_data);
      } catch (e) {
        ctx.throw(400, "Predict error", e);
      }
    });

    // Update new routes
    app.use(router.routes()).use(router.allowedMethods());
  }
}

module.exports = HttpAPI;
