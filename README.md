# StockML-TF
Tensorflow JS implementation under NodeJS for server side applications, with simple HTTP API for remote control

# Install:
```
- npm install
```

# HTTP API
- GET http://localhost:3000/model_name/shapes // Input and Output tensor shapes for training and Input useable for prediction
- GET http://localhost:3000/model_name/status // Loaded,Training,Ready when it is Ready your Model is ready for accept prediction
- POST http://localhost:3000/model_name/predict + {input: your_data} // Accept data for prediction/evaluation and response the result

# Available modells:
- LSTM for Timeseries data

# Example data structure:
- Useable for utils.js -> trade_singal_extractor(name, indicator_count)

```
{"buy_price":30.7616,"sell_price":30.2791,"buy_in":[31.65006498237831,-0.02152908007145273,-164.0231458138285,33.83191977547463,-0.02217128703825609,-152.61796439795648,33.70383786148567,-0.0207593912906259,-122.46201115453322,33.25975977033626,-0.017922392328029904,-98.03832965631184,28.04732220819858,-0.018634852510045485,-129.56704879371398,29.3365361912319,-0.02005154022042621,-120.29750479846271,27.909354975574956,-0.018944985110844206,-107.876343513387,24.164254673616057,-0.022498746826151994,-166.69364468310226]}
```
