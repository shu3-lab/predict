# predict

This is REST API for machine learning written by python.(and using Flask framework)
The library used for machine learning is  scikit-learn.

## How to use this API

The design of this API is written by OpenAPI 3.0(swagger).<br>
[Please access the swagger UI.](https://shu3-lab.github.io/predict/distribution/index.html)

### Run this API on docker container

Run `docker-compose up -d --build`. 

### Monitor access log

Watch Kibana dashboard via access `http://localhost:5601`.

Logs of API are collected by fluentd ans sended to elasticsearch.
Logs stored in elasticsearch can be watched by Kibana.

<img src=https://user-images.githubusercontent.com/56756975/83962800-54122180-a8db-11ea-82d4-28c4014ba50e.png width=65% >
