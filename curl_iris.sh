curl -X 'POST' \
  'http://127.0.0.1:8000/predict_mlflow' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "sepal_length": 1.8,
  "sepal_width": 1.4,
  "petal_length": 0.5,
  "petal_width": 0.2
}'