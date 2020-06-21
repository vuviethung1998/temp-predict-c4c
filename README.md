# Utility search

## Temperature prediction  

API trả về list nhiệt độ của ngày, tuần, tháng  khi nhập vào ngày, tháng, năm, type 

__Url:__ `http://0.0.0.0:1998/temp_predict`

__Method:__ `POST`


__Header:__

Key | Value
--- | ---
Content-type | application/json

__Body:__

Key | Value
--- | ---
Day | Int
Month | Int
Year | Int
Type | Str ('day','week', 'month)

__Request example:__
```
curl -X POST \
  http://localhost:1998/temp_predict \
  -H 'cache-control: no-cache' \
  -H 'content-type: application/json' \
  -H 'postman-token: 9c1037ae-923e-216c-cb52-16d7b0d822ab' \
  -d '{
	"day": 5,
	"month": 12,
	"year": 2017,
	"type": "week"
}'
```

__Response example:__

```json
[
  21.408254623413086, 
  21.340089797973633, 
  21.260210037231445, 
  21.15755844116211, 
  21.089624404907227, 
  21.066020965576172, 
  21.07756805419922
]


```


## Power prediction  

API trả về list điện năng của ngày, tuần, tháng  khi nhập vào ngày, tháng, năm, type 

__Url:__ `http://0.0.0.0:1998/power_predict`

__Method:__ `POST`


__Header:__

Key | Value
--- | ---
Content-type | application/json

__Body:__

Key | Value
--- | ---
Day | Int
Month | Int
Year | Int
Type | Str ('day','week', 'month)

__Request example:__
```
curl -X POST \
  http://localhost:1998/power_predict \
  -H 'cache-control: no-cache' \
  -H 'content-type: application/json' \
  -H 'postman-token: 9c1037ae-923e-216c-cb52-16d7b0d822ab' \
  -d '{
	"day": 5,
	"month": 12,
	"year": 2017,
	"type": "week"
}'
```

__Response example:__

```json
[
  67047.6796875, 
  66478.8828125, 
  63386.40234375, 
  58972.43359375, 
  55210.91796875, 
  52868.859375, 
  51916.6171875
]
```

