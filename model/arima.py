from pmdarima.arima import auto_arima
# ARIMA 模型
yhat = DataFrame()
train = values[:n_train,configure['OutputAttributes']]
model = [0 for i in range(len(configure['OutputAttributes']))]
for _attr in range(len(configure['OutputAttributes'])):
    model[_attr] = auto_arima(train[:,_attr], trace=True, error_action='ignore', suppress_warnings=True)
    result = DataFrame()
    for i in range(math.ceil((configure['Length'] - n_train)/configure['n_periods'])):
        if n_train + i * configure['n_periods'] <= configure['Length']:
            training_length = n_train + i * configure['n_periods']
        else:
            training_length = configure['Length']
        print(training_length)
        train = values[:training_length,configure['OutputAttributes']]
        model[_attr].fit(train[:,_attr])
        forecast = model[_attr].predict(n_periods=configure['n_periods'])
        result = result.append(DataFrame(forecast),ignore_index=True)
    yhat[str(_attr)] = result.values.reshape(-1)
    
yhat.to_csv(configure['OutputFilePath'],index=False)