# General Modeling Starting Point

fork this repo for new predictive modeling projects

## Basic Idea

### Data Classes
Data classes are the building blocks

Each data class should be some logical/semantic group of data:
  * IP addresses
  * credit attributes
  * sales data
  * etc...

Each data class should have:
  * init where you set the value of some common key to be used in query filtering
  * gather_data_submethod where data is gathered through DB queries, file reads, etc
  * process_data_submethod where any data cleaning/transforming is done

### Class Dictionary
A class dict specifies how the data should be joined together
```
my_class_dict = {
  DataClass1: ['key1', 'key3']
  DataClass2: ['key2']
}
```
### Modeling Data
ModelingData takes a class dict and a list of ids and will build the modeling dataset
```
my_data = ModelingData(class_dict=my_class_dict, ids=[8234, 9314, 2828])
my_data.build()
```

### Model Wrapper
The ModelWrapper implements the actual predictive modeling components
* tuning hyperparameters
* fitting a model
* saving a model

ModelWrapper takes a ModelingData object which is the data the model will learn from and a model_type which needs to correspond to a value in ModelLookup
```
my_model = ModelWrapper(data=my_data, model_type='xgboost', model_name='my_xgboost_model')
my_model.tune_hyperparams(n_calls = 20)
my_model.fit()
my_model.save(dir='~/saved_models')
```

### Model Lookup
ModelLookup is basically a library of models that can be used seamlessly (most of the time) in the modeling framework

### Scorer
Scorer takes a trained ModelWrapper object (or path to a pickled ModelWrapper object) and some new data and allows you to make predictions
```
new_data = ModelingData(class_dict=my_class_dict, ids=[1084, 3939, 9423])
new_data.build()
my_scorer = Scorer(model='~/saved_models/my_xgboost_model', data=new_data)
my_scorer.score()
```

## Workflow

* Define some data classes
* Create a class_dict where the key is the class and the val is a list of keys to join on
* Build some modeling data
* Create a model wrapper and train a model
* Get some new data
* Pass a trained model and new data to a Scorer
* Make predictions
* ????
* PROFIT

### TODO

* incorporate more robust joins in modeling_data.build() to allow for different foreign keys across data_classes


### Additional Notes
xgboost package needs extra love on MacOS https://medium.com/@lenguyenthedat/installing-xgboost-on-os-x-1f63c1ed042
