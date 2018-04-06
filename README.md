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
m = ModelingData(class_dict=my_class_dict, ids=[8234, 9314, 2828])
m.build()
```

### Model Wrapper
The ModelWrapper implements the actual predictive modeling components
* tuning hyperparameters
* fitting a model
* saving a model

ModelWrapper takes a ModelingData object which is the data the model will learn from and a model_type which needs to correspond to a value in ModelLookup

### Model Lookup
ModelLookup is basically a library of models that can be used seamlessly (most of the time) in the modeling framework

## Workflow

* Define some data classes
* Create a class_dict where the key is the class and the val is a list of keys to join on
* Build some modeling data
* Create a model wrapper and train a model
* ????
* PROFIT

### TODO

* incorporate more robust joins in modeling_data.build() to allow for different foreign keys across data_classes


### Additional Notes
xgboost package needs extra love in MacOS https://medium.com/@lenguyenthedat/installing-xgboost-on-os-x-1f63c1ed042
