# Domain Adaptation
Domain Adaptation implementations using tensorflow.

# Installation
```bash
pip install git+https://github.com/kthfan/Domain-Adaptation.git
```

# Usage

## import package
```python
from domainadaptation import VADA, DIRTT
```

## A DIRT-t approach to unsupervised domain adaptation
The implementation ofA DIRT-t approach to unsupervised domain adaptation.

Train data using tf.data.Dataset:
```python
train_ds = tf.data.Dataset.zip((x_source_ds, y_source_ds, x_target_ds))
```
Note that `x_source_ds` is image dataset in source domain, `y_source_ds` is label dataset in source domain and `x_target_ds` is image dataset in target domain.
All `x_source_ds`, `y_source_ds` and `x_target_ds` should be instance of `tf.data.Dataset`.

Create Virtual Adversarial Domain Adaptation (VADA) model:
```python
image_input, feature_output, domain_output, classification_output = VADA.get_default_model()
vada = VADA(image_input, feature_output, domain_output, classification_output)
```

Train model:
```python
vada.compile(tf.keras.optimizers.Adam(learning_rate=1e-5), loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
vada.fit(train_ds)
```

Decision-boundary Iterative Refinement Training with a Teacher (DIRT-T):
The second phase of train process.
```python
dirtt = vada.switch_to_dirtt()
dirtt.fit(train_ds)

pred_y = dirtt.predict(X_test)
```


Save model:
```python
dirtt.save("./model.h5")
config = dirtt.get_config()

with open('config.json', 'w') as f:
    json.dump(config, f)

```

Load model:
```python
with open('config.json') as f:
    config = json.load(f)

dirtt = DIRTT.load("./model.h5", **config)
```


# References
1. A DIRT-t approach to unsupervised domain adaptation  
https://arxiv.org/pdf/1802.08735.pdf

2. Implementation of A DIRT-T Approach to Unsupervised Domain Adaptation    
https://github.com/RuiShu/dirt-t    
https://github.com/ozanciga/dirt-t
