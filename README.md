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

Prediction:
```python
pred_y = seam.predict(X_test)
```


Save model:
```python
seam.save("./model.h5")
config = seam.get_config()

with open('config.json', 'w') as f:
    json.dump(config, f)

```

Load model:
```python
with open('config.json') as f:
    config = json.load(f)

seam = SEAM.load("./model.h5", **config)
```

## Pixel2Prototype
The implementation of Weakly Supervised Semantic Segmentation by Pixel-to-Prototype Contrast.  

Similars to SEAM:
```python
img_input = tf.keras.layers.Input((224, 224, 3))
x = tf.keras.layers.Conv2D(64, 3, padding="same")(img_input)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation("relu")(x)
x = tf.keras.layers.Conv2D(128, 3, padding="same")(x)
proj = tf.keras.layers.Conv2D(128, 1)(x)
        
p2p = Pixel2Prototype(img_input, x, 10, project_feature=proj)
p2p.compile(tf.keras.optimizers.Adam(learning_rate=1e-4), loss=tf.keras.losses.binary_crossentropy, metrics=['accuracy'])
p2p.fit(X_train, y_train)

```

## Affine
Read image:

```python
I = cv2.imread("./img.jpg").astype(np.float32) / 255
I = np.expand_dims(I, 0) # [batch_size, height, width, n_channel]
```

Generate affine code and arguments:
```python
affine = Affine()
code, args = affine.random_affine_code()
inv_args = affine.inverse_affine_code(code, args)
```

Apply affine:
```python
A_I = affine.apply_affine(I, code, args)     # transformed image
R_I = affine.apply_affine(I, code, inv_args) # inverse transformed image
```


# References
1. Non-local Neural Networks  
https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Non-Local_Neural_Networks_CVPR_2018_paper.pdf  
https://arxiv.org/pdf/1711.07971.pdf  
https://ieeexplore.ieee.org/document/8578911  

2. Self-supervised Equivariant Attention Mechanism for Weakly Supervised Semantic Segmentaion  
https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Self-Supervised_Equivariant_Attention_Mechanism_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2020_paper.pdf  
https://ieeexplore.ieee.org/abstract/document/9157474  
https://arxiv.org/pdf/2004.04581.pdf  

3. The implementation of Self-supervised Equivariant Attention Mechanism for Weakly Supervised Semantic Segmentaion  
https://github.com/YudeWang/SEAM

4. Weakly Supervised Semantic Segmentation by Pixel-to-Prototype Contrast  
https://arxiv.org/pdf/2110.07110.pdf  
https://openaccess.thecvf.com/content/CVPR2022/papers/Du_Weakly_Supervised_Semantic_Segmentation_by_Pixel-to-Prototype_Contrast_CVPR_2022_paper.pdf  

5. The Pytorch implementation of Weakly Supervised Semantic Segmentation by Pixel-to-Prototype Contrast  
https://github.com/usr922/wseg
