import tensorflow as tf
import tensorflow_addons as tfa

class _VADA_DIRTT(tf.keras.models.Model):
    '''
    Implementation of A DIRT-t approach to unsupervised domain adaptation (https://arxiv.org/pdf/1802.08735.pdf) in tensorflow.
    
    # Arguments:
        image_input: The image input of the model, a KerasTensor.
        feature_output: The output feature of the feature extractor, a KerasTensor.
        domain_output: The output label of domain discriminator, a KerasTensor.
        classification_output: The output label of classifiet, a KerasTensor.
        lambda_d: Coefficient for domain discriminate loss.
        lambda_s: Coefficient for source domain locally-Lipschitz constraint.
        lambda_t: Coefficient for target domain conditional entropy loss and locally-Lipschitz constraint.
        lipschitz_radius: Perturbation 2-norm ball radius that is used to ensure locally Lipschitz. 
        beta: Only used in dirt-t, coefficient for ensure parameterization-invariant.
        ema_decay: Decay rate of exponential moving average. Constraint updates of teacher model and ensure parameterization-invariant.
        constraint_loss: Recommend kl_divergence or categorical_cross_entropy. Ensure parameterization-invariant.
        epsilon: Small float added to variance to avoid dividing by zero. 
    # Usage:
        # training data should be Dataset: source images, source labels, target images
        train_ds = tf.data.Dataset.zip((x_source_ds, y_source_ds, x_target_ds))

        # build VADA model
        image_input, feature_output, domain_output, classification_output = VADA.get_default_model()
        vada = VADA(image_input, feature_output, domain_output, classification_output)

        # train vada model
        vada.compile(tf.keras.optimizers.Adam(learning_rate=1e-5), loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
        vada.fit(train_ds)

        # switch to DIRT-T
        dirtt = vada.switch_to_dirtt()
        dirtt.fit(train_ds)
        
        pred_y = dirtt.predict(X_test)
        
        # save and load model
        config = dirtt.get_config()
        dirtt.save("./model.h5")
        dirtt = DIRTT.load_model("./model.h5", **config)
        
    '''
    def __init__(self, image_input, feature_output, domain_output, classification_output,
                 lambda_d=1e-2, lambda_s=1, lambda_t=1e-2, lipschitz_radius=3.5, beta=1e-2, ema_decay=0.998,
                 constraint_loss=tf.losses.kl_divergence, epsilon=1e-7, name="_vada_dirt-t", **kwargs):
        super(_VADA_DIRTT, self).__init__(name=name, **kwargs)
        
        self.lambda_d = lambda_d
        self.lambda_s = lambda_s
        self.lambda_t = lambda_t
        self.beta = beta
        self.lipschitz_radius = lipschitz_radius
        self.ema_decay = ema_decay
        self.epsilon = epsilon
        self.constraint_loss = getattr(tf.losses, constraint_loss) if isinstance(constraint_loss, str) else constraint_loss
        
#         if type(self) == _VADA_DIRTT:
        self._build_models(image_input, feature_output, domain_output, classification_output)
        
    @staticmethod
    def get_default_model(input_shape=(32, 32, 3), output_shape=10, sig=1, p=0.5, use_instance_norm=True):
        x = image_input = tf.keras.layers.Input(input_shape)
        if use_instance_norm: 
            x = tfa.layers.InstanceNormalization()(x)
        
        for i in range(3): x = tf.nn.leaky_relu(tf.keras.layers.Conv2D(64, 3, padding='same')(x), 0.1)
        x = tf.keras.layers.MaxPool2D(2)(x)
        x = tf.keras.layers.Dropout(p)(x)
        x = tf.keras.layers.GaussianNoise(sig)(x)
        
        for i in range(3): x = tf.nn.leaky_relu(tf.keras.layers.Conv2D(64, 3, padding='same')(x), 0.1)
        x = tf.keras.layers.MaxPool2D(2)(x)
        x = tf.keras.layers.Dropout(p)(x)
        feature_output = x = tf.keras.layers.GaussianNoise(sig)(x)
    
        for i in range(3): x = tf.nn.leaky_relu(tf.keras.layers.Conv2D(64, 3, padding='same')(x), 0.1)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        
        classification_output = tf.keras.layers.Dense(output_shape, 'softmax')(x)
        
        domain_output = tf.keras.layers.Flatten()(feature_output)
        domain_output = tf.keras.layers.Dense(100, 'relu')(domain_output)
        domain_output = tf.keras.layers.Dense(1, 'sigmoid')(domain_output)
        
        return image_input, feature_output, domain_output, classification_output
    
    def _build_models(self, image_input, feature_output, domain_output, classification_output):
        
        self.model = tf.keras.models.Model(inputs=image_input, 
                                           outputs=[feature_output, domain_output, classification_output])
        
        # model outputs feature map, domain label and class label
        self.feature_extractor = tf.keras.models.Model(inputs=image_input, outputs=feature_output)
        self.domain_discriminator = tf.keras.models.Model(inputs=image_input, outputs=domain_output)
        self.classifier = tf.keras.models.Model(inputs=image_input, outputs=classification_output)
        
        # extract domain input
        feature_layer = next(filter(lambda layer: layer.output.ref() == feature_output.ref(), self.classifier.layers))
        _, self._domain_discriminator = self._split_network(self.domain_discriminator, feature_layer.name)
            
        # dirt-t teacher model
        self._teacher = tf.keras.models.Model.from_config(self.classifier.get_config())
            
    ## https://github.com/nrasadi/split-keras-tensorflow-model/
    def _split_network(self, model2split, split_layer_name):
        head_layer_name = split_layer_name
        split_layer = model2split.get_layer(split_layer_name)
        tail_input_layer = next(filter(lambda layer: any([node.inbound_layers == split_layer for node in layer.inbound_nodes]), model2split.layers))
        split_layer_name = tail_input_layer.name
        tail_input = tf.keras.layers.Input(batch_shape=tail_input_layer.get_input_shape_at(0))

        layer_outputs = {}
        def _find_backwards(layer):
            if layer.name in layer_outputs:
                return layer_outputs[layer.name]

            if layer.name == split_layer_name:
                out = layer(tail_input)
                layer_outputs[layer.name] = out
                return out

            prev_layers = [node.inbound_layers for node in layer.inbound_nodes]
            prev_layers = [layers if isinstance(layers, list) else [layers] for layers in prev_layers]
            prev_layers = [layer for layers in prev_layers for layer in layers]

            pl_outs = [_find_backwards(pl) for pl in prev_layers]
            pl_outs = [plos if isinstance(plos, list) else [plos] for plos in pl_outs]
            pl_outs = [plo for plos in pl_outs for plo in plos]

            ref_sets = set([plo.ref() for plo in pl_outs])
            pl_outs = [ref.deref() for ref in list(ref_sets)]

            out = layer(pl_outs[0] if len(pl_outs) == 1 else pl_outs)
            layer_outputs[layer.name] = out
            return out

        tail_output = _find_backwards(model2split.layers[-1])

        head_model = tf.keras.models.Model(model2split.input, model2split.get_layer(head_layer_name).output)
        tail_model = tf.keras.models.Model(tail_input, tail_output)

        return head_model, tail_model
    
    def _vat_loss(self, x, y_pred):
        y_pred = tf.stop_gradient(y_pred)
        
        # find r = argmax KL(h(x)||h(x+r))
        eps = self.epsilon * tf.nn.l2_normalize(tf.random.normal(tf.shape(x)), axis=range(1, len(x.shape)))
        with tf.GradientTape() as tape:
            tape.watch(eps)
            x_perturb = x + eps
            _, _, y_perturb = self.model(x_perturb, training=True)
            L_perturb = tf.reduce_mean(self.constraint_loss(y_pred, y_perturb))
        r = self.lipschitz_radius * tf.nn.l2_normalize(tape.gradient(L_perturb, eps), axis=range(1, len(x.shape)))
        
        # compute vat loss
        x_adv = tf.stop_gradient(x + r)
        _, _, y_adv = self.model(x_adv, training=True)
        L_v = tf.reduce_mean(self.constraint_loss(y_pred, y_adv))
        
        return L_v
    
    def switch_to_vada(self):
        image_input = self.inputs[0]
        feature_output, domain_output, classification_output = self.outputs
        config = self.get_config(); config.pop('model')
        vada = VADA(image_input, feature_output, domain_output, classification_output, **config)
        
        return vada
    
    def switch_to_dirtt(self, auto_compile=True):
        image_input = self.model.inputs[0]
        feature_output, domain_output, classification_output = self.model.outputs
        config = self.get_config(); config.pop('model')
        dirtt = DIRTT(image_input, feature_output, domain_output, classification_output, **config)
        if auto_compile:
            dirtt.compile(self.optimizer, loss=self.constraint_loss)
        return dirtt
    
    def compile(self, optimizer, **kwargs):
        self.domain_discriminator.compile(optimizer, loss=tf.keras.losses.binary_crossentropy, metrics=kwargs.get('metrics', ['accuracy']))
        self.classifier.compile(optimizer, **kwargs)
        self._ema = tf.train.ExponentialMovingAverage(decay=self.ema_decay)
        super(_VADA_DIRTT, self).compile(optimizer=optimizer, **kwargs)
        self.optimizers = {'h': self.optimizer, 
                           'd':type(self.optimizer)(**self.optimizer.get_config()),
                           't':type(self.optimizer)(**self.optimizer.get_config())}
        
    def predict_step(self, data):
        return self.model.predict_step(data)

    def get_config(self):
        
        return {"name": self.name,
                "epsilon": self.epsilon,
                "lambda_d": self.lambda_d,
                "lambda_s": self.lambda_s,
                "lambda_t": self.lambda_t,
                "beta": self.beta,
                "lipschitz_radius": self.lipschitz_radius,
                "ema_decay": self.ema_decay,
                "constraint_loss": self.constraint_loss.__name__,
                "model": self.model.get_config()
               }
    
    def save(self, filepath, **kwds):
        return self.model.save(filepath, **kwds)
    
    @staticmethod
    def from_config(config):
        model = tf.keras.models.Model.from_config(config.pop("model"))
        image_input = model.inputs[0]
        feature_output, domain_output, classification_output = model.outputs
        _vada_dirtt = _VADA_DIRTT(image_input, feature_output, domain_output, classification_output, **config)
        return _vada_dirtt
    
    @staticmethod
    def load_model(filepath, **kwds):
        model = tf.keras.models.load_model(filepath)
        image_input = model.inputs[0]
        feature_output, domain_output, classification_output = model.outputs
        _vada_dirtt = _VADA_DIRTT(image_input, feature_output, domain_output, classification_output, **kwds)
        return _vada_dirtt