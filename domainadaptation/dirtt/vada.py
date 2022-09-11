import tensorflow as tf
from ._vada_dirtt import _VADA_DIRTT


class VADA(_VADA_DIRTT):
    def __init__(self, image_input, feature_output, domain_output, classification_output, name='vada', **kwargs):
        super(VADA, self).__init__(image_input, feature_output, domain_output, classification_output, 
                                   name=name, **kwargs)

    def _loss(self, x_source, y_source, x_target, f_s, d_s, y_s, f_t, d_t, y_t):
           
        # ordinary cross entropy loss for classification
        L_y = self.compiled_loss(y_source, y_s, regularization_losses=None)
        
        # update domain_discriminator
        L_d = tf.reduce_mean(tf.losses.binary_crossentropy(tf.ones_like(d_s), d_s)) + tf.reduce_mean(tf.losses.binary_crossentropy(tf.zeros_like(d_t), d_t))
        # update feature_extractor
        L_d_f = tf.reduce_mean(tf.losses.binary_crossentropy(tf.zeros_like(d_s), d_s)) + tf.reduce_mean(tf.losses.binary_crossentropy(tf.ones_like(d_t), d_t))
        L_d, L_d_f = 0.5*L_d, 0.5*L_d_f
        
        # conditional entropy
        L_c = tf.reduce_mean(self.loss(y_t, y_t))
               
        # locally-Lipschitz constraint
        L_v_s = self._vat_loss(x_source, y_s)
        L_v_t = self._vat_loss(x_target, y_t)
        
        # losses are multiplied by coefficients
        L_d, L_d_f = self.lambda_d*L_d, self.lambda_d*L_d_f
        L_v_s = self.lambda_s*L_v_s
        L_c, L_v_t = self.lambda_t*L_c, self.lambda_t*L_v_t
            
        return (L_y, L_d, L_d_f, L_c, L_v_s, L_v_t)
            
    def call(self, x_source, y_source=None, x_target=None, training=False):
        if training and y_source is not None and x_target is not None :
            # compute feature map, domain prediction and label prediction
            f_s, d_s, y_s = self.model(x_source, training=True)
            f_t, d_t, y_t = self.model(x_target, training=True)
        
            # compute losses
            losses = self._loss(x_source, y_source, x_target, f_s, d_s, y_s, f_t, d_t, y_t)
            
            L_y, L_d, L_d_f, L_c, L_v_s, L_v_t = losses
            
            self.add_loss(L_y)
            self.add_loss(L_d)
            self.add_loss(L_d_f)
            self.add_loss(L_c)
            self.add_loss(L_v_s)
            self.add_loss(L_v_t)
            
            L_all = sum(losses)
            self.add_metric(L_all, name='L_all')
            self.add_metric(L_y, name='L_y')
            self.add_metric(L_d, name='L_d')
            self.add_metric(L_d_f, name='L_d_f')
            self.add_metric(L_c, name='L_c')
            self.add_metric(L_v_s, name='L_v_s')
            self.add_metric(L_v_t, name='L_v_t')
            
            return y_s
            
        elif y_source is None:
            return self.classifier(x_source)
        else:
            return self.model(x_source)
    
    def train_step(self, data):
        x_source, y_source, x_target = data
        with tf.GradientTape(persistent=True) as tape:
            pred_y = self(x_source, y_source=y_source, x_target=x_target, training=True)
            L_y, L_d, L_d_f, L_c, L_v_s, L_v_t = self.losses
            L_feature, L_domain = L_y+L_d_f+L_c+L_v_s+L_v_t, L_d
        
        # update feature extractor and classifier
        self.optimizers['h'].minimize(L_feature, self.classifier.trainable_variables, tape=tape)
        # update domain discriminator
        self.optimizers['d'].minimize(L_domain, self._domain_discriminator.trainable_variables, tape=tape)
        
        self.compiled_metrics.update_state(y_source, pred_y)
        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
            
        return return_metrics  
    
