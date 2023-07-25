import tensorflow as tf
import tensorflow_probability as tfp

import six
import functools

from tensorflow.python.keras import backend
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine import keras_tensor, data_adapter
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.keras.layers import InputSpec

# from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.keras.utils import conv_utils
# from tensorflow.python.keras.utils import generic_utils
# from tensorflow.python.keras.utils import tf_inspect
# from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator

from tensorflow.python.util.tf_export import get_canonical_name_for_symbol
from tensorflow.python.util.tf_export import get_symbol_from_name
from tensorflow.python.util.tf_export import keras_export
from tensorflow.keras.layers import Dense, Layer
from tensorflow.keras import Model

from numpy import product
import numpy as np


def dense(inputs, kernel, bias=None, activation=None, dtype=None):
    inputs = math_ops.cast(inputs, dtype=dtype)

    rank = inputs.shape.rank

    outputs = tf.einsum('...pi,pij->...pj', inputs, kernel)

#     if not context.executing_eagerly():
#         shape = inputs.shape.as_list()
#         output_shape = shape[:-1] + [kernel.shape[0]] + [kernel.shape[-1]]


    if bias is not None:
        outputs = outputs + bias

    if activation is not None:
        outputs = activation(outputs)

    return outputs

class SteinModel(Model):
    def __init__(self, *args, **kwargs):
        super(SteinModel, self).__init__(*args, **kwargs)

    def train_step(self, data):
        particle_vars = []
        standard_vars = []

        particle_grad = []
        standard_grad = []

        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

#         with tf.GradientTape(persistent=True) as hess_tape:
        with tf.GradientTape(persistent=False) as tape1:
            tape1.watch(self.trainable_variables)
            y_pred = self(x, training=True)
            loss = self.compiled_loss(tf.expand_dims(y, 1), y_pred, sample_weight, regularization_losses=self.losses)
        gradient = tape1.gradient(loss, self.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
#             prior_gradient = tape1.gradient(prior_loss, self.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)


        for grad_, var_ in zip(gradient, self.trainable_variables):
            if ("kernel" in var_.name) or ("bias" in var_.name):
                n_particles = var_.shape[0]

                var_flat = tf.reshape(var_, [n_particles, -1])
                grad_flat = tf.reshape(grad_, [n_particles, -1])
                d = var_flat.shape[1]
#                 M = hess_tape.jacobian(grad_, var_, unconnected_gradients=tf.UnconnectedGradients.ZERO)[0, :, 0, 0, :, 0]# + (self.prior_std)**-2 * tf.eye(d)
#                 print(M.shape)

                M = tf.linalg.adjoint(grad_flat) @ grad_flat / n_particles
#                 M = tf.eye(d)
                sign_diff = var_flat - tf.expand_dims(var_flat, 1)
                Msd = sign_diff @ M
                dMsd = tf.reduce_sum(sign_diff * Msd, 2)
                h = tfp.stats.percentile(dMsd, 50.)**2 / np.log(n_particles)
#                 h = tf.reduce_mean(dMsd)**2 / np.log(n_particles)
#                 h = d
                if n_particles == 1:
                    h = 1

                kern  = tf.exp(-dMsd / (2.*h))
                gkern = -tf.einsum('ijd, ij->idj', Msd, kern) / h
                mgJ = (kern @ grad_flat - tf.reduce_sum(gkern, 2))# / n_particles

#                 B = tf.linalg.inv(M)
#                 print(M.shape, mgJ.shape)
                Q = tf.reshape(mgJ, var_.shape)

                particle_grad.append(Q)
                particle_vars.append(var_)

            else:
                standard_vars.append(var_)
                standard_grad.append(grad_)

#         print('applying particle grads')
        self.optimizer.apply_gradients(zip(particle_grad, particle_vars))
#         print('applying standard grads')
        if len(standard_grad):
            self.optimizer.apply_gradients(zip(standard_grad, standard_vars))

        self.compiled_metrics.update_state(tf.expand_dims(y, 1), y_pred, sample_weight)
        # Collect metrics to return
        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return return_metrics


class SteinDense(Dense):
    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='random_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 kernel_projection=None,
                 kernel_projection_initializer=None,
                 bias_projection=None,
                 n_particles=2,
                 **kwargs):

            super(Dense, self).__init__(
                activity_regularizer=activity_regularizer, **kwargs)

            self.units = int(units) if not isinstance(units, int) else units
            self.activation = activations.get(activation)
            self.use_bias = use_bias
            self.kernel_initializer = initializers.get(kernel_initializer)
            self.bias_initializer = initializers.get(bias_initializer)
            self.kernel_regularizer = regularizers.get(kernel_regularizer)
            self.bias_regularizer = regularizers.get(bias_regularizer)
            self.kernel_constraint = constraints.get(kernel_constraint)
            self.bias_constraint = constraints.get(bias_constraint)
            self.n_particles=n_particles

            self.input_spec = InputSpec()
            self.supports_masking = True



    def build(self, input_shape):
        dtype = dtypes.as_dtype(self.dtype or K.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('Unable to build `Dense` layer with non-floating point '
                          'dtype %s' % (dtype,))

        input_shape = tensor_shape.TensorShape(input_shape)
        last_dim = tensor_shape.dimension_value(input_shape[-1])

        if last_dim is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                           'should be defined. Found `None`.')
        self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})

        self.kernel = self.add_weight(
        'kernel',
        shape=[self.n_particles, last_dim, self.units],
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        dtype=self.dtype,
        trainable=True)

        if self.use_bias:
            self.bias = self.add_weight(
              'bias',
              shape=[self.n_particles, self.units,],
              initializer=self.bias_initializer,
              regularizer=self.bias_regularizer,
              constraint=self.bias_constraint,
              dtype=self.dtype,
              trainable=True)
        else:
            self.bias = None

    def call(self, inputs):
#         self.full_kernel = tf.reshape(tf.linalg.matmul(self.reduced_kernel, self.kernel_projection) + self.kernel_map, [self.n_particles, self.last_dim, self.units])
#         self.add_loss(self.kernel_prior(self.full_kernel))
        # return tf.einsum('...pi,pij->...pj', inputs, self.kernel)
        return dense(
            inputs,
            self.kernel,
            self.bias,
            self.activation,
            dtype=self._compute_dtype_object)


class SteinConv(Layer):
    def __init__(self,
                             rank,
                             filters,
                             kernel_size,
                             strides=1,
                             padding='valid',
                             data_format=None,
                             dilation_rate=1,
                             groups=1,
                             activation=None,
                             use_bias=True,
                             kernel_initializer='random_normal',
                             bias_initializer='zeros',
                             kernel_regularizer=None,
                             bias_regularizer=None,
                             activity_regularizer=None,
                             kernel_constraint=None,
                             bias_constraint=None,
                             trainable=True,
                             name=None,
                             conv_op=None,
                             n_particles=1,
                             **kwargs):
        super(SteinConv, self).__init__(
                trainable=trainable,
                name=name,
                activity_regularizer=regularizers.get(activity_regularizer),
                **kwargs)
        self.rank = rank
        self.n_particles = n_particles

        if isinstance(filters, float):
            filters = int(filters)
        self.filters = filters
        self.groups = groups or 1
        self.kernel_size = conv_utils.normalize_tuple(
                kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(
                dilation_rate, rank, 'dilation_rate')

        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec()

        self._validate_init()
        self._is_causal = self.padding == 'causal'
        self._channels_first = self.data_format == 'channels_first'
        self._tf_data_format = conv_utils.convert_data_format(
                self.data_format, self.rank + 2)

    def _validate_init(self):
        if self.filters is not None and self.filters % self.groups != 0:
            raise ValueError(
                    'The number of filters must be evenly divisible by the number of '
                    'groups. Received: groups={}, filters={}'.format(
                            self.groups, self.filters))

        if not all(self.kernel_size):
            raise ValueError('The argument `kernel_size` cannot contain 0(s). '
                                             'Received: %s' % (self.kernel_size,))

        if (self.padding == 'causal' and not isinstance(self,
                                                                                                        (Conv1D, SeparableConv1D))):
            raise ValueError('Causal padding is only supported for `Conv1D`'
                                             'and `SeparableConv1D`.')

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_channel = self._get_input_channel(input_shape)
        if input_channel % self.groups != 0:
            raise ValueError(
                    'The number of input channels must be evenly divisible by the number '
                    'of groups. Received groups={}, but the input has {} channels '
                    '(full input shape is {}).'.format(self.groups, input_channel,
                                                                                         input_shape))
        kernel_shape = (self.n_particles,) + self.kernel_size + (input_channel // self.groups, self.filters)

        self.kernel = self.add_weight(
                name='reduced_kernel',
                shape=kernel_shape,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                trainable=True,
                dtype=self.dtype)
        if self.use_bias:
            self.bias = self.add_weight(
                    name='bias',
                    shape=(self.filters,),
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint,
                    trainable=True,
                    dtype=self.dtype)
        else:
            self.bias = None
        channel_axis = self._get_channel_axis()
        self.input_spec = InputSpec(min_ndim=self.rank + 2,
                                                                axes={channel_axis: input_channel})

        # Convert Keras formats to TF native formats.
        if self.padding == 'causal':
            tf_padding = 'VALID'    # Causal padding handled in `call`.
        elif isinstance(self.padding, six.string_types):
            tf_padding = self.padding.upper()
        else:
            tf_padding = self.padding
        tf_dilations = list(self.dilation_rate)
        tf_strides = list(self.strides)

        tf_op_name = self.__class__.__name__
        if tf_op_name == 'Conv1D':
            tf_op_name = 'conv1d'    # Backwards compat.

        self._convolution_op = functools.partial(
                nn_ops.convolution_v2,
                strides=tf_strides,
                padding=tf_padding,
                dilations=tf_dilations,
                data_format=self._tf_data_format,
                name=tf_op_name)
        self.built = True

    def call(self, inputs):
        if self._is_causal:    # Apply causal padding to inputs for Conv1D.
            inputs = array_ops.pad(inputs, self._compute_causal_padding(inputs))

        output_list = []
        for n in range(self.n_particles):
            output_list.append(tf.expand_dims(self._convolution_op(inputs[:, n], self.kernel[n]), 1))
        outputs = tf.concat(output_list, axis=1)

        if self.use_bias:
            output_rank = outputs.shape.rank
            if self.rank == 1 and self._channels_first:
                # nn.bias_add does not accept a 1D input tensor.
                bias = array_ops.reshape(self.bias, (1, self.filters, 1))
                outputs += bias
            else:
                # Handle multiple batch dimensions.
                if output_rank is not None and output_rank > 2 + self.rank:

                    def _apply_fn(o):
                        return nn.bias_add(o, self.bias, data_format=self._tf_data_format)

                    outputs = nn_ops.squeeze_batch_dims(
                            outputs, _apply_fn, inner_rank=self.rank + 1)
                else:
                    outputs = nn.bias_add(
                            outputs, self.bias, data_format=self._tf_data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def _spatial_output_shape(self, spatial_input_shape):
        return [
                conv_utils.conv_output_length(
                        length,
                        self.kernel_size[i],
                        padding=self.padding,
                        stride=self.strides[i],
                        dilation=self.dilation_rate[i])
                for i, length in enumerate(spatial_input_shape)
        ]

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        batch_rank = len(input_shape) - self.rank - 1
        if self.data_format == 'channels_last':
            return tensor_shape.TensorShape(
                    input_shape[:batch_rank]
                    + self._spatial_output_shape(input_shape[batch_rank:-1])
                    + [self.filters])
        else:
            return tensor_shape.TensorShape(
                    input_shape[:batch_rank] + [self.filters] +
                    self._spatial_output_shape(input_shape[batch_rank + 1:]))

    def _recreate_conv_op(self, inputs):    # pylint: disable=unused-argument
        return False

    def get_config(self):
        config = {
                'filters': self.filters,
                'kernel_size': self.kernel_size,
                'strides': self.strides,
                'padding': self.padding,
                'data_format': self.data_format,
                'dilation_rate': self.dilation_rate,
                'groups': self.groups,
                'activation': activations.serialize(self.activation),
                'use_bias': self.use_bias,
                'kernel_initializer': initializers.serialize(self.kernel_initializer),
                'bias_initializer': initializers.serialize(self.bias_initializer),
                'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                'kernel_constraint': constraints.serialize(self.kernel_constraint),
                'bias_constraint': constraints.serialize(self.bias_constraint)}
        base_config = super(Conv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _compute_causal_padding(self, inputs):
        """Calculates padding for 'causal' option for 1-d conv layers."""
        left_pad = self.dilation_rate[0] * (self.kernel_size[0] - 1)
        if getattr(inputs.shape, 'ndims', None) is None:
            batch_rank = 1
        else:
            batch_rank = len(inputs.shape) - 2
        if self.data_format == 'channels_last':
            causal_padding = [[0, 0]] * batch_rank + [[left_pad, 0], [0, 0]]
        else:
            causal_padding = [[0, 0]] * batch_rank + [[0, 0], [left_pad, 0]]
        return causal_padding

    def _get_channel_axis(self):
        if self.data_format == 'channels_first':
            return -1 - self.rank
        else:
            return -1

    def _get_input_channel(self, input_shape):
        channel_axis = self._get_channel_axis()
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                                             'should be defined. Found `None`.')
        return int(input_shape[channel_axis])

    def _get_padding_op(self):
        if self.padding == 'causal':
            op_padding = 'valid'
        else:
            op_padding = self.padding
        if not isinstance(op_padding, (list, tuple)):
            op_padding = op_padding.upper()
        return op_padding


    
class Pooling1D(Layer):
    def __init__(self, pool_function, pool_size, strides,
                             padding='valid', data_format='channels_last',
                             name=None, **kwargs):
        super(Pooling1D, self).__init__(name=name, **kwargs)
        if data_format is None:
            data_format = backend.image_data_format()
        if strides is None:
            strides = pool_size
        self.pool_function = pool_function
        self.pool_size = conv_utils.normalize_tuple(pool_size, 1, 'pool_size')
        self.strides = conv_utils.normalize_tuple(strides, 1, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)

    def call(self, inputs):
        n_particles = inputs.shape.as_list()[1]
        pad_axis = 2 if self.data_format == 'channels_last' else 3
        output_list = []
        for n in range(n_particles):
            inputs_ = tf.expand_dims(inputs[:, n], pad_axis)
            outputs_ = self.pool_function(
                    inputs_,
                    self.pool_size + (1,),
                    strides=self.strides + (1,),
                    padding=self.padding,
                    data_format=self.data_format)
            output_list.append(tf.expand_dims(tf.squeeze(outputs_, pad_axis), 1))
        return tf.concat(output_list, axis=1)

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_first':
            steps = input_shape[2]
            features = input_shape[1]
        else:
            steps = input_shape[1]
            features = input_shape[2]
        length = conv_utils.conv_output_length(steps,
                                               self.pool_size[0],
                                               self.padding,
                                               self.strides[0])
        if self.data_format == 'channels_first':
            return tensor_shape.TensorShape([input_shape[0], input_shape[1], features, length])
        else:
            return tensor_shape.TensorShape([input_shape[0], input_shape[1], length, features])

    def get_config(self):
        config = {
                'strides': self.strides,
                'pool_size': self.pool_size,
                'padding': self.padding,
                'data_format': self.data_format}
        base_config = super(Pooling1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    
class Pooling2D(Layer):
    def __init__(self, pool_function, pool_size, strides,
                             padding='valid', data_format='channels_last',
                             name=None, **kwargs):
        super(Pooling2D, self).__init__(name=name, **kwargs)
        if data_format is None:
            data_format = backend.image_data_format()
        if strides is None:
            strides = pool_size
        self.pool_function = pool_function
        self.pool_size = conv_utils.normalize_tuple(pool_size, 2, 'pool_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=5)

    def call(self, inputs):
        n_particles = inputs.shape.as_list()[1]
        pad_axis = 2 if self.data_format == 'channels_last' else 3
        output_list = []
        for n in range(n_particles):
            inputs_ = inputs[:, n]
            outputs_ = self.pool_function(
                    inputs_,
                    self.pool_size,
                    strides=self.strides,
                    padding=self.padding,
                    data_format=self.data_format)
            output_list.append(tf.expand_dims(outputs_, 1))
        return tf.concat(output_list, axis=1)

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_first':
            steps = input_shape[2]
            features = input_shape[1]
        else:
            steps = input_shape[1]
            features = input_shape[2]
        length = conv_utils.conv_output_length(steps,
                                               self.pool_size[0],
                                               self.padding,
                                               self.strides[0])
        if self.data_format == 'channels_first':
            return tensor_shape.TensorShape([input_shape[0], input_shape[1], features, length])
        else:
            return tensor_shape.TensorShape([input_shape[0], input_shape[1], length, features])

    def get_config(self):
        config = {
                'strides': self.strides,
                'pool_size': self.pool_size,
                'padding': self.padding,
                'data_format': self.data_format}
        base_config = super(Pooling1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class MaxPooling1D(Pooling1D):
    def __init__(self, pool_size=2, strides=None,
               padding='valid', data_format='channels_last', **kwargs):

        super(MaxPooling1D, self).__init__(
            functools.partial(backend.pool2d, pool_mode='max'),
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            **kwargs)
        
class MaxPooling2D(Pooling2D):
    def __init__(self, pool_size=2, strides=None,
               padding='valid', data_format='channels_last', **kwargs):

        super(MaxPooling2D, self).__init__(
            functools.partial(backend.pool2d, pool_mode='max'),
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            **kwargs)
        

class SteinDenseProjected(Dense):
    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 kernel_projection=None,
                 bias_projection=None,
                 kernel_prior_particles=None,
                 kernel_posterior_particles=None,                 
                 bias_prior_particles=None,
                 bias_posterior_particles=None,
                 n_particles=1,
                 **kwargs
                ):

            super(Dense, self).__init__(
                activity_regularizer=activity_regularizer, **kwargs)
            self.units = int(units) if not isinstance(units, int) else units
            self.activation = activations.get(activation)
            self.use_bias = use_bias
            self.kernel_initializer = initializers.get(kernel_initializer)
            self.bias_initializer = initializers.get(bias_initializer)
            self.kernel_regularizer = regularizers.get(kernel_regularizer)
            self.bias_regularizer = regularizers.get(bias_regularizer)
            self.kernel_constraint = constraints.get(kernel_constraint)
            self.bias_constraint = constraints.get(bias_constraint)
            self.n_particles = n_particles
            self.kernel_projection = tf.constant(kernel_projection, dtype=self.dtype)
            self.kernel_prior_particles = kernel_prior_particles
            self.kernel_posterior_particles = kernel_posterior_particles
            self.bias_projection = tf.constant(bias_projection, dtype=self.dtype)
            self.bias_prior_particles = bias_prior_particles
            self.bias_posterior_particles = bias_posterior_particles
            self.input_spec = InputSpec(min_ndim=2)
            self.supports_masking = True

            self.kernel_r_truncated = kernel_projection.shape[0]
            self.bias_r_truncated = bias_projection.shape[0]


    def build(self, input_shape):

        input_shape = tensor_shape.TensorShape(input_shape)
        self.last_dim = tensor_shape.dimension_value(input_shape[-1])

        if self.last_dim is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                           'should be defined. Found `None`.')
        self.input_spec = InputSpec(min_ndim=2, axes={-1: self.last_dim})

        self.kernel = self.add_weight(
        'kernel',
        shape=[self.n_particles, self.kernel_r_truncated],
        initializer=initializers.get(self.kernel_initializer),
        regularizer=regularizers.get(None),
        constraint=self.kernel_constraint,
        dtype=self.dtype,
        trainable=True)


        if self.use_bias:
            self.bias = self.add_weight(
              'bias',
              shape=[self.n_particles, self.bias_r_truncated],
              initializer=self.bias_initializer,
              regularizer=regularizers.get(None),
              constraint=self.bias_constraint,
              dtype=self.dtype,
              trainable=True)
        else:
            self.bias = None
            
        #kernel
        if self.kernel_prior_particles is not None:
            self.prior_kernel = tf.constant(self.kernel_prior_particles, dtype=self.dtype)
        else:
            self.prior_kernel = tf.Variable(np.random.normal(0., 1., [self.n_particles, self.last_dim*self.units]),
                                        dtype=self.dtype, trainable=False, name='prior_kernel')

        self.kernel_prior_compliment = self.prior_kernel @ (tf.eye(self.last_dim*self.units) - tf.linalg.adjoint(self.kernel_projection) @ self.kernel_projection)
        self.kernel.assign(tf.constant(self.kernel_posterior_particles, dtype=self.dtype) @ tf.linalg.adjoint(self.kernel_projection))
        self.kernel_higher_projection_var = tf.Variable(tf.reshape(self.kernel @ self.kernel_projection, [self.n_particles, self.last_dim, self.units]), trainable=False, dtype=self.dtype, name='kernel_higher_projection_var')
        
        #bias
        if self.bias_prior_particles is not None:
            self.prior_bias = tf.constant(self.bias_prior_particles, dtype=self.dtype)
        else:
            self.prior_bias = tf.Variable(np.random.normal(0., 1., [self.n_particles, self.units]),
                                        dtype=self.dtype, trainable=False, name='prior_bias')

        self.bias_prior_compliment = self.prior_bias @ (tf.eye(self.units) - tf.linalg.adjoint(self.bias_projection) @ self.bias_projection)
        self.bias.assign(tf.constant(self.bias_posterior_particles, dtype=self.dtype) @ tf.linalg.adjoint(self.bias_projection))
        self.bias_higher_projection_var = tf.Variable(tf.reshape(self.bias @ self.bias_projection, [self.n_particles, self.units]), trainable=False, dtype=self.dtype, name='bias_higher_projection_var')

        self.built = True


    def call(self, inputs):
        kernel_higher_projection = tf.reshape(self.kernel @ self.kernel_projection + self.kernel_prior_compliment, [self.n_particles, self.last_dim, self.units])
        self.add_loss(self.kernel_regularizer(kernel_higher_projection))
        self.kernel_higher_projection_var.assign(kernel_higher_projection)
        
        bias_higher_projection = tf.reshape(self.bias @ self.bias_projection + self.bias_prior_compliment, [self.n_particles, self.units])
        self.add_loss(self.bias_regularizer(bias_higher_projection))
        self.bias_higher_projection_var.assign(bias_higher_projection)
        return dense(
            inputs,
            kernel_higher_projection,
            bias_higher_projection,
            self.activation,
            dtype=self._compute_dtype_object)

class SteinConvProjected(Layer):
    def __init__(self,
                             rank,
                             filters,
                             kernel_size,
                             strides=1,
                             padding='valid',
                             data_format=None,
                             dilation_rate=1,
                             groups=1,
                             activation=None,
                             use_bias=True,
                             kernel_initializer='glorot_uniform',
                             bias_initializer='zeros',
                             kernel_regularizer=None,
                             bias_regularizer=None,
                             activity_regularizer=None,
                             kernel_constraint=None,
                             bias_constraint=None,
                             kernel_projection=None,
                             kernel_map=None,
                             trainable=True,
                             name=None,
                             conv_op=None,
                             n_particles=1,
                             prior_particles=None,
                             posterior_particles=None,
                             **kwargs):
        super(SteinConvProjected, self).__init__(
                trainable=trainable,
                name=name,
                activity_regularizer=regularizers.get(activity_regularizer),
                **kwargs)
        self.rank = rank
        self.n_particles = n_particles

        if isinstance(filters, float):
            filters = int(filters)
        self.filters = filters
        self.groups = groups or 1
        self.kernel_size = conv_utils.normalize_tuple(
                kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(
                dilation_rate, rank, 'dilation_rate')

        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=self.rank + 2)

        self.kernel_projection = tf.constant(kernel_projection, dtype=self.dtype)
        self.prior_particles = prior_particles
        self.posterior_particles = posterior_particles
        self.d = kernel_projection.shape[0]

        self._validate_init()
        self._is_causal = self.padding == 'causal'
        self._channels_first = self.data_format == 'channels_first'
        self._tf_data_format = conv_utils.convert_data_format(
                self.data_format, self.rank + 2)

    def _validate_init(self):
        if self.filters is not None and self.filters % self.groups != 0:
            raise ValueError(
                    'The number of filters must be evenly divisible by the number of '
                    'groups. Received: groups={}, filters={}'.format(
                            self.groups, self.filters))

        if not all(self.kernel_size):
            raise ValueError('The argument `kernel_size` cannot contain 0(s). '
                                             'Received: %s' % (self.kernel_size,))

        if (self.padding == 'causal' and not isinstance(self,
                                                                                                        (Conv1D, SeparableConv1D))):
            raise ValueError('Causal padding is only supported for `Conv1D`'
                                             'and `SeparableConv1D`.')

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_channel = self._get_input_channel(input_shape)
        if input_channel % self.groups != 0:
            raise ValueError(
                    'The number of input channels must be evenly divisible by the number '
                    'of groups. Received groups={}, but the input has {} channels '
                    '(full input shape is {}).'.format(self.groups, input_channel,
                                                                                         input_shape))
        self.kernel_shape = (self.n_particles,) + self.kernel_size + (input_channel // self.groups, self.filters)

        self.kernel = self.add_weight(
                name='reduced_kernel',
                shape=[self.n_particles, self.d],
                initializer=self.kernel_initializer,
                regularizer=regularizers.get(None),
                constraint=self.kernel_constraint,
                trainable=True,
                dtype=self.dtype)

        if self.use_bias:
            self.bias = self.add_weight(
                    name='bias',
                    shape=(self.filters,),
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint,
                    trainable=True,
                    dtype=self.dtype)
        else:
            self.bias = None

        if self.prior_particles is not None:
            self.prior_kernel = tf.constant(self.prior_particles, dtype=self.dtype)
        # else:
        #     self.prior_kernel = tf.Variable(np.random.normal(0., 1., [self.n_particles, self.last_dim*self.units]),
        #                                 dtype=self.dtype, trainable=False, name='prior_kernel')

        print(self.prior_kernel.shape, (tf.linalg.adjoint(self.kernel_projection) @ self.kernel_projection).shape)
        self.prior_compliment = self.prior_kernel @ (1. - tf.linalg.adjoint(self.kernel_projection) @ self.kernel_projection)
        self.kernel.assign(tf.constant(self.posterior_particles, dtype=self.dtype) @ tf.linalg.adjoint(self.kernel_projection))

        self.higher_projection_var = tf.Variable(tf.reshape(self.kernel @ self.kernel_projection, self.kernel_shape), trainable=False, dtype=self.dtype, name='higher_projection_var')

        channel_axis = self._get_channel_axis()
        self.input_spec = InputSpec(min_ndim=self.rank + 2,
                                                                axes={channel_axis: input_channel})

        # Convert Keras formats to TF native formats.
        if self.padding == 'causal':
            tf_padding = 'VALID'    # Causal padding handled in `call`.
        elif isinstance(self.padding, six.string_types):
            tf_padding = self.padding.upper()
        else:
            tf_padding = self.padding
        tf_dilations = list(self.dilation_rate)
        tf_strides = list(self.strides)

        tf_op_name = self.__class__.__name__
        if tf_op_name == 'Conv1D':
            tf_op_name = 'conv1d'    # Backwards compat.

        self._convolution_op = functools.partial(
                nn_ops.convolution_v2,
                strides=tf_strides,
                padding=tf_padding,
                dilations=tf_dilations,
                data_format=self._tf_data_format,
                name=tf_op_name)
        self.built = True

    def call(self, inputs):
        higher_projection = tf.reshape(self.kernel @ self.kernel_projection + self.prior_compliment, self.kernel_shape)
        self.add_loss(self.kernel_regularizer(higher_projection))
        self.higher_projection_var.assign(higher_projection)

        if self._is_causal:    # Apply causal padding to inputs for Conv1D.
            inputs = array_ops.pad(inputs, self._compute_causal_padding(inputs))

        output_list = []
        for n in range(self.n_particles):
            output_list.append(tf.expand_dims(self._convolution_op(inputs[:, n], higher_projection[n]), 1))
        outputs = tf.concat(output_list, axis=1)

        if self.use_bias:
            output_rank = outputs.shape.rank
            if self.rank == 1 and self._channels_first:
                # nn.bias_add does not accept a 1D input tensor.
                bias = array_ops.reshape(self.bias, (1, self.filters, 1))
                outputs += bias
            else:
                # Handle multiple batch dimensions.
                if output_rank is not None and output_rank > 2 + self.rank:

                    def _apply_fn(o):
                        return nn.bias_add(o, self.bias, data_format=self._tf_data_format)

                    outputs = nn_ops.squeeze_batch_dims(
                            outputs, _apply_fn, inner_rank=self.rank + 1)
                else:
                    outputs = nn.bias_add(
                            outputs, self.bias, data_format=self._tf_data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def _spatial_output_shape(self, spatial_input_shape):
        return [
                conv_utils.conv_output_length(
                        length,
                        self.kernel_size[i],
                        padding=self.padding,
                        stride=self.strides[i],
                        dilation=self.dilation_rate[i])
                for i, length in enumerate(spatial_input_shape)
        ]

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        batch_rank = len(input_shape) - self.rank - 1
        if self.data_format == 'channels_last':
            return tensor_shape.TensorShape(
                    input_shape[:batch_rank]
                    + self._spatial_output_shape(input_shape[batch_rank:-1])
                    + [self.filters])
        else:
            return tensor_shape.TensorShape(
                    input_shape[:batch_rank] + [self.filters] +
                    self._spatial_output_shape(input_shape[batch_rank + 1:]))

    def _recreate_conv_op(self, inputs):    # pylint: disable=unused-argument
        return False

    def get_config(self):
        config = {
                'filters': self.filters,
                'kernel_size': self.kernel_size,
                'strides': self.strides,
                'padding': self.padding,
                'data_format': self.data_format,
                'dilation_rate': self.dilation_rate,
                'groups': self.groups,
                'activation': activations.serialize(self.activation),
                'use_bias': self.use_bias,
                'kernel_initializer': initializers.serialize(self.kernel_initializer),
                'bias_initializer': initializers.serialize(self.bias_initializer),
                'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                'kernel_constraint': constraints.serialize(self.kernel_constraint),
                'bias_constraint': constraints.serialize(self.bias_constraint)}
        base_config = super(Conv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _compute_causal_padding(self, inputs):
        """Calculates padding for 'causal' option for 1-d conv layers."""
        left_pad = self.dilation_rate[0] * (self.kernel_size[0] - 1)
        if getattr(inputs.shape, 'ndims', None) is None:
            batch_rank = 1
        else:
            batch_rank = len(inputs.shape) - 2
        if self.data_format == 'channels_last':
            causal_padding = [[0, 0]] * batch_rank + [[left_pad, 0], [0, 0]]
        else:
            causal_padding = [[0, 0]] * batch_rank + [[0, 0], [left_pad, 0]]
        return causal_padding

    def _get_channel_axis(self):
        if self.data_format == 'channels_first':
            return -1 - self.rank
        else:
            return -1

    def _get_input_channel(self, input_shape):
        channel_axis = self._get_channel_axis()
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                                             'should be defined. Found `None`.')
        return int(input_shape[channel_axis])

    def _get_padding_op(self):
        if self.padding == 'causal':
            op_padding = 'valid'
        else:
            op_padding = self.padding
        if not isinstance(op_padding, (list, tuple)):
            op_padding = op_padding.upper()
        return op_padding
