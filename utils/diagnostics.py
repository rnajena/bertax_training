# stuff to get raw memory usage of a model
# NOTE: seems to underestimate based on a test


def get_model_memory_usage(batch_size, model):
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        if (isinstance(l.output_shape, list)):
            sum_ = 0
            for shape in l.output_shape:
                single_layer_mem = 1
                for s in shape:
                    if s is None:
                        continue
                    single_layer_mem *= s
                sum_ += single_layer_mem
            single_layer_mem = sum_
        else:
            single_layer_mem = 1
            for s in l.output_shape:
                if s is None:
                    continue
                single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p)
                              for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p)
                                  for p in model.non_trainable_weights])

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size*(batch_size*shapes_mem_count
                                + trainable_count + non_trainable_count)
    gbytes = (np.round(total_memory / (1024.0 ** 3), 3)
              + internal_model_mem_count)
    return gbytes



get_model_memory_usage(500, model)
import numpy
shapes_count = int(numpy.sum([numpy.prod(numpy.array([s if isinstance(s, int) else 1 for s in l.output_shape])) for l in model.layers]))

memory = shapes_count * 4
from keras import backend as K

trainable_count = int(numpy.sum([K.count_params(p) for p in model.trainable_weights]))

non_trainable_count = int(numpy.sum([K.count_params(p) for p in model.non_trainable_weights]))
