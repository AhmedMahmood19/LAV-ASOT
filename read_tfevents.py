import tensorflow as tf
from tensorflow.python.framework import tensor_util
from tensorflow.python.summary.summary_iterator import summary_iterator

# Path to the tf.events file
events_file = "log/eval_logs/events.out.tfevents.1729878791.770685c713b5.460694.7eval_logs"

# Read and iterate through the events in the file
for summary in summary_iterator(events_file):
    for value in summary.summary.value:
        if value.HasField('simple_value'):
            print(f"Tag: {value.tag}, Value: {value.simple_value}")
        elif value.HasField('tensor'):
            tensor_value = tensor_util.MakeNdarray(value.tensor)
            print(f"Tag: {value.tag}, Tensor Value: {tensor_value}")