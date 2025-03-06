import onnx
import tensorflow as tf
import tf2onnx

path = "data/ECG"
keras_model_name = "model.keras"
onnx_model_name = "ecg_classifier.onnx"

keras_model = tf.keras.models.load_model(f"{path}/{keras_model_name}")

# Convert to ONNX
spec = (tf.TensorSpec((None, 1000, 12), tf.float32, name="input"),)
onnx_model, _ = tf2onnx.convert.from_keras(keras_model, input_signature=spec, opset=14)

# Change GlobalAveragePool node to ReduceMean
# since it is not supported by Concrete ML
# GlobalAveragePool is a special case of ReduceMean, but the conversion
# below might not be correct
# TODO: Double check conversion
for i, node in enumerate(onnx_model.graph.node):
    if node.op_type == "GlobalAveragePool":
        input_name = node.input[0]
        output_name = node.output[0]
        reducemean_node = onnx.helper.make_node(
            "ReduceMean",
            inputs=[input_name],
            outputs=[output_name],
            axes=[2],
            keepdims=1,
        )
        onnx_model.graph.node.remove(node)
        onnx_model.graph.node.insert(i, reducemean_node)

onnx.save(onnx_model, f"{path}/{onnx_model_name}")
