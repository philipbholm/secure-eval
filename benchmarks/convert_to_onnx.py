import onnx
import tensorflow as tf
import tf2onnx

path = "data/ECG"
keras_model_name = "model.keras"
onnx_model_name = "ecg_classifier"

keras_model = tf.keras.models.load_model(f"{path}/{keras_model_name}")

# Convert to ONNX
spec = (tf.TensorSpec((None, 1000, 12), tf.float32, name="input"),)
onnx_model, _ = tf2onnx.convert.from_keras(keras_model, input_signature=spec, opset=14)

# Save model before processing
onnx.save(onnx_model, f"{path}/{onnx_model_name}_base.onnx")

# Change GlobalAveragePool node to AveragePool
# since it is not supported by Concrete ML
for i, node in enumerate(onnx_model.graph.node):
    if node.op_type == "GlobalAveragePool":
        input_name = node.input[0]
        output_name = node.output[0]
        new_node = onnx.helper.make_node(
            "AveragePool",
            inputs=[input_name],
            outputs=[output_name],
            kernel_shape=[1000],
        )
        onnx_model.graph.node.remove(node)
        onnx_model.graph.node.insert(i, new_node)

onnx.save(onnx_model, f"{path}/{onnx_model_name}.onnx")
