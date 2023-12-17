import onnx_matcher
import onnx.helper as helper
import onnx

name = "yolov5s.onnx"
model = onnx.load(name)

# Define a replace policy function.
def conv_swish_to_conv_relu(i, subgraph):
    conv = subgraph[0]
    mul  = subgraph[2]
    relu = helper.make_node("Relu", inputs=conv.output, outputs=mul.output, name=f"{conv.output[0]}_relu")
    return [conv, relu], []

# Define a subgraph pattern.
subgraph_matcher = onnx_matcher.Matcher(
    """
    Conv(?, c0)
    Sigmoid(c0, s0)
    Mul([s0, c0], ?)
    """
)

# Print all matched subgraph to the current console.
subgraph_matcher.print_match(model)

# Use a specific policy(to_conv_relu) to build new subgraphs and replace matching subgraphs.
num_replaced_graph = subgraph_matcher.replace(model, conv_swish_to_conv_relu)

print(f"Done for replace {num_replaced_graph} nodes.")
onnx.save(model, "relu_yolov5.onnx")