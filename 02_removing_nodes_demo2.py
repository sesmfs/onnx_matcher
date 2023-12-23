import onnx_matcher
import onnx.helper as helper
import onnx

name = "yolov5s.onnx"
model = onnx.load(name)

# Define a replace policy function.
def removing_fuction(model, i, subgraph):
    subgraph[-1].input[0] = subgraph[0].output[0]
    return [subgraph[0],subgraph[-1]], []

# Define a subgraph pattern to delete Conv, Reshape, and Transpose.
subgraph_matcher = onnx_matcher.Matcher(
    """
    Mul(?,a0)
    Conv(a0, b0)
    Reshape(b0, c0)
    Transpose(c0, d0)
    Sigmoid(d0, ?)
    """
)

# Print all matched subgraph to the current console.
subgraph_matcher.print_match(model)

# Use a specific policy to build new subgraphs and replace matching subgraphs.
num_replaced_graph = subgraph_matcher.replace(model, removing_fuction)
print(f"Done for replace {num_replaced_graph} nodes.")
onnx.save(model, "remove02.onnx")