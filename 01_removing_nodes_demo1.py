import onnx_matcher
import onnx.helper as helper
import onnx

name = "yolov5s.onnx"
model = onnx.load(name)

# Define a replace policy function.
def removing_fuction(model, i, subgraph):
    parent = onnx_matcher.find_node_by_output(model, subgraph[0].input[0])
    child  = onnx_matcher.find_node_by_input(model, subgraph[-1].output[0])
    parent.output[0] = child.input[0]
    return [], []

# Define a subgraph pattern to delete Conv, Reshape, and Transpose.
subgraph_matcher = onnx_matcher.Matcher(
    """
    Conv(?, b0)
    Reshape(b0, c0)
    Transpose(c0, ?)
    """
)

# Print all matched subgraph to the current console.
subgraph_matcher.print_match(model)

# Use a specific policy to build new subgraphs and replace matching subgraphs.
num_replaced_graph = subgraph_matcher.replace(model, removing_fuction)
print(f"Done for replace {num_replaced_graph} nodes.")
onnx.save(model, "remove01.onnx")




