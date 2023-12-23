import onnx_matcher
import onnx.helper as helper
import onnx

name = "yolov5s.onnx"
model = onnx.load(name)

# Define a replace policy function.
def replacing_fuction(model, i, subgraph):
    parent = subgraph[0]
    sub    = subgraph[1]
    add    = subgraph[2]
    mul    = subgraph[3]
    
    sigmoid = helper.make_node("Sigmoid", inputs=[sub.output[0]], outputs=[f"Custom_sigmoid_{i}"], name=f"Custom_sigmoid_{i}")
    mul.input[0] = sub.output[0]

    # remove old const node
    onnx_matcher.remove_costnode_by_tensor(model, mul.input[1])
    mul.input[1] = sigmoid.output[0]
    
    return [parent, sub, sigmoid, mul], []

# Define a subgraph pattern, deleting Add, adding Sigmod.
# change to Sub->Mul, Sub->Sigmod->Mul
# Replacing result in picture 03_replacing_a_subgraph.jpg 
subgraph_matcher = onnx_matcher.Matcher(
    """
    ?(?, i0)
    Sub(i0, a0)
    Add(a0, b0)
    Mul(b0, ?)
    """
)

# Print all matched subgraph to the current console.
subgraph_matcher.print_match(model)

# Use a specific policy to build new subgraphs and replace matching subgraphs.
num_replaced_graph = subgraph_matcher.replace(model, replacing_fuction)

print(f"Done for replace {num_replaced_graph} nodes.")
onnx.save(model, "replace03.onnx")


