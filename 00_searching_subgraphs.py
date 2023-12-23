import onnx_matcher
import onnx.helper as helper
import onnx

name = "yolov5s.onnx"
model = onnx.load(name)

# Define a subgraph pattern.
subgraph_matcher_demo1 = onnx_matcher.Matcher(
    """
    Sigmoid(?, x0)
    Slice(x0, a0)
    Mul(a0, b0)
    Pow(b0, ?)
    """
)
# Define a subgraph pattern.
subgraph_matcher_demo2 = onnx_matcher.Matcher(
    """
    Conv(?, ?)
    ?(?, ?)
    ?(?, ?)
    Conv(?, ?)
    """
)

# Print all matched subgraph to the current console.
subgraph_matcher_demo1.print_match(model)
# Print all matched subgraph to the current console.
subgraph_matcher_demo2.print_match(model)
