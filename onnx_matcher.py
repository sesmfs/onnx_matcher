# MIT License
# Copyright (c) 2023 Zyy
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE

import re
from copy import deepcopy
import inspect

def log(msg):
    lineno = inspect.stack()[1].lineno
    print(f"[ONNX_Matcher:{lineno}]: {msg}")

def find_node_by_output(model, output):
    for node in model.graph.node:
        if node.op_type == "Constant":
            continue
        
        if output in node.output:
            return node
        
def find_node_by_input(model, input):
    for node in model.graph.node:
        if node.op_type == "Constant":
            continue
        
        if input in node.input:
            return node

def find_nodes_by_input(model, input):
    nodes = []
    for node in model.graph.node:
        if node.op_type == "Constant":
            continue
        
        if input in node.input:
            nodes.append(node)
    return nodes

def find_nodes_by_output(model, output):
    nodes = []
    for node in model.graph.node:
        if node.op_type == "Constant":
            continue
        
        if output in node.output:
            nodes.append(node)
    return nodes

def find_consts(model, name):
    nodes = []
    for node in model.graph.node:
        if name in node.output and node.op_type == "Constant":
            nodes.append(node)
    return nodes

def find_initializers(model, name):
    nodes = []
    for node in model.graph.initializer:
        if node.name == name:
            nodes.append(node)
    return nodes

def remove_node_and_init_by_indexs(model, inodes, inints):
    inodes = sorted(inodes, reverse=True)
    inints = sorted(inints, reverse=True)
    for i in inodes:
        del model.graph.node[i]
        
    for i in inints:
        del model.graph.initializer[i]
        
def remove_costnode_by_tensor(model, tensor_name):
    for i, node in enumerate(model.graph.node):
        if node.op_type == "Constant":
            if tensor_name in node.output:
                log(f"Remove a constant node: {node.name}")
                del model.graph.node[i]
                return True
    return False

def remove_node_and_info(model, node):
    nidxs = []
    iidxs = []
    lnodes = list(model.graph.node)
    linits = list(model.graph.initializer)
    for input in node.input:
        consts = find_consts(model, input)

        for n in consts:
            nidxs.append(lnodes.index(n))
        
        inits = find_initializers(model, input)
        for n in inits:
            iidxs.append(linits.index(n))
    
    remove_node_and_init_by_indexs(model, nidxs, iidxs)

def cleanup(model):
    in_graph_tensors = set()
    already_pass = set()
    output_names = set([item.name for item in model.graph.output])
    tensors = [[item.name] for item in model.graph.input]
    for node in model.graph.node:
        if len(node.input) == 0:
            tensors.extend(list(node.output))

    already_pass_tensors = []
    while len(tensors) > 0:
        names = tensors.pop()
        tensor = names[-1]
        if tensor in already_pass:
            already_pass_tensors.append(names)
            continue
        
        already_pass.add(tensor)
        if tensor in output_names:
            in_graph_tensors.update(names)
            continue
        
        nodes = find_nodes_by_input(model, tensor)
        for node in nodes:
            for output in node.output:
                tensors.append(names + list(node.input) + [output])
    
    for names in already_pass_tensors:
        tensor = names[-1]
        if tensor in in_graph_tensors:
            in_graph_tensors.update(names)

    del_nodes = []
    del_inits = []
    for inode, node in enumerate(model.graph.node):
        in_graph = any([output in in_graph_tensors for output in node.output])
        if not in_graph:
            log(f"Remove a floating node: {node.name}, the node output is: {node.output}")
            del_nodes.append(inode)

    for iinit, init in enumerate(model.graph.initializer):
        in_graph = init.name in in_graph_tensors
        if not in_graph:
            log(f"Remove a unused initializer: {init.name}")
            del_inits.append(iinit)
    
    remove_node_and_init_by_indexs(model, del_nodes, del_inits)

class Lexer:
    def __init__(self, pattern):
        
        # Compile the extraction regular expression.
        extract_name_and_argument = re.compile("([\W\w]+)\(([\W\w]+)\)")
        #     Slice(c2, ?)

        
        # Remove spaces and split patterns by the break line.
        lines = [item for item in pattern.replace(" ", "").split("\n") if item != ""]

        
        # Parsing patterns by lexical analyzer.
        self.pattern  = pattern
        self.lines    = lines
        self.patterns = []
        for line in lines:
            
            names_and_arguments = extract_name_and_argument.findall(line)
            
            assert len(names_and_arguments) == 1, f"Unexpected line: {line}. The valid symbol is: name(input_argument, output_argument)"
            operator_names, argumants = names_and_arguments[0]
            inputs, outputs = self.parse_arguments(argumants)
    
     
            self.patterns.append([operator_names.split("/"), inputs, outputs]) 
        
    def parse_variable(self):
        
        variable_name = ""
        while self.itoken < len(self.symbols):
            self.token = self.symbols[self.itoken]
            
            # If a valid token(alpha/number/_ or ?) for variable.
            if self.token.isalnum() or self.token == "?" or self.token == "_":
                variable_name += self.token
                
            else:
                break
            
            self.itoken += 1
        return variable_name
    
    def parse_list(self):
        self.itoken += 1
        lists = [self.parse_variable()]
        while self.itoken < len(self.symbols):
            self.token = self.symbols[self.itoken]
            if self.token == ",":
                self.itoken += 1
                name = self.parse_variable()
                lists.append(name)
                continue
            elif self.token == "]":
                self.itoken += 1
                break
            else:
                raise ValueError(f"Unexpected token: {self.token}")
        assert self.token == "]", f"Unexpected end token for list: ], pos: {self.itoken}"
        return lists
        
    def parse_arguments(self, symbols):
        self.itoken = 0
        self.symbols = symbols

        
        lists = []
        while self.itoken < len(symbols):
            self.token = symbols[self.itoken]
            if self.token == "[":
                lists.append(self.parse_list())
                log(self.parse_list())
            else:
                lists.append([self.parse_variable()])      
            self.itoken += 1
        assert len(lists) == 2, f"Unexpected number of params: {len(lists)}"
        return lists

class Matcher:
    def __init__(self, pattern):
        self.lexer = Lexer(pattern)
    
    def _match_io(self, input_params, input_names, variables):
        for item in input_params:
            if item != "?" and variables[item] not in input_names:
                return False
        return True
    
    def _try_to_match(self, model, anchor):
        matched_paths = []
        params_stack = [[[anchor], 0, dict()]]
        while len(params_stack) > 0:
            path, icondition, variables = params_stack.pop()
            anchor = path[-1]
            allowed_op_types, inputs, outputs = self.lexer.patterns[icondition]
            if not (anchor.op_type in allowed_op_types or "?" in allowed_op_types):
                # if icondition > 1:
                #     path_string = ", ".join([item.name for item in path])
                #     print(f"Can not match type[{path_string}], icondition={icondition}, anchor={anchor.name}[{anchor.op_type}]")
                continue

            if not self._match_io(inputs, anchor.input, variables):
                # if icondition > 1:
                #     path_string = ", ".join([item.name for item in path])
                #     print(f"Can not match io[{path_string}], icondition={icondition}, anchor={anchor.name}[{anchor.op_type}]")
                continue
            
            if icondition == len(self.lexer.patterns) - 1:
                # last condition
                matched_paths.append(path)
                continue
            
            variables = deepcopy(variables)
            for i, item in enumerate(outputs):
                if item != "?":
                    variables[item] = anchor.output[i]
            
            for output in anchor.output:
                for item in find_nodes_by_input(model, output):
                    params_stack.append([path + [item], icondition+1, variables])
        return matched_paths

    def match(self, model):
        all_matched_pairs = []
        for node in model.graph.node:
            if node.op_type == "Constant":
                continue
            
            all_matched_pairs.extend(self._try_to_match(model, node))
        return all_matched_pairs
    
    def print_match(self, model):
        print("=====================================================================")
        matched_subgraphs = self.match(model)
        log(f"Found {len(matched_subgraphs)} subgraphs:")
        for i, subgraph in enumerate(self.match(model)):
            subgraph_names = ", ".join([f"{item.name}({item.op_type})" for item in subgraph])
            print(f"\tSubgraph{i}: {subgraph_names}")
            
        pattern_text = "\n\t".join(self.lexer.lines)
        log(f"Pattern is:\n\t{pattern_text}")
        print("=====================================================================")
    
    # delete some subgraph
    def delete(self, model):
        self.replace(model, None)
    
    # replace some subgraph to new
    def replace(self, model, new_graph_fn=None):
        matched_subgraphs = self.match(model)
        for i, subgraph in enumerate(matched_subgraphs):
            if new_graph_fn is not None:
                new_nodes, new_initializers = new_graph_fn(model, i, subgraph)
            else:
                new_nodes, new_initializers = [], []
                
            newgraph_names = ", ".join([f"{item.name}({item.op_type})" for item in new_nodes])
            subgraph_names = ", ".join([f"{item.name}({item.op_type})" for item in subgraph])
            if len(new_nodes) > 0:
                log(f"Replace subgraph{i}: [{subgraph_names}] to: [{newgraph_names}]")
            else:
                log(f"Delete subgraph{i}: {subgraph_names}")
            
            lnodes = list(model.graph.node)
            idxs = sorted([lnodes.index(item) for item in subgraph], reverse=True)
            for i in idxs:
                del model.graph.node[i]
                
            for n in subgraph:
                # Remove the node and its corresponding information if it is not in new_nodes
                if n not in new_nodes:
                    remove_node_and_info(model, n)
            
            if len(new_nodes) == 0:
                input_node = subgraph[0]
                output_node = subgraph[-1]
                assert len(input_node.input) == len(output_node.output) and new_graph_fn is None or new_graph_fn is not None, f"Invalid replace"
                
                i2o = {a:b for a, b in zip(input_node.input, output_node.output)}
                for input_name in input_node.input:
                    parents = find_nodes_by_output(model, input_name)
                    for p in parents:
                        p.output[list(p.output).index(input_name)] = i2o[input_name]
            else:
                insert_point = idxs[-1]
                for node in new_nodes:
                    model.graph.node.insert(insert_point, node)
                    insert_point += 1
                
            for init in new_initializers:
                model.graph.initializer.append(init)
        return len(matched_subgraphs)