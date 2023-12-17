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

def find_node_by_output(model, output):
    for node in model.graph.node:
        if output in node.output:
            return node
        
def find_node_by_input(model, input):
    for node in model.graph.node:
        if input in node.input:
            return node

def find_nodes_by_input(model, input):
    nodes = []
    for node in model.graph.node:
        if input in node.input:
            nodes.append(node)
    return nodes

def find_nodes_by_output(model, output):
    nodes = []
    for node in model.graph.node:
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
    
    nidxs = sorted(nidxs, reverse=True)
    iidxs = sorted(iidxs, reverse=True)
    for i in nidxs:
        del model.graph.node[i]
        
    for i in iidxs:
        del model.graph.initializer[i]


class Lexer:
    def __init__(self, pattern):
        
        # Compile the extraction regular expression.
        extract_name_and_argument = re.compile("([\W\w]+)\(([\W\w]+)\)")
        
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
        variables = dict()
        matched_pairs = []
        for inode, (allowed_op_types, inputs, outputs) in enumerate(self.lexer.patterns):
            found_node = None
            if inode == 0:
                candiates = [anchor]
            else:
                candiates = model.graph.node
                
            for node in candiates:
                is_allowed_op = False
                for optype in allowed_op_types:
                    if optype == "?":
                        is_allowed_op = True
                        break
                    
                    if node.op_type == optype:
                        is_allowed_op = True
                        
                if not is_allowed_op:
                    continue

                if not self._match_io(inputs, node.input, variables):
                    continue
                
                for i, item in enumerate(outputs):
                    if item != "?":
                        variables[item] = node.output[i]
                        
                found_node = node
                break
                    
            if found_node is None:
                return None
            
            matched_pairs.append(found_node)
        return matched_pairs

    def match(self, model):
        all_matched_pairs = []
        for node in model.graph.node:
            m = self._try_to_match(model, node)
            if m is not None:
                all_matched_pairs.append(m)
        return all_matched_pairs
    
    def print_match(self, model):
        print("=====================================================================")
        matched_subgraphs = self.match(model)
        print(f"Found {len(matched_subgraphs)} subgraphs:")
        for i, subgraph in enumerate(self.match(model)):
            subgraph_names = ", ".join([f"{item.name}({item.op_type})" for item in subgraph])
            print(f"\tSubgraph{i}: {subgraph_names}")
            
        pattern_text = "\n\t".join(self.lexer.lines)
        print(f"Pattern is:\n\t{pattern_text}")
        print("=====================================================================")
    
    # delete some subgraph
    def delete(self, model):
        self.replace(model, None)
    
    # replace some subgraph to new
    def replace(self, model, new_graph_fn=None):
        matched_subgraphs = self.match(model)
        for i, subgraph in enumerate(matched_subgraphs):
            if new_graph_fn is not None:
                new_nodes, new_initializers = new_graph_fn(i, subgraph)
            else:
                new_nodes, new_initializers = [], []
                
            newgraph_names = ", ".join([f"{item.name}({item.op_type})" for item in new_nodes])
            subgraph_names = ", ".join([f"{item.name}({item.op_type})" for item in subgraph])
            if len(new_nodes) > 0:
                print(f"Replace subgraph{i}: [{subgraph_names}] to: [{newgraph_names}]")
            else:
                print(f"Delete subgraph{i}: {subgraph_names}")
            
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
                assert len(input_node.input) == len(output_node.output), f"Invalid replace"
                
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