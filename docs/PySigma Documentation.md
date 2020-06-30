# PySigma Documentation

## External Libraries Used
  - **NetworkX** - easy graph representation and manipulation
  - **PyTorch** - support for tensor operation, GPU acceleration, and neural models
  - ***(optional)* Pyro** - probabilistic inference and learning that support continuous parameterized distributions 
  - **Matplotlib / pygraphviz** - Interactive GUI


## (Sub)Structures defined by `namedtuple`

### `PredicateArgument`

### `PredicatePattern`

### `Evidence`


## Class System

### Components of the Cognitive Architecture

#### `Sigma`
  - The Sigma model class. Instance of this class specify a Sigma model. 
  - **Attributes**:
  - **Methods**:

#### `Type`
  - Specify a Sigma type. Can be symbolic or discrete (will not support continuous type at this point). If symbolic, must provide a list of symbols. If discrete, must specify the range of values through `min` and `max`. Note that `min` is inclusive while `max` is exclusive, i.e., the actual range would be $[min, max)$
  - **Attributes**:
  - **Methods**:
    ```Python
    __init__(type_name, value_type, min=None, max=None, symbol_list=None)
    ```
      - `type_name`: `str` type. Name of the Sigma type
      - `value_type`: 'str' type. `"symbolic"` or `"discrete"`.
      - `min`: `int` type. The lowest value of a discrete type. Must specify if type is `"discrete"`.
      - `max`: `int` type. The highest value + 1 of a discrete type. Must specify if type is `"discrete"`.
      - `symbol_list`: `list` type. A list of symbols. Must specify if type is `"symbol"`. 

#### `Predicate`
  - Specify a Sigma predicate. 
  - **Attributes**:
  - **Methods**:
    ```Python
    __init__(predicate_name, arguments, world='open', exponential=False, no_normalize=True, perception=False, function=None)
    ```
      - `predicate_name`: `str` type
      - `arguments`: `list` type. List of **argument tuples**. This is to specify the **working memory variables** (formal arguments) corresponding to this predicate. 
        - Each argument tuple is a tuple of two or three `str` of the form `(argument_name, type_name)` or `(argument_name, type_name, unique_symbol)`
        - Valid `unique_symbol`s are:
          - `'!'`: Select best
          - `'%'`: Maintain distribution
          - `'$'`: Select expected value
          - `'^'`: Maintain exponential transform of distribution
          - `'='`: Select by probability matching
        - **NOTE**: `unique_symbol` must be specified if to declare this variable to be unique. Otherwise will be treated as a universal variable. You can use `%` to specify that the distribution should not be changed during selection phase. 
      - `world`: `open` or `closed`. Default to `open`.
      - `exponential`: `bool` type. Whether to exponentiate outgoing messages from Working Memory Variable Node (WMVN). Default to `False`.
      - `no_normalize`: `bool` type. Whether not to normalize outgoing messages from WMVN. Default to `True`.
      - `perception`: `bool` type. Whether this predicate can receive perceptual information.
      - `function`: 
        - `None`: no function specified at this predicate
        - `int` or `float`: a constant function
        - `torch.tensor`: Use the given tensor as the predicate function. Note that the shape of the tensor must agree with dimensions of the variables and in the order as they are specified. 
        - `str`: Name of another predicate. In this case they shares the same function. 

#### `Conditional`


### Components of the Graphical Architecture

#### `Message` inherits `torch.tensor`
  - A subclass of pytorch tensor. Stores the actual content as a tensor and can be directly manipulated as a regular tensor, but stores extra bookkeeping information for message processing mechanism in Sigma. 
  - **Attributes**:
  - **Methods**: 

#### `Node`
  - The super class of `FactorNode` and `VariableNode`. It declares common attributes between `FactorNode` and `VariableNode`, for example a list of working memory variables. During construction of the graph, its instance will be passed to `NetworkX` methods to instantiate a node. 
  - **Attributes**:
    - `id`: `int` type. Specify the integer id of this node in the graph.
    - `variables`: `list` type. Specify a list of **working memory variable names** corresponding to this node. For a variable node, it is just its declared working memory variables. For a factor node, it is the union of working memory variables from all its adjacent variable nodes. This will be used to name the dimensions of the factor function tensor. 
  - **Methods**:


#### `LinkData`
  - Identify the *data* of a ***directed*** link between a factor node and a variable node. Stores intermediate messages in the **message memory**. Two of such links should be specified with opposite directions to represent a bidirectional link between a factor node and a variable node, typically in the case of condacts. During construction of the graph, its instance will be passed to `NetworkX` methods as the edge data to instantiate an edge. 
  - **Attributes**:
    - `id`: `int` type. Specify the integer id of this link in the graph
    - `message_memory`: `Message` type. Temporarily stores the message sent from one node to the other. 
  - **Methods**:


#### `FactorNode` inherits `Node`
  - Specify a **factor node**, with pytorch tensor as optional factor node function (default to a constant function of 1 everywhere effectively). It is the super class of factor nodes of various subtypes, such as alpha, beta, delta, gamma nodes. 
  - **Attributes**:
    - `function`: `Message` type. Specify the factor node function. Default to a constant function of 1's in all slots. 
  - **Methods**:

#### `VariableNode` inherits `Node`
  - Specify a **variable node**. 
  - **Attributes**:
  - **Methods**:

#### `Graph` inherits `networkx.DiGraph`
  - The actual graph instance representing a compiled graphical architecture. Can be manipulated directly as a `networkx.DiGraph` but store extra bookkeeping information. 


###### tags: `Research` `Sigma` `Cognitive Architecture` `Implementation` `Documentation`