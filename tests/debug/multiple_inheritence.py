class Node:
    def __init__(self, name):
        self.name = name
        print("Hello from Node class! My name is {}".format(self.name))


class VariableNode(Node):
    def __init__(self, name, var_a, var_b):
        super(VariableNode, self).__init__(name)
        self.var_a = var_a
        self.var_b = var_b
        print("Hello from VariableNode! I have variables {} and {}".format(self.var_a, self.var_b))


class BetaNode(Node):
    def __init__(self, name, *args, **kwargs):
        super(BetaNode, self).__init__(name, *args, **kwargs)
        print("Hello from BetaNode!")


class TestNode(BetaNode, VariableNode):
    def __init__(self, name, var_a, var_b):
        super(TestNode, self).__init__(name, var_a, var_b)
        print("Hello from TestNode!")


if __name__ == "__main__":
    my_name = "Tester"
    my_var_a = "Variable_A"
    my_var_b = "Variable_B"
    tn = TestNode(my_name, my_var_a, my_var_b)

