from graphviz import Digraph

def _render_tree(self, node, dot=None):
    if dot is None:
        dot = Digraph(comment='Decision Tree')
        dot.node(str(id(node)), str(node['question']))

    for label, child_node in node['children'].items():
        dot.node(str(id(child_node)), str(child_node['question']))
        dot.edge(str(id(node)), str(id(child_node)), label=str(label))

        if child_node['children']:
            self._render_tree(child_node, dot)

    return dot

def viz(self):
    root = self.tree  # Assuming you store your tree in a 'tree' attribute
    dot = self.render_tree(root)
    dot.render('decision_tree', view=True)
