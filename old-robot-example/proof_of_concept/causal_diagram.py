from collections import defaultdict
import numpy as np
import networkx as nx


class NodeFunction:
    pass


class DAG:
    """
    """
    def __init__(self, V, F):
        """
        Send a message to a recipient

        :param dict V: A dict of vertex and its associated values
        :param dict F: A dict of edges (key) and its associated function (value).
                       {(v_sink,v_src1,v_src2,...):NodeFunction, ...}
        """
        self._set_vertices_edges(V, F)

    def _set_vertices_edges(self, V, F):
        """
        """
        self.graph = nx.DiGraph()

        # deal with vertices
        for vn, val in V.items():
            self.graph.add_node(vn, value=val)

        # deal with edges
        for vs, f in F.items():
            self.graph.add_edges_from([(vs[x],vs[0]) for x in range(1,len(vs))])
            self.graph.nodes[vs[0]]['node_function'] = f

        # specify random values for root nodes
        root_nodes = self.get_root_nodes()
        self.intervene({node:np.random.randn(*self.graph.nodes[node]['value'].shape) for node in root_nodes})
        self.run()

    def get_root_nodes(self):
        return [node for node in nx.topological_sort(self.graph) if len(list(self.graph.predecessors(node)))==0]

    def intervene(self, vertex_value_dict):
        """
        """
        for k,v in vertex_value_dict.items():
            self.graph.nodes[k]['value'] = v

    def run(self, N=1, reset_root=False):
        """
        """
        root_nodes = self.get_root_nodes()
        nonroot_nodes = [x for x in nx.topological_sort(self.graph) if x not in root_nodes]

        res = defaultdict(list)
        for n in range(N):
            if reset_root:
                self.intervene({node:np.random.randn(*self.graph.nodes[node]['value'].shape) for node in root_nodes})
            for node in nonroot_nodes:
                parent_vals = {x:self.graph.nodes[x]['value'] for x in self.graph.predecessors(node)}
                if 'node_function' in self.graph.nodes[node]:
                    self.graph.nodes[node]['value'] = self.graph.nodes[node]['node_function'](**parent_vals)
            for node in self.graph.nodes:
                res[node].append(self.graph.nodes[node]['value'])
        for k in res:
            res[k] = np.concatenate(res[k], axis=0)
        return res

    def visualize(self):
        """
        """
        pass
