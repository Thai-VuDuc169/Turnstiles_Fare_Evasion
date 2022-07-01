import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
from tensorflow.python.platform import gfile
GRAPH_PB_PATH = r"/media/vee/Docs/thaivd4/Projects/Turnstiles_Fare_Evasion/models/deep_sort/mars-small128.pb"
### refer: https://stackoverflow.com/questions/50632258/how-to-restore-tensorflow-model-from-pb-file-in-python
# with tf.Session() as sess:
#     print("load graph")
#     with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read())
#         sess.graph.as_default()
#         tf.import_graph_def(graph_def, name='')
#         graph_nodes=[n for n in graph_def.node]
#         names = []
#         for t in graph_nodes:
#             names.append(t.name)
#         print(names)


### refer: https://stackoverflow.com/questions/51278213/what-is-the-use-of-a-pb-file-in-tensorflow-and-how-does-it-work
def load_pb(path_to_pb):
    with gfile.GFile(path_to_pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        graph_nodes=[n for n in graph_def.node]
        names = []
        for t in graph_nodes:
            names.append(t.name)
        return graph, names

graph, names = load_pb(GRAPH_PB_PATH)
print("names of graph nodes:")
print(names)
print(graph.get_operation_by_name('images:0'))
