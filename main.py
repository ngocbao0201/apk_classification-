from androguard.misc import AnalyzeAPK
import dgl
import torch
import FCG as F
from GIN import gin
from RF import RF

a1 = "Dataset//Begin//begin (2202).apk"
a2 = "Dataset//Malware//malware (2204).apk"
#dịch ngược file apk
a, d, dx = AnalyzeAPK(a1)
        
FCG = dx.get_call_graph()
edge_list = F.getEdgeList(FCG = FCG)
features_matrix = F.getFeatureMatrix(FCG = FCG)

g = dgl.graph(edge_list)
gin_model = gin.GIN(input_dim = 40, hidden_dim=16, output_dim=40)
h = torch.tensor(features_matrix)
gin_vec = gin_model.forward(g, h)

RF.predict(a.get_permissions(), gin_vec.detach().numpy())