import pandas as pd
import dgl
import torch
from androguard.misc import AnalyzeAPK
import FCG as F
from GIN import gin

# Train 2000 file begin + 2000 file malware
for i in range(1, 2001):
    try :
        a2 = "Dataset//Malware//malware (" + str(i) + ").apk"

        a, d, dx = AnalyzeAPK(a2)
        output_file = "Log/common_info_" + str(i) + ".txt"

        with open(output_file, "w") as f:
            f.write("Package Name: " + a.get_package() + "\n")
            f.write("Version Name: " + a.get_androidversion_name() + "\n")
            f.write("Permissions: " + str(a.get_permissions()) + "\n")
                
        FCG = dx.get_call_graph()

        edge_list = F.getEdgeList(FCG = FCG)
        features_matrix = F.test2(FCG = FCG)

        g = dgl.graph(edge_list)
        input_dim = 40

        gin_model = gin.GIN(input_dim = input_dim, hidden_dim=16, output_dim=40)
        h = torch.tensor(features_matrix)
        gin_vec = gin_model.forward(g, h)

        labels = [0]
        df = pd.DataFrame(gin_vec.detach().numpy(), columns=[f'feature_{j+1}' for j in range(gin_vec.shape[1])])
        df['label'] = labels
        df.to_csv('csv/malware_' + str(i) + '.csv', index=False)
    except :
        print(i)

#Tạo DataFrame chứa thông tin từ common_info.txt và đặc trưng từ gin_vec cho tất cả các file APK
merged_df_list = []
for i in range(1, 2001):
    with open('Log/common_info_' + str(i) + '.txt', 'r') as f:
        common_info = f.readlines()
    #package_name = common_info[0].strip().split(": ")[1]
    #version_name = common_info[1].strip().split(": ")[1]
    permissions = common_info[2].strip().split(": ")[1]

    common_info_df = pd.DataFrame({
        #'Package Name': [package_name],
        #'Version Name': [version_name],
        'Permissions': [permissions]
    })

    df_gin_vec = pd.read_csv('csv/malware_' + str(i) + '.csv')

    merged_df = pd.concat([common_info_df, df_gin_vec], axis=1)
    merged_df_list.append(merged_df)

final_df = pd.concat(merged_df_list, ignore_index=True)
final_df.to_csv('csv/malware_final.csv',mode='a',header=False, index=False)