import numpy as np      
import networkx as nx
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1/(1+np.exp(-x))

input_layer=3
hidden_layer=4 
output_layer=2


input_layer_value = np.array([1, -0.5, 2])


hidden_layer_weight = np.random.rand(3, 4)  
hidden_layer_output = sigmoid(np.dot(input_layer_value, hidden_layer_weight))


output_layer_weight = np.random.rand(4, 2)  # 4 gizli katman düğümü, 2 çıktı düğümü
output_layer_output = sigmoid(np.dot(hidden_layer_output, output_layer_weight))

G = nx.DiGraph()


for i in range(input_layer):
    G.add_node(f'Input {i + 1}', value=input_layer_value[i]) 

for i in range(hidden_layer):
    round1= round(hidden_layer_output[i],2) #Using round causes overload and errors due to conflict, so it is useful to write and name round variables separately.
    G.add_node(f'Hidden {i + 1}', value=round1)  

for i in range(output_layer):
    round2 = round(output_layer_output[i],2)
    G.add_node(f'Output {i + 1}', value=round2)  


for i in range(input_layer):
    for j in range(hidden_layer):
        weight = hidden_layer_weight[i, j]
        G.add_edge(f'Input {i + 1}', f'Hidden {j + 1}', weight=weight)

for i in range(hidden_layer):
    for j in range(output_layer):
        weight=output_layer_weight[i,j]
        G.add_edge(f'Hidden {i + 1}', f'Output {j + 1}', weight=weight)


pos = {
    'Input 1': (0, 2),
    'Input 2': (0, 1),
    'Input 3': (0, 0),
    'Hidden 1': (1, 3),
    'Hidden 2': (1, 2),
    'Hidden 3': (1, 1),
    'Hidden 4': (1, 0),
    'Output 1': (2, 2),
    'Output 2': (2, 0)
}


edge_labels = {(i, j): f'{G[i][j]["weight"]:.2f}' for i, j in G.edges()}
node_labels = {node: f'{G.nodes[node]["value"]}' for node in G.nodes()}


nx.draw(G, pos, with_labels=True, labels=node_labels, node_size=1800, node_color='skyblue', font_size=8, font_color='black', font_weight='bold', arrowsize=20)


for (i, j), label in edge_labels.items():
    x, y = (pos[i][0] * 0.25 + pos[j][0] * 0.75, pos[i][1] * 0.25 + pos[j][1] * 0.75)
    plt.text(x, y, label, color='red', fontsize=12, ha='center', va='center')


print("Girdi katman:",input_layer_value)
print("Gizli katman ağırlık:",hidden_layer_weight)
print("Gizli katman değerleri:",hidden_layer_output)
print("Çıktı katman ağırlık:",output_layer_weight)
print("Çıktı katman değer",output_layer_output)


plt.title('Yapay Sinir Ağı Görselleştirmesi')
plt.show()

