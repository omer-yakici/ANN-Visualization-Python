import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Sinir ağı parametreleri
input_size = 3
hidden_size = 4
output_size = 2

input_layer = np.array([1, -0.5, 2])

# Ağırlıklar
weights_input_hidden = np.random.rand(input_size, hidden_size)
weights_hidden_output = np.random.rand(hidden_size, output_size)

hidden_layer = np.dot(input_layer, weights_input_hidden)
output_layer = np.dot(hidden_layer, weights_hidden_output)
# Görselleştirme için ağ oluşturma
G = nx.DiGraph()

# Input layer düğümleri
for i in range(input_size):
    G.add_node(f'Input {i + 1}', value=input_layer[i])  # Giriş düğüm değerleri

# Hidden layer düğümleri
for i in range(hidden_size):
    #rounded_value = round(hidden_layer[i], 2)  # Değerleri yuvarla
    G.add_node(f'Hidden {i + 1}', value=hidden_layer[i]) # Gizli düğüm değerleri

# Output layer düğümleri
for i in range(output_size):
    #rounded_value = round(output_layer[i], 2)  # Değerleri yuvarla
    G.add_node(f'Output {i + 1}', value=output_layer[i]) # Çıkış düğüm değerleri

# Ağırlıkları görselleştirmek için düğümleri ve bağlantıları ekleme
for i in range(input_size):
    for j in range(hidden_size):
        weight = weights_input_hidden[i, j]
        G.add_edge(f'Input {i + 1}', f'Hidden {j + 1}', weight=weight)

for i in range(hidden_size):
    for j in range(output_size):
        weight = weights_hidden_output[i, j]
        G.add_edge(f'Hidden {i + 1}', f'Output {j + 1}', weight=weight)

# Ağırlıkların uzunluğunu ayarlamak için pozisyonları elle belirleme
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

# Görselleştirmeyi çiz
nx.draw(G, pos, with_labels=True, labels=node_labels, node_size=1800, node_color='skyblue', font_size=8, font_color='black', font_weight='bold', arrowsize=20)

# Ağırlıkları okların ortasına değil biraz gerisine ekleme (%75'lik dilim)
for (i, j), label in edge_labels.items():
    x, y = (pos[i][0] * 0.25 + pos[j][0] * 0.75, pos[i][1] * 0.25 + pos[j][1] * 0.75)
    plt.text(x, y, label, color='red', fontsize=12, ha='center', va='center')

print("Giriş Verisi:",input_layer)
print("weights_input_hidden",weights_input_hidden)
print("weights_hidden_output:",weights_hidden_output)
print("hidden_layer",hidden_layer)
print("output_layer",output_layer)

plt.title('Yapay Sinir Ağı Görselleştirmesi')
plt.show()


