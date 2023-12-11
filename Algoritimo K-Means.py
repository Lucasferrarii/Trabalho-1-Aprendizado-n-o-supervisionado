import cv2
import numpy as np
import os

# Função para calcular a quantidade de cores únicas na imagem
def count_unique_colors(image):
    pixels = image.reshape(-1, 3)
    unique_colors_str = set(map(tuple, pixels))
    unique_colors_count = len(unique_colors_str)
    return unique_colors_count

# Função para aplicar o algoritmo K-Means em uma imagem com um valor específico de k
def apply_kmeans(image, k):
    pixels = image.reshape((-1, 3))
    pixels = np.float32(pixels)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)

    return segmented_image

# Função para obter o tamanho do arquivo em KB ou MB
def get_file_size(path, unit='MB'):
    size_in_bytes = os.path.getsize(path)

    if unit == 'KB':
        size_in_unit = size_in_bytes / 1024
        unit_str = 'KB'
    elif unit == 'MB':
        size_in_unit = size_in_bytes / (1024 * 1024)
        unit_str = 'MB'
    else:
        size_in_unit = size_in_bytes
        unit_str = 'bytes'

    return size_in_unit, unit_str

# Carregar a imagem de entrada
image_path = 'Imagem3.png'
original_image = cv2.imread(image_path)

# Obter informações sobre a imagem original
resolution_before = original_image.shape[:2]
size_bytes_before = os.path.getsize(image_path)
size_kb_before, unit_before = get_file_size(image_path, unit='KB')
colors_before = count_unique_colors(original_image)

print("### Antes da Aplicação do K-Means ###")
print(f"Resolução: {resolution_before}")
print(f"Tamanho: {size_kb_before:.2f} {unit_before}")
print(f"Quantidade de Cores Únicas: {colors_before}")
print("\n")

# Definir uma lista de valores de k para testar
k_values = [100]

# Criar um diretório para salvar as imagens segmentadas
output_directory = 'imagens_segmentadas'
os.makedirs(output_directory, exist_ok=True)

# Crie uma lista para armazenar todas as imagens segmentadas
segmented_images = []

# Aplicar o algoritmo K-Means para diferentes valores de k
for k in k_values:
    segmented_image = apply_kmeans(original_image, k)
    segmented_images.append(segmented_image)

    # Obter a quantidade de cores únicas após a aplicação do K-Means
    colors_after = count_unique_colors(segmented_image)

    # Salvar as imagens segmentadas no formato PNG dentro da pasta 'imagens_segmentadas'
    output_path = os.path.join(output_directory, f'imagem_segmentada_k_{k}.png')
    cv2.imwrite(output_path, segmented_image)

    # Obter informações após cada segmentação
    resolution_after = segmented_image.shape[:2]
    size_bytes_after, unit_after = get_file_size(output_path, unit='KB')
    print(f"### Após K-Means (k = {k}) ###")
    print(f"Resolução: {resolution_after}")
    print(f"Tamanho: {size_bytes_after:.2f} {unit_after}")
    print(f"Quantidade de Cores Únicas: {colors_after}")
    print("\n")

print("Imagens segmentadas salvas no diretório:", output_directory)
