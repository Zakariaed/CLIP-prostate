import pandas as pd
import os

# Path to your NCT-CRC-HE-100K dataset
dataset_path = 'datasets/NCT-CRC-HE-100K/NCT-CRC-HE-100K'

# Class descriptions based on tissue type
class_descriptions = {
    "ADI": "Adipose tissue: clusters of large, empty-looking cells with thin cytoplasm and nuclei pushed to the periphery.",
    "BACK": "Background: areas with no tissue, usually white or pale with no cellular structure.",
    "DEB": "Debris: fragmented or necrotic tissue remnants, irregular in shape and staining.",
    "LYM": "Lymphocytes: small, round cells with dense dark nuclei and very little cytoplasm.",
    "MUC": "Mucus: extracellular mucin pools, pale or lightly stained regions often without nuclei.",
    "MUS": "Muscle: elongated, fibrous tissue with striations and cigar-shaped nuclei.",
    "NORM": "Normal colon mucosa: organized epithelial glands with uniform nuclei and clear cytoplasm.",
    "STR": "Stroma: connective tissue with fibroblasts, extracellular matrix, and scattered nuclei.",
    "TUM": "Tumor: disorganized epithelial cells with pleomorphic nuclei and increased mitotic figures."
}

all_files = []
texts = []
labels = []

# Enumerate through class folders
for label_idx, class_name in enumerate(class_descriptions.keys()):
    class_path = os.path.join(dataset_path, class_name)
    files = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.lower().endswith('.tif')]

    all_files.extend(files)
    texts.extend([class_descriptions[class_name]] * len(files))
    labels.extend([label_idx] * len(files))

# Create DataFrame
df = pd.DataFrame({
    'images': all_files,
    'text': texts,
    'labels': labels
})

# Save to CSV
output_csv = './nct_crc_he_100k.csv'
df.to_csv(output_csv, index=False)

print(f"Length of the exported data: {len(df)}")
print(f"CSV saved to {output_csv}")
