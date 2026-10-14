from sklearn.datasets import load_iris
from sklearn.manifold import TSNE

# Load dataset
X, y = load_iris(return_X_y=True)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X)

print("Original Shape:", X.shape)
print("Reduced Shape:", X_tsne.shape)