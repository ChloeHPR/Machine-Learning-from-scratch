import numpy as np

# 1. Choix des données d'entrée et vraies valeurs
# 2 échantillons, 2 caractéristiques (features) chacun -> Shape (2, 2)
X = np.array([[1, 0.2],
              [0.9, 0.3]])
y_true = np.array([[0],
                   [1]]) # Shape (2, 1) pour correspondre aux matrices

lr = 0.5
n = len(y_true)

# 2. Initialisation des poids et biais
# Couche cachée : 2 entrées vers 16 neurones -> W1 shape (2, 16), b1 shape (1, 16)
np.random.seed(42) # Pour des résultats reproductibles
W1 = np.random.randn(2, 16)
b1 = np.zeros((1, 16))

# Couche de sortie : 16 entrées vers 1 neurone -> W2 shape (16, 1), b2 shape (1, 1)
W2 = np.random.randn(16, 1)
b2 = np.zeros((1, 1))

for epoch in range(500):
    # 3. Forward pass (Propagation avant)
    Z1 = np.dot(X, W1) + b1       
    A1 = 1 / (1 + np.exp(-Z1))     # Activation Sigmoid

    Z2 = np.dot(A1, W2) + b2      
    A2 = 1 / (1 + np.exp(-Z2))     # Prédiction finale y_pred, shape (2, 1)

    print("Prédiction initiale (y_pred) :")
    print(A2.T)

   
    # 4. Backward pass (Rétropropagation)
   
    # Erreur sur la couche de sortie
    error_output = A2 - y_true     

    
    dW2 = np.dot(A1.T, error_output) / n   # Shape (16, 1)
    db2 = np.sum(error_output, axis=0, keepdims=True) / n  # Shape (1, 1)

    # Rétropropagation de l'erreur vers la couche cachée
    # On multiplie par la dérivée de la fonction Sigmoid : A1 * (1 - A1)
    error_hidden = np.dot(error_output, W2.T) * A1 * (1 - A1)  # Shape (2, 16)


    dW1 = np.dot(X.T, error_hidden) / n    # Shape (2, 16)
    db1 = np.sum(error_hidden, axis=0, keepdims=True) / n  # Shape (1, 16)

    # 5. Descente de gradient
   
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1


# 6. Forward pass final

Z1_new = np.dot(X, W1) + b1
A1_new = 1 / (1 + np.exp(-Z1_new))
Z2_new = np.dot(A1_new, W2) + b2
y_pred_new = 1 / (1 + np.exp(-Z2_new))

print("\nPrédiction après 1000 mise à jour :")
print(y_pred_new.T)
