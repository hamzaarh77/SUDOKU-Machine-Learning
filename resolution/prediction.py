import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def prediction(model, sudoku_grid, device='cpu'):
   
    model.eval()  # Met le modèle en mode évaluation
    
    
    # Convertir la grille en un Tensor PyTorch, ajouter une dimension batch, et envoyer au dispositif approprié
    sudoku_tensor = th.tensor(sudoku_grid, dtype=th.float32).unsqueeze(0).to(device)
    
    with th.no_grad():  # Désactive le calcul des gradients
        output = model(sudoku_tensor)
        _, predictions = th.max(output, dim=3)  # Obtenir les indices des classes prédites
        
    # Ajuster les indices pour une indexation basée à 1, convertir en numpy array, et redimensionner à 4x4
    predicted_sudoku = (predictions.squeeze().cpu().numpy() + 1).reshape((9, 9))
    
    return predicted_sudoku

# teste sur les predictions :
def correspondance(x, y):
    cpt = np.sum(x == y)
    percentage = (cpt / np.prod(x.shape)) * 100
    print("percentage de correspondance :",str(percentage),"%")