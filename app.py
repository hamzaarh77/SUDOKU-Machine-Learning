import torch
import numpy as np
from resolution import *
from extraction import *
from outils import *

def resultat(chemin):
    grille=numerisation_image(chemin)

    # traitement sur la grille pour pouvoir etre utilisable pour la resolution
    grille=np.array(grille).reshape(9,9)
    grille=to_one_hot(grille)

    # on importe le modele de resolution de sudoku
    model = model2.SudokuCNN()
    model.load_state_dict(torch.load('/home/etud/Bureau/sudoku_9x9/resolution/sudoku_model.pth'))
    model.eval()

    solution=prediction.prediction(model,grille)
    print(solution)


resultat("images/sudoku.png")




