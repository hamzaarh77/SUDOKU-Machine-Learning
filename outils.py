import numpy as np

def to_one_hot(grille,num_classes=9):
    # creation d'un nouveau tableau
    one_hot_grids = np.zeros((grille.shape[0], grille.shape[1], num_classes))
    
    for i in range(grille.shape[0]):  # for each colonne
        for j in range(grille.shape[1]):  # for each row
            number = grille[i, j]
            if number != 0:  # 
                one_hot_grids[i, j, number-1] = 1  # number-1 to shift index (1-4) -> (0-3)
    
    return one_hot_grids