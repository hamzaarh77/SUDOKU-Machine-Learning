import pandas as pd 
import numpy as np 
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch as th
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import model2 as m 
from entrainement import train
from evaluation import evaluate
from prediction import prediction,correspondance

# importation des données 
grilles= pd.read_csv("data/sudoku_matrice.csv")
solutions= pd.read_csv("data/sudoku_solutions.csv")

# traitement des données 
# conversion en tableaux numpy
grilles = grilles.values
solutions = solutions.values

# redimensionnement et préparation des grilles et solutions
grilles_reshaped = []
solutions_reshaped = []

for grille in tqdm(grilles, desc="grille"):
    reshaped = np.array(grille).reshape((9, 9))  
    grilles_reshaped.append(reshaped)

for solution in tqdm(solutions, desc="solutions"):
    reshaped = np.array(solution).reshape((9, 9))  # les solutions
    solutions_reshaped.append(reshaped)

grilles_reshaped = np.array(grilles_reshaped)
solutions_reshaped = np.array(solutions_reshaped)

# séparation en jeux d'entraînement et de test
x_train, x_test, y_train, y_test = train_test_split(grilles_reshaped, solutions_reshaped, test_size=0.2, random_state=42)


print("dimensions de : x_train, x_test, y_train, y_test")
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# transformation en one hot encoding 
def convert_to_one_hot(grids, num_classes=9):
    # creation d'un nouveau tableau
    one_hot_grids = np.zeros((grids.shape[0], grids.shape[1], grids.shape[2], num_classes))
    
    for i in range(grids.shape[0]):  # for each grid
        for j in range(grids.shape[1]):  # for each row
            for k in range(grids.shape[2]):  # for each column
                number = grids[i, j, k]
                if number != 0:  # 
                    one_hot_grids[i, j, k, number-1] = 1  # number-1 to shift index (1-4) -> (0-3)
    
    return one_hot_grids


x_train = convert_to_one_hot(x_train)
x_test = convert_to_one_hot(x_test)

print("dimensions de x_train et x_test en one hot :")
print(x_train.shape)
print(x_test.shape)


# transformation des données en tenseur puis dataset puis dataloader 
x_train_tensor = th.tensor(x_train, dtype=th.float32)
y_train_tensor = th.tensor(y_train -1, dtype=th.long).view(-1,81)  
x_test_tensor = th.tensor(x_test, dtype=th.float32)
y_test_tensor = th.tensor(y_test -1, dtype=th.long).view(-1,81)

print("dimensions des tenseur :")
print(y_test_tensor.shape)
print(x_test_tensor.shape)

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(x_test_tensor , y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# modele 
model = m.SudokuCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
num=10


#entrainement du model
train(train_loader, model, criterion, optimizer, num)


# sauvegarde du modele 
th.save(model.state_dict(), 'sudoku_model.pth')















# #evaluation du model 
# evaluate(model, test_loader)




# pred= prediction(model, x_test[3])
# attendus =y_test[3]

# print("resultat attendus => \n",attendus)
# print("resulttat predis => \n", pred)


# correspondance(pred,attendus)
