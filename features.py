import pandas as pd
import torch
from torch import tensor
from tqdm import tqdm
from torch.utils.data import Dataset

def cabin_deck(row):
    return row['Cabin'].split('/')[0] if row['Cabin'] != '???' else '?'

def cabin_num(row):
    return row['Cabin'].split('/')[1] if row['Cabin'] != '???' else '?'

def cabin_side(row):
    return row['Cabin'].split('/')[2] if row['Cabin'] != '???' else '?'

def first_name(row):
    return row['Name'].split(' ')[0] if row['Name'] != '???' else '?'

def surname(row):
    return row['Name'].split(' ')[1] if row['Name'] != '???' else '?'

def surname_first_letter(row):
    return row['Name'].split(' ')[1][0] if row['Name'] != '???' else '?'

def encode_name(name):
    if len(name) > 12:
        name = name[:12]
    elif len(name) < 12:
        name = name + ('?'*(12-len(name)))
    name = name.upper()

    return torch.stack([
        torch.tensor([1 if LETTERS.index(letter) == i else 0 for i in range(len(LETTERS))])
        for letter in name
    ])

PLANETS = ['Earth', 'Europa', 'Mars', '???']
DESTINATIONS = ['55 Cancri e', 'PSO J318.5-22', 'TRAPPIST-1e', '???']
DECKS = ['A','B','C','D','E','F','G','T','?']
SIDES = ['S', 'P', '?']
LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '?']

def read_csv(csv_file):
    """Reads a csv into a pandas dataframe."""
    data = pd.read_csv(csv_file)
    data["Cabin Deck"] = data.apply(cabin_deck, axis=1)
    data["Cabin Num"] = data.apply(cabin_num, axis=1)
    data["Cabin Side"] = data.apply(cabin_side, axis=1)

    data['First Name'] = data.apply(first_name, axis=1)
    data['Surname'] = data.apply(surname, axis=1)
    data['Surname First Letter'] = data.apply(surname_first_letter, axis=1)

    return data
  

def categorical_feature(row, feature_name, domain):
    """Creates binary features from a categorical feature."""    
    result = [0.0] * len(domain)
    result[domain.index(row[feature_name])] = 1.0    
    return tensor(result)


def create_features(row):
    """Creates a feature vector from a row of a dataframe."""  

    return torch.cat([
        categorical_feature(row, 'HomePlanet', PLANETS),
        categorical_feature(row, 'Destination', DESTINATIONS),
        categorical_feature(row,'Cabin Deck', DECKS),
        categorical_feature(row,'Cabin Side', SIDES),
 
        tensor([row["CryoSleep"]]),
        tensor([row["VIP"]]), 

        # normalizing numerical features
        tensor([(row["Age"]-28.88) / 14.38]),
        tensor([(row["RoomService"]-220.51) / 649.12]),
        tensor([(row["FoodCourt"]-478.14) / 1690.64]),
        tensor([(row["ShoppingMall"]-173.71) / 611.09]),
        tensor([(row["Spa"]-317.16) / 1146.32]),
        tensor([(row["VRDeck"]-305.19) / 1140.75])

    ], dim=0)
 

class SpaceshipZitanicData(Dataset):
    def __init__(self, csv_file, test_set=False):
        data = read_csv(csv_file)
        self.x = []
        self.y = []
        for _, row in tqdm(data.iterrows()):              
            self.x.append({f: row[f] for f in set(data.columns) - {'Transported'}})
            if test_set:
                self.y.append(-1)
            else:
                self.y.append(row['Transported'])        
                      
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return ((create_features(self.x[idx]), self.y[idx]), encode_name(self.x[idx]['Surname']))

    
if __name__ == "__main__":             
    train_set = SpaceshipZitanicData('data/train.csv')
    print(train_set[0])
    