# Spaceship Zitanic

This project is adapted from the [Spaceship Titanic Kaggle competition](https://www.kaggle.com/competitions/spaceship-titanic). The task: predict whether passengers were transported to another dimension after a catastrophic event aboard the cruise ship Spaceship Zitanic. I designed and implemented this machine learning pipeline as my final project for CS 381: Deep Learning at Williams College.

---

## Key Contributions

Data preprocessing & feature engineering
- Encoded categorical variables such as `HomePlanet`, `Destination`, and `Cabin` (deck, number, side).
- Engineered name-based features (first name, surname, surname initial).
- Handled missing data with custom transformations.

Model design in PyTorch
  - Implemented a custom `SpaceshipZitanicData` dataset class for clean data loading.
 - Built a hybrid neural network (`CombinedNetwork`) that integrates:
    - Feedforward layers for numerical + categorical embeddings.
    - A recurrent-style component to capture character-level surname patterns.
  - Trained models with batch processing (`DataLoader`) and tracked performance.

Analysis & visualization
- Used matplotlib and seaborn for exploratory data analysis (`explore.py`).
- Compared feature distributions and visualized transport outcomes.

---

For more information on the assignment, please see [class-README.md](https://github.com/maddyandersen/spaceship-zitanic/blob/main/class-README.md).
