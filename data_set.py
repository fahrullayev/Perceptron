import pandas as pd
import numpy as np


data = {
    'Yosh': np.random.randint(20, 70, 2000),
    'Vazn': np.random.randint(50, 100, 2000),
    'SystolicQonbosim': np.random.randint(110, 160, 2000),
    'DiastolicQonbosim': np.random.randint(70, 100, 2000),
    'Xoleterin': np.random.randint(150, 250, 2000),
    'Glukoza': np.random.randint(70, 150, 2000)
}

df = pd.DataFrame(data)

df['Target'] = np.where((df['Age'] + df['Weight']) > 100, 1, 0)

df.to_csv('malumotlar.csv', index=False)

print("Dataset yaratildi.")
