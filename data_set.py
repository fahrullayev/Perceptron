import pandas as pd

data = {
    'Yosh': [25, 45, 35, 50, 40, 60, 30, 55, 42, 38],
    'Vazni': [70, 80, 60, 90, 75, 85, 65, 82, 68, 74],
    'SistolikQonBosimi': [120, 130, 125, 140, 135, 150, 115, 145, 125, 130],
    'DiastolikQonBosimi': [80, 85, 82, 90, 88, 95, 78, 92, 80, 83],
    'Xolesterin': [180, 210, 190, 220, 200, 240, 170, 230, 180, 200],
    'Glyukoza': [90, 110, 100, 120, 105, 130, 85, 125, 95, 100],
    'Target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 0]
}

df = pd.DataFrame(data)
df.to_csv('perceptron_dataset.csv', index=False)

print("CSV faylga saqlandi.")
