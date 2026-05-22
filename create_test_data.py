import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

NUM_TRAIN = 200
NUM_VAL = 50
NUM_TEST = 50

np.random.seed(42)

rows = []
for i in range(NUM_TRAIN):
    pixels = ' '.join(str(x) for x in np.random.randint(0, 256, 48*48))
    rows.append({'emotion': np.random.randint(0, 7), 'pixels': pixels, 'Usage': 'Training'})
for i in range(NUM_VAL):
    pixels = ' '.join(str(x) for x in np.random.randint(0, 256, 48*48))
    rows.append({'emotion': np.random.randint(0, 7), 'pixels': pixels, 'Usage': 'PrivateTest'})
for i in range(NUM_TEST):
    pixels = ' '.join(str(x) for x in np.random.randint(0, 256, 48*48))
    rows.append({'emotion': np.random.randint(0, 7), 'pixels': pixels, 'Usage': 'PublicTest'})

os.makedirs('datasets/fer2013', exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv('datasets/fer2013/fer2013.csv', index=False)
print(f"Synthetic dataset created: {len(rows)} samples")
