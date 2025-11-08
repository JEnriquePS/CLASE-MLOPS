import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error, confusion_matrix, precision_recall_fscore_support
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Backend for saving plots without display
import matplotlib.pyplot as plt
import seaborn as sns

import re
import pickle

DATA_PATH = Path(__file__).resolve().parent / "data" / "base_encuestados_v2.csv"
df = pd.read_csv(DATA_PATH)     

df = df[['Comentarios','NPS']].dropna().copy()
df['Comentarios'] = df['Comentarios'].apply(lambda x: x.lower())
df['Comentarios'] = df['Comentarios'].apply(lambda x: re.sub(r'[^a-zA-z0-9\s]', '', x))


le = LabelEncoder()
df['NPS_encoded'] = le.fit_transform(df['NPS'])
# integer labels for modeling
y = df['NPS_encoded'].values


max_features = 1000
tokenizer = Tokenizer(num_words=max_features, split = ' ')
tokenizer.fit_on_texts(df['Comentarios'].values)
X = tokenizer.texts_to_sequences(df['Comentarios'].values)
X = pad_sequences(X)
print(X.shape)

emdeb_dim = 50
model = Sequential()
model.add(Embedding(max_features, emdeb_dim, input_length = X.shape[1]))
model.add(LSTM(10))
model.add(Dense(len(df['NPS_encoded'].unique()), activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
print(model.summary())


y = pd.get_dummies(df['NPS_encoded']).values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1901)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


model.fit(X_train, y_train, epochs=5, verbose=1, validation_data=(X_test, y_test))

test = ['El servicio fue excelente y muy rápido']
test = tokenizer.texts_to_sequences(test)
test = pad_sequences(test, maxlen=X.shape[1], dtype='int32', value=0)
print(model.predict(test))
sentiment = model.predict(test)[0]
if(np.argmax(sentiment) == 0):
    print("Detractor")
elif (np.argmax(sentiment) == 1):
    print("Pasivo")
else:
    print("Promotor")




with open('models/tokenizer.pickle', 'wb') as tk:
    pickle.dump(tokenizer, tk, protocol=pickle.HIGHEST_PROTOCOL)

model_json = model.to_json()
with open("models/model.json", "w") as js:
    js.write(model_json)

model.save_weights('models/.model.weights.h5')



# --- New evaluation on X_test and saving metrics to file ---
# Convert one-hot y_test back to label indices
y_test_labels = np.argmax(y_test, axis=1)

# Predict on X_test
y_preds_probs = model.predict(X_test)
y_preds_labels = np.argmax(y_preds_probs, axis=1)

# Compute metrics
acc = accuracy_score(y_test_labels, y_preds_labels)
mae_val = np.round(float(mean_absolute_error(y_test_labels, y_preds_labels)), 2)
mse_val = np.round(float(mean_squared_error(y_test_labels, y_preds_labels)), 2)
conf_mat = confusion_matrix(y_test_labels, y_preds_labels)
# classification report with original label names (in encoder order)
label_names = list(le.classes_)

class_report = classification_report(y_test_labels, y_preds_labels, target_names=label_names, zero_division=0)

metrics_text = []
metrics_text.append(f"Accuracy = {acc:.4f}")
metrics_text.append(f"Mean Absolute Error = {mae_val}")
metrics_text.append(f"Mean Squared Error = {mse_val}")
metrics_text.append("\nClassification Report:")
metrics_text.append(class_report)
metrics_text.append("\nConfusion Matrix:")
metrics_text.append(np.array2string(conf_mat))

metrics_output = "\n".join(metrics_text)
print("\nEvaluation results:\n", metrics_output)

with open('metrics.txt', 'w', encoding='utf-8') as outfile:
    outfile.write(metrics_output)

# --- Generate plot comparing metrics per category ---
# Extract precision, recall, f1-score per class
precision, recall, f1, support = precision_recall_fscore_support(y_test_labels, y_preds_labels, zero_division=0)

# Set style for better visualization
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10

# Create DataFrame with metrics
metrics_df = pd.DataFrame({
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1
}, index=label_names)

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot grouped bar chart
x = np.arange(len(label_names))
width = 0.25

bars1 = ax.bar(x - width, metrics_df['Precision'], width, label='Precision', 
               color='#3498db', edgecolor='black', linewidth=0.7)
bars2 = ax.bar(x, metrics_df['Recall'], width, label='Recall', 
               color='#2ecc71', edgecolor='black', linewidth=0.7)
bars3 = ax.bar(x + width, metrics_df['F1-Score'], width, label='F1-Score', 
               color='#e74c3c', edgecolor='black', linewidth=0.7)

# Customize plot
ax.set_xlabel('Categorías NPS', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Comparación de Métricas por Categoría', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(label_names, fontsize=11)
ax.set_ylim(0, 1.1)
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig('metrics_by_category.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("\nGráfica guardada en: metrics_by_category.png")
