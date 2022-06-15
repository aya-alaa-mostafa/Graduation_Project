import os
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import shap
from joblib import load
from tensorflow.keras.models import load_model
from featurize import generate_descriptors, generate_fingerprints, extract_smiles, extract_smile_structures
shap.initjs()
import warnings
warnings.filterwarnings('ignore')
test_set = pd.read_csv('data/fda_test.csv', low_memory=False)
mordred_features = generate_descriptors(test_set.smile.to_list())
fingerprints = generate_fingerprints(test_set.smile)
test_set = pd.concat([test_set, mordred_features, fingerprints], axis=1)
features = test_set.drop(['name', 'smile', 'type'], axis=1).astype(float)
smile_features = extract_smiles(test_set.smile, max_length=250).astype(float)
smile_structure = extract_smile_structures(test_set.smile, resolution=100, scale=(-15, 15)).astype(float)
meta = test_set[['name', 'type']]
def image_plot(shap_values, labels, figsize):
fig, axes = plt.subplots(nrows=shap_values.shape[0], ncols=1, figsize=figsize)
for row in range(shap_values.shape[0]):
abs_vals = np.abs(shap_values.sum(-1)).flatten()
max_val = np.nanpercentile(abs_vals, 99.9)
axes[row].set_title(labels[row], fontsize=11)
sv = shap_values[row].sum(-1)
im = axes[row].imshow(sv, cmap=shap.plots.colors.red_transparent_blue, vmin=-max_val, vmax=max_val)
for label in (axes[row].get_xticklabels() + axes[row].get_yticklabels()):
label.set_fontsize(8)
fig.subplots_adjust(wspace=0, hspace=0.3)
cb = fig.colorbar(im, ax=np.ravel(axes).tolist(), label="SHAP value", orientation="horizontal",
aspect=figsize[0]/0.2, pad=0.08)
cb.ax.xaxis.label.set_fontsize(10)
for label in (cb.ax.get_xticklabels()):
label.set_fontsize(8)
cb.outline.set_visible(False)
def plot_radar(values_best, values_worse, categories):
fig = go.Figure()
fig.add_trace(go.Scatterpolar(r=values_best, theta=categories, fill='toself', name='Best Prediction'))
fig.add_trace(go.Scatterpolar(r=values_worse, theta=categories, fill='toself', name='Worse Prediction'))
fig.update_layout(showlegend=True, autosize=False, width=500, height=500)
fig.show()
model_paths = os.listdir('models')
results = np.ndarray((meta.shape[0] * len(model_paths), 4), dtype=object)
for model_index, model_path in enumerate(model_paths):
print(model_path)
if '.h5' in model_path:
model = load_model(f'models/{model_path}')
if '2d' in model_path:
predictions = model.predict_proba(smile_structure)[:, 0]
sorted_indices = np.argsort(predictions, axis=0)
index = [sorted_indices[-1], sorted_indices[1]]
explainer = shap.GradientExplainer(model, smile_structure)
shap_values = explainer.shap_values(smile_structure)
plt.figure(figsize=(2.5, 2.5))
plt.matshow(np.amax(smile_structure[index[0],:,:,:], 2), cmap=plt.cm.gray_r, fignum=1)
plt.savefig(f"images/example_structure_1.svg", dpi=300)
plt.figure(figsize=(2.5, 2.5))
plt.matshow(np.amax(smile_structure[index[1],:,:,:], 2), cmap=plt.cm.gray_r, fignum=2)
plt.savefig(f"images/example_structure_2.svg", dpi=300)
image_plot(shap_values[0][index], meta.name[index].values, figsize=(3,8))
plt.savefig(f"images/2d_shap.svg", dpi=300)
plt.show()
else:
predictions = model.predict_proba(smile_features)[:, 0]
else:
model = load(f'models/{model_path}')
predictions = model.predict_proba(features)[:, 0]
sorted_indices = np.argsort(predictions, axis=0)
index = [sorted_indices[-1], sorted_indices[1]]
explainer = shap.KernelExplainer(model.predict_proba, features)
shap_values = explainer.shap_values(features, nsamples=50)
best = True
for i in index:
print(meta.name[i])
shap.force_plot(explainer.expected_value[0], shap_values[0][i,:], features.iloc[i,:].values,
list(features.columns.astype(str)), matplotlib=True, show=False, figsize=(20, 3.5), text_rotation=10)
plt.title(meta.name[i], fontsize=12)
plt.tight_layout()
plt.savefig(f"images/{model_path.split('.')[0]}_force_{'best' if best else 'worst'}.svg", dpi=300)
plt.show()
best = False
for pred_index, pred in enumerate(predictions):
index = pred_index + (meta.shape[0] * model_index)
results[index, :] = [meta.iloc[pred_index, 0], meta.iloc[pred_index, 1], model_path.split('-')[0], pred]
results = pd.DataFrame(results, columns=['name', 'type', 'model', 'inhibitor_conf'])
heatmap = None
for strength in ['strong', 'moderate', 'weak']:
temp = results[results.type == strength]
temp = temp.pivot(index='name', columns='model', values='inhibitor_conf').reset_index()
temp['name'] = temp.name + ' [' + strength + ']'
heatmap = pd.concat([heatmap, temp])
plt.figure(figsize=(6, 6))
plt.rcParams.update({'font.size': 7})
x = ['CNN Structure', 'CNN SMILE', 'LR', 'NN', 'RF', 'SVM']
y = heatmap['name'].to_list()
z = heatmap.values[:, 1:].astype(float)
im = plt.imshow(z, aspect='auto')
plt.xticks(np.arange(len(x)), x)
plt.yticks(np.arange(len(y)), y)
for i in range(len(y)):
for j in range(len(x)):
text = plt.text(j, i, round(z[i, j], 2), ha="center", va="center", color="w")
plt.tight_layout()
plt.savefig(f"images/fda_test.svg", dpi=300)
plt.show()