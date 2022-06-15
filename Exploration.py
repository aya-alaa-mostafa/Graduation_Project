import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
import shap
import os
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data/cyp3a4_labeled_data.csv', low_memory=False)
features = df.drop(['smile', 'label'], axis=1).astype(float)

def adjust_plot(title, colorbar_label):
    plt.title(title, fontsize=11)
    ax = plt.gcf().axes[0]
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(8)
    for label in [ax.xaxis.label, ax.yaxis.label]:
        label.set_fontsize(10)
    ax = plt.gcf().axes[1]
    for label in (ax.get_yticklabels()):
        label.set_fontsize(8)
    if colorbar_label:
        ax.yaxis.label.set_fontsize(10)
    else:
        ax.set_ylabel('')
    plt.tight_layout()
    
def shap_feature_importance(features, model, title, save_path, sample_size=100):
    samples = shap.sample(features, sample_size)
    explainer = shap.KernelExplainer(model.predict_proba, samples)
    shap_values = explainer.shap_values(samples, nsamples=5, l1_reg="aic")
    shap.summary_plot(shap_values[0], samples, feature_names=samples.columns, max_display=5,
    plot_size=(6, 2), show=False)
    plt.xlabel('SHAP Value')
    adjust_plot(title, False)
    plt.savefig(f"{save_path}_feature_importance.svg", dpi=300)
    plt.show()
    shap.dependence_plot('rank(0)', shap_values[0], samples, interaction_index='rank(1)', show=False)
    adjust_plot(title, True)
    plt.savefig(f"{save_path}_dependence.svg", dpi=300)
    plt.show()

titles = ['Logistic Regression', 'Neural Network', 'Random Forest', 'Support Vector Machine']
model_paths = os.listdir('models')
index = 0
for model_path in model_paths:
    if '.h5' in model_path:
        pass
    else:
        model = load(f'models/{model_path}')
        print(model_path)
        shap_feature_importance(features, model, titles[index], f"images/{model_path.split('.')[0]}")
        index += 1