import shap
from lime.lime_tabular import LimeTabularExplainer
import dice_ml
import pandas as pd

def run_shap(clf, X_train_scaled, X_test_scaled, X_test_df, feature_names):
    explainer = shap.KernelExplainer(clf.predict_proba, X_train_scaled[:200])
    shap_values = explainer.shap_values(X_test_scaled[:100])
    shap.summary_plot(shap_values[1], X_test_df.iloc[:100], feature_names=feature_names)

def run_lime(clf, X_train, X_test, feature_names):
    explainer = LimeTabularExplainer(training_data=X_train.values, feature_names=feature_names,
                                     class_names=['NonDiabetic','Diabetic'], discretize_continuous=True)
    explanation = explainer.explain_instance(X_test.values[0], clf.predict_proba, num_features=6)
    explanation.show_in_notebook()

def run_dice(clf, X, y, feature_names):
    data_df = pd.DataFrame(X, columns=feature_names)
    data_df['Diabetes_binary'] = y.reset_index(drop=True)

    dice_data = dice_ml.Data(dataframe=data_df,
                             continuous_features=[col for col in feature_names],
                             outcome_name='Diabetes_binary')
    dice_model = dice_ml.Model(model=clf, backend='sklearn')
    exp = dice_ml.Dice(dice_data, dice_model, method='random')
    cf = exp.generate_counterfactuals(data_df.iloc[:1], total_CFs=3, desired_class="opposite")
    return cf.visualize_as_dataframe(show_only_changes=True)