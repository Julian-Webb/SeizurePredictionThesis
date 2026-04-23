import pandas as pd
from pandas import IndexSlice

from config import PATHS


def qualify_model_features(
        alpha: float = 0.05,
        roc_auc_thresh: float = 0.75,
        model_names: list[str] = ('CNN', 'ensemble'),
):
    models = pd.read_pickle(PATHS.per_model_comparison_table.pickle)
    cycle_extr = pd.read_pickle(PATHS.cycle_extraction_metrics_per_ptnt_table.pickle)
    circ_comp = pd.read_pickle(PATHS.circular_comparison_per_ptnt_table.pickle)
    combined = cycle_extr.join(circ_comp)

    # Check that the patient's seizures are phase locked with features
    t = combined[[('seizures', 'p_rayleigh_bh')]].copy()
    t[('seizures', 'significant')] = t[('seizures', 'p_rayleigh_bh')] < alpha

    for model_name in model_names:
        # Check that the model has appropriate ROC AUC
        roc_auc_by_ptnt = models.loc[IndexSlice[:, model_name], ('roc_auc', 'test')].droplevel(1)
        roc_auc = t.index.get_level_values(0).map(roc_auc_by_ptnt)  # Broadcast to entire index
        t[(model_name, 'roc_auc')] = roc_auc
        t[(model_name, 'roc_auc_met')] = roc_auc >= roc_auc_thresh

        # Check if FPs per model are phase locked with features
        p_ray = combined[(f'{model_name} FPs', 'p_rayleigh_bh')]
        t[(model_name, 'p_rayleigh_bh')] = p_ray
        t[(model_name, 'significant')] = p_ray < alpha

        # Check if FPs vs seizures per model come from the same distribution
        p_perm = combined[(f'seizures vs {model_name} FPs', 'p_perm_bh')]
        t[(model_name, 'p_perm_bh')] = p_perm
        t[(model_name, 'met')] = p_perm >= alpha

        t[(model_name, 'qualified')] = (
                t[('seizures', 'significant')] &
                t[(model_name, 'roc_auc_met')] &
                t[(model_name, 'significant')] &
                t[(model_name, 'met')]
        )

    # ---- Save DataFrame
    path = PATHS.model_feature_qualifications_table
    t.to_pickle(path.pickle)
    # Color boolean columns and save as Excel with formatting
    t.style.map(lambda val: 'background-color: lightgreen' if val else 'background-color: salmon',
                subset=[col for col in t.columns if t[col].dtype == bool]
                ).to_excel(path.xlsx, engine='openpyxl')
    return t


if __name__ == '__main__':
    qualify_model_features()
