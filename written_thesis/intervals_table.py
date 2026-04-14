from pandas import DataFrame, Index

from config.intervals import SEGMENT, CLIP, INTER_PRE, PREICTAL, SPH, INTERVENTION, POSTICTAL, INTER_POST
from utils.utils import safe_float_to_int

ivs = {iv.label: iv for iv in [SEGMENT, CLIP, INTER_PRE, PREICTAL, SPH, INTERVENTION, POSTICTAL, INTER_POST]}
idx = Index(ivs.keys(), name='Interval')

df = DataFrame(index=idx)
df['approx_dur'] = [iv.approx_dur for iv in ivs.values()]
df['mult_seg'] = df['approx_dur'] / SEGMENT.approx_dur
df['exact_dur'] = [iv.exact_dur for iv in ivs.values()]

# Make it look nice
r = DataFrame(index=idx)
r['Approximate\nduration [h]'] = df['approx_dur'].apply(lambda v: str(v.to_pytimedelta()))
r['Segments\ncontained'] = df['mult_seg'].apply(lambda v: safe_float_to_int(v))
r['Exact\nduration [h]'] = df['exact_dur'].apply(lambda v: str(v.to_pytimedelta()))

# r.to_excel(r"/Users/julian/Desktop/Master's Thesis/thesis_data/tables/intervals_table_generated.xlsx")
# r.to_csv(r"/Users/julian/Desktop/Master's Thesis/thesis_data/tables/intervals_table_generated.csv")
r.to_latex(r"/Users/julian/Desktop/Master's Thesis/thesis_data/tables/intervals_table_generated.tex")