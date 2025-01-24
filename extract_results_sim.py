import pandas as pd

# Define the model names and Datasets
models = ["Autoformer", "Crossformer", "FEDformer", "PatchTST", "iTransformer", "TimeXer", "Transformer"]
Datasets = ["idp", "dp02", "dp04", "dp06", "dp08", "dp10"]
seeds = ['_42','_123','_456']
plens = ['pl96','pl192','pl336','pl720']
variants = ['noSkipTrue','FDCTrue']

# Read the input text file and process it
input_file = "./eval_results_delta/mi_results_tasks/max/mi_results_synthetic.txt"  # Replace with your text file name
records = []
with open(input_file, 'r') as file:
    lines = file.readlines()
    for i in range(0, len(lines), 2):
        # Extract the model and Dataset from the first line
        model_info = lines[i].strip()
        # print(model_info)
        performance_info = lines[i + 1].strip()

        # Find model and Dataset
        model = next((m for m in models if m in model_info), None)
        Dataset = next((b for b in Datasets if b in model_info), None)
        seed = next((s for s in seeds if s in model_info),None)
        plen = next((pl for pl in plens if pl in model_info),None)
        vt = [v for v in variants if v in model_info]
        

        if model and Dataset and seed and plen:
            # Parse the performance metrics
            performance = dict(item.split(':') for item in performance_info.split(','))
            performance = {k: float(v) for k, v in performance.items()}
            if len(vt)>0:
                v = '_'.join(vt)
                performance['Model'] = model+'_'+v
            else:
                performance['Model'] = model
            performance['Dataset'] = Dataset
            performance['Seed'] = seed
            performance['PLen'] = plen
            records.append(performance)

# Create a pandas DataFrame
df = pd.DataFrame(records)

# df_hierarchical = (
#     df.set_index(["Dataset", "Model"])
#     .unstack("Model")
#     .swaplevel(axis=1)
#     .sort_index(axis=1)
# )
# Reshape the data for comparison
# comparison_table = df.pivot(index="Dataset", columns="Model", values=["self_mi", "cross_mi", "mse", "mae"])

df.to_csv('./mi_extracted_synthetic3.csv')