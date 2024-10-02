import json
from json import JSONDecodeError

import pandas as pd

filepaths = [ r"..\res\earrmpp_data\earrmpp_cache.json"]
for filepath in filepaths:
    file = open(filepath)
    pv_model_runs = json.load(file)
    print(filepath)
    new_json = {}

    data_dict = pv_model_runs["PV Panel Type"]
    data_type = list(data_dict.keys())[0]
    new_json["Dataset Type"] = {data_type:{}}
    new_json["Dataset Type"][data_type]["Best Model"] = data_dict[data_type]["Best Model"]
    new_json["Dataset Type"][data_type]["Best Model Score"] = data_dict[data_type]["Best Model Score"]
    new_json["Dataset Type"][data_type]["Best Model SI"] = data_dict[data_type]["Best Model SI"]
    models = data_dict[data_type]["Models"]
    new_models_dict = {}
    for key in models:
        #reform base model
        new_models_dict[key] = {}
        new_models_dict[key]["best_params"] = models[key]["param_variations"][0]["params"]
        new_models_dict[key]["best_rmse"] = models[key]["param_variations"][0]["RMSE"]
        new_models_dict[key]["best_si"] = models[key]["param_variations"][0]["SI"]
        new_models_dict[key]["param_variations"] = [models[key]["param_variations"][0]]


        # reform tuned model entry
        new_models_dict[key+"_tuned"] = {}
        new_models_dict[key + "_tuned"]["best_params"] = models[key]["best_tuned_params"]
        new_models_dict[key + "_tuned"]["best_rmse"] = models[key]["best_tuned_rmse"]
        new_models_dict[key + "_tuned"]["best_si"] = models[key]["best_tuned_si"]
        if len(models[key]["param_variations"]) > 1:
            new_models_dict[key + "_tuned"]["param_variations"] = models[key]["param_variations"][1:]
    new_json["Dataset Type"][data_type]["Models"] = new_models_dict
    new_fp = filepath.split('\\')[-1]
    with open(new_fp, 'w') as f:
        json.dump(new_json, f, indent=4)



