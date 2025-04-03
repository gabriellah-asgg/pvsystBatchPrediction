import json
from json import JSONDecodeError
from modelWrapper import BaseModel, TunedModel, TFModel

import pandas as pd


def check_models_to_run(model_name, model, data_type, filepath=r"../res/cache.json"):
    model_found = False
    file = open(filepath)
    try:
        pv_model_runs = json.load(file)
    except JSONDecodeError as e:
        pv_model_runs = {}
    if not pv_model_runs.get('Dataset Type'):
        pv_model_runs['Dataset Type'] = {}
    if not pv_model_runs['Dataset Type'].get(data_type):
        pv_model_runs['Dataset Type'][data_type] = {
            "Best Model": "",
            "Best Model Score": None,
            "Best Model SI": None,
            "Models": {
            }
        }
    if not pv_model_runs["Dataset Type"][data_type]["Models"].get(model_name):
        # add the model name and param_variation dictionary
        pv_model_runs["Dataset Type"][data_type]["Models"][model_name] = {"best_params": {},
                                                                        "best_rmse": None, "best_si": None,
                                                                        "param_variations": []}
    
    for param_var in pv_model_runs["Dataset Type"][data_type]["Models"][model_name]["param_variations"]:
        param_to_check = param_var.get("params")
        model_found = model.equals(param_to_check)
        if model_found:
            break
    if not model_found:
        with open(filepath, 'w') as f:
            json.dump(pv_model_runs, f, indent=4)
    return model_found


def params_equal(input_params, params):
    match = False
    if len(input_params) != len(params):
        return match
    for key in input_params:
        # the input params may have tuples that need to be converted to lists
        if isinstance(input_params[key], tuple):
            input_params[key] = list(input_params[key])
        if isinstance(input_params[key], list):
            input_params[key] = sorted(input_params[key], key=lambda x: (x is None, x))

    for key in params:
        if isinstance(params[key], list):
            params[key] = sorted(input_params[key], key=lambda x: (x is None, x))
    if input_params == params:
        match = True
    return match


def add_model_params(data_type, model, filepath=r"../res/cache.json"):
    export = True
    file = open(filepath)
    pv_model_runs = json.load(file)
    serialized_params = model.serialize_parameters()
    best_params = model.best_params
    rmse = model.rmse
    si = model.si
    pv_model_runs["Dataset Type"][data_type]["Models"][model.model_type]["param_variations"].append(
        {"params": serialized_params, "RMSE": rmse, "SI": si})
    improved_rmse = pv_model_runs["Dataset Type"][data_type]["Models"][model.model_type]["best_rmse"] is None or \
                    pv_model_runs["Dataset Type"][data_type]["Models"][model.model_type]["best_rmse"] > rmse
    if improved_rmse:
        pv_model_runs["Dataset Type"][data_type]["Models"][model.model_type]["best_rmse"] = rmse
        pv_model_runs["Dataset Type"][data_type]["Models"][model.model_type]["best_si"] = si
        if best_params is None:
            best_params = serialized_params
        pv_model_runs["Dataset Type"][data_type]["Models"][model.model_type]["best_params"] = best_params
    # check if model is improved tuning (should be exported)
    else:
        if not improved_rmse:
            export = False

    if pv_model_runs["Dataset Type"][data_type].get("Best Model Score") is None or pv_model_runs["Dataset Type"][
        data_type].get("Best Model Score") > rmse:
        pv_model_runs["Dataset Type"][data_type]["Best Model"] = model.model_type
        pv_model_runs["Dataset Type"][data_type]["Best Model Score"] = rmse
        pv_model_runs["Dataset Type"][data_type]["Best Model SI"] = si

    with open(filepath, 'w') as f:
        json.dump(pv_model_runs, f, indent=4)
    return export


def export_to_csv(pv_type, filepath=r"../res/cache.json"):
    file = open(filepath)
    pv_model_runs = json.load(file)
    results_path = '../res/' + pv_type + '/' + 'model_results.csv'
    model_names = []
    rmse_list = []
    si_list = []
    for model in pv_model_runs["Dataset Type"][pv_type]["Models"]:
        model_names.append((model + "_tuned"))
        model_dict = pv_model_runs["Dataset Type"][pv_type]["Models"][model]
        rmse_list.append(model_dict["best_rmse"])
        si_list.append(model_dict["best_si"])
        for i in range(0, len(model_dict["param_variations"])):
            check_params = model_dict["param_variations"][i]
            if check_params == {}:
                model_names.append(model)
                rmse_list.append(model_dict["param_variations"][i]["RMSE"])
                si_list.append(model_dict["param_variations"][i]["SI"])
                break
    results_data = {'Model Name': model_names, 'Model RMSE': rmse_list, 'Model Scatter Index': si_list}
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(results_path, index=False)
