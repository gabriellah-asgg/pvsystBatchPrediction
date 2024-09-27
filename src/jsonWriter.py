import json

import pandas as pd

file = open(r"../res/cache.json")
x = json.load(file)
print(x["PV Panel Type"])


def check_models_to_run(model, model_params, pv_type):
    model_found = False
    model_name = str(model.__class__.__name__)
    file = open(r"../res/cache.json")
    pv_model_runs = json.load(file)
    if not pv_model_runs['PV Panel Type'].get(pv_type):
        pv_model_runs['PV Panel Type'][pv_type] = {
            "Best Model": "",
            "Best Model Score": None,
            "Best Model SI": None,
            "Models": {
            }
        }
    if not pv_model_runs["PV Panel Type"][pv_type]["Models"].get(model_name):
        # add the model name and param_variation dictionary
        pv_model_runs["PV Panel Type"][pv_type]["Models"][model_name] = {"best_tuned_params": {},
                                                                         "best_tuned_rmse": None, "best_tuned_si": None,
                                                                         "param_variations": []}
    for param_var in pv_model_runs["PV Panel Type"][pv_type]["Models"][model_name]["param_variations"]:
        param_to_check = param_var.get("params")
        model_found = params_equal(model_params, param_to_check)
        if model_found:
            break

    with open(r"../res/cache.json", 'w') as f:
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
            input_params[key] = sorted(input_params[key])

    for key in params:
        if isinstance(params[key], list):
            params[key] = sorted(params[key])
    if input_params == params:
        match = True
    return match


def add_model_params(pv_type, params, model, rmse, si, best_params=None):
    export = True
    file = open(r"../res/cache.json")
    model_name = model.replace("_tuned", "")
    pv_model_runs = json.load(file)
    pv_model_runs["PV Panel Type"][pv_type]["Models"][model_name]["param_variations"].append(
        {"params": params, "RMSE": rmse, "SI": si})
    improved_tuned_rmse = pv_model_runs["PV Panel Type"][pv_type]["Models"][model_name].get(
        "best_tuned_rmse") is None or pv_model_runs["PV Panel Type"][pv_type]["Models"][model_name].get(
        "best_tuned_rmse") > rmse
    if best_params is not None and improved_tuned_rmse:
        pv_model_runs["PV Panel Type"][pv_type]["Models"][model_name]["best_tuned_rmse"] = rmse
        pv_model_runs["PV Panel Type"][pv_type]["Models"][model_name]["best_tuned_si"] = si
        pv_model_runs["PV Panel Type"][pv_type]["Models"][model_name]["best_tuned_params"] = best_params
    # check if model is improved tuning (should be exported)
    else:
        if best_params is not None and not improved_tuned_rmse:
            export = False

    if pv_model_runs["PV Panel Type"][pv_type].get("Best Model Score") is None or pv_model_runs["PV Panel Type"][
        pv_type].get("Best Model Score") > rmse:
        pv_model_runs["PV Panel Type"][pv_type]["Best Model"] = model
        pv_model_runs["PV Panel Type"][pv_type]["Best Model Score"] = rmse
        pv_model_runs["PV Panel Type"][pv_type]["Best Model SI"] = si
    with open(r"../res/cache.json", 'w') as f:
        json.dump(pv_model_runs, f, indent=4)
    return export


def export_to_csv(pv_type):
    file = open(r"../res/cache.json")
    pv_model_runs = json.load(file)
    results_path = '../res/' + pv_type + '/' + 'model_results.csv'
    model_names = []
    rmse_list = []
    si_list = []
    for model in pv_model_runs["PV Panel Type"][pv_type]["Models"]:
        model_names.append((model + "_tuned"))
        model_dict = pv_model_runs["PV Panel Type"][pv_type]["Models"][model]
        rmse_list.append(model_dict["best_tuned_rmse"])
        si_list.append(model_dict["best_tuned_si"])
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
