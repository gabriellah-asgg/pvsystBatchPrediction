import json

file = open(r"C:\Users\gabriellahoover\PycharmProjects\pvsystBatchPrediction\res\cache.json")
x = json.load(file)
print(x["PV Panel Type"])


def check_models_to_run(model, model_params, pv_type):
    model_found = False
    model_name = str(model.__class__.__name__)
    file = open(r"..\res\cache.json")
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
        pv_model_runs["PV Panel Type"][pv_type]["Models"][model_name] = {"param_variations": []}
    for param_var in pv_model_runs["PV Panel Type"][pv_type]["Models"][model_name]["param_variations"]:
        param_to_check = param_var.get("params")
        model_found = params_equal(model_params, param_to_check)

    with open(r"..\res\cache.json", 'w') as f:
        json.dump(pv_model_runs, f, indent=4)
    return model_found


def params_equal(input_params, params):
    match = False
    if len(input_params) != len(params):
        return match
    for key in input_params:
        if isinstance(input_params[key], list):
            input_params[key] = sorted(input_params[key])

    for key in params:
        if isinstance(params[key], list):
            params[key] = sorted(params[key])
    if input_params == params:
        match = True
    return match


def add_model_params(pv_type, params, model, rmse, si):
    file = open(r"..\res\cache.json")
    pv_model_runs = json.load(file)
    pv_model_runs["PV Panel Type"][pv_type]["Models"][model.replace("_tuned", "")]["param_variations"].append(
        {"params": params, "RMSE": rmse, "SI": si})
    if pv_model_runs["PV Panel Type"][pv_type].get("Best Model Score") is None or pv_model_runs["PV Panel Type"][pv_type].get("Best Model Score") > rmse:
        pv_model_runs["PV Panel Type"][pv_type]["Best Model"] = model
        pv_model_runs["PV Panel Type"][pv_type]["Best Model Score"] = rmse
        pv_model_runs["PV Panel Type"][pv_type]["Best Model SI"] = si
    with open(r"..\res\cache.json", 'w') as f:
        json.dump(pv_model_runs, f, indent=4)
