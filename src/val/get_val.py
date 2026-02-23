from src.val.Bert import val_Bert

def get_val(model_name, data_name, state_dict_full, logger):
    if model_name != "Bert":
        raise ValueError(f"Bert-only mode: unsupported model-name '{model_name}'")
    val_Bert(model_name, data_name, state_dict_full, logger)
    return True
