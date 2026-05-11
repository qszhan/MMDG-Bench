

def compute_overfit(val_loss, train_loss, prev_val_loss, prev_train_loss):
    return abs((val_loss - train_loss) - (prev_val_loss - prev_train_loss))

def compute_gen(val_loss, prev_val_loss):
    return abs(val_loss - prev_val_loss)

def compute_gblend_coef(train_loss, val_loss, prev_train_loss, prev_val_loss):
    overfit_growth = compute_overfit(val_loss, train_loss, prev_val_loss, prev_train_loss)
    gen_gain = compute_gen(val_loss, prev_val_loss)
    if overfit_growth == 0:
        return 0.0
    return gen_gain / (overfit_growth ** 2 + 1e-6)  # avoid division by zero

def compute_gblend_coef_power(train_loss, val_loss, prev_train_loss, prev_val_loss):
    overfit_growth = compute_overfit(val_loss, train_loss, prev_val_loss, prev_train_loss)
    gen_gain = compute_gen(val_loss, prev_val_loss)
    if overfit_growth == 0:
        return 0.0
    return gen_gain** 2 / (overfit_growth ** 2 + 1e-6)  # avoid division by zero

def normalize_weights(weights):
    total = sum(weights)
    if total>0:
        normed_weights = [w / total for w in weights]
    else:
        normed_weights = [1.0 / len(weights)] * len(weights)
    return  normed_weights


