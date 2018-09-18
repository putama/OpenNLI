from torch import optim

def build_optimizer(model, arguments):
    params = filter(lambda p: p.requires_grad, model.parameters())
    if arguments.optim == "adam":
        optimizer = optim.Adam(params, lr=arguments.learning_rate)
    else:
        raise Exception("invalid optimization method")
    return optimizer


def adjust_learning_rate(optimizer, learning_rate):
    for pg in optimizer.param_groups:
        pg['lr'] = learning_rate
