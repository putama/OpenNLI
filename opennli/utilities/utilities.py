import torch


def to_cuda(*objects):
    if torch.cuda.is_available():
        if len(objects) == 1:
            return objects[0].cuda()
        return [obj.cuda() for obj in objects]
    else:
        if len(objects) == 1:
            return objects[0]
        else:
            return objects