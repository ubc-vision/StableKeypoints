import re
import os
import shutil

import torch
import torch.nn.functional as F
import numpy as np

r'''
    source code from GLU-Net
    https://github.com/PruneTruong/GLU-Net
'''
def save_checkpoint(state, is_best, save_path, filename):
    torch.save(state, os.path.join(save_path,filename))
    if is_best:
        shutil.copyfile(os.path.join(save_path,filename), os.path.join(save_path,'model_best.pth'))

r'''
    source code from GLU-Net
    https://github.com/PruneTruong/GLU-Net
'''
def load_checkpoint(model, optimizer, scheduler, filename='checkpoint.pth.tar'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    best_val=-1
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        try:
            best_val=checkpoint['best_loss']
        except:
            best_val=-1
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, scheduler, start_epoch, best_val


# Argument parsing
def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def log_args(args):
    r"""Log program arguments"""
    print('\n+================================================+')
    for arg_key in args.__dict__:
        print('| %20s: %-24s |' % (arg_key, str(args.__dict__[arg_key])))
    print('+================================================+\n')


def parse_list(list_str):
    r"""Parse given list (string -> int)"""
    return list(map(int, re.findall(r'\d+', list_str)))


def where(predicate):
    r"""Predicate must be a condition on nd-tensor"""
    matching_indices = predicate.nonzero()
    if len(matching_indices) != 0:
        matching_indices = matching_indices.t().squeeze(0)
    return matching_indices

    
def visualie_correspondences(initial_image, final_image, source, target, name, correct_ids=None, save_folder = "outputs", line_width=5):
    r"""Visualize correspondences. Show initial image on the left, final image on the right and correspondences connecting corresponding points"""
    
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.colors as colors
    import matplotlib.cm as cmx
    import numpy as np
    
    display_img = torch.cat([initial_image, final_image], dim=2)
    
    
    # make the figure without a border
    fig = plt.figure(frameon=False)
    fig.set_size_inches(20, 10)
    
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    plt.imshow(display_img.permute(1, 2, 0).detach().cpu().numpy(), aspect='auto')
    
    source = source.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    
    for i in range(source.shape[2]):
        if source[0, 0, i] == -1:
            continue
        if correct_ids is not None and i in correct_ids:
            color = "blue"
        else:
            color = "orange"
        plt.plot([source[0, 0, i], target[0, 0, i]+512], [source[0, 1, i], target[0, 1, i]], color=color, linewidth=line_width)
        
    plt.axis('on')

    plt.savefig(f'{save_folder}/{name}.png')
    plt.close()

    