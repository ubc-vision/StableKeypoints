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


def flow2kps(trg_kps, flow, n_pts, upsample_size=(512, 512)):
    _, _, h, w = flow.size()
    
    flow = F.interpolate(flow, upsample_size, mode='bilinear') * (upsample_size[0] / h)
    
    
    src_kps = []
    for trg_kps, flow, n_pts in zip(trg_kps.long(), flow, n_pts):
        size = trg_kps.size(1)

        kp = torch.clamp(trg_kps.narrow_copy(1, 0, n_pts), 0, upsample_size[0] - 1)
        estimated_kps = kp + flow[:, kp[1, :], kp[0, :]]
        estimated_kps = torch.cat((estimated_kps, torch.ones(2, size - n_pts).cuda() * -1), dim=1)
        src_kps.append(estimated_kps)
        

    return torch.stack(src_kps)


# def visualize_image_with_points(image, point, name):
#     import matplotlib.pyplot as plt
    
#     print("plotting")
#     print(name)
    
#     # make the figure without a border
#     fig = plt.figure(frameon=False)
#     fig.set_size_inches(10, 10)
    
#     ax = plt.Axes(fig, [0., 0., 1., 1.])
#     ax.set_axis_off()
#     fig.add_axes(ax)
    
#     plt.imshow(image.permute(1, 2, 0).detach().cpu().numpy(), aspect='auto')
    
#     # plot point on image
#     plt.scatter(point[:, 0], point[:, 1], s=10, marker='o', c='r')
    
    
#     plt.savefig(f'outputs/{name}.png', dpi=300)
#     plt.close()
    
    

def visualie_flow(initial_image, final_image, flow, name):
    r"""Visualize flow. Show initial image on the left, final image on the right and flow connecting corresponding points"""
    
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.colors as colors
    import matplotlib.cm as cmx
    import numpy as np
    
    display_img = torch.cat([final_image, initial_image], dim=2)

    # visualize initial image
    # plt.subplot(1, 2, 1)
    plt.imshow(display_img.permute(1, 2, 0).detach().cpu().numpy())

    # # visualize final image
    # plt.subplot(1, 2, 2)
    # plt.imshow(final_image.permute(1, 2, 0).detach().cpu().numpy())
    # plt.axis('off')
    
    flow = flow.detach().cpu().numpy()
    
    
    
    # flow is a 2xHxW tensor
    # flow[0] is the horizontal component
    # flow[1] is the vertical component
    # visualize flow as arrows connecting first image to second
    H, W = flow.shape[1], flow.shape[2]
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    
    x = x.reshape(-1)
    y = y.reshape(-1)
    
    x = (x+0.5)*16
    y = (y+0.5)*16
    # x = x*16
    # y = y*16
    
    flow = flow.reshape(2, -1)
    
    for i in range(flow.shape[1]):
        if flow[0, i] == 0 and flow[1, i] == 0:
            continue
        plt.plot([x[i], x[i]+flow[0, i]*16+512], [y[i], y[i]+flow[1, i]*16], color="red", linewidth=1)
        
    plt.axis('on')

    
    
    
    # plt.plot(x, y, color="white", linewidth=3)
    # plt.quiver(x*16, y*16, flow[0]/32, flow[1]/32, color='r', scale=1)
    plt.savefig(f'outputs/{name}.png')
    
    
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
    
    # print("display_img.permute(1, 2, 0).detach().cpu().numpy().shape")
    # print(display_img.permute(1, 2, 0).detach().cpu().numpy().shape)
    # exit()
    
    

    # visualize initial image
    # plt.subplot(1, 2, 1)
    plt.imshow(display_img.permute(1, 2, 0).detach().cpu().numpy(), aspect='auto')
    
    
    # # visualize final image
    # plt.subplot(1, 2, 2)
    # plt.imshow(final_image.permute(1, 2, 0).detach().cpu().numpy())
    # plt.axis('off')
    
    source = source.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    
    # print("correspondences")
    # print(correspondences)
    # print("correspondences.shape")
    # print(correspondences.shape)
    
    # import ipdb; ipdb.set_trace()
    
    for i in range(source.shape[2]):
        if source[0, 0, i] == -1:
            continue
        if correct_ids is not None and i in correct_ids:
            color = "blue"
        else:
            color = "orange"
        plt.plot([source[0, 0, i], target[0, 0, i]+512], [source[0, 1, i], target[0, 1, i]], color=color, linewidth=line_width)
        
    plt.axis('on')

    
    
    
    # plt.plot(x, y, color="white", linewidth=3)
    
    # print("x.shape")
    # print(x.shape)
    # print("x")
    # print(x)
    # print("y.shape")
    # print(y.shape)
    # print("y")
    # print(y)
    # plt.quiver(x*16, y*16, flow[0]/32, flow[1]/32, color='r', scale=1)
    plt.savefig(f'{save_folder}/{name}.pdf')
    plt.close()

    