import torch
import os
import glob
import time
from flask import send_from_directory
from flask import Flask, render_template
import argparse

app = Flask(__name__)


parser = argparse.ArgumentParser(description='Correspondence Visualization')

# Dataset
parser.add_argument('--image_folder_path', type=str, default='outputs', help='path to the folder containing the images to be served')
args = parser.parse_args()

args.image_folder_path = os.path.abspath(args.image_folder_path)

@app.route('/serve_image/<path:image_path>')
def serve_image(image_path):
    return send_from_directory(args.image_folder_path, image_path)

@app.route('/images/<int:img_num>/<line_tag>')
def show_images(img_num, line_tag):
    cache_buster = int(time.time())
    image_files = get_image_files(line_tag, img_num)
    return render_template('image_gallery.html', images=image_files, img_num=img_num, cache_buster=cache_buster)

def get_image_files(line_tag, img_num=0):
    image_files = [
        f"{img_num:03d}_initial_point_{line_tag}.png",
        f"{img_num:03d}_largest_loc_src_{line_tag}_00.png",
        f"{img_num:03d}_largest_loc_src_{line_tag}_01.png",
        f"{img_num:03d}_largest_loc_src_{line_tag}_02.png",
        f"{img_num:03d}_largest_loc_src_{line_tag}_03.png",
        f"{img_num:03d}_largest_loc_src_{line_tag}_mean.png",
        f"{img_num:03d}_target_point_{line_tag}.png",
        f"{img_num:03d}_largest_loc_trg_{line_tag}_00.png",
        f"{img_num:03d}_largest_loc_trg_{line_tag}_01.png",
        f"{img_num:03d}_largest_loc_trg_{line_tag}_02.png",
        f"{img_num:03d}_largest_loc_trg_{line_tag}_03.png",
        f"{img_num:03d}_largest_loc_trg_{line_tag}_mean.png",
    ]

    return image_files


@app.route('/<int:img_num>')
@app.route('/', defaults={'img_num': 0})
def index(img_num):
    cache_buster = int(time.time())
    return render_template('index.html', lines=get_lines(img_num), img_num=img_num, cache_buster=cache_buster)


def get_lines(img_num=0):

    data = torch.load(
        f"{args.image_folder_path}/correspondence_data_{(img_num):03d}.pt")
    
    est_keypoints = data['est_keypoints'][0]
    correct_ids = data['correct_ids']
    src_kps = data['src_kps'][0]
    trg_kps = data['trg_kps'][0]

    # count the number of keypoints that arent -1
    num_keypoints = torch.sum(est_keypoints[0] != -1)

    lines = []

    for i in range(num_keypoints):

        tag = f"{i:02d}"
        x1 = src_kps[0, i].item()
        y1 = src_kps[1, i].item()
        x2 = est_keypoints[0, i].item()+512
        y2 = est_keypoints[1, i].item()
        color = 'blue' if i in correct_ids else 'orange'

        lines.append({'tag': tag, 'x1': x1, 'y1': y1,
                     'x2': x2, 'y2': y2, 'color': color})

    return lines


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
