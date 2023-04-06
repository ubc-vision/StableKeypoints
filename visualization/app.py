

from flask import Flask, render_template, url_for, send_from_directory
from glob import glob
import os
import time

app = Flask(__name__)


@app.route('/custom_static/<path:filename>')
def custom_static(filename):
    
    print('/scratch/iamerich/prompt-to-prompt/outputs/' + filename)
    
    return send_from_directory('/scratch/iamerich/prompt-to-prompt/outputs/', filename)


@app.route('/')
def main():
    
    
    # gallery_names is a list of the folder names in outputs
    gallery_names = glob("/scratch/iamerich/prompt-to-prompt/outputs/*")
    # current gallery_names is list of pngs and folders, only take the folders
    gallery_names = [x for x in gallery_names if os.path.isdir(x)]
    
    gallery_names = [x.split("/")[-1] for x in gallery_names]
    
    print("gallery_names")
    print(gallery_names)

    return render_template('main.html', gallery_names=gallery_names)


@app.route('/gallery/<gallery_id>')
def gallery(gallery_id):
    cache_buster = int(time.time())

    # static_folder = os.path.join(
    #     app.root_path, f'static/static_{gallery_id:02d}', f"correspondences_estimated_*.png")
    static_folder = f"/scratch/iamerich/prompt-to-prompt/outputs/{gallery_id}/correspondences_estimated_*.png"
    
    print("static_folder")
    print(static_folder)

    matching_files = glob(static_folder)
    n = len(matching_files)

    return render_template('index.html', gallery_id=gallery_id, n=n, cache_buster=cache_buster)


@app.route('/gallery/<gallery_id>/view/<int:image_id>')
def view(gallery_id, image_id):
    cache_buster = int(time.time())

    # static_folder = os.path.join(
    #     app.root_path, f'static/static_{gallery_id:02d}', f"{image_id:03d}_largest_loc_trg_*_00.png")
    static_folder = f"/scratch/iamerich/prompt-to-prompt/outputs/{image_id:03d}_largest_loc_trg_*_00.png"

    matching_files = glob(static_folder)
    n = len(matching_files)
    
    print("n")
    print(n)

    return render_template('view.html', gallery_id=gallery_id, image_id=image_id, x=n, y=3, cache_buster=cache_buster)


if __name__ == '__main__':
    # app.run(host='127.0.0.1', port=5000, debug=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
