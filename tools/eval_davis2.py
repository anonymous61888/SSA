import os
import time
import argparse
import cv2
from PIL import Image
import json
import numpy as np
from metrics import db_eval_iou, db_eval_boundary
import multiprocessing as mp

NUM_WOEKERS = 32


def eval_queue(q, rank, out_dict, davis_anno_path, davis_pred_path):
    while not q.empty():
        # print(q.qsize())
        vid_name, exp = q.get()

        vid = exp_dict[vid_name]

        exp_name = f'{vid_name}_{exp}'

        if not os.path.exists(f'{davis_pred_path}/{vid_name}'):
            print(f'{vid_name} not found')
            out_dict[exp_name] = [0, 0]
            continue

        pred_0_path = f'{davis_pred_path}/{vid_name}/{exp}/00000.png'
        pred_0 = cv2.imread(pred_0_path, cv2.IMREAD_GRAYSCALE)
        h, w = pred_0.shape
        vid_len = len(vid['frames'])
        gt_masks = np.zeros((vid_len, h, w), dtype=np.uint8)
        pred_masks = np.zeros((vid_len, h, w), dtype=np.uint8)

        obj_id = vid['expressions'][exp]['obj_id']

        for frame_idx, frame_name in enumerate(vid['frames']):

            mask = np.array(Image.open(f'{davis_anno_path}/{vid_name}/{frame_name}.png'))

            a = np.unique(mask)

            gt_mask = (mask == int(obj_id)).astype(np.uint8) * 255

            b = np.unique(gt_mask)

            gt_masks[frame_idx] = gt_mask

            pred_masks[frame_idx] = cv2.imread(f'{davis_pred_path}/{vid_name}/{exp}/{frame_name}.png', cv2.IMREAD_GRAYSCALE)

            c = np.unique(pred_masks[frame_idx])

        j = db_eval_iou(gt_masks, pred_masks).mean()
        f = db_eval_boundary(gt_masks, pred_masks).mean()
        out_dict[exp_name] = [j, f]
        # print(f'{vid_name} {exp}: {j:.4f} {f:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--davis_exp_path", type=str, default="meta_expressions.json")
    parser.add_argument("--davis_anno_path", type=str, default="Annotations")
    parser.add_argument("--davis_pred_path", type=str, default="inference")
    parser.add_argument("--save_name", type=str, default="davis_test.json")
    args = parser.parse_args()
    queue = mp.Queue()
    exp_dict = json.load(open(args.davis_exp_path))['videos']

    shared_exp_dict = mp.Manager().dict(exp_dict)
    output_dict = mp.Manager().dict()

    for vid_name in exp_dict:
        vid = exp_dict[vid_name]
        for exp in vid['expressions']:
            queue.put([vid_name, exp])

    start_time = time.time()
    processes = []
    for rank in range(NUM_WOEKERS):
        p = mp.Process(target=eval_queue, args=(queue, rank, output_dict, args.davis_anno_path, args.davis_pred_path))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    with open(args.save_name, 'w') as f:
        json.dump(dict(output_dict), f)

    j = [output_dict[x][0] for x in output_dict]
    f = [output_dict[x][1] for x in output_dict]

    print(f'J: {np.mean(j)}')
    print(f'F: {np.mean(f)}')
    print(f'J&F: {(np.mean(j) + np.mean(f)) / 2}')

    end_time = time.time()
    total_time = end_time - start_time
    print("time: %.4f s" %(total_time))