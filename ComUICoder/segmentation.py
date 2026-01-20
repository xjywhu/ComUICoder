import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json

def find_split_lines_color(img_path, min_ratio=0.35, max_ratio=0.65, threshold=30,
                           max_height=2500,
                           basename=None,
                           save_dir=None):
    os.makedirs(save_dir, exist_ok=True)
    #os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    img = cv2.imread(img_path)  # BGR
    h, w, c = img.shape
    print(f"original height: {h}")

    # color difference
    diff = np.linalg.norm(img[1:, :, :] - img[:-1, :, :], axis=2)  # shape: (h-1, w)

    split_lines = []

    def recursive_split(top, bottom, idx=0):

        height = bottom - top
        if height <= max_height:
            return

        lower_bound = int(top + min_ratio * height)
        upper_bound = int(top + max_ratio * height)

        best_line = None
        best_run = -1

        for y in range(lower_bound, upper_bound):
            row_diff = diff[y, :]
            mask = row_diff > threshold

            run = 0
            max_run = 0
            for val in mask:
                if val:
                    run += 1
                else:
                    max_run = max(max_run, run)
                    run = 0
            max_run = max(max_run, run)

            if max_run > best_run:
                best_run = max_run
                best_line = y

        if best_line is not None:
            split_lines.append(best_line)
            print(f"Find the split line: {best_line}px (段高度={height})")

            recursive_split(top, best_line, idx + 1)
            recursive_split(best_line, bottom, idx + 1)

    recursive_split(0, h)

    # draw split lines
    img_with_lines = img.copy()
    for y in split_lines:
        cv2.line(img_with_lines, (0, y), (w, y), (0, 0, 255), 3)

    save_path=os.path.join(save_dir,f"{basename}_with_segment_position.png")
    cv2.imwrite(save_path, img_with_lines)
    print(f"Image with split lines is save at {save_path}")


    split_lines_sorted = sorted([0] + split_lines + [h])
    for i in range(len(split_lines_sorted) - 1):
        top, bottom = split_lines_sorted[i], split_lines_sorted[i + 1]
        part = img[top:bottom, :]
        save_path=os.path.join(save_dir, f"{basename}_split")
        os.makedirs(save_path, exist_ok=True)
        out_path = os.path.join(save_path,f"split_{i}.png")
        cv2.imwrite(out_path, part)
        print(f"Segmentation part saved: {out_path}, height={bottom - top}")

    json_path = os.path.join(save_dir, "split_lines.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "image_height": h,
            "split_lines": sorted(split_lines)
        }, f, indent=4, ensure_ascii=False)

    # display
    # plt.figure(figsize=(10, 10))
    # plt.imshow(cv2.cvtColor(img_with_lines, cv2.COLOR_BGR2RGB))
    # plt.axis("off")
    # plt.title("All split lines")
    # plt.show()

    return split_lines

if __name__ == "__main__":
    for i in range(1,119):
        find_split_lines_color(f"./data/{i}.png",
                               basename=f"{i}",
                               save_dir=f"./output_seg/{i}/")
