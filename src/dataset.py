import os
import cv2
import argparse
import numpy as np

def generate_random_psf(psf_cnt: int) -> np.ndarray:
    """
    Generates random PSF coefficients of length psf_cnt that sum to 1.
    """
    psf_v = np.random.rand(psf_cnt)
    psf_v /= np.sum(psf_v)  # normalize so sum(psf_v) = 1
    return psf_v

def blur_image(img: np.ndarray, psf_v: np.ndarray, psf_x: np.ndarray, psf_y: np.ndarray) -> np.ndarray:
    """
    Blurs the input image using the given PSF coefficients and offsets.
    
    :param img:    Input image as a NumPy array (H x W x 3).
    :param psf_v:  PSF coefficients (length = psf_cnt).
    :param psf_x:  Offsets in the X direction (length = psf_cnt).
    :param psf_y:  Offsets in the Y direction (length = psf_cnt).
    :return:       Blurred image (uint8).
    """
    h, w, _ = img.shape 
    blurred = np.zeros_like(img, dtype=np.float64)
    
    for y in range(h):
        for x in range(w):
            pixel_val = img[y, x, :]
            for k in range(len(psf_v)):
                tx = x + psf_x[k]
                ty = y + psf_y[k]
                if 0 <= tx < w and 0 <= ty < h:
                    blurred[ty, tx, :] += psf_v[k] * pixel_val
    
    blurred = np.clip(blurred, 0, 255).astype(np.uint8)
    return blurred

def process_images(
    input_dir: str,
    output_blurred_dir: str,
    output_labels_dir: str,
    psf_x: np.ndarray,
    psf_y: np.ndarray,
    random_psf: bool = True
):
    """
    Processes images from input_dir, applies blur, and saves results to output_blurred_dir and output_labels_dir.
    """
    os.makedirs(output_blurred_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    psf_cnt = len(psf_x)  # Expecting 9 if using [-4..4]
    uniform_psf = np.full(psf_cnt, 1.0 / psf_cnt)

    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    image_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith(valid_extensions)
    ]

    for img_name in image_files:
        img_path = os.path.join(input_dir, img_name)
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"Warning: Could not read {img_path}. Skipping.")
            continue

        # Convert to float64 for more accurate accumulation before final clip
        img_bgr = img_bgr.astype(np.float64)

        # Generate random PSF or use uniform
        if random_psf:
            psf_v = generate_random_psf(psf_cnt)
        else:
            psf_v = uniform_psf

        # Apply blur
        blurred_bgr = blur_image(img_bgr, psf_v, psf_x, psf_y)

        # Construct output paths
        base_name, _ = os.path.splitext(img_name)
        output_img_path = os.path.join(output_blurred_dir, f"{base_name}_blurred.png")
        output_psf_path = os.path.join(output_labels_dir, f"{base_name}_psf.txt")

        # Save blurred image and PSF coefficients
        cv2.imwrite(output_img_path, blurred_bgr)
        np.savetxt(output_psf_path, psf_v, header="PSF Coefficients")

        print(f"[INFO] Blurred image saved to {output_img_path}")
        print(f"[INFO] PSF coefficients saved to {output_psf_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Blur images with a specified PSF kernel and save PSF coefficients as labels."
    )
    parser.add_argument(
        "--train_input_dir",
        default="../dataset/images/train",
        help="Directory for the training images."
    )
    parser.add_argument(
        "--test_input_dir",
        default="../dataset/images/test",
        help="Directory for the test images."
    )
    parser.add_argument(
        "--train_output_blurred_dir",
        default="../dataset/blurred_images/train",
        help="Directory to save blurred training images."
    )
    parser.add_argument(
        "--test_output_blurred_dir",
        default="../dataset/blurred_images/test",
        help="Directory to save blurred test images."
    )
    parser.add_argument(
        "--train_output_labels_dir",
        default="../dataset/labels/train",
        help="Directory to save PSF labels for training images."
    )
    parser.add_argument(
        "--test_output_labels_dir",
        default="../dataset/labels/test",
        help="Directory to save PSF labels for test images."
    )
    parser.add_argument(
        "--random_psf",
        action="store_true",
        help="If set, generate a random PSF for each image. Otherwise, use a uniform PSF."
    )

    args = parser.parse_args()

    # Define PSF offsets (e.g., 9-tap kernel with offsets [-4..4])
    psf_x = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4], dtype=int)
    psf_y = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4], dtype=int)

    # Process TRAIN set
    process_images(
        input_dir=args.train_input_dir,
        output_blurred_dir=args.train_output_blurred_dir,
        output_labels_dir=args.train_output_labels_dir,
        psf_x=psf_x,
        psf_y=psf_y,
        random_psf=args.random_psf
    )

    # Process TEST set
    process_images(
        input_dir=args.test_input_dir,
        output_blurred_dir=args.test_output_blurred_dir,
        output_labels_dir=args.test_output_labels_dir,
        psf_x=psf_x,
        psf_y=psf_y,
        random_psf=args.random_psf
    )

if __name__ == "__main__":
    main()
