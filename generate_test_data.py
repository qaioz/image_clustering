import numpy as np
import cv2


def create_noisy_image(width=600, height=600):
    # Create an empty image (white background)
    image = np.ones((height, width, 3), dtype=np.uint8) * 255  # white background

    # Define color patches (BGR format for OpenCV)
    blue_patch = (255, 0, 0)
    green_patch = (0, 255, 0)
    red_patch = (0, 0, 255)

    # Draw three large color patches
    image[:, :width // 3] = blue_patch   # Blue patch on the left
    image[:, width // 3: 2 * width // 3] = green_patch  # Green patch in the center
    image[:, 2 * width // 3:] = red_patch  # Red patch on the right

    # noisy points are at the corners these are squares of size 10x10
    square_size = 50
    noisy_color = 0
    #left top corner
    image[:square_size, :square_size] = [noisy_color, noisy_color, noisy_color]
    #left bottom corner
    image[height-square_size:, :square_size] = [noisy_color, noisy_color, noisy_color]
    #right bottom corner
    image[height-square_size:, width-square_size:] = [noisy_color, noisy_color, noisy_color]
    #right top corner
    image[:square_size, width-square_size:] = [noisy_color, noisy_color, noisy_color]
    # middle bottom
    image[height-square_size:, width//2-square_size//2:width//2+square_size//2] = [noisy_color, noisy_color, noisy_color]
    # middle top
    image[:square_size, width//2-square_size//2:width//2+square_size//2] = [noisy_color, noisy_color, noisy_color]

    # Save the image as .bmp
    cv2.imwrite("input_images/noisy_image.bmp", image)

    # Display the image (optional)
    cv2.imshow("Noisy Image with Lines", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Call the function to create the noisy image with lines
create_noisy_image()
