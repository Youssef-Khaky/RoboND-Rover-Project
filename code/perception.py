import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only


def color_thresh_nav(img, rgb_thresh=(160, 160, 160)):#lower thresh
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    
    mask = np.zeros_like(img[:,:,0])
    #mask[int(mask.shape[0]/2):mask.shape[0]-1,int(mask.shape[1]/2):mask.shape[1]-1] = 1
    mask[int(mask.shape[0]/2):mask.shape[0]-1,:] = 1
    
        	


    
    
    
    
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    navigable = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[navigable] = 1
    
    color_select = cv2.bitwise_and(color_select,color_select,mask = mask)

    # Return the binary image
    return color_select

def color_thresh_rock(img, rgb_lower=(60, 60, 0), rgb_upper=(255, 255 ,30)):
    color_select = np.zeros_like(img[:,:,0])
    rock = (img[:,:,0] > rgb_lower[0]) \
                & (img[:,:,1] > rgb_lower[1]) \
                & (img[:,:,2] > rgb_lower[2]) \
                & (img[:,:,0] < rgb_upper[0]) \
                & (img[:,:,1] < rgb_upper[1]) \
                & (img[:,:,2] < rgb_upper[2])
    color_select[rock] = 1
    return color_select


def color_thresh_obs(img, rgb_thresh=(160, 160, 160)):#lower thresh
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    navigable = (img[:,:,0] < rgb_thresh[0]) \
                & (img[:,:,1] < rgb_thresh[1]) \
                & (img[:,:,2] < rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[navigable] = 1

    # Return the binary image
    return color_select

# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the
    # center bottom of the image.
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle)
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))

    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result
    return xpix_rotated, ypix_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale):
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image

    return warped


# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    image = Rover.img

    xpos = Rover.pos[0]
    ypos = Rover.pos[1]
    yaw = Rover.yaw

    dst_size = 5
    bottom_offset = 6
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],
                      [image.shape[1]/2 + dst_size, image.shape[0] - bottom_offset],
                      [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                      [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                      ])

    
    # 2) Apply perspective transform
    warped = perspect_transform(image, source, destination)
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    navigable = color_thresh_nav(warped) #bin image of eagle eye
    
    
    rock = color_thresh_rock(warped)
    obs = color_thresh_obs(warped)

    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
    # Rover.vision_image[:,:,0] = obs*255 #obstacle color-thresholded binary image
    # Rover.vision_image[:,:,1] = rock*255 #rock_sample color-thresholded binary image
    Rover.vision_image[:,:,2] = navigable*255 #navigable terrain color-thresholded binary image

    # 5) Convert map image pixel values to rover-centric coords
    nav_rover_x, nav_rover_y= rover_coords(navigable)
    rock_rover_x, rock_rover_y = rover_coords(rock)
    obs_rover_x, obs_rover_y = rover_coords(obs)

    # 6) Convert rover-centric pixel values to world coordinates
    nav_world_x, nav_world_y = pix_to_world(nav_rover_x, nav_rover_y, xpos, ypos, yaw, 200, 10)
    rock_world_x, rock_world_y = pix_to_world(rock_rover_x, rock_rover_y, xpos, ypos, yaw, 200, 10)
    obs_world_x, obs_world_y = pix_to_world(obs_rover_x, obs_rover_y, xpos, ypos, yaw, 200, 10)

    # Rover.worldmap[obs_world_y, obs_world_x, 0] += 1
    # Rover.worldmap[rock_world_y, rock_world_x, :] = 255
    Rover.worldmap[nav_world_y, nav_world_x, 2] += 10


    dist, angle = to_polar_coords(nav_rover_x, nav_rover_y)


    # 7) Update Rover worldmap (to be displayed on right side of screen)

    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
    Rover.nav_dists = dist
    Rover.nav_angles = angle
#    mean_dir = np.mean(angle)
    
    
#    threshed = color_thresh_nav(warped)
    
    # Calculate pixel values in rover-centric coords and distance/angle to all pixels
#    xpix, ypix = rover_coords(threshed)
#    dist, angles = to_polar_coords(xpix, ypix)
#    mean_dir = np.mean(angles)
    # Do some plotting
#    fig = plt.figure(figsize=(12,12))
#    plt.subplot(221)
#    plt.imshow(image)
#    plt.subplot(222)
#    plt.imshow(warped)
#    plt.subplot(223)
#    plt.imshow(navigable, cmap='gray')
#    plt.subplot(224)
#    plt.plot(nav_rover_x, nav_rover_y, '.')
#    plt.ylim(-160, 160)
#    plt.xlim(0, 160)
#    arrow_length = 100
#    x_arrow = arrow_length * np.cos(mean_dir)
#    y_arrow = arrow_length * np.sin(mean_dir)
#    plt.arrow(0, 0, x_arrow, y_arrow, color='red', zorder=2, head_width=10, width=2)
#    plt.ion()
#    
#    plt.show()
#    plt.pause(0.0001)
#    plt.close()
#
#




    return Rover
