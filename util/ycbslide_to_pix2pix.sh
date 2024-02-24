#!/bin/bash

# Define source and target directories
src_dir="/proj/vondrick/shared/touch/YCB_Slide/sim"
target_dir="/proj/vondrick/shared/touch/touch2depth"

# Create target subdirectories for touch and depth images if they don't already exist
mkdir -p "$target_dir/touch"
mkdir -p "$target_dir/depth"

# A list of directories under YCB_Slide/sim to iterate over
objects="004_sugar_box 005_tomato_soup_can 006_mustard_bottle 021_bleach_cleanser 025_mug 035_power_drill 037_scissors 042_adjustable_wrench 048_hammer 055_baseball"

# Iterate over each object directory
for object in $objects; do
    # Extract object name without the leading numbers for naming
    object_name=$(echo $object | sed 's/^[0-9]*_//')
    
    # Iterate over the subdirectories (00, 01, 02, 03, 04)
    for subdir in {00..04}; do
        # Define source paths for touch and depth images
        touch_src="$src_dir/$object/$subdir/tactile_images"
        depth_src="$src_dir/$object/$subdir/gt_heightmaps"
        
        # Check if the source directories exist before proceeding
        if [[ -d "$touch_src" && -d "$depth_src" ]]; then
            # Copy and rename touch images
            for touch_file in "$touch_src"/*.jpg; do
                touch_filename=$(basename "$touch_file" .jpg) # Remove the .jpg extension for renaming
                cp "$touch_file" "$target_dir/touch/${object_name}_${subdir}_${touch_filename}_touch.jpg"
            done
            
            # Copy and rename depth images
            for depth_file in "$depth_src"/*.jpg; do
                depth_filename=$(basename "$depth_file" .jpg) # Remove the .jpg extension for renaming
                cp "$depth_file" "$target_dir/depth/${object_name}_${subdir}_${depth_filename}_depth.jpg"
            done
        else
            echo "Warning: Directory $touch_src or $depth_src does not exist. Skipping."
        fi
    done
done

echo "Copy and rename process completed."

