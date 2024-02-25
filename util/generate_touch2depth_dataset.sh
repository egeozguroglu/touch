#!/bin/bash

# Define source and target directories
src_dir="/proj/vondrick/shared/touch/YCB_Slide/sim"
target_dir="/proj/vondrick/shared/touch/touch2depth"

# Number of test pairs per object
num_test_pairs=4

# Create target subdirectories for touch and depth images in both train and test sets
mkdir -p "$target_dir/train/touch"
mkdir -p "$target_dir/train/depth"
mkdir -p "$target_dir/test/touch"
mkdir -p "$target_dir/test/depth"

# A list of directories under YCB_Slide/sim to iterate over
objects="004_sugar_box 005_tomato_soup_can 006_mustard_bottle 021_bleach_cleanser 025_mug 035_power_drill 037_scissors 042_adjustable_wrench 048_hammer 055_baseball"
objects_array=($objects)
total_objects=${#objects_array[@]}

function progress_bar {
    let _progress=(${1}*100/${2}*100)/100
    let _done=(${_progress}*4)/10
    let _left=40-$_done
    _fill=$(printf "%${_done}s")
    _empty=$(printf "%${_left}s")
    printf "\rProgress : [${_fill// /#}${_empty// /-}] ${_progress}%%"
}

for i in "${!objects_array[@]}"; do
    object=${objects_array[$i]}
    # Extract object name without the leading numbers for naming
    object_name=$(echo $object | sed 's/^[0-9]*_//')
    
    # Update and display progress
    progress_bar $((i+1)) $total_objects
    
    # Iterate over the subdirectories (00, 01, 02, 03, 04)
    for subdir in {00..04}; do
        touch_src="$src_dir/$object/$subdir/tactile_images"
        depth_src="$src_dir/$object/$subdir/gt_heightmaps"
        
        if [[ -d "$touch_src" && -d "$depth_src" ]]; then
            touch_files=($(ls $touch_src/*.jpg))
            depth_files=($(ls $depth_src/*.jpg))
            
            # Shuffle the array indices
            indices=($(shuf -i 0-$((${#touch_files[@]} - 1))))
            
            # Preparing to select test and train indices
            total_files=${#touch_files[@]}
            test_indices=("${indices[@]:0:$num_test_pairs}")
            train_indices=("${indices[@]:$num_test_pairs}")

            # Copy files to their respective train/test directories
            for idx in "${!touch_files[@]}"; do
                base_filename=$(basename "${touch_files[$idx]}" .jpg)
                
                if [[ " ${test_indices[@]} " =~ " ${idx} " ]]; then
                    # Test set
                    cp "${touch_files[$idx]}" "$target_dir/test/touch/${object_name}_${subdir}_${base_filename}_touch.jpg"
                    cp "${depth_files[$idx]}" "$target_dir/test/depth/${object_name}_${subdir}_${base_filename}_depth.jpg"
                else
                    # Train set
                    cp "${touch_files[$idx]}" "$target_dir/train/touch/${object_name}_${subdir}_${base_filename}_touch.jpg"
                    cp "${depth_files[$idx]}" "$target_dir/train/depth/${object_name}_${subdir}_${base_filename}_depth.jpg"
                fi
            done
        else
            echo "Warning: Directory $touch_src or $depth_src does not exist. Skipping."
        fi
    done
done

progress_bar $total_objects $total_objects

echo "touch2depth dataset created!"
