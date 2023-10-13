# perform morphological dilation and erosion operations respectively 
# to ensure static_masks sufficeintly cover the regions of moving objects,
# and the regions from dynamic_masks are within the true regions of moving objects.

import os, sys, cv2

def generate_mask(input_mask, ksize=(3, 3)):
    '''
    return static_mask, dynamic_mask
    '''
    static_mask = cv2.erode(input_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize), iterations=1)
    dynamic_mask = cv2.dilate(input_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize), iterations=1)
    
    return static_mask, dynamic_mask
    
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python generate_mask.py input_dir output_dir')
        sys.exit()
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    dynamic_masks_dir = os.path.join(output_dir, 'dynamic_masks')
    static_masks_dir = os.path.join(output_dir, 'static_masks')

    os.makedirs(dynamic_masks_dir, exist_ok=True)
    os.makedirs(static_masks_dir, exist_ok=True)
    for i, file in enumerate(sorted(os.listdir(input_dir))):
        if file.endswith('.png'):
                
            input_mask_path = os.path.join(input_dir, file)
            input_mask = cv2.imread(input_mask_path, 0)
            input_mask = cv2.resize(input_mask, (512, 288))     # do resize
            if not file.endswith('.jpg.png'):
                # invert white and black
                input_mask = 255 - input_mask

            static_mask, dynamic_mask = generate_mask(input_mask)
            
            static_mask_path = os.path.join(static_masks_dir, f'{i}.png')
            dynamic_mask_path = os.path.join(dynamic_masks_dir, f'{i}.png')
            
            cv2.imwrite(static_mask_path, static_mask)
            cv2.imwrite(dynamic_mask_path, dynamic_mask)
            
            print(f'generate frame {i}: {input_mask_path} done.')
            
            