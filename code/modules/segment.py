from unipath import Path, DIRS_NO_LINKS
import uuid
import cv2

def segment(img, annotation, work_dir):
    sides = ['left', 'top', 'right', 'bottom']
    # Parse the given sentences
    for sentence in annotation:
        for word in sentence:
            for char in word:
                c = char.get('text')
                cdir = Path(work_dir, c)
                cdir.mkdir()
                f = Path(cdir, str(uuid.uuid1()) + '.ppm')

                rect = {side: int(char.get(side)) for side in sides}
                # Correct for swapped coordinates
                if rect['top'] > rect['bottom']:
                    rect['top'], rect['bottom'] = rect['bottom'], rect['top']
                if rect['left'] > rect['right']:
                    rect['left'], rect['right'] = rect['right'], rect['left']
                cropped_im = img[rect['top']:rect['bottom'], rect['left']:rect['right']]

                # Remove rows from the top if they're white
                while cropped_im.shape[0] > 0 and min(cropped_im[0,:]) == 255:
                    cropped_im = cropped_im[1:,:]
                # Remove from the bottom
                while cropped_im.shape[0] > 0 and min(cropped_im[-1,:]) == 255:
                    cropped_im = cropped_im[:-1,:]
                # Remove from the left
                while cropped_im.shape[1] > 0 and min(cropped_im[:,0]) == 255:
                    cropped_im = cropped_im[:,1:]
                # Remove from the right
                while cropped_im.shape[1] > 0 and min(cropped_im[:,-1]) == 255:
                    cropped_im = cropped_im[:,:-1]
                cv2.imwrite(f, cropped_im)
