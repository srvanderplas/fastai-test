import cv2
from random import randint
from collections import Counter


parser = VOCBBoxParser(annotations_dir="Modified Data/Modified Annotation", 
                       images_dir="Modified Data/Images")
t, v = parser.parse()

def get_bndbox(bbox):
    xmin = int(bbox.xmin)
    ymin = int(bbox.ymin)
    xmax = int(bbox.xmax)
    ymax = int(bbox.ymax)
    return xmin, ymin, xmax, ymax
  
def get_info(record_obj):
  # '''
  # Returns three lists containing information of bboxes, labels, path and the non-unique labels, respectively.
  # '''
    all_info = list(record_obj.components)
    str_info = [str(i) for i in all_info]
    
    bboxes_index = [('BBox' in i) for i in str_info].index(1)
    labels_index = [('Labels' in i) for i in str_info].index(1)
    path_index = [('Filepath' in i) for i in str_info].index(1)
    
    bboxes = all_info[bboxes_index].bboxes
    labels = all_info[labels_index].labels
    path = all_info[path_index].filepath
    filename = str(path.stem)
    offset = dict(Counter([get_bndbox(i)[0] for i in bboxes]))
    offsets = {k:v for k,v in offset.items() if v > 1}
    
    return bboxes, labels, path, filename, offsets
    
def visualize(record_obj):    
    args = {'color': (0, 255, 0),
            'thickness': 2,
            'fontFace': int(cv2.FONT_HERSHEY_SIMPLEX),
            'fontScale': 0.8,
            'x_offset': 10,
            'y_offset': 20}
        
    bboxes, labels, path, filename, offsets = get_info(record_obj)
    
    counts = sum([[1] * len(offsets)], [])
    counts = dict(zip(offsets.keys(), counts))
    
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    for i in range(len(bboxes)):
        offset = 1
        xmin, ymin, xmax, ymax = get_bndbox(bboxes[i])
        if xmin in counts.keys():
            offset = counts[xmin]
            counts[xmin] += 1
            
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
            args.get('color'),
            args.get('thickness'))
                  
        cv2.putText(img, 
            labels[i],
            (xmin + args.get('x_offset'), (ymin + offset * args.get('y_offset'))),
            args.get('fontFace'),
            args.get('fontScale'),
            args.get('color'),
            args.get('thickness'))
                
    # cv2.imshow(filename, img) # plotting
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)
    return img

# t_paths = list(set(lkns_files))
# records = train_records+valid_records

records = t + v
t_paths = [str(i) for i in paths[:5]]

for t_path in t_paths:
    for i in records:
        bboxes, labels, path, filename, offsets = get_info(i)
        if filename in t_path:
            visualize(i)
            
            
for i in [0, 4, 6, 78, 90, 91, 92, 479, 780, 1213]:
    img = visualize(train_records[i])
    cv2.imwrite('Vis_{}.png'.format(i), img) # save the image

