from shiny import *
# import FasterRCNN.py # Look into this
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import cv2
import shiny_vis

           
app_ui = ui.page_fluid(
    ui.input_slider("n_box", "Number of Boxes", value=1, min=0, max=20),
    ui.input_checkbox("randomness", "Random generate"),
    ui.input_slider("idx", "Number between 0 and 897", value=23, min=0, max=897),
    
    ui.output_plot('origin'),
    ui.output_text("input_index")
)

def server(input, output, session):
    @output
    @render.text
    def classes():
        return f"The value of randomess is {input.randomness()}"
    
    @output
    @render.plot
    def origin():
        if input.randomness() == True:
            idx = str(random.randint(0, 897))
        if input.randomness() == False:
            idx = input.idx()
        
        origin_name = 'Modified Data/Valid_pred/origin' + str(idx) + '.pt'
        batch = torch.load(origin_name)

        tensor_img = batch['images'][0]
        img = tensor_img.permute(1,2,0).cpu().numpy()
        img_copy = img.copy()
        
        classes = ['logo', 'polygon', 'chevron', 'circle', 'text', 'quad', 'other', 
           'triangle', 'exclude', 'star', 'bowtie', 'line', 'ribbon']
        labels = [classes[i] for i in batch['targets'][0]['labels']]

        tensor_boxes = batch['targets'][0]['boxes']
        np_boxes = tensor_boxes.cpu().detach().numpy().astype(np.int32)
        l_bboxes = np_boxes.tolist()
        bboxes = np.column_stack((np_boxes, labels))
        [l_bboxes[i].append(labels[i]) for i in range(len(labels))]
        
        label_color=(255,0,255)
        corner_color = (0,255,255)
        box_color=(220, 0, 0)
        length=20
        
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        draw_bboxes_and_label(l_bboxes, img_copy)
        ax.set_axis_off()
        ax.imshow(img_copy)
        # plt.show()
        plt.close()
        return fig
        
    def draw_bboxes_and_label(l_bboxes, img_copy,label_color=(255,0,255), corner_color = (0,255,255), box_color=(220, 0, 0), length=20):
        for bbox in l_bboxes:
            label = bbox[-1]
            label_size = cv2.getTextSize(label + '0' , cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0] # cv2.getTextSize('Test', font, fontScale, thickness) for width and height
            cv2.rectangle(img_copy,
                              (bbox[0], bbox[1]),
                              (bbox[2], bbox[3]),
                              box_color, 3)
            if bbox[1] - label_size[1] - 3 < 0:
                cv2.rectangle(img_copy,
                              (bbox[0], bbox[1] + 2),
                              (bbox[0] + label_size[0], bbox[1] + label_size[1] + 3),
                              color=label_color,
                              thickness=-1
                              )
                cv2.putText(img_copy, label,
                            (bbox[0], bbox[1] + label_size + 3),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 0),
                            thickness=2
                            )
            else:
                cv2.rectangle(img_copy,
                              (bbox[0] , bbox[1] - label_size[1] - 3),
                              (bbox[0] + label_size[0], bbox[1] - 3),
                              color=label_color,
                              thickness=-1 
                              )
                cv2.putText(img_copy, label,
                            (bbox[0], bbox[1] - 3),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 0),
                            thickness=2
                            )
            
        cv2.line(img_copy, (bbox[0], bbox[1]), (bbox[0] + length, bbox[1]), corner_color, thickness=3)
        cv2.line(img_copy, (bbox[0], bbox[1]), (bbox[0], bbox[1] + length), corner_color, thickness=3)
        # Top Right
        cv2.line(img_copy, (bbox[2], bbox[1]), (bbox[2] - length, bbox[1]), corner_color, thickness=3)
        cv2.line(img_copy, (bbox[2], bbox[1]), (bbox[2], bbox[1] + length), corner_color, thickness=3)
        # Bottom Left
        cv2.line(img_copy, (bbox[0], bbox[3]), (bbox[0] + length, bbox[3]), corner_color, thickness=3)
        cv2.line(img_copy, (bbox[0], bbox[3]), (bbox[0], bbox[3] - length), corner_color, thickness=3)
        # Bottom Right
        cv2.line(img_copy, (bbox[2], bbox[3]), (bbox[2] - length, bbox[3]), corner_color, thickness=3)
        cv2.line(img_copy, (bbox[2], bbox[3]), (bbox[2], bbox[3] - length), corner_color, thickness=3)
        
        


app = App(app_ui, server)


