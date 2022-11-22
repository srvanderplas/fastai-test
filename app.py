from shiny import *
# import FasterRCNN.py # Look into this
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import cv2

classes = ['logo', 'polygon', 'chevron', 'circle', 'text', 'quad', 'other', 
           'triangle', 'exclude', 'star', 'bowtie', 'line', 'ribbon']
           
app_ui = ui.page_fluid(
    # ui.input_checkbox("logo", "logo"),
    # ui.input_checkbox("polygon", "polygon"),
    # ui.input_checkbox("chevron", "chevron"),
    # ui.input_checkbox("circle", "circle"),
    # ui.input_checkbox("text", "text"),
    # ui.input_checkbox("quad", "quad"),
    # ui.input_checkbox("other", "other"),
    # ui.input_checkbox("triangle", "triangle"),
    # ui.input_checkbox("exclude", "exclude"),
    # ui.input_checkbox("star", "star"),
    # ui.input_checkbox("bowtie", "bowtie"),
    # ui.input_checkbox("line", "line"),
    # ui.input_checkbox("ribbon", "ribbon"),
    ui.input_slider("n_box", "Number of Boxes", value=1, min=0, max=20),
    # ui.input_checkbox("randomness", "Random generate"),
    ui.output_plot('origin'),
)

def server(input, output, session):
    @output
    @render.plot
    def origin():
        # if input.randomness:
        #     x_rand = 100 + 15 * np.random.randn(437)
        #     fig, ax = plt.subplots()
        #     ax.hist(x_rand, int(input.n_box()))
        idx = random.randint(0, 897)
        origin_name = 'Modified Data/Valid_pred/origin' + str(idx) + '.pt'
        # else:
            # origin_name = 'Modified Data/Valid_pred/origin' + str(23) + '.pt'
        batch = torch.load(origin_name)

        model = torch.load('stat_all_model2.pt')
        tensor_img = batch['images'][0]
        img = tensor_img.permute(1,2,0).cpu().numpy()

        tensor_boxes = batch['targets'][0]['boxes']
        np_boxes = tensor_boxes.cpu().detach().numpy().astype(np.int32)

        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        img_copy = img.copy()
        for box in np_boxes:
            cv2.rectangle(img_copy,
                          (box[0], box[1]),
                          (box[2], box[3]),
                          (220, 0, 0), 3)

        ax.set_axis_off()
        ax.imshow(img_copy)
        plt.show()
        plt.close()
        return fig
        

app = App(app_ui, server)


