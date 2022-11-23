from shiny import *
# import FasterRCNN.py # Look into this
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import cv2
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


classes = ['logo', 'polygon', 'chevron', 'circle', 'text', 'quad', 'other', 
           'triangle', 'exclude', 'star', 'bowtie', 'line', 'ribbon']
           
app_ui = ui.page_fluid(
    ui.panel_title('Origin and Prediction'),
    
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.input_slider('n_boxes', 'Number of Boxes', value=1, min=0, max=20),
            ui.input_numeric('idx', 'Number between 0 and 896', value=10),
            ui.output_text("threshold"),
            ui.navset_tab_card(
                ui.nav(
                    'Confusion Matrix',
                    ui.output_plot('confusion_matrix_plot'),
                ),
                ui.nav(
                    'Table',
                    ui.output_table('origin_and_pred_classes_table'),
                ),
                ui.nav(
                    'Score',
                    ui.output_table('scores'),
                ),
            ),
        ),
        
        ui.panel_main(
            ui.output_plot('pred'),
            ui.output_plot('origin'),
        ),
    ),
)

def server(input, output, session):
    @output
    @render.text
    def threshold():
        pred_batch = batches()[1]
        threshold = format(pred_batch[0]['scores'][input.n_boxes()], '.3f')
        return f'The threshold is {threshold}'
    
    @output
    @render.plot
    def confusion_matrix_plot():
        origin_batch, pred_batch = batches()
        origin_labels = [classes[i] for i in origin_batch['targets'][0]['labels']]
        pred_labels = [classes[i] for i in pred_batch[0]['labels']][:len(origin_labels)]
        
        labels = list(set(origin_labels + pred_labels))
        
        sns.set()
        fig, ax = plt.subplots()
        # cf_matrix = pd.crosstab(origin_labels,pred_labels)
        # sns.heatmap(cf_matrix, linewidths=1, annot=True, ax=ax, fmt='g')
        matrix = confusion_matrix(origin_labels, pred_labels, labels = labels)
        sns.heatmap(matrix, annot=True, fmt = 'g', ax=ax, cmap='crest')
        ax.xaxis.set_ticklabels(labels)
        ax.yaxis.set_ticklabels(labels)
        
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predict')
        ax.set_ylabel('True')
        
        return fig
        
    @output
    @render.table
    def origin_and_pred_classes_table():
        df_all = pd.DataFrame(classes)
        df_all['origin_freq'] = 0
        df_all.set_index(0, inplace=True)
        df_all['classes'] = df_all.index
        df_all['pred_freq'] = 0
        df_all.loc['Total'] = [0, 'Total', input.n_boxes()]
        
        origin_batch, pred_batch = batches()
        
        origin_labels = pd.DataFrame([classes[i] for i in origin_batch['targets'][0]['labels']])
        pred_labels = pd.DataFrame([classes[i] for i in pred_batch[0]['labels']])[:input.n_boxes()]
        
        origin_freq = origin_labels.value_counts()

        for i in origin_freq.index:
            df_all.loc[i[0], 'origin_freq']=origin_freq[[i][0]]
        df_all.iloc[-1, 0] = sum(df_all['origin_freq'])
        
        pred_freq = pred_labels.value_counts()
        for i in pred_freq.index:
            df_all.loc[i[0], 'pred_freq']=pred_freq[[i][0]]
        
        return df_all
        
    def batches():
        idx = input.idx()
        
        origin_name = 'Modified Data/Valid_pred/origin' + str(idx) + '.pt'
        pred_name = 'Modified Data/Valid_pred/pred' + str(idx) + '.pt'
        
        origin_batch = torch.load(origin_name)
        pred_batch = torch.load(pred_name)
        
        return [origin_batch, pred_batch]
    
    @output
    @render.table
    def scores():
        origin_batch, pred_batch = batches()
        
        pred_scores = pred_batch[0]['scores'][:input.n_boxes()]
        np_scores = pred_scores.cpu().detach().numpy().astype(np.float64)
        l_scores = np_scores.tolist()
        
        pred_labels = [classes[i] for i in pred_batch[0]['labels']][:input.n_boxes()]
        
        df_all = pd.DataFrame({'Labels' : pred_labels, 
                               'Scores' : l_scores})
        return df_all
    
    @output
    @render.plot
    def pred():
        origin_batch, pred_batch = batches()

        tensor_img = origin_batch['images'][0]
        img = tensor_img.permute(1,2,0).cpu().numpy()
        img_copy = img.copy()
        
        labels = [classes[i] for i in pred_batch[0]['labels'][:input.n_boxes()]]
        
        tensor_boxes = pred_batch[0]['boxes'][:input.n_boxes()]
        np_boxes = tensor_boxes.cpu().detach().numpy().astype(np.int32)
        l_bboxes = np_boxes.tolist()
        bboxes = np.column_stack((np_boxes, labels))
        [l_bboxes[i].append(labels[i]) for i in range(len(labels))]
        
        label_color=(0,255,0)
        corner_color = (0,255,255)
        box_color=(220, 0, 0)
        length=20
        
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        draw_bboxes_and_label(l_bboxes, img_copy, label_color=label_color)
        ax.set_axis_off()
        ax.imshow(img_copy)
        # plt.show()
        plt.close()
        return fig
        
        
    @output
    @render.plot
    def origin():
        origin_batch = batches()[0]

        tensor_img = origin_batch['images'][0]
        img = tensor_img.permute(1,2,0).cpu().numpy()
        img_copy = img.copy()
        
        labels = [classes[i] for i in origin_batch['targets'][0]['labels']]

        tensor_boxes = origin_batch['targets'][0]['boxes']
        np_boxes = tensor_boxes.cpu().detach().numpy().astype(np.int32)
        l_bboxes = np_boxes.tolist()
        bboxes = np.column_stack((np_boxes, labels))
        [l_bboxes[i].append(labels[i]) for i in range(len(labels))]
        
        label_color=(255,0,255)
        corner_color = (0,255,255)
        box_color=(220, 0, 0)
        length=20
        
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        draw_bboxes_and_label(l_bboxes, img_copy,label_color=(100,100,0))
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


