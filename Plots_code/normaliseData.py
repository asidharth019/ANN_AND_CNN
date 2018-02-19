import numpy as np


def normaliseData(input_data, type_norm, type_fea_img, means, stds):
    # this function is used to normalize the input data

        if type_fea_img == "image":
            if type_norm == 1:
                norm_input = input_data - means
            elif type_norm == 2:
                norm_input = (input_data - means) / stds
            elif type_norm == 3:
                max_value = np.max(input_data, axis=1).reshape(input_data.shape[0], 1)
                min_value = np.min(input_data, axis=1).reshape(input_data.shape[0], 1)
                norm_input = (input_data - min_value)/(max_value-min_value)

        elif type_fea_img == "feature":
            if type_norm == 1:
                norm_input = input_data - means
            elif type_norm == 2:
                norm_input = (input_data - means) / stds
            elif type_norm == 3:
                max_value = np.max(input_data, axis=0).reshape(1, input_data.shape[1])
                min_value = np.min(input_data, axis=0).reshape(1, input_data.shape[1])
                norm_input = (input_data - min_value) / (max_value - min_value)

        elif type_fea_img == "255":
            norm_input = input_data/255

        return norm_input