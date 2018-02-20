# NN-with-all-Variants-of-GD
The code will work for the varinat so gd(adam,nag,momentum,gd).
Also it supports the batch gd.

To run this code just run the train.py with the parameters given in the run.sh file. In the parameters you have to pass certain things such as lr, momentum, input files, etc.

You can see the supported.txt file to check which all features this code support.
This code is made in the vectorised form rather than loop.
But you can find the loop format also available in the optimizerLoop, and all Loop files.

Further you can also make plot of the error during training and validation for each epoch for 
different factors. Factors like on the basis of variants, batch_size, output_loss(ce,sq), activation function(sigmoid,tanh).
You can observe the plots in the dl_assignment.pdf file.
All the ploting code is lying in the plots folder. And to run that you have to go in different plot*.py file to have a specific 
plot.
