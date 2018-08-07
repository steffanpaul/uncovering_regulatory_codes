# Applying First Order Mutagenesis to Uncovering Regulatory Codes

#### *BE ADVISED: I just reworked the structure of these files, so the paths may not match up to where the modules are. Remember to change the path when using a new file.*

Motivation: see if First Order Mutagenesis (FoM) can produce better visuals than guided backprop.

**Progress as of 08/07/2018:**
* Motif recovery performance
  * Created files that perform FoM on every single sequence for every single model. Outputs large file that is accessible on google drive.
  Use FoM model recovery performance.ipynb to extract data. This is unprocessed for TPS and FPS calculation

* Layer saliency for LocalNet and DistNet
  * Created a file for mining sequences which shows the FoM saliency for multiple sequences along with the model. Sequences were selected by eye that had
  the best saliency visual for both DistNet and LocalNet. Pay attention to when sequences are being indexed by their order in Test or their order by plot_index (sequences ordered by DistNet score).
  * So far, I've identified 3 sequences that have *(in my opinion)* quite good visuals for both sequence and have viable saliency at different layers.
  * Created a separate folder for each sequence. Within each folder is a notebook that mines through the layers to identify the best ones.
    * The first file just shows the heatmaps of every layer ordered by their relative score for a given sequence. This is a faster visual from which better candidate layers can be selected by eye.
    * The candidate sequences can then be visualized as logos and the best four selected.
  * The files Plot_FoM_layers can be used to identify the best sequence to use for the paper.
    * Play around with the different normfactors for visualizing each layer's saliency. It could clean up the saliencies. Or it might not. Who knows?
  * This has been quite tedious and after this, I don't think FoM is a viable alternative to guided backprop.
  
* Expressivity
  * This has not been started. However, the last layers for any sequence have tended to all give basically the same saliency. So using code from the Mining files in the Layers folder, it could be 
  quite easy to loop over sequences and visualize FoM of just a few last layers for both DistNet and LocalNet.
  
  
