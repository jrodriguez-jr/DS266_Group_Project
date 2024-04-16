README


Note: trying with two  head two layers and drop rate 0.6 Running 12 epochs with 90% 10% split in train/val and dff = 256
These embeddings were loaded from FastText Twitter pre-trained embeddings.
These  embeddings are associated with the model checkpoints saved during training pertaining to the model with
base name 'cl_enh_more_dat_bl_tx_w_tw_embed_2h_2l_20ep_drophigh_low_dff', these word embeddings
can be used for a pre- and post- training analysis of the training effects to each word embedding.
The initial state of the  word embeddings before training are saved in the './Final_Work/enh_clean_more_bl_tx_2_h_2l_twitter_emb_highdrop_low_dff/' directory.
The file 'initial_embeddings.csv' contains the embeddings in CSV format.

The hyperparameters for this model ara saved on this directory in hyperparameters.csv file
The results of this training run will be saved under CSV_Results-cl_enh_more_dat_bl_tx_w_tw_embed_2h_2l_20ep_drophigh_low_dff'.csv file.
The learning schedule plot, training loss plot and training model summary will also saved in this directory.
