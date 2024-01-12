# Cartoonized Face Overlay (Viola Jones)

## Reference of Data

### CBCL FACE DATABASE #1:
http://cbcl.mit.edu/software-datasets/FaceData2.html

### Non face dataset 1 from repo: (**augmented**)
https://github.com/aparande/FaceDetection


### LFWcrop Face Dataset: 
https://conradsanderson.id.au/lfwcrop/

### BG-20k Dataset v1.0
https://drive.google.com/drive/folders/1ZBaMJxZtUNHIuGj8D8v3B9Adn8dbHwSS

### Scene Understanding Datasets (**augmented**)
http://dags.stanford.edu/projects/scenedataset.html

### Numbers
Train faces:  (2429, 6066) \
Train faces 1:  (13233, 6066) \
Train non faces:  (9096, 6066) \
Train non faces 1:  (3008, 6066) \
Train non faces 2:  (2860, 6066) \
Faces:  15662 \
Non Faces:  14964 \
Features: 6066 \
Selected Features after training using ANOVA F1-score with 10%:  607 \
X_train shape:  (30626, 607) \
X_val shape:  (24045, 607) \
X_test shape:  (698, 607)