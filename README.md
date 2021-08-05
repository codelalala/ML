# ML snippets
This is where I save some useful ML code and snippets from different sources.
### 1. Huggingface nlp+transformers+imdb->sentiment analysis
This is based on:
> https://www.youtube.com/watch?v=G3pOvrKkFuk&list=PL1v8zpldgH3pQwRz1FORZdChMaNZaR3pu&index=17

And the address:
> https://github.com/yk/huggingface-nlp-demo/blob/master/demo.py

This is a useful demo for huggingface tools, including:
- nlp for datasets
- transformers for model zoo

Current Performance ~90% validation acc

### 2. DETR Facebook object detection

This is based on:
> https://www.youtube.com/watch?v=LfUsGv-ESbc

And:
> https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_demo.ipynb#scrollTo=bawoKsBC7oBh

This official DETR github:
> https://github.com/facebookresearch/detr

What is interesting:
- Cnn+transformers
- Get around those computational extensive bounding boxes per pixel calculation and use transformers to propose bounding boxes

Limitation:
- Need to have a score threshold

### 3. DETR Panoptic segmentation detection

This is based on:
> https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/DETR_panoptic.ipynb

Need to see that is the difference between this and Umap, but it looks fantastic!

-Detectron2 is broken on colab even with its original colab code, I'll try to use numpy and matplotlib to fix this
