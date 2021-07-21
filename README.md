# A PyTorch reimplementation of Local Implicit Grid Representations for 3D Scenes

This project is a PyTorch implementation of [LIG](http://maxjiang.ml/proj/lig).
The codes is based on authors' Tensorflow implementation [here](https://github.com/tensorflow/graphics/tree/master/tensorflow_graphics/projects/local_implicit_grid),

## Prepare Environment
```
  pip install -r requirements.txt
  python setup.py build_ext --inplace
```

## Perform Reconstruction
```
  python main/run_lig.py --input_ply demo_data/living_room_33_1000_per_m2.ply --output_ply test.ply
```