# <a href="https://zzlab.net/RustNet/"><img src = "docs/img/logo.png" width = 40> RustNet </a>   
A neural network to detect wheat stripe rust with images from drones, smartphones and videos.

# To training a model

usage:

python tf_resnet_cl.py [-h] [--gpu GPU] [--data DATA] [--out OUT]

Define the data folder, gpu and out name

optional arguments:
```

  --help, -h            show this help message and exit
   
  --gpu GPU, -g GPU     number for gpu:1, 2, 3
  
  --data DATA, -d DATA  the data folder
  
  --out OUT, -o OUT     prefix for the output files (parameters file and training accuracy png file)
```

The parameters can be download from <a href="https://zzlab.net/RustNet/"> here </a>


# Training and labeling with Rooster
The RustNet can be used together with a label tool, <a href="https://github.com/12HuYang/Rooster"> Rooster </a>.

# launch a web app in your own computer to detect the stripe rust

```

streamlit run app.py

```

