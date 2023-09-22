# 🧭 Learning layouts in Path of Exile with Vision Transformers: A proof of concept

https://github.com/kweimann/poe-learning-layouts/assets/8287691/af505d71-ea64-4840-a9bc-375efcad62a5 

Where's the exit? This question often crosses the minds of both newcomers and seasoned players alike. The key lies in understanding the game's layouts, especially during the campaign when taking a wrong turn can significantly slow you down. Our project aims to solve this challenge through machine learning.

We've developed a proof-of-concept for learning layouts in Path of Exile using Vision Transformers. We trained a Vision Transformer to predict the direction of the exit in the A3 Marketplace, relying solely on a video of the minimap. You can see the model in action in the video above: the red arrow indicates the predicted exit direction, while the green arrow shows the actual direction.

You can download the model at https://huggingface.co/kweimann/poe-learning-layouts.

Furthermore, we provide training and test data at https://huggingface.co/datasets/kweimann/poe-learning-layouts. Find out more about the datasets below.

### Using the model

The model expects a video of the minimap in the top-right corner. If you captured a full-screen video, you need to provide the coordinates to crop the minimap. In the example above, we captured the video at 2560x1440 resolution and provided the script with the coordinates to crop the minimap, which are from a width range of 2190 to 2550 and a height range of 10 to 370.

```shell
python predict_video.py path/to/video.mp4 --model compass.pt --crop 2190 10 2550 370
```

Note that we trained and tested our model using minimap settings shown in the picture below.

![](https://github.com/kweimann/poe-learning-layouts/assets/8287691/072a7ded-ad2d-4956-85af-e6133827d054)

### Testing and visualizing results

You can measure the performance of our model on a test set of 50 minimap videos. The script writes the results to `predictions.pt`, which you can later visualize.

```shell
python test_model.py test_data.npz --model compass.pt
```

We measure the performance by counting how often the difference between predicted and actual direction is smaller than α. Below we show the performance of our model measured on the test set.

| α   | accuracy |
|-----|----------|
| 15° | 72.02%   |
| 30° | 85.52%   |
| 45° | 94.25%   |
| 60° | 97.26%   |
| 90° | 99.42%   |

During the development of our method, we focused on the accuracy at α=45°, which indicates how well the model predicts general directions: forward, left, right, back. At 94.25% accuracy, we believe that the model can learn the task very well. Interestingly, our model can often predict the correct direction shortly after entering the zone.

To visualize the results, for each video of a minimap in the test set, the following script stitches the frames and optionally draws predictions. It makes for a good visual representation of the performance of the model.

```shell
python visualize_data.py test_data.npz --predictions predictions.pt --out viz_data
```
![](https://github.com/kweimann/poe-learning-layouts/assets/8287691/23e12439-4df8-4933-a891-851d68dbf1a9)

You can visualize which frames are important for making predictions. To achieve this, we highlight the top frames to which the model attends in the last layer. Note that this is a rather simplistic approach to explaining the model.

```shell
python visualize_attention.py test_data.npz --model compass.pt --out viz_attn
```

https://github.com/kweimann/poe-learning-layouts/assets/8287691/154f29f7-5c2f-4d4e-9768-512729d4375a

Browsing the attention videos, you may notice that our model often puts importance on the early frames up to the tunnel, which connects the two rooms in the A3 Marketplace, and near the entrance to the A3 Catacombs.

Finally, the code snippet below shows how to visualize a video from the test set.

```python
import skvideo.io
import utils
data = utils.data.load_data('test_data.npz')
file, (_, video, _) = data.popitem()
video = utils.transforms.resize(video, size=360)
skvideo.io.vwrite(
  file, video, inputdict={'-r': '10'},
  outputdict={'-pix_fmt': 'yuv420p'}
)
```

https://github.com/kweimann/poe-learning-layouts/assets/8287691/423ff8cc-5a1f-4c30-8f67-6c3a55974e11

### Training your own model

You can replicate our work by training (and optionally validating) your own model with the script below. The training dataset contains 300 minimap videos.

```shell
python train_model.py training_data.npz 
```

If you wish to create your own dataset, you should record videos focused solely on the minimap in the top-right corner. Each video should begin upon entering the zone and conclude just prior to reaching the exit. This approach enables the algorithm to accurately annotate the exit direction. To ensure the accuracy of the videos, you can draw the minimap using the script below, which stitches the frames together.

```shell
python draw_minimap.py path/to/data/video.mp4
```
![](https://github.com/kweimann/poe-learning-layouts/assets/8287691/9e27d5d8-1eed-48a3-aa06-5968a3745563)

Once you collect enough videos, run the script below to create a dataset.

```shell
python create_dataset.py path/to/data/ --append --num-workers 4
```

### How does the model learn?

In the early stages of our project, we set a crucial requirement: data collection should be simple and not require manual labeling. We accomplish this by first stitching together frames from a minimap video and then for each frame calculating the direction of the exit frame. For this method to work effectively, the video must end precisely at the exit point. We calculate the exit's direction by measuring the angle between the x-axis and the line connecting the center points of a given frame and the exit frame.

Our model takes a video as input and learns to predict the angle at each frame. To prevent overfitting, we deliberately exclude a significant number of frames, which adds complexity to the task. During data collection, we discard frames to ensure that the difference between consecutive frames is approximately one-quarter of the screen. During training, we further mask approximately 80% of the frames and perform backpropagation solely on the unmasked frames.

### Installation

1. Download Python 3.9 or higher and setup a virtual environment.
    ```shell
    python -m venv venv
    source venv/bin/activate
    python -m pip install --upgrade pip setuptools
    pip install -r requirements.txt
    ```
2. Download FFmpeg (https://ffmpeg.org/) and put the executables in the root directory of the project.

### Limitations

We showed that our model can learn the layout of A3 Marketplace very well, and we believe it is capable of learning the layouts of other zones. However, there are still some limitations.

1. There is no mechanism to represent multiple directions within a single layout, e.g., pick up a quest item and find the exit. Currently, this could be achieved by segmenting the video, enabling the algorithm to accurately annotate various directions.
2. Our model has no concept of walls or otherwise obstructed paths. Consequently, in more complex scenarios, such as labyrinthine layouts, the predictions may not prove particularly useful.

### Want to contribute?

Naturally, the next stage in the evolution of this project is to extend it to more zones. If that sounds good, or you have another idea in mind, open up an issue or contact us directly.
