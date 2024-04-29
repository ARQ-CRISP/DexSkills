# DexSkill: Skill Segmentation Using Haptic Data for Learning Autonomous Long-Horizon Robotic Manipulation Tasks


<details>
<summary><strong><em> PAPER WEBSITE:</em></strong></summary>

<div style="background-color: #f2f2f2; padding: 10px;">
https://arq-crisp.github.io/DexSkills/
</div>
</details>

<details>
<summary><strong><em>ABSTRACT:</em></strong></summary>

<div style="background-color: #f2f2f2; padding: 10px; text-align: justify;">
Effective execution of long-horizon tasks with dexterous robotic hands remains a significant challenge in real-world problems. While learning from human demonstrations have shown encouraging results, they require extensive data collection for training. Hence, decomposing long-horizon tasks into reusable primitive skills is a more efficient approach. To achieve so, we developed DexSkills, a novel supervised learning framework that addresses long-horizon dexterous manipulation tasks using primitive skills. DexSkills is trained to recognize and replicate a select set of skills using human demonstration data, which can then segment a demonstrated long-horizon dexterous manipulation task into a sequence of primitive skills to achieve one-shot execution by the robot directly. Significantly, DexSkills operates solely on proprioceptive and tactile data, i.e., haptic data. Our real-world robotic experiments show that DexSkills can accurately segment skills, thereby enabling autonomous robot execution of a diverse range of tasks.
</div>



</details>






<details>
<summary><strong><em> DEMONSTRATION: </em></strong></summary>

<div style="background-color: #f2f2f2; padding: 10px;">
The dataset includes data of 20 haptic skils (10 repetitions each):

| Skill Number | Skill Name          | Skill Number | Skill Name          | Skill Number | Skill Name          | Skill Number | Skill Name          | Skill Number | Skill Name          |
|--------------|---------------------|--------------|---------------------|--------------|---------------------|--------------|---------------------|--------------|---------------------|
| 1            | Reach               | 2            | Setup Position      | 3            | PreTouch            | 4            | Touch               | 5            | Flip                |
| 6            | Wipe Forth          | 7            | Wipe Back           | 8            | PreGrasp            | 9            | Grasp               | 10           | Lift with Grasp     |
| 11           | Transport Forward   | 12           | Place               | 13           | PreRotate           | 14           | Rotate              | 15           | Shake Up            |
| 16           | Shake Down          | 17           | Twist               | 18           | Vertical Place      | 19           | Pour                | 20           | Release             |

And 20 Long Tasks executed as a sequence of skills.

| Task | I | II | III | IV | V | VI | VII | VIII | IX | X |
|------|---|----|-----|----|---|----|------|-------|----|---|
| A (s)| 1 | 5  | 3   | 4  | 7 | 6  | 8    | 9     | 10 | 20|
| B (t)| 4 | 7  | 8   | 9  | 10| 11 | 12   | 2     |    |   |
| C (b)| 13| 14 | 10  | 15 | 16| 17 | 18   |       |    |   |
| D (s)| 6 | 7  | 6   | 7  | 6 | 7  |      |       |    |   |
| E (b)| 5 | 8  | 9   | 10 | 15| 19 |      |       |    |   |
| F (b)| 8 | 9  | 10  | 17 |   |    |      |       |    |   |
| G (b)| 1 | 5  | 8   | 9  |   |    |      |       |    |   |
| H (t)| 15| 16 | 15  | 12 |   |    |      |       |    |   |
| I (s)| 16| 15 | 16  | 20 |   |    |      |       |    |   |
| J (b)| 9 | 10 | 17  | 20 |   |    |      |       |    |   |
| K (t)| 4 | 8  | 9   |    |   |    |      |       |    |   |
| L (s)| 13| 14 | 17  |    |   |    |      |       |    |   |
| M (s)| 9 | 20 | 2   |    |   |    |      |       |    |   |
| N (s)| 17| 10 | 16  |    |   |    |      |       |    |   |
| O (b)| 10| 17 | 19  |    |   |    |      |       |    |   |
| P (t)| 19| 17 | 18  |    |   |    |      |       |    |   |
| Q (s)| 5 | 8  | 2   |    |   |    |      |       |    |   |
| R (b)| 1 | 13 | 2   |    |   |    |      |       |    |   |
| S (s)| 18| 10 | 20  |    |   |    |      |       |    |   |
| T (b)| 10| 17 | 18  |    |   |    |      |       |    |   |

</div>
</details>

<details>
<summary><strong><em> DATASET:</em></strong></summary>
<div style="background-color: #f2f2f2; padding: 10px; text-align: justify;"> 

The dataset provides the following modalities:

 - Proprioception
 - Tactile Sensing

The dataset files are organised as following:

```
DexSkill_dataset
    └─ dataset / Long-horizon task dataset
         └── data_0.pt
         └── ...
         └── data_i.pt
         │   ├── state_input
         │   ├── state_output
         │   ├── feature_input
         │   ├── feature_output
         │   ├── label
    

```

 There are 60 dataset files for the training, each consisting of a batch size of 256 with data shuffled. The dataset is saved in a dictionary style.  

 - `data['state_input']` contains the raw haptic data, including the end-effector state, filtered tactile information, filtered contact indicators, and the AH joint state.
 -  `data['feature_input']` includes proposed features while excluding the raw haptic data.
 -  `data['state_output']` and `data['features_output']` is the proposed feature at the next timestep, which is used to train the auto-regressive autoencoder.
 -   `data['label']` includes the skill name for the recorded task.

The `.pt` files located within the `/DexSkill_dataset/dataset` directory encompass a comprehensive collection of recorded demonstrations across 20 primitive skills. Additionally, within the `/DexSkill_dataset/LH_dataset` folder, each `.pt` file correspond to a specific long-horizon manipulation task, with no shuffling involved to preserve the time-series sequence of these tasks. 

All trained policies, including those of our framework and comparative works, are inside the `trained_policy` folder. Furthermore, the `json_file` within this dataset provides human-labeled task segmentation for all long-horizon tasks, serving as a ground truth. 

The demo code for load the dataset and train the classifier is in the file `/code/train_classifier.py`

</div>
</details>


<details>
<summary><strong><em>VIDEO AND DEMO:</em></strong></summary>
<div style="background-color: #f2f2f2; padding: 10px;">

https://github.com/ARQ-CRISP/DexSkill/assets/62802841/f60f5a8a-f3ac-4726-8840-50d545b14b38

https://github.com/ARQ-CRISP/DexSkills/assets/62802841/f7fc40b3-19dd-41a2-95a5-e7b9d42a9b8d

</div>
</details>
