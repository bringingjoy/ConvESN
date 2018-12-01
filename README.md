# ConvESN: Convolutional Echo State Network for human action recognition
Implementation in Keras is based on the ConvESN network described in <a href="https://www.ijcai.org/proceedings/2017/0342.pdf"><i>WALKING WALKing walking: Action Recognition from Action Echoes</i></a>[1]. <br>

The <a href="http://users.eecs.northwestern.edu/~jwa368/my_data.html"><b>MSRDailyActivity3D dataset</b></a> was used to evaluate the model. This dataset contains videos of 10 subjects, each performing 16 types of activities: drink, eat, read book, call cellphone,write on a paper, use laptop, use vacuum cleaner, cheer up, sit still, toss paper, play game, lay down on sofa, walk, play guitar, stand up, sit down. Each subject performs each activity in two different poses: “sitting on sofa” and “standing”. It contains 320 total activity sequences.[2]<br>


The ConvESN-MSMC (multi-step & multi-channel) model was evaluated using 5 holdout sets of MSRDailyActivity. <br>
The average classification results for each activity across the 5 holdout sets is 66.25%.


<table style="width:100%">
  <tr>
    <th>Activity</th>
    <th>Accuracy (% correct)</th> 
  </tr>
  <tr>
    <td>drink</td>
    <td>80.0%</td>
  </tr>
  <tr>
    <td>eat</td>
    <td>80.0%</td>
  </tr>
  <tr>
    <td>read book</td>
    <td>60.0%</td>
  </tr>
  <tr>
    <td>call cellphone</td>
    <td>40.0%</td>
  </tr>
  <tr>
    <td>write on a paper</td>
    <td>50.0%</td>
  </tr>
  <tr>
    <td>use laptop</td>
    <td>50.0%</td>
  </tr>
  <tr>
    <td>use vacuum cleaner</td>
    <td>58.3%</td>
  </tr>
  <tr>
    <td>cheer up</td>
    <td>100.0%</td>
  </tr>
  <tr>
    <td>sit still</td>
    <td>58.30%</td>
  </tr>
  <tr>
    <td>toss paper</td>
    <td>58.3%</td>
  </tr>
  <tr>
    <td>play game</td>
    <td>58.3%</td>
  </tr>
  <tr>
    <td>lie down on sofa</td>
    <td>58.30%</td>
  </tr>
  <tr>
    <td>walk</td>
    <td>80.0%</td>
  </tr>
  <tr>
    <td>play guitar</td>
    <td>50.0%</td>
  </tr>
  <tr>
    <td>stand up</td>
    <td>90.0%</td>
  </tr>
  <tr>
    <td>sit down</td>
    <td>80.0%</td>
  </tr>
</table>

The following hyperparameters were used: <br>
Size of the reservoir: <br>
> n_res = n_in * 8 = 96 <br>

Input scaling (used in initialization of input weights to the reservoir): 
> IS = 0.1 <br>

Spectral radius of the reservoir weight matrix: 
> SR = 0.99 <br>

Level of connection sparsity in the reservoir
> sparsity = 0.1 <br>

Leaky rate 
> leakyrate = 0.9


#### Discussion of hyperparameters:
<b>Input scaling:</b> <br>
"Codetermines the degree of nonlinearity of the reservoir dynamics. In one extreme, with very small effective input amplitudes the reservoir behaves almost like a linear medium, while in the other extreme, very large input amplitudes drive the neurons to the saturation of the sigmoid and a binary switching dynamics results."[3] <br>

<b>Spectral radius:</b> <br>
"Codetermines (i) the effective time constant of the echo state network (larger spectral radius implies slower decay of impulse response) and (ii) the amount of nonlinear interaction of input components through time (larger spectral radius implies longer-range interactions)."[3] <br>

<b>Sparsity:</b> <br>
"Sparse connectivity → decomposition of reservoir dynamics into loosely coupled subsystems → large variation among the reservoir signals (desirable). However, contrary to this intuition, many authors have reported that fully connected reservoirs work as well as sparsely connected ones. Considering that sparsely but randomly connected networks have small-world properties, it appears plausible that a sparse random wiring does not lead to a dynamical decoupling, so the original intuitions are misguided. A more practically important aspect of a sparse connectivity is that it engenders linear scaling of computational complexity. If reservoirs are set up such that each neuron on average connects to a fixed number K of other neurons, regardless of network size N , the computational cost of running the trained networks grows only linearly with N."[3]

### References:
[1] Ma, Qianli, Shen, Lifeng, Chen, Enhuan, Tian, Shuai, Wang, Jiabing, and Cottrell, Garrison W. (2017) WALKING WALKing walking: Action recognition from action echoes. In the Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence (IJCAI-17), pp. 2627-2633. <br>
[2] Mining Actionlet Ensemble for Action Recognition with Depth Cameras, Jiang Wang, Zicheng Liu, Ying Wu, Junsong Yuan, IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2012), Providence, Rhode Island, June 16-21, 2012. <br>
[2] Jaeger, Herbert. “Echo State Network.” Scholarpedia, www.scholarpedia.org/article/Echo_state_network.

