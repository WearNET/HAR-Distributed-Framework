# HAR-Distributed-Framework
HAR Distributed System for the Classification of a Finite Set of Human Activities

<img width="15906" height="9843" alt="electronics-3760176" src="https://github.com/user-attachments/assets/94f1e0dc-2d3a-4656-89ec-749dd85fc1fc" />

## **[Central Node Model](Central_Node_Model)**  
This node is responsible for executing the central neural network that combines all the probabilities from each of the sensors used as input for the MLP and generates the overall classification of the activity performed. It also has the script responsible for communicating with the GUI.

The folder contains the scripts to run the local neural network, the communication script with the GUI developed in Unity, and the .pth files of the neural network model weights along with the scaler used by it.

This node can be executed with the following command:

```bash
python3 central_node_ros_main.py
python3 har_aggregator_and_relay.py
python3 har_start_latch_relay.py
python3 experiment_manager_node.py
python3 results_exporter_node.py
```

You can also create your own ROS package from the last 4 scripts by creating its executable and placing it in a launch file and running it as follows:

```bash
roslaunch name_your_package name_asigned.launch
```

## **[Chest Distributed](Chest_Distributed)**  
This local node handles data acquisition from the **chest-mounted** sensor, including accelerometer and gyroscope signals, executes the corresponding neural network for activity classification, and extracts quaternion data for visualization in the graphical user interface (GUI).

The folder contains the scripts to run the central neural network, the .pth files of the neural network model weights along with the scaler used by it.

This node can be executed with the following command:

```bash
python3 chest_node_ros.py
```


## **[Left Hand Node](Left_Hand_Distributed/)**  
This local node handles data acquisition from the **left-hand-mounted** sensor, including quaternion signals, executes the corresponding neural network for activity classification and send data for visualization in the graphical user interface (GUI).

The folder contains the scripts to run the central neural network, the .pth files of the neural network model weights along with the scaler used by it.

This node can be executed with the following command:

```bash
python3 left_hand_node_ros.py
```

## **[Left Knee Node](Left_Knee_Distributed/)**  
This local node handles data acquisition from the **left-knee-mounted** sensor, including accelerometer and gyroscope signals, executes the corresponding neural network for activity classification, and extracts quaternion data for visualization in the graphical user interface (GUI).

The folder contains the scripts to run the central neural network, the .pth files of the neural network model weights along with the scaler used by it.

This node can be executed with the following command:

```bash
python3 left_knee_node_ros.py
```

## **[Right Hand Node](Right_Hand_Distributed/)**  
This local node handles data acquisition from the **right-hand-mounted** sensor, including accelerometer and gyroscope signals, executes the corresponding neural network for activity classification, and extracts quaternion data for visualization in the graphical user interface (GUI).

The folder contains the scripts to run the central neural network, the .pth files of the neural network model weights along with the scaler used by it.

This node can be executed with the following command:

```bash
python3 right_hand_node_ros.py
```

## **[Right Knee Node](Right_Knee_Distributed/)**  
This local node handles data acquisition from the **right-knee-mounted** sensor, including quaternion signals, executes the corresponding neural network for activity classification and send data for visualization in the graphical user interface (GUI).

The folder contains the scripts to run the central neural network, the .pth files of the neural network model weights along with the scaler used by it.

This node can be executed with the following command:

```bash
python3 right_knee_node_ros.py
```
