# HAR-Distributed-Framework
HAR Distributed System for the Classification of a Finite Set of Human Activities

<img 
  width="15906" 
  height="9843"
  alt="Architecture of the distributed wearable computing framework for real-time human activity recognition" 
  src="https://github.com/user-attachments/assets/fb047dbf-3620-473f-a589-7da16560f825" 
  title="Distributed Computing Wearable System Overview" 
  />

## 1. **[Central Node Model](Central_Node_Model)**  
This node is responsible for executing the central neural network that combines all the probabilities from each of the sensors used as input for the MLP and generates the overall classification of the activity performed. It also has the script responsible for communicating with the GUI.

2. **[Chest Distributed](Chest_Distributed)**  
This local node handles data acquisition from the **chest-mounted** sensor, including accelerometer and gyroscope signals, executes the corresponding neural network for activity classification, and extracts quaternion data for visualization in the graphical user interface (GUI).

3. **[Left Hand Node](Left_Hand_Distributed/)**  
This local node handles data acquisition from the **left-hand-mounted** sensor, including quaternion signals, executes the corresponding neural network for activity classification and send data for visualization in the graphical user interface (GUI).

4. **[Left Knee Node](Left_Knee_Distributed/)**  
This local node handles data acquisition from the **left-knee-mounted** sensor, including accelerometer and gyroscope signals, executes the corresponding neural network for activity classification, and extracts quaternion data for visualization in the graphical user interface (GUI).

5. **[Right Hand Node](Right_Hand_Distributed/)**  
This local node handles data acquisition from the **right-hand-mounted** sensor, including accelerometer and gyroscope signals, executes the corresponding neural network for activity classification, and extracts quaternion data for visualization in the graphical user interface (GUI).

6. **[Right Knee Node](Right_Knee_Distributed/)**  
This local node handles data acquisition from the **right-knee-mounted** sensor, including quaternion signals, executes the corresponding neural network for activity classification and send data for visualization in the graphical user interface (GUI).