# Cloth Animation with Time-dependent Persistent Wrinkles
In our Eurographics 2025 paper <em>Cloth Animation with Time-dependent Persistent Wrinkles</em>, we propose a noval method for simulating the cloth wrinkles' time-dependence obversed on the real clothes.

![image](/images/teaser.jpg)


## Compiling

The program was developed by using x64 MSVC (19.36.32537) and C++ 20 standard. It also needs the extra libraries listed below:

- CUDA
- [Eigen](https://eigen.tuxfamily.org/)
- [Alglib](https://www.alglib.net/)
- [JSON-cpp](https://github.com/open-source-parsers/jsoncpp)

## Usage

The simulation settings are defined through the JSON file [Config.json](./SimConfig/Config.json). Alter the gravity and the pin stiffness by 

    "Environment": {
        "Gravity": [ 0.0, 0.0, -9.8 ],
        "Handle_Stiffness": 1e5
    }

The cloth mesh and physical parameters (including the elastic parameters, friction parameters, and plastic parameters) are defined in

    "Cloth":{
        "MeshPath": "./Path_of_Cloth_Mesh",
        "Density": 0.1,
        "Bending": 1e-6,
        "Stretching": [ 200.0, 0.2, 200.0, 20.0 ],
        "Damping": 0.0
        ...
    }

The obstacles, e.g., human body, are defined in 

    "Obstacles":[
        {
            "MeshPath": "./Path_of_Obstacle_Mesh",
            ...
        }
    ]

## Cite

    @article{
        title={Cloth Animation with Time-dependent Persistent Wrinkles},
        author={Deshan, Gong and Ying, Yang, and Tianjia, Shao and He, Wang},
        year={2025}
    }

