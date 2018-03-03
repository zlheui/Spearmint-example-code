language: PYTHON
name: "gaussian-train"

variable {
 name: "batch_size"
 type: ENUM
 size: 1
 options:  "8"
 options:  "16"
 options:  "32"
 options:  "64"
 options:  "128"
}

variable {
 name: "dropout"
 type: ENUM
 size: 2
 options: "0.3"
 options: "0.5"
 options: "0.8"
}

variable {
 name: "weight_decay"
 type: ENUM
 size: 1
 options: "0.001"
 options: "0.0001"
 options: "0.00001"
}

variable {
 name: "init_std"
 type: ENUM
 size: 1
 options: "0.1"
 options: "0.05"
 options: "0.01"
 options: "0.005"
}

variable {
 name: "lr"
 type: ENUM
 size: 1
 options: "1"
 options: "0.5"
 options: "0.1"
 options: "0.05"
 options: "0.01"
 options: "0.005"
 options: "0.001"
 options: "0.0005"
 options: "0.0001"
}

variable {
 name: "momentum"
 type: ENUM
 size: 1
 options: "0.5"
 options: "0.9"
}

variable {
 name: "activation"
 type: ENUM
 size: 1
 options: "relu"
 options: "tanh"
 options: "sigmoid"
}