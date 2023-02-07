# PTX-EMU
> under development

PTX-EMU is a simple emulator for [PTX](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html).

You can use it to generate image below like on GPU.

![](assets/pic/output.bmp)

# dependence
- cmake 
- make 
- cuda
- gcc

# Usage

## Set up env

```
. setup
```

## Run test
run full test
```
make test
```
run single test
```
# make <name of benchmark>
make RAY
```

## Run single program
After setting up the env, just run it.
```
./bin/RAY 256 256
```

# Mode

## release (default)
Fast execution time
```
make lib
```

## debug
Used with gdb
```
make Dlib
```

## step
Similar to gdb mode.
```
make Slib
```
